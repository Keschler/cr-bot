"""Reproducible V4 controlled learning-curve experiments.

This module is deliberately a thin experiment harness around the validated V4
PPO and frontier evaluation APIs.  It owns the experiment protocol (seeds,
dataset separation, paired statistics, artifact hashes, and checkpoint
bookkeeping); it does not own PPO or simulator semantics.

The public :func:`run_learning_curve` function accepts injectable hooks.  The
default hooks call :mod:`simulator.rl.v4.v4_ppo` and
:mod:`simulator.rl.v4.frontier_ppo`, while tests and small protocol checks can
use a tiny fake backend without loading a simulator model or frontier cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np


LEARNING_CURVE_SCHEMA_VERSION = "v4-learning-curve-v1"
LEARNING_CURVE_VERSION = "v4-controlled-learning-curve-6"
# ``learning_curve.py`` now lives at ``simulator/rl/v4``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNNER_PATH = Path(__file__).resolve()
_CLI_PATH = _REPO_ROOT / "scripts" / "run_v4_learning_curve.py"
_RUNTIME_SOURCE_PATHS = (
    _RUNNER_PATH,
    _CLI_PATH,
    _REPO_ROOT / "simulator" / "rl" / "v4" / "v4_ppo",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "frontier_ppo",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "frontier.py",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "raw_ppo.py",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "policy_opponent.py",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "real_decks.py",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "real_decks.json",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "frontier_eval.py",
    _REPO_ROOT / "scripts" / "run_v4_raw_ppo_starts.py",
    _REPO_ROOT / "scripts" / "run_v4_random_actor.py",
    _REPO_ROOT / "simulator" / "rl" / "v4" / "model_v4.py",
    _REPO_ROOT / "simulator" / "rl" / "foundation" / "opponent_pool.py",
    _REPO_ROOT / "simulator" / "public_state_estimator.py",
    _REPO_ROOT / "simulator" / "observation_v3.py",
)
DEFAULT_UPDATE_SCHEDULE: tuple[int, ...] = (0, 1, 2, 4, 8)
DEFAULT_EVAL_SEED_START = 10_000
DEFAULT_EVAL_MATCHES = 24
DEFAULT_EVAL_MAX_DECISIONS = 1200
DEFAULT_ROLLOUT_DECISIONS = 128
DEFAULT_ROLLOUT_LANES = 8
DEFAULT_EPISODE_CAP = 128
# The V4 rollout env advances one decision every 250 ms. ``reward_mode=terminal``
# can only pay when an episode actually reaches a match outcome, so the cap must
# cover a whole regulation period; anything shorter yields a zero reward and a
# zero advantage on every step.
DEFAULT_DECISION_INTERVAL_US = 250_000
MIN_TERMINAL_BUDGET_US = 180_000_000


def _json_default(value: Any) -> Any:
    """Convert common experiment values to deterministic JSON values."""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        # Full observations are intentionally never placed in provenance, but
        # this makes small fake hook reports easy to serialize.
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return stable JSON used for config and generated-artifact hashes."""

    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a path-independent identity record for a local artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def object_fingerprint(value: Any) -> str:
    """Hash a small generated object without relying on object identity."""

    try:
        payload = canonical_json(value).encode("utf-8")
    except (TypeError, ValueError):
        payload = repr(value).encode("utf-8")
    return sha256_bytes(payload)


def _git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        value = "unknown"
    return value or "unknown"


def _git_output(*args: str) -> str:
    """Return a git query result without making provenance collection fatal."""

    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def dirty_source_fingerprint(
    paths: Sequence[str | os.PathLike[str]] = _RUNTIME_SOURCE_PATHS,
) -> dict[str, Any]:
    """Fingerprint source bytes and the dirty git state used by a run.

    ``git rev-parse HEAD`` alone is insufficient for an experiment started
    from a dirty checkout.  The source files are hashed directly so modified
    and untracked runner/runtime files are covered, while the worktree and
    index patches make the dirty state explicit in the provenance record.
    """

    source_paths: set[Path] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        source_paths.update(path.rglob("*.py") if path.is_dir() else (path,))
    resolved_paths = tuple(sorted(source_paths, key=str))
    file_hashes: dict[str, str] = {}
    for path in resolved_paths:
        if not path.is_file():
            continue
        try:
            relative = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            relative = str(path)
        file_hashes[relative] = sha256_file(path)

    relative_paths = []
    for path in resolved_paths:
        try:
            relative_paths.append(str(path.relative_to(_REPO_ROOT)))
        except ValueError:
            relative_paths.append(str(path))
    status = _git_output(
        "status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths
    )
    worktree_diff = _git_output("diff", "--binary", "--", *relative_paths)
    index_diff = _git_output("diff", "--cached", "--binary", "--", *relative_paths)
    payload = {
        "git_revision": _git_revision(),
        "status": status,
        "worktree_diff": worktree_diff,
        "index_diff": index_diff,
        "files": file_hashes,
    }
    return {
        "sha256": object_fingerprint(payload),
        "dirty": bool(status.strip() or worktree_diff or index_diff),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
        "files": file_hashes,
    }


def _runtime_provenance(runtime_version: str) -> dict[str, Any]:
    """Collect runtime versions, observation contract identity, and source hashes."""

    versions: dict[str, Any] = {"v4_runtime": str(runtime_version)}
    try:
        from .v4_ppo import V4_RUNTIME_VERSION

        versions["v4_ppo"] = str(V4_RUNTIME_VERSION)
    except (ImportError, AttributeError):
        versions["v4_ppo"] = "unavailable"
    try:
        from .frontier_ppo import FRONTIER_PPO_VERSION

        versions["frontier_ppo"] = str(FRONTIER_PPO_VERSION)
    except (ImportError, AttributeError):
        versions["frontier_ppo"] = "unavailable"
    try:
        from simulator.public_state_estimator import PUBLIC_ESTIMATOR_VERSION

        versions["public_state_estimator"] = str(PUBLIC_ESTIMATOR_VERSION)
    except (ImportError, AttributeError):
        versions["public_state_estimator"] = "unavailable"
    try:
        from simulator.observation_v3 import (
            OBSERVATION_V3_CONTRACT_HASH,
            OBSERVATION_V3_SCHEMA_VERSION,
        )

        versions["observation_v3_schema"] = str(OBSERVATION_V3_SCHEMA_VERSION)
        versions["observation_v3_contract_hash"] = str(OBSERVATION_V3_CONTRACT_HASH)
    except (ImportError, AttributeError):
        versions["observation_v3_schema"] = "unavailable"
        versions["observation_v3_contract_hash"] = "unavailable"
    source = dirty_source_fingerprint()
    return {
        "versions": versions,
        "source_files": source["files"],
        "dirty_source_fingerprint": source,
    }


def _set_reproducible_seed(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, and torch when available; return what was seeded."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # These flags are harmless on CPU and make the protocol explicit.  A
        # caller can still select a backend/device for which deterministic
        # kernels are unavailable; the report records the request.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch_seeded = True
    except (ImportError, RuntimeError):
        pass
    return {"python": seed, "numpy": seed % (2**32), "torch": torch_seeded}


def _phase_seed(seed: int, phase: str, index: int = 0) -> int:
    """Derive an independent deterministic RNG stream for one protocol phase."""

    digest = hashlib.sha256(
        canonical_json({"seed": int(seed), "phase": str(phase), "index": int(index)}).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % (2**31 - 1)


def _normalise_schedule(schedule: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(v) for v in schedule)
    if not values:
        raise ValueError("update_schedule must contain at least update 0")
    if values[0] != 0:
        values = (0,) + values
    if any(v < 0 for v in values):
        raise ValueError("update_schedule values must be non-negative")
    if tuple(sorted(set(values))) != values:
        raise ValueError("update_schedule must be strictly increasing")
    return values


def parse_update_schedule(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse ``0,1,2,4`` style schedules, always including baseline 0."""

    if isinstance(value, str):
        if not value.strip():
            raise ValueError("update schedule cannot be empty")
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        values = [int(part) for part in value]
    return _normalise_schedule(values)


@dataclass(frozen=True, slots=True)
class LearningCurveConfig:
    """All knobs that affect one controlled curve.

    ``rollout_decisions`` and ``rollout_lanes`` are protocol fields rather
    than incidental implementation values.  To compare either duration,
    rollout length, or lane count, run the same config twice while changing
    one field and retain both reports.
    """

    checkpoint: Path | None = None
    train_entries: Path | None = None
    training_source: str = "frontier"
    sealed_entries: Path | None = None
    output_dir: Path = Path("outputs/v4/learning_curve")
    update_schedule: tuple[int, ...] = DEFAULT_UPDATE_SCHEDULE
    seed: int = 0
    eval_seed_start: int = DEFAULT_EVAL_SEED_START
    eval_matches: int = DEFAULT_EVAL_MATCHES
    eval_max_decisions: int = DEFAULT_EVAL_MAX_DECISIONS
    eval_seeds: tuple[int, ...] | None = None
    rollout_decisions: int = DEFAULT_ROLLOUT_DECISIONS
    rollout_lanes: int = DEFAULT_ROLLOUT_LANES
    episode_cap: int = DEFAULT_EPISODE_CAP
    choice_replicas: int = 8
    choice_horizon: int = 128
    choice_adaptive_replicas: bool = False
    choice_max_replicas: int = 32
    choice_se_k: float = 2.0
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    bc_coef: float = 0.05
    ppo_epochs: int = 1
    device: str = "cpu"
    deterministic_policy: bool = False
    reward_mode: str = "tower-delta"
    smoke: bool = False
    # A JSON-compatible ModelConfigV4 field mapping.  ``None`` selects the
    # checked-in frontier architecture through the real backend.  A free-form
    # model-size label is intentionally not accepted: model comparisons must
    # carry the dimensions that actually construct the policy.
    model_config: Mapping[str, Any] | None = None
    comparison_checkpoints: tuple[Path, ...] = ()
    opponent_specs: tuple[dict[str, Any], ...] | None = None
    evaluation_only: bool = False
    allow_scripted_frontier_entries: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "train_entries", "sealed_entries", "output_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(
            self,
            "comparison_checkpoints",
            tuple(Path(value) for value in self.comparison_checkpoints),
        )
        object.__setattr__(self, "update_schedule", _normalise_schedule(self.update_schedule))
        if self.eval_seeds is not None:
            object.__setattr__(self, "eval_seeds", tuple(int(v) for v in self.eval_seeds))
        for name in (
            "eval_matches",
            "eval_max_decisions",
            "rollout_decisions",
            "rollout_lanes",
            "episode_cap",
            "choice_replicas",
            "choice_horizon",
            "choice_max_replicas",
            "ppo_epochs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.eval_seeds is not None and not self.eval_seeds:
            raise ValueError("eval_seeds cannot be empty")
        for name in ("actor_lr", "critic_lr", "gamma", "gae_lambda", "bc_coef", "choice_se_k"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not self.device:
            raise ValueError("device cannot be empty")
        if self.reward_mode not in ("terminal", "tower-delta", "truncation"):
            raise ValueError("reward_mode must be terminal, tower-delta, or truncation")
        if self.reward_mode == "terminal" and not self.smoke:
            # ``terminal`` pays the env's sparse outcome only, and that outcome
            # arrives when a match ends. An episode is cut at ``episode_cap``
            # decisions, so the cap has to span a whole match or the reward is
            # identically zero and the advantages with it. The default cap of 128
            # decisions at the 250 ms decision cadence is 32 s against a 180 s
            # regulation, so this combination trained nothing while passing
            # configuration validation.
            budget_us = self.episode_cap * DEFAULT_DECISION_INTERVAL_US
            if budget_us < MIN_TERMINAL_BUDGET_US:
                raise ValueError(
                    f"reward_mode=terminal needs an episode long enough to finish a match: "
                    f"episode_cap={self.episode_cap} at {DEFAULT_DECISION_INTERVAL_US} us per decision "
                    f"covers {budget_us} us, below the {MIN_TERMINAL_BUDGET_US} us minimum. "
                    f"Raise episode_cap to at least "
                    f"{-(-MIN_TERMINAL_BUDGET_US // DEFAULT_DECISION_INTERVAL_US)}, or use "
                    f"reward_mode=truncation or tower-delta, which pay on a cut episode"
                )
        if self.training_source not in ("frontier", "match-start"):
            raise ValueError("training_source must be frontier or match-start")
        if self.training_source == "match-start" and self.bc_coef != 0.0:
            raise ValueError("match-start reward-only PPO requires bc_coef=0")
        if self.model_config is not None and not isinstance(self.model_config, Mapping):
            raise ValueError("model_config must be a JSON object or None")
        if type(self.allow_scripted_frontier_entries) is not bool:
            raise ValueError("allow_scripted_frontier_entries must be boolean")

    def resolved_eval_seeds(self) -> tuple[int, ...]:
        if self.eval_seeds is not None:
            return self.eval_seeds
        return tuple(self.eval_seed_start + i for i in range(self.eval_matches))

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["checkpoint"] = str(self.checkpoint) if self.checkpoint is not None else None
        result["train_entries"] = str(self.train_entries) if self.train_entries is not None else None
        result["sealed_entries"] = str(self.sealed_entries) if self.sealed_entries is not None else None
        result["output_dir"] = str(self.output_dir)
        result["comparison_checkpoints"] = [str(p) for p in self.comparison_checkpoints]
        result["eval_seeds"] = list(self.eval_seeds) if self.eval_seeds is not None else None
        result["model_config"] = (
            _serialise(self.model_config) if self.model_config is not None else None
        )
        if self.opponent_specs is not None:
            result["opponent_specs"] = list(self.opponent_specs)
        return result


@dataclass(slots=True)
class TrainingState:
    """Mutable state held between frontier rollouts and PPO updates."""

    critic: Any = None
    actor_optimizer: Any = None
    critic_optimizer: Any = None


@dataclass(slots=True)
class LearningCurveHooks:
    """Backend boundary used by :func:`run_learning_curve`.

    All callbacks are keyword-friendly.  A callback may return ordinary
    dictionaries, dataclass reports, or lightweight test values; the runner
    serializes them at the artifact boundary.
    """

    load_policy: Callable[..., Any]
    clone_policy: Callable[[Any], Any]
    make_training_state: Callable[..., TrainingState]
    collect_rollout: Callable[..., Any]
    update: Callable[..., Any]
    evaluate_matches: Callable[..., Sequence[dict[str, Any]]]
    evaluate_choice: Callable[..., Sequence[dict[str, Any]]] | None = None
    save_checkpoint: Callable[..., Any] | None = None
    runtime_version: str = "unknown"


# Short alias for callers that prefer the experiment terminology.
ExperimentHooks = LearningCurveHooks


class _SmokePolicy:
    """Tiny randomly initialized policy used only by ``--smoke``.

    It intentionally has no simulator semantics.  Its vector is enough to
    prove seed handling, update/checkpoint plumbing, and paired reporting
    without implying any game quality.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng_seed = int(seed)
        self.weights = np.random.default_rng(self.rng_seed).normal(0.0, 0.1, size=8)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"smoke_weights": np.asarray(self.weights, dtype=np.float32).copy()}


def smoke_hooks(seed: int = 0) -> LearningCurveHooks:
    """Return a deterministic lightweight backend for protocol smoke runs."""

    def load_policy(*, path: Any = None, config: LearningCurveConfig | None = None, **_: Any) -> _SmokePolicy:
        return _SmokePolicy(int(seed))

    def clone_policy(policy: _SmokePolicy) -> _SmokePolicy:
        return copy.deepcopy(policy)

    def make_training_state(**_: Any) -> TrainingState:
        return TrainingState()

    def collect(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"update": int(kwargs.get("update", 0)), "seed": int(kwargs.get("seed", 0))},
            {"episodes": 1, "smoke": True},
        )

    def update(**kwargs: Any) -> dict[str, Any]:
        policy: _SmokePolicy = kwargs["policy"]
        update_index = int(kwargs.get("update", 0))
        rng = np.random.default_rng(int(seed) + update_index * 1_000_003)
        policy.weights += rng.normal(0.0, 0.01, size=policy.weights.shape)
        return {"smoke_update": update_index, "quarantined": False}

    def matches(**kwargs: Any) -> list[dict[str, Any]]:
        policy: _SmokePolicy = kwargs["policy"]
        records: list[dict[str, Any]] = []
        policy_signal = float(policy.weights.mean())
        for spec in kwargs["opponent_specs"]:
            for match_seed in kwargs["seeds"]:
                match_seed = int(match_seed)
                # One explicit censored match exercises the report contract.
                if match_seed % 11 == 0:
                    records.append({
                        "archetype": spec.get("archetype", "smoke"),
                        "seed": match_seed,
                        "result": "truncated",
                        "decisions": int(kwargs["config"].eval_max_decisions),
                    })
                    continue
                value = policy_signal + ((match_seed % 7) - 3) * 0.01
                result = "win" if value >= 0 else "loss"
                records.append({
                    "archetype": spec.get("archetype", "smoke"),
                    "seed": match_seed,
                    "result": result,
                    "decisions": 4,
                })
        return records

    def choices(**kwargs: Any) -> list[dict[str, Any]]:
        policy: _SmokePolicy = kwargs["policy"]
        champion: _SmokePolicy = kwargs["champion"]
        delta = float(policy.weights.mean() - champion.weights.mean())
        return [
            {
                "entry_id": entry.get("entry_id", f"smoke-{index}"),
                "same_action": abs(delta) < 1e-12,
                "delta": round(delta, 6),
                "paired_mean": round(delta, 6),
                "paired_se": None,
                "paired_deltas": [round(delta, 6)],
                "sign": "same" if abs(delta) < 1e-12 else ("improved" if delta > 0 else "regressed"),
            }
            for index, entry in enumerate(kwargs["entries"])
        ]

    return LearningCurveHooks(
        load_policy=load_policy,
        clone_policy=clone_policy,
        make_training_state=make_training_state,
        collect_rollout=collect,
        update=update,
        evaluate_matches=matches,
        evaluate_choice=choices,
        save_checkpoint=_default_save_checkpoint,
    )


def _coerce_model_config(spec: Any) -> Any:
    """Construct the real V4 model config from a dataclass or JSON mapping."""

    if spec is None:
        from .frontier import frontier_model_config

        return frontier_model_config()
    try:
        from .model_v4 import ModelConfigV4
    except ImportError:
        return spec
    if isinstance(spec, ModelConfigV4):
        return spec
    if not isinstance(spec, Mapping):
        raise ValueError("model_config must be a ModelConfigV4 or JSON object")
    allowed = {field.name for field in fields(ModelConfigV4)}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"unknown V4 model config fields: {unknown}")
    return ModelConfigV4(**dict(spec))


def _model_config_payload(spec: Any) -> dict[str, Any] | None:
    """Return a serializable record of the dimensions used to build a policy."""

    if spec is None:
        try:
            return _serialise(asdict(_coerce_model_config(None)))
        except (ImportError, RuntimeError, TypeError, ValueError):
            return {"preset": "frontier"}
    if is_dataclass(spec):
        return _serialise(asdict(spec))
    if isinstance(spec, Mapping):
        return _serialise(dict(spec))
    return {"repr": repr(spec)}


def _default_hooks() -> LearningCurveHooks:
    """Build lazy real-simulator hooks, keeping torch optional at import."""

    from .frontier import frontier_model_config
    from .frontier_ppo import (
        collect_frontier_rollout,
        full_match_record,
        sealed_choice_regret,
    )
    from .model_v4 import V4ValueHead
    from .v4_ppo import load_frozen_anchor, update_v4_ppo
    try:
        from . import v4_ppo as _v4_runtime_module

        runtime_version = str(getattr(_v4_runtime_module, "V4_RUNTIME_VERSION", "legacy"))
    except ImportError:
        runtime_version = "legacy"

    def load_policy(path: Path, *, model_config: Any, device: str) -> Any:
        config = _coerce_model_config(model_config)
        policy = load_frozen_anchor(str(path), config, device=device)
        # Newer V4 runtimes make the fixed WAIT semantics explicit.  Keep the
        # harness compatible with the pre-runtime API used by older checkouts.
        try:
            from .v4_ppo import configure_v4_runtime

            configure_v4_runtime(policy)
        except (ImportError, AttributeError):
            pass
        policy.train()
        for parameter in policy.parameters():
            parameter.requires_grad_(True)
        return policy

    def clone_policy(policy: Any) -> Any:
        clone = copy.deepcopy(policy)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone

    def make_training_state(policy: Any, *, model_config: Any, config: LearningCurveConfig) -> TrainingState:
        import torch

        model = _coerce_model_config(model_config)
        critic = V4ValueHead(model.gru_hidden_dim).to(config.device)
        actor_optimizer = torch.optim.Adam(policy.parameters(), lr=config.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
        return TrainingState(critic, actor_optimizer, critic_optimizer)

    def collect(**kwargs: Any) -> Any:
        config: LearningCurveConfig = kwargs.pop("config")
        return collect_frontier_rollout(
            kwargs["policy"], kwargs["critic"], kwargs["anchor"], kwargs["entries"],
            n_decisions=config.rollout_decisions,
            n_lanes=config.rollout_lanes,
            episode_cap=config.episode_cap,
            device=config.device,
            deterministic_policy=False,
            seed=kwargs["seed"],
            truncation_reward=config.reward_mode == "truncation",
            step_tower_reward=config.reward_mode == "tower-delta",
        )

    def update(**kwargs: Any) -> Any:
        config: LearningCurveConfig = kwargs.pop("config")
        return update_v4_ppo(
            kwargs["policy"], kwargs["critic"], kwargs["actor_optimizer"],
            kwargs["critic_optimizer"], kwargs["batch"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            bc_coef=config.bc_coef,
            ppo_epochs=config.ppo_epochs,
        )

    def matches(**kwargs: Any) -> Sequence[dict[str, Any]]:
        config: LearningCurveConfig = kwargs.pop("config")
        return full_match_record(
            policy=kwargs["policy"], opponent_specs=kwargs["opponent_specs"],
            seeds=kwargs["seeds"], device=config.device,
            deterministic=config.deterministic_policy,
            max_decisions=config.eval_max_decisions,
        )

    def choices(**kwargs: Any) -> Sequence[dict[str, Any]]:
        config: LearningCurveConfig = kwargs.pop("config")
        if not kwargs["entries"]:
            return []
        return sealed_choice_regret(
            policy=kwargs["policy"], champion=kwargs["champion"],
            entries=kwargs["entries"], n_replicas=config.choice_replicas,
            horizon=config.choice_horizon, device=config.device,
            adaptive_replicas=config.choice_adaptive_replicas,
            max_replicas=config.choice_max_replicas, se_k=config.choice_se_k,
        )

    return LearningCurveHooks(
        load_policy=load_policy,
        clone_policy=clone_policy,
        make_training_state=make_training_state,
        collect_rollout=collect,
        update=update,
        evaluate_matches=matches,
        evaluate_choice=choices,
        save_checkpoint=_default_save_checkpoint,
        runtime_version=runtime_version,
    )


def _default_save_checkpoint(
    *, policy: Any, critic: Any, path: Path, metadata: Mapping[str, Any]
) -> None:
    """Write a plain actor state dict plus an optional critic sidecar.

    Keeping the actor file as a plain state dict preserves compatibility with
    ``load_frozen_anchor`` and makes each curve point independently usable.
    Metadata lives in the curve report, so no checkpoint promotion is needed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        if hasattr(policy, "state_dict"):
            torch.save(
                {
                    key: value.detach().cpu().clone() if hasattr(value, "detach") else value
                    for key, value in policy.state_dict().items()
                },
                path,
            )
        else:
            path.write_text(canonical_json(policy), encoding="utf-8")
        if critic is not None and hasattr(critic, "state_dict"):
            critic_path = path.with_name(path.stem + ".critic.pt")
            torch.save(
                {
                    key: value.detach().cpu().clone() if hasattr(value, "detach") else value
                    for key, value in critic.state_dict().items()
                },
                critic_path,
            )
    except ImportError:
        state = policy.state_dict() if hasattr(policy, "state_dict") else policy
        path.write_text(canonical_json(state) + "\n", encoding="utf-8")


def _load_entries(path: Path | None, *, label: str, required: bool) -> tuple[list[Any], dict[str, Any]]:
    if path is None:
        if required:
            raise FileNotFoundError(f"{label} entries are required; pass the corresponding path")
        return [], {"source": None, "count": 0, "sha256": None}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} entries file does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix in (".json", ".jsonl"):
        if suffix == ".jsonl":
            payload = [json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
    else:
        try:
            import torch

            payload = torch.load(resolved, map_location="cpu", weights_only=False)
        except ImportError as exc:
            raise RuntimeError(f"cannot load {label} entries without PyTorch: {resolved}") from exc
    if isinstance(payload, Mapping):
        for key in ("entries", "train", "sealed", "data", "records"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"{label} entries must be a list/tuple, got {type(payload).__name__}")
    return list(payload), {
        **artifact_fingerprint(resolved),
        "count": len(payload),
    }


def _nested_value(entry: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Find a provenance field in the common frontier-entry containers."""

    containers: list[Mapping[str, Any]] = [entry]
    for name in ("provenance", "config", "origin", "match", "opponent"):
        value = entry.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
    return None


def _entry_state_identity(entry: Mapping[str, Any]) -> str | None:
    value = _nested_value(
        entry,
        ("state_hash", "initial_state_hash", "state_fingerprint", "state_id"),
    )
    if value is not None:
        return str(value)
    # Synthetic fixtures sometimes carry only the event-inclusive snapshot.
    # Hashing it gives the same strict state identity without trusting entry_id.
    initial_state = entry.get("initial_state")
    if initial_state is not None:
        return object_fingerprint(initial_state)
    return None


def _entry_match_origin(entry: Mapping[str, Any]) -> str | None:
    value = _nested_value(
        entry,
        (
            "match_origin",
            "match_id",
            "source_match_id",
            "match_seed",
            "seed",
            "origin_id",
        ),
    )
    if value is None:
        return None
    return canonical_json(value)


def _canonical_cards(value: Any) -> tuple[str, ...] | None:
    """Return a stable deck identity from a sequence of card names."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    cards = tuple(str(card) for card in value)
    if not cards or any(not card for card in cards):
        return None
    return tuple(sorted(cards))


def _real_deck_catalog() -> tuple[dict[str, Any], ...]:
    """Load the checked-in catalog lazily for split validation."""

    from .real_decks import load_real_decks

    return tuple(
        {
            "deck_id": str(deck.deck_id),
            "split": str(deck.split),
            "archetype": str(deck.archetype),
            "cards": tuple(sorted(str(card) for card in deck.cards)),
        }
        for deck in load_real_decks()
    )


def _catalog_deck_maps() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    rows = _real_deck_catalog()
    return (
        {row["deck_id"]: row for row in rows},
        {row["cards"]: row for row in rows},
    )


def _entry_deck_info(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Extract explicit and catalog-derived opponent deck identity."""

    explicit_id = _nested_value(
        entry, ("deck_id", "opponent_deck_id", "real_deck_id")
    )
    cards_value = _nested_value(
        entry,
        (
            "opponent_deck",
            "opponent_deck_cards",
            "deck_cards",
            "deck",
            "cards",
        ),
    )
    declared_split = _nested_value(
        entry, ("real_deck_split", "deck_split", "opponent_deck_split")
    )
    if explicit_id is None and cards_value is None and declared_split is None:
        return {
            "identities": set(),
            "canonical_identity": None,
            "deck_ids": set(),
            "cards": None,
            "declared_split": None,
            "catalog_split": None,
            "catalog_id": None,
        }
    cards = _canonical_cards(cards_value)
    by_id, by_cards = _catalog_deck_maps()
    id_row = by_id.get(str(explicit_id)) if explicit_id is not None else None
    cards_row = by_cards.get(cards) if cards is not None else None
    if explicit_id is not None and id_row is None and cards_row is not None:
        raise ValueError(
            f"deck ID {explicit_id!r} does not match the catalog card identity "
            f"{cards_row['deck_id']}"
        )
    if id_row is not None and cards is not None and cards != id_row["cards"]:
        raise ValueError(
            f"deck ID {explicit_id!r} does not match its supplied card list"
        )
    catalog_row = id_row or cards_row
    identities: set[str] = set()
    if explicit_id is not None:
        identities.add(f"id:{explicit_id}")
    if cards is not None:
        identities.add("cards:" + canonical_json(cards))
    if catalog_row is not None:
        identities.add(f"id:{catalog_row['deck_id']}")
        identities.add("cards:" + canonical_json(catalog_row["cards"]))
    catalog_split = catalog_row["split"] if catalog_row is not None else None
    if declared_split is not None and str(declared_split) not in ("train", "eval"):
        raise ValueError(f"invalid real-deck split identity: {declared_split!r}")
    if declared_split is not None and catalog_row is None:
        raise ValueError(
            "real-deck rows require a catalog deck ID or catalog-matching card list"
        )
    if (
        declared_split is not None
        and catalog_split is not None
        and str(declared_split) != catalog_split
    ):
        raise ValueError(
            f"deck split identity {declared_split!r} disagrees with catalog deck "
            f"{catalog_row['deck_id']} ({catalog_split})"
        )
    canonical_identity = (
        f"id:{catalog_row['deck_id']}"
        if catalog_row is not None
        else (
            f"id:{explicit_id}"
            if explicit_id is not None
            else "cards:" + canonical_json(cards)
        )
    )
    return {
        "identities": identities,
        "canonical_identity": canonical_identity,
        "deck_ids": {
            identity[3:]
            for identity in identities
            if identity.startswith("id:")
        },
        "cards": cards,
        "declared_split": str(declared_split) if declared_split is not None else None,
        "catalog_split": catalog_split,
        "catalog_id": catalog_row["deck_id"] if catalog_row is not None else None,
    }


def _opponent_specs_deck_info(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _entry_deck_info(spec)


def _catalog_ids(split: str) -> set[str]:
    if split not in ("train", "eval"):
        raise ValueError(f"unknown catalog split: {split}")
    return {
        row["deck_id"] for row in _real_deck_catalog() if row["split"] == split
    }


def _split_identities(
    entries: Sequence[Any],
    *,
    label: str,
    strict: bool,
    require_deck_identity: bool = False,
    require_catalog_identity: bool = False,
) -> dict[str, Any]:
    """Validate and summarize state/origin/deck identities for one split."""

    states: dict[str, int] = {}
    origins: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    decks: dict[str, int] = {}
    deck_ids: dict[str, int] = {}
    deck_splits: dict[str, int] = {}
    missing: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            if strict:
                raise ValueError(f"{label} entry {index} must be an object")
            continue
        state = _entry_state_identity(raw)
        origin = _entry_match_origin(raw)
        deck_info = _entry_deck_info(raw)
        entry_decks = (
            {deck_info["canonical_identity"]}
            if deck_info["canonical_identity"] is not None
            else set()
        )
        decks.update({identity: decks.get(identity, 0) + 1 for identity in entry_decks})
        for deck_id in deck_info["deck_ids"]:
            deck_ids[deck_id] = deck_ids.get(deck_id, 0) + 1
        if deck_info["catalog_split"] is not None:
            split = str(deck_info["catalog_split"])
            deck_splits[split] = deck_splits.get(split, 0) + 1
        elif deck_info["declared_split"] is not None:
            split = str(deck_info["declared_split"])
            deck_splits[split] = deck_splits.get(split, 0) + 1
        missing_fields = [
            name
            for name, value in (
                ("state", state),
                ("match_origin", origin),
                ("deck", entry_decks),
                (
                    "catalog_deck",
                    deck_info["catalog_id"] if require_catalog_identity else True,
                ),
            )
            if value is None or value == set()
        ]
        if (
            state is None
            or origin is None
            or (require_deck_identity and not entry_decks)
            or (require_catalog_identity and deck_info["catalog_id"] is None)
        ):
            missing.append({
                "index": index,
                "entry_id": raw.get("entry_id"),
                "missing": missing_fields,
            })
            continue
        states[state] = states.get(state, 0) + 1
        origins[origin] = origins.get(origin, 0) + 1
        pair = (state, origin)
        pairs[pair] = pairs.get(pair, 0) + 1
    duplicate_states = sorted(key for key, count in states.items() if count > 1)
    duplicate_pairs = sorted(
        [list(key) for key, count in pairs.items() if count > 1]
    )
    if strict and missing:
        missing_detail = ", ".join(
            f"{row['index']}:{'/'.join(row['missing'])}"
            for row in missing
        )
        raise ValueError(
            f"{label} entries require state_hash/initial_state and match origin provenance; "
            f"missing at indices {[row['index'] for row in missing]} ({missing_detail})"
        )
    if strict and duplicate_states:
        raise ValueError(
            f"{label} entries contain duplicate states ({len(duplicate_states)}); "
            "deduplicate frontier states before training/evaluation"
        )
    if strict and duplicate_pairs:
        raise ValueError(f"{label} entries contain duplicate state/origin pairs")
    if strict and label in ("train", "sealed") and deck_splits.get("eval", 0):
        raise ValueError(
            f"{label} development entries contain {deck_splits['eval']} "
            "reserved evaluation-deck identities"
        )
    return {
        "states": set(states),
        "origins": set(origins),
        "pairs": set(pairs),
        "decks": set(decks),
        "deck_ids": set(deck_ids),
        "deck_splits": dict(deck_splits),
        "count": len(entries),
        "identified_count": len(states),
        "identified_deck_count": len(decks),
        "missing": missing,
        "duplicate_state_count": len(duplicate_states),
        "duplicate_pair_count": len(duplicate_pairs),
    }


def validate_entry_split(
    train_entries: Sequence[Any],
    sealed_entries: Sequence[Any],
    *,
    strict: bool = True,
    require_deck_identity: bool = False,
    require_catalog_identity: bool = False,
) -> dict[str, Any]:
    """Enforce disjoint frontier states, origins, and deck identities.

    Multiple states from one match are valid within a split.  A state itself
    may appear only once, and no source match origin may feed both train and
    sealed data.  Development diagnostics may reuse a training-catalog deck
    identity across those disjoint matches; reserved evaluation decks may not
    enter either development split.  ``require_catalog_identity`` is enabled
    by the real comparison runner so arbitrary deck IDs cannot cross its
    split boundary; direct callers may leave it disabled for synthetic data.
    """

    train = _split_identities(
        train_entries,
        label="train",
        strict=strict,
        require_deck_identity=require_deck_identity,
        require_catalog_identity=require_catalog_identity,
    )
    sealed = _split_identities(
        sealed_entries,
        label="sealed",
        strict=strict,
        require_deck_identity=require_deck_identity,
        require_catalog_identity=require_catalog_identity,
    )
    shared_states = sorted(train["states"] & sealed["states"])
    shared_origins = sorted(train["origins"] & sealed["origins"])
    shared_decks = sorted(train["decks"] & sealed["decks"])
    if strict and shared_states:
        raise ValueError(
            f"train/sealed overlap in {len(shared_states)} state identities"
        )
    if strict and shared_origins:
        raise ValueError(
            f"train/sealed share {len(shared_origins)} match origins; sealed origins must be held out"
        )
    # Development train and sealed frontier diagnostics may intentionally use
    # different matches from the same training-catalog deck.  State and match
    # origin identities remain disjoint; train entries are separately barred
    # from the reserved evaluation catalog above.
    return {
        "train": {
            key: (sorted(value) if isinstance(value, set) else value)
            for key, value in train.items()
            if key not in ("pairs",)
        },
        "sealed": {
            key: (sorted(value) if isinstance(value, set) else value)
            for key, value in sealed.items()
            if key not in ("pairs",)
        },
        "shared_state_count": len(shared_states),
        "shared_origin_count": len(shared_origins),
        "shared_deck_count": len(shared_decks),
        "strict": strict,
        "require_deck_identity": require_deck_identity,
        "require_catalog_identity": require_catalog_identity,
    }


def _serialise(value: Any) -> Any:
    """Best-effort conversion for reports, including torch dataclasses."""

    if is_dataclass(value):
        return _serialise(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        result = value.item()
        return None if isinstance(result, float) and not np.isfinite(result) else result
    if hasattr(value, "detach"):
        try:
            return _serialise(value.detach().cpu().tolist())
        except Exception:
            pass
    return repr(value)


def _capture_pre_update_state(
    *,
    policy: Any,
    critic: Any,
    actor_optimizer: Any,
    critic_optimizer: Any,
) -> dict[str, Any] | None:
    """Copy the parameters and optimizer state an update would overwrite.

    Quarantine diagnostics are computed after ``optimizer.step()``, so a run
    that trips one has already moved its weights. This snapshot lets the loop
    undo the update instead of halting with a policy that failed its own checks.

    Any component a hook did not supply is left out rather than dereferenced:
    the hooks boundary permits a caller to run without a critic or an
    optimizer, and a missing component is nothing to restore.
    """

    snapshot: dict[str, Any] = {}
    for name, component in (
        ("policy", policy),
        ("critic", critic),
        ("actor_optimizer", actor_optimizer),
        ("critic_optimizer", critic_optimizer),
    ):
        state_dict = getattr(component, "state_dict", None)
        if callable(state_dict):
            snapshot[name] = copy.deepcopy(state_dict())
    return snapshot or None


def _restore_pre_update_state(
    *,
    policy: Any,
    critic: Any,
    actor_optimizer: Any,
    critic_optimizer: Any,
    snapshot: Mapping[str, Any] | None,
) -> None:
    """Restore a snapshot captured by :func:`_capture_pre_update_state`."""

    if not snapshot:
        return
    for name, component in (
        ("policy", policy),
        ("critic", critic),
        ("actor_optimizer", actor_optimizer),
        ("critic_optimizer", critic_optimizer),
    ):
        if name not in snapshot:
            continue
        load_state_dict = getattr(component, "load_state_dict", None)
        if callable(load_state_dict):
            load_state_dict(snapshot[name])


def _quarantine_status(report: Any) -> tuple[bool, list[str]]:
    """Read the V4 update quarantine contract from a dataclass or mapping."""

    if is_dataclass(report):
        report = asdict(report)
    if not isinstance(report, Mapping):
        return False, []
    quarantined = bool(report.get("quarantined", False))
    reasons = report.get("quarantine_reasons", report.get("reasons", []))
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, (list, tuple)):
        reasons = [repr(reasons)]
    return quarantined, [str(reason) for reason in reasons]


def _invoke(callback: Callable[..., Any], **kwargs: Any) -> Any:
    """Call hooks with useful kwargs while allowing minimal fake signatures."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(**kwargs)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return callback(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return callback(**accepted)


def _match_result(record: Mapping[str, Any], *, max_decisions: int) -> tuple[str, bool]:
    """Normalize one match result while keeping truncation distinct from draw."""

    raw = str(record.get("result", "")).lower().strip()
    if raw in {"timeout", "censored", "unfinished", "incomplete"}:
        raw = "truncated"
    cap_reached = False
    decisions = record.get("decisions")
    if decisions is not None:
        try:
            cap_reached = int(decisions) >= int(max_decisions)
        except (TypeError, ValueError):
            cap_reached = False
    explicit_truncated = bool(record.get("truncated", False)) or bool(record.get("cap_reached", False))
    explicit_terminated = record.get("terminated") is True or record.get("completed") is True
    if raw == "truncated" or explicit_truncated:
        return "truncated", False
    # Some older evaluators emitted ``draw`` with winner=None at the hard
    # decision cap.  A capped, non-terminated row is censored, never a draw.
    if raw == "draw" and cap_reached and not explicit_terminated and record.get("winner") is None:
        return "truncated", False
    if raw in {"win", "loss", "draw"}:
        return raw, True
    winner = record.get("winner")
    if winner == 0:
        return "win", True
    if winner == 1:
        return "loss", True
    # Older frontier evaluators returned winner=None/result=draw at the hard
    # cap.  Infer censoring from the explicit decision budget instead of
    # silently turning an unfinished match into a draw.
    if cap_reached:
        return "truncated", False
    return "draw", True


def _score(result: str) -> float | None:
    return {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(result)


def _record_key(record: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    seed = _nested_value(record, ("seed", "match_seed"))
    seed_present = seed is not None
    if not seed_present:
        # Never use list position or archetype as an implicit pairing key.  In
        # particular, two openings from one archetype can have different
        # hidden state while exposing the same human-readable label.
        return ("__invalid_pair_identity__", "missing-seed", index)

    if seed is None:
        return ("__invalid_pair_identity__", "missing-seed", index)

    # ``full_match_record`` includes ``deck_id=None`` for scripted opponents,
    # so the presence of that optional field alone does not make a row a real
    # catalog row.  A real-deck marker with no catalog ID is ambiguous: the
    # same archetype/seed can still refer to different openings.
    real_deck_row = any(
        _nested_value(record, (key,)) is not None
        for key in (
            "real_deck_split",
            "real_deck_id",
            "opponent_deck_id",
            "deck_profile",
        )
    ) or _nested_value(record, ("profile",)) == "real"
    deck_id = _nested_value(record, ("deck_id", "opponent_deck_id", "real_deck_id"))
    if real_deck_row and deck_id is None:
        return ("__invalid_pair_identity__", "missing-deck-id", index)

    # Keep the established report shape for valid rows: callers may inspect
    # these keys in saved learning-curve artifacts.
    opponent = record.get("opponent")
    if isinstance(opponent, (Mapping, list, tuple, set, frozenset)):
        opponent = canonical_json(opponent)
    return (record.get("archetype"), deck_id, seed, opponent)


def paired_match_statistics(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    max_decisions: int,
) -> dict[str, Any]:
    """Pair match outcomes by opponent/seed and report censoring uncertainty.

    A truncated match is censored: it contributes to the truncation count and
    paired metadata but never becomes a draw or a score delta.
    """

    if not baseline_records or not candidate_records:
        raise ValueError(
            "paired evaluation requires non-empty baseline and candidate records"
        )

    def normalise(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], list[int]]]:
        rows: list[dict[str, Any]] = []
        by_key: dict[tuple[Any, ...], list[int]] = {}
        for index, raw in enumerate(records):
            row = dict(raw)
            key = _record_key(row, index)
            if key and key[0] == "__invalid_pair_identity__":
                reason = key[1]
                if reason == "missing-deck-id":
                    raise ValueError(
                        "paired evaluation real-deck records require an explicit deck ID"
                    )
                raise ValueError(
                    "paired evaluation records must include an explicit seed identity"
                )
            result, completed = _match_result(row, max_decisions=max_decisions)
            row["result"] = result
            row["completed"] = completed
            row["score"] = _score(result)
            rows.append(row)
            by_key.setdefault(key, []).append(index)
        return rows, by_key

    base, base_keys = normalise(baseline_records)
    cand, cand_keys = normalise(candidate_records)
    duplicate_base = {key for key, indices in base_keys.items() if len(indices) > 1}
    duplicate_cand = {key for key, indices in cand_keys.items() if len(indices) > 1}
    if duplicate_base or duplicate_cand:
        raise ValueError(
            "paired evaluation contains duplicate match identities: "
            f"baseline={len(duplicate_base)}, candidate={len(duplicate_cand)}"
        )
    missing_candidate = set(base_keys) - set(cand_keys)
    candidate_only = set(cand_keys) - set(base_keys)
    if missing_candidate or candidate_only:
        raise ValueError(
            "paired evaluation match identities mismatch: "
            f"missing_candidate={len(missing_candidate)}, candidate_only={len(candidate_only)}"
        )

    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts = {label: sum(1 for row in rows if row["result"] == label) for label in ("win", "loss", "draw", "truncated")}
        scores = [float(row["score"]) for row in rows if row["score"] is not None]
        mean = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "n": len(rows),
            "wins": counts["win"],
            "losses": counts["loss"],
            "draws": counts["draw"],
            "truncated": counts["truncated"],
            "completed": len(scores),
            "score_mean_completed": mean,
            "records": rows,
        }

    paired: list[dict[str, Any]] = []
    diffs: list[float] = []
    diff_seeds: list[Any] = []
    used_candidate: set[int] = set()
    for base_index, base_row in enumerate(base):
        key = _record_key(base_row, base_index)
        candidates = cand_keys.get(key, [])
        candidate_index = next((idx for idx in candidates if idx not in used_candidate), None)
        if candidate_index is None:
            paired.append({"key": list(key), "comparable": False, "status": "missing_candidate"})
            continue
        used_candidate.add(candidate_index)
        cand_row = cand[candidate_index]
        comparable = base_row["score"] is not None and cand_row["score"] is not None
        delta = (float(cand_row["score"]) - float(base_row["score"])) if comparable else None
        if delta is not None:
            diffs.append(delta)
            # A seed is shared across the deck pool in the controlled final
            # protocol.  Keep it as the uncertainty cluster instead of
            # pretending same-seed deck outcomes are independent replicas.
            diff_seeds.append(key[2])
        paired.append({
            "key": list(key),
            "baseline_result": base_row["result"],
            "candidate_result": cand_row["result"],
            "comparable": comparable,
            "status": "paired" if comparable else "truncated",
            "score_delta": delta,
        })
    n = len(diffs)
    n_censored = sum(1 for row in paired if not row["comparable"])
    if n == 0:
        raise ValueError(
            "paired evaluation produced zero completed pairs; "
            "all matched outcomes are censored"
        )
    mean = sum(diffs) / n if n else 0.0
    naive_std = float("inf")
    naive_se = float("inf")
    if n >= 2:
        import statistics

        naive_std = statistics.stdev(diffs)
        naive_se = naive_std / (n**0.5)
    else:
        # One completed pair has an observed delta but no estimable sampling
        # uncertainty.  Zero would falsely turn a singleton into a narrow
        # confidence interval.
        naive_std = naive_se = float("inf")
    by_seed: dict[Any, list[float]] = {}
    for seed, delta in zip(diff_seeds, diffs):
        by_seed.setdefault(seed, []).append(delta)
    cluster_values = list(by_seed.values())
    cluster_means = [sum(values) / len(values) for values in cluster_values]
    cluster_counts = [len(values) for values in cluster_values]
    cluster_weighted_mean = sum(
        cluster_mean * count
        for cluster_mean, count in zip(cluster_means, cluster_counts)
    ) / n
    if len(cluster_means) >= 2 and n_censored == 0:
        import statistics
        import math

        cluster_std = statistics.stdev(cluster_means)
        # Weight each independent match-seed cluster by its observed number
        # of deck rows.  This keeps the estimand equal to the completed-pair
        # mean even when the deck counts differ by seed, while the variance
        # still has one independent contribution per seed cluster.
        total = float(sum(cluster_counts))
        proportions = [count / total for count in cluster_counts]
        weighted_variance = (
            len(cluster_means) / (len(cluster_means) - 1)
        ) * sum(
            proportion * proportion * (cluster_mean - cluster_weighted_mean) ** 2
            for proportion, cluster_mean in zip(proportions, cluster_means)
        )
        se = math.sqrt(weighted_variance)
    else:
        cluster_std = float("inf")
        se = float("inf")
    if n_censored:
        # A censored pair contributes an unknown score delta in [-1, 1].
        # Keep the observed completed-pair mean for diagnostics, but expose a
        # finite censoring interval and suppress inferential SE/claims.
        total_pairs = n + n_censored
        censoring_lower = (sum(diffs) - n_censored) / total_pairs
        censoring_upper = (sum(diffs) + n_censored) / total_pairs
        naive_se = float("inf")
    else:
        total_pairs = n
        censoring_lower = censoring_upper = mean
    inference_valid = n_censored == 0 and len(cluster_means) >= 2
    return {
        "baseline": summary(base),
        "candidate": summary(cand),
        "paired": paired,
        "paired_deltas": diffs,
        "n_paired": n,
        "n_censored": n_censored,
        "paired_mean_score_delta": mean,
        "paired_mean_score_delta_completed": mean,
        "paired_std_score_delta": naive_std,
        "paired_se_score_delta": se,
        "paired_se_score_delta_naive": naive_se,
        "paired_cluster_count": len(cluster_means),
        "paired_cluster_mean_score_delta": (
            sum(cluster_means) / len(cluster_means)
            if cluster_means else None
        ),
        "paired_cluster_weighted_mean_score_delta": cluster_weighted_mean,
        "paired_censoring_bounds": {
            "lower": censoring_lower,
            "upper": censoring_upper,
            "n_total_pairs": total_pairs,
        },
        "paired_inference_valid": inference_valid,
        "paired_inference_reason": (
            "complete match-seed clusters"
            if inference_valid
            else (
                "censored pairs; score-delta bounds are descriptive"
                if n_censored
                else "fewer than two match-seed clusters"
            )
        ),
        "paired_uncertainty_scope": "match-seed-clustered",
    }


def _fallback_choice_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("delta", row.get("paired_mean", 0.0))) for row in rows]
    return {
        "n": len(values),
        "mean_delta": sum(values) / len(values) if values else 0.0,
        "n_changed": sum(1 for row in rows if not row.get("same_action", False)),
    }


def _choice_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from .frontier_ppo import summarize_choice_regret

        return summarize_choice_regret(rows)
    except (ImportError, KeyError, TypeError, ValueError):
        return _fallback_choice_summary(rows)


def _default_opponents() -> tuple[dict[str, Any], ...]:
    """Return the development opponent pool from the training catalog."""

    from .frontier import _stable_seed
    from .real_decks import REAL_DECK_ARCHETYPES, real_decks_for

    return tuple(
        {
            "archetype": deck.archetype,
            "strategy": deck.archetype,
            "deck": list(deck.cards),
            "deck_id": deck.deck_id,
            "source_url": deck.source_url,
            "real_deck_split": "train",
            "controller_seed": _stable_seed("real-train", deck.deck_id) & 0xFFFFFFFF,
        }
        for archetype in REAL_DECK_ARCHETYPES
        for deck in real_decks_for("train", archetype)
    )


def _default_final_opponents() -> tuple[dict[str, Any], ...]:
    """Return the reserved catalog pool used only by final evaluation."""

    from .frontier_ppo.environment import real_eval_opponent_decks

    return tuple(dict(value) for value in real_eval_opponent_decks())


def _validate_opponent_specs(
    specs: Sequence[Mapping[str, Any]], *, final: bool, smoke: bool = False
) -> dict[str, Any]:
    """Require explicit deck identities and enforce catalog split boundaries."""

    if not specs:
        raise ValueError("evaluation requires at least one opponent deck")
    if smoke:
        return {"deck_ids": [], "identities": [], "split": "synthetic"}
    eval_ids = _catalog_ids("eval")
    train_ids = _catalog_ids("train")
    identities: set[str] = set()
    deck_ids: set[str] = set()
    unknown: list[int] = []
    wrong_split: list[int] = []
    duplicate: set[str] = set()
    for index, raw in enumerate(specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"opponent spec {index} must be an object")
        info = _opponent_specs_deck_info(raw)
        if not info["identities"]:
            raise ValueError(
                f"opponent spec {index} requires an explicit deck identity"
            )
        before = len(identities)
        identities.update(info["identities"])
        if len(identities) == before:
            duplicate.update(info["identities"])
        deck_ids.update(info["deck_ids"])
        catalog_split = info["catalog_split"]
        declared_split = info["declared_split"]
        if final:
            if (
                info["catalog_id"] is None
                or catalog_split != "eval"
                or not (info["deck_ids"] & eval_ids)
            ):
                wrong_split.append(index)
            if declared_split is not None and declared_split != "eval":
                wrong_split.append(index)
        else:
            if info["deck_ids"] & eval_ids or catalog_split == "eval":
                wrong_split.append(index)
            if declared_split == "eval":
                wrong_split.append(index)
            if info["catalog_id"] is None or not (info["deck_ids"] & train_ids):
                unknown.append(index)
    if unknown:
        raise ValueError(
            "opponent specs contain deck IDs outside the pinned real-deck catalog: "
            f"indices {sorted(set(unknown))}"
        )
    if wrong_split:
        required = "evaluation" if final else "development/train"
        raise ValueError(
            f"opponent specs must use reserved {required} deck identities; "
            f"invalid indices {sorted(set(wrong_split))}"
        )
    if duplicate:
        raise ValueError(
            "opponent specs contain duplicate deck identities: "
            f"{sorted(duplicate)}"
        )
    return {
        "deck_ids": sorted(deck_ids),
        "identities": sorted(identities),
        "split": "eval" if final else "train",
    }


def _validate_match_records(
    records: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    label: str,
    final: bool,
    smoke: bool = False,
) -> None:
    """Fail closed if an evaluator drops the deck/seed pairing metadata."""

    if smoke:
        return
    expected_info = _validate_opponent_specs(specs, final=final)
    expected_ids = set(expected_info["deck_ids"])
    if not records:
        raise ValueError(f"{label} evaluation returned no match records")
    observed: set[tuple[str, int]] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} match record {index} must be an object")
        seed = _nested_value(raw, ("seed", "match_seed"))
        deck_id = _nested_value(raw, ("deck_id", "opponent_deck_id", "real_deck_id"))
        if seed is None:
            raise ValueError(f"{label} match record {index} is missing explicit seed identity")
        if deck_id is None:
            raise ValueError(
                f"{label} match record {index} is missing explicit deck ID identity"
            )
        deck_id = str(deck_id)
        if deck_id not in expected_ids:
            raise ValueError(
                f"{label} match record {index} deck ID {deck_id!r} is outside its opponent split"
            )
        key = (deck_id, int(seed))
        if key in observed:
            raise ValueError(f"{label} evaluation contains duplicate deck/seed identities")
        observed.add(key)
    expected = {(str(deck_id), int(seed)) for deck_id in expected_ids for seed in seeds}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} evaluation deck/seed identities mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def select_development_checkpoint(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select only from complete development full-match outcomes.

    Sealed/frontier choice diagnostics are deliberately ignored.  The helper
    accepts serialized curve points so a selection report can be audited
    independently of the training process.
    """

    candidates: list[dict[str, Any]] = []
    for point in points:
        evaluation = point.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        full_match = evaluation.get("full_match")
        if not isinstance(full_match, Mapping):
            continue
        candidate = full_match.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        baseline = full_match.get("baseline")
        baseline_truncated = (
            int(baseline.get("truncated", 0) or 0)
            if isinstance(baseline, Mapping)
            else 0
        )
        completed = int(candidate.get("completed", 0) or 0)
        truncated = int(candidate.get("truncated", 0) or 0)
        # Checkpoint selection has no valid winner interpretation when a
        # point contains censored matches.  A later point with more wins can
        # simply have exposed a different truncation pattern.
        if truncated or baseline_truncated:
            raise ValueError(
                "development checkpoint selection refuses censored full-match points"
            )
        if completed < 1:
            continue
        score = candidate.get("score_mean_completed")
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        update = int(point.get("update", 0))
        cost_value = point.get("cumulative_training_cost", point.get("training_cost"))
        try:
            cost = float(cost_value) if cost_value is not None else float("inf")
        except (TypeError, ValueError):
            cost = float("inf")
        candidates.append({
            "update": update,
            "checkpoint": point.get("checkpoint"),
            "score_mean_completed": score,
            "completed": completed,
            "cumulative_training_cost": None if not np.isfinite(cost) else cost,
        })
    if not candidates:
        raise ValueError(
            "development checkpoint selection requires at least one completed full-match point"
        )
    candidates.sort(
        key=lambda row: (
            -float(row["score_mean_completed"]),
            float("inf") if row["cumulative_training_cost"] is None else float(row["cumulative_training_cost"]),
            int(row["update"]),
        )
    )
    return {
        "performed": True,
        "scope": "development-complete-matches",
        "metric": "full_match.candidate.score_mean_completed",
        "excludes": ["sealed_choice", "reserved_eval_decks", "truncated_points"],
        "candidates": candidates,
        "selected_update": candidates[0]["update"],
        "selected_checkpoint": candidates[0]["checkpoint"],
    }


def _validate_input_paths(config: LearningCurveConfig) -> None:
    if config.smoke:
        pass
    else:
        if config.checkpoint is None:
            raise FileNotFoundError(
                "V4 learning-curve training needs --checkpoint. The historical "
                "simx12 checkpoint is not present in this checkout; no improvement "
                "claim can be fabricated. Pass a real checkpoint or use --smoke."
            )
        if not config.checkpoint.expanduser().is_file():
            raise FileNotFoundError(
                f"V4 checkpoint does not exist: {config.checkpoint}. The historical "
                "simx12 checkpoint is absent; pass a supplied checkpoint or use --smoke."
            )
        if config.train_entries is None and not config.evaluation_only:
            raise FileNotFoundError("training requires --train-entries (or use --smoke)")
    for path in config.comparison_checkpoints:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"comparison checkpoint does not exist: {path}")


def _validate_final_input_paths(config: LearningCurveConfig) -> None:
    """Validate the frozen-actor-only final protocol inputs."""

    if config.smoke:
        raise ValueError("final evaluation cannot use the synthetic smoke protocol")
    if config.allow_scripted_frontier_entries:
        raise ValueError(
            "final evaluation cannot use --allow-scripted-frontier-entries"
        )
    if config.evaluation_only:
        raise ValueError("final evaluation requires a normal frozen actor checkpoint")
    if config.checkpoint is None:
        raise FileNotFoundError(
            "final evaluation needs --checkpoint (the frozen actor to evaluate)"
        )
    if not config.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"final-evaluation checkpoint does not exist: {config.checkpoint}")
    if config.train_entries is not None or config.sealed_entries is not None:
        raise ValueError(
            "final evaluation accepts frozen actors only; omit --train-entries and --sealed-entries"
        )
    for path in config.comparison_checkpoints:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"comparison checkpoint does not exist: {path}")


def _validate_output_dir(path: Path) -> Path:
    """Resolve a run directory and fail closed if it already contains output."""

    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"learning-curve output is not a directory: {resolved}")
    if resolved.is_dir() and any(resolved.iterdir()):
        raise FileExistsError(
            f"learning-curve output already exists and is non-empty: {resolved}; "
            "choose a new --out directory"
        )
    return resolved


def _load_policy_for_config(
    hooks: LearningCurveHooks,
    config: LearningCurveConfig,
    *,
    model_config: Any = None,
    checkpoint: Path | None = None,
) -> Any:
    checkpoint = config.checkpoint if checkpoint is None else checkpoint
    return _invoke(
        hooks.load_policy,
        path=checkpoint,
        model_config=model_config,
        device=config.device,
        config=config,
    )


def _freeze_policy(policy: Any) -> Any:
    """Put a loaded final actor in inference mode without requiring Torch."""

    if hasattr(policy, "eval"):
        policy.eval()
    parameters = getattr(policy, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            requires_grad = getattr(parameter, "requires_grad_", None)
            if callable(requires_grad):
                requires_grad_(False)
    return policy


def _save_point(
    hooks: LearningCurveHooks,
    *,
    policy: Any,
    state: TrainingState,
    output_dir: Path,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_dir / f"checkpoint_update_{update:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint artifact: {path}")
    if hooks.save_checkpoint is None:
        _default_save_checkpoint(policy=policy, critic=state.critic, path=path, metadata=metadata)
    else:
        _invoke(
            hooks.save_checkpoint,
            policy=policy,
            critic=state.critic,
            path=path,
            metadata=dict(metadata),
        )
    if not path.is_file():
        raise RuntimeError(f"checkpoint hook did not create {path}")
    return artifact_fingerprint(path)


def _load_runtime_entries(config: LearningCurveConfig) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]:
    train, train_meta = _load_entries(
        config.train_entries, label="train", required=not config.evaluation_only and not config.smoke
    )
    if config.training_source == "match-start" and not config.evaluation_only and not config.smoke:
        from .raw_ppo import RAW_PPO_STARTS_VERSION
        from .real_decks import catalog_sha256, real_decks_for

        if not train:
            raise ValueError("match-start PPO requires at least one opening entry")
        for index, entry in enumerate(train):
            provenance = entry.get("provenance") if isinstance(entry, Mapping) else None
            initial_state = entry.get("initial_state") if isinstance(entry, Mapping) else None
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("generator") != RAW_PPO_STARTS_VERSION
                or not isinstance(initial_state, Mapping)
                or initial_state.get("tick") != 0
                or "verdict" in entry
            ):
                raise ValueError(
                    f"train entry {index} is not a search-free raw PPO match opening"
                )
            if (
                provenance.get("profile") == "real"
                and entry.get("opponent", {}).get("kind") != "v4-checkpoint"
            ):
                deck_id = provenance.get("real_deck_id")
                archetype = entry.get("opponent", {}).get("archetype")
                valid = (
                    provenance.get("real_deck_catalog_sha256") == catalog_sha256()
                    and any(
                        deck.deck_id == deck_id
                        and tuple(entry.get("opponent_deck", ())) == deck.cards
                        for deck in real_decks_for("train", archetype)
                    )
                )
                if not valid:
                    raise ValueError(
                        f"train entry {index} does not match the pinned real-deck train catalog"
                    )
    sealed, sealed_meta = _load_entries(
        config.sealed_entries, label="sealed", required=False
    )
    return train, sealed, train_meta, sealed_meta


def run_learning_curve(
    config: LearningCurveConfig,
    *,
    hooks: LearningCurveHooks | None = None,
    model_config: Any = None,
) -> dict[str, Any]:
    """Run one controlled curve and write its artifacts.

    The baseline at update zero and every later point use identical
    development opponent specs and fresh evaluation seeds.  Training entries
    are passed only to the collector.  Optional sealed entries are secondary
    diagnostics and never feed checkpoint selection. ``evaluation_only``
    evaluates one supplied checkpoint and performs no rollout/update calls.
    """

    from simulator.viewer.recording import publish_progress

    output_dir = _validate_output_dir(config.output_dir)
    _validate_input_paths(config)
    hooks = hooks or _default_hooks()
    seed_info = _set_reproducible_seed(config.seed)
    train_entries, sealed_entries, train_meta, sealed_meta = _load_runtime_entries(config)
    if config.smoke:
        if not train_entries:
            train_entries = [
                {"entry_id": f"smoke-train-{index}", "synthetic": True}
                for index in range(4)
            ]
            train_meta = {
                "source": "generated-smoke",
                "count": len(train_entries),
                "sha256": object_fingerprint(train_entries),
            }
        if not sealed_entries:
            sealed_entries = [
                {"entry_id": f"smoke-sealed-{index}", "synthetic": True}
                for index in range(3)
            ]
            sealed_meta = {
                "source": "generated-smoke",
                "count": len(sealed_entries),
                "sha256": object_fingerprint(sealed_entries),
            }
    split_meta = validate_entry_split(
        train_entries,
        sealed_entries,
        strict=not config.smoke,
        require_deck_identity=not config.smoke,
        require_catalog_identity=(
            not config.smoke and not config.allow_scripted_frontier_entries
        ),
    )
    if config.smoke:
        opponents = tuple(config.opponent_specs or ({"archetype": "smoke"},))
    else:
        opponents = tuple(config.opponent_specs or _default_opponents())
    opponent_split_meta = _validate_opponent_specs(
        opponents, final=False, smoke=config.smoke
    )
    eval_seeds = config.resolved_eval_seeds()
    train_origins = set(split_meta["train"].get("origins", []))
    sealed_origins = set(split_meta["sealed"].get("origins", []))
    eval_origin_overlap = {
        canonical_json(seed)
        for seed in eval_seeds
        if canonical_json(seed) in train_origins | sealed_origins
    }
    if not config.smoke and eval_origin_overlap:
        raise ValueError(
            "evaluation seeds overlap train/sealed match origins; use fresh held-out seeds: "
            f"{sorted(eval_origin_overlap)}"
        )
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("evaluation seeds must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)

    viewer_progress: dict[str, Any] = {
        "source": "training", "run_id": str(output_dir.resolve()),
        "output_dir": str(output_dir.resolve()), "status": "running",
        "phase": "initializing", "current_update": 0,
        "config": config.as_dict(), "updates": {}, "points": [],
    }
    publish_progress(viewer_progress)

    resolved_model_config = model_config if model_config is not None else config.model_config
    model_config_record = _model_config_payload(resolved_model_config)

    policy = _load_policy_for_config(
        hooks,
        config,
        model_config=resolved_model_config,
    )
    baseline = hooks.clone_policy(policy)
    anchor = hooks.clone_policy(baseline)
    state = TrainingState()
    if not config.evaluation_only:
        state = _invoke(
            hooks.make_training_state,
            policy=policy,
            model_config=resolved_model_config,
            config=config,
        )
        if state is None:
            state = TrainingState()

    def evaluate(candidate: Any, *, evaluation_index: int) -> dict[str, Any]:
        # Evaluation may use torch/NumPy randomness in the simulator.  Give
        # every update point its own fixed stream so inserting an intermediate
        # curve point cannot change a later training or evaluation result.
        _set_reproducible_seed(_phase_seed(config.seed, "evaluation", evaluation_index))
        candidate_matches = list(_invoke(
            hooks.evaluate_matches,
            policy=candidate,
            opponent_specs=opponents,
            seeds=eval_seeds,
            config=config,
        ))
        _validate_match_records(
            candidate_matches,
            specs=opponents,
            seeds=eval_seeds,
            label=f"candidate update {evaluation_index}",
            final=False,
            smoke=config.smoke,
        )
        match_eval = paired_match_statistics(
            baseline_match_records,
            candidate_matches,
            max_decisions=config.eval_max_decisions,
        )
        if hooks.evaluate_choice is None or not sealed_entries:
            choice_rows: list[dict[str, Any]] = []
        else:
            choice_rows = [dict(row) for row in _invoke(
                hooks.evaluate_choice,
                policy=candidate,
                champion=baseline,
                entries=sealed_entries,
                config=config,
            )]
        return {
            "full_match": match_eval,
            "sealed_choice": {
                "rows": choice_rows,
                "summary": _choice_summary(choice_rows),
            },
        }

    # The fixed baseline is evaluated once and reused as the left side of
    # every pair.  This keeps the paired protocol stable across curve points.
    _set_reproducible_seed(_phase_seed(config.seed, "evaluation-baseline"))
    baseline_match_records = list(_invoke(
        hooks.evaluate_matches,
        policy=baseline,
        opponent_specs=opponents,
        seeds=eval_seeds,
        config=config,
    ))
    _validate_match_records(
        baseline_match_records,
        specs=opponents,
        seeds=eval_seeds,
        label="baseline",
        final=False,
        smoke=config.smoke,
    )
    baseline_choice_rows: list[dict[str, Any]] = []
    if hooks.evaluate_choice is not None and sealed_entries:
        baseline_choice_rows = [dict(row) for row in _invoke(
            hooks.evaluate_choice,
            policy=baseline,
            champion=baseline,
            entries=sealed_entries,
            config=config,
        )]

    points: list[dict[str, Any]] = []
    update_reports: dict[int, Any] = {}
    schedule = (0,) if config.evaluation_only else config.update_schedule
    current_update = 0
    quarantine_stop: dict[str, Any] | None = None
    for target_update in schedule:
        stopped = False
        while current_update < target_update:
            current_update += 1
            # Snapshot before the update so a quarantined update can be undone:
            # the diagnostics that quarantine on are measured after the step.
            pre_update_state = _capture_pre_update_state(
                policy=policy,
                critic=state.critic,
                actor_optimizer=state.actor_optimizer,
                critic_optimizer=state.critic_optimizer,
            )
            # Reset all backend RNGs from the update identity.  Evaluation
            # hooks are then unable to perturb a later update when the caller
            # changes the set of reported curve points.
            _set_reproducible_seed(_phase_seed(config.seed, "train-update", current_update))
            rollout_seed = _phase_seed(config.seed, "rollout", current_update)
            collected = _invoke(
                hooks.collect_rollout,
                policy=policy,
                critic=state.critic,
                anchor=anchor,
                entries=train_entries,
                config=config,
                update=current_update,
                seed=rollout_seed,
            )
            if isinstance(collected, tuple) and len(collected) == 2:
                batch, rollout_info = collected
            else:
                batch, rollout_info = collected, {}
            report = _invoke(
                hooks.update,
                policy=policy,
                critic=state.critic,
                actor_optimizer=state.actor_optimizer,
                critic_optimizer=state.critic_optimizer,
                batch=batch,
                config=config,
                update=current_update,
            )
            serialised_update = {
                "rollout": _serialise(rollout_info),
                "update": _serialise(report),
            }
            update_reports[current_update] = serialised_update
            viewer_progress.update(phase="update", current_update=current_update)
            viewer_progress["updates"][str(current_update)] = serialised_update
            publish_progress(viewer_progress)
            is_quarantined, reasons = _quarantine_status(report)
            if is_quarantined:
                # Quarantine is detected after the optimizer has stepped, so
                # the weights that tripped it are already in the model. Restore
                # the pre-update parameters and optimizer state before halting,
                # otherwise a quarantined run leaves a policy that failed its own
                # diagnostics in memory and in any later export from it.
                _restore_pre_update_state(
                    policy=policy,
                    critic=state.critic,
                    actor_optimizer=state.actor_optimizer,
                    critic_optimizer=state.critic_optimizer,
                    snapshot=pre_update_state,
                )
                quarantine_stop = {
                    "stopped_at_update": current_update,
                    "reasons": reasons,
                    "weights_restored": True,
                    "update_report": serialised_update,
                }
                viewer_progress.update(status="quarantined", quarantine_stop=quarantine_stop)
                publish_progress(viewer_progress)
                stopped = True
                break
        if stopped:
            break
        evaluation = evaluate(policy, evaluation_index=target_update)
        checkpoint = _save_point(
            hooks,
            policy=policy,
            state=state,
            output_dir=output_dir,
            update=target_update,
            metadata={"update": target_update, "config": config.as_dict()},
        )
        points.append({
            "update": target_update,
            "checkpoint": checkpoint,
            # The collector consumes exactly this fixed tensor budget per
            # update.  Record it explicitly so selection ties can use a
            # deterministic cumulative training cost even when hooks do not
            # provide wall-clock accounting.
            "cumulative_training_cost": int(
                current_update * config.rollout_lanes * config.rollout_decisions
            ),
            "training_cost_unit": "rollout_decisions",
            "evaluation": _serialise(evaluation),
            "updates_since_previous_point": {
                str(key): value for key, value in update_reports.items() if key <= target_update
            },
        })
        viewer_progress.update(phase="evaluation", current_update=target_update)
        viewer_progress["points"] = _serialise(points)
        publish_progress(viewer_progress)

    # Comparison checkpoints are evaluated through the same held-out hooks;
    # they are never silently reduced to a hash-only annotation.  A different
    # architecture must be supplied with a matching --model-config and is
    # usually clearer as its own output directory.
    comparisons: list[dict[str, Any]] = []
    comparison_eval_index = points[-1]["update"] if points else 0
    for path in config.comparison_checkpoints:
        comparison_policy = _load_policy_for_config(
            hooks,
            config,
            model_config=resolved_model_config,
            checkpoint=path,
        )
        comparison_eval = evaluate(
            comparison_policy,
            evaluation_index=comparison_eval_index,
        )
        versus_final = None
        if points:
            final_matches = points[-1]["evaluation"]["full_match"]["candidate"]["records"]
            comparison_matches = comparison_eval["full_match"]["candidate"]["records"]
            versus_final = {
                "curve_update": comparison_eval_index,
                "full_match": paired_match_statistics(
                    final_matches, comparison_matches,
                    max_decisions=config.eval_max_decisions,
                ),
            }
        comparisons.append({
            "artifact": artifact_fingerprint(path),
            "role": "independently trained comparison checkpoint",
            "evaluated_in_this_curve": True,
            "evaluation": _serialise(comparison_eval),
            "versus_final_curve_point": _serialise(versus_final),
        })

    if config.smoke:
        checkpoint_selection = {
            "performed": False,
            "scope": "development-complete-matches",
            "reason": "synthetic smoke run",
        }
    else:
        checkpoint_selection = select_development_checkpoint(points)

    checkpoint_meta = None
    if config.checkpoint is not None and config.checkpoint.expanduser().is_file():
        checkpoint_meta = artifact_fingerprint(config.checkpoint)
    runtime = _runtime_provenance(hooks.runtime_version)
    provenance = {
        "schema_version": LEARNING_CURVE_SCHEMA_VERSION,
        "runner_version": LEARNING_CURVE_VERSION,
        "git_revision": _git_revision(),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "runtime": runtime,
        "v4_runtime_version": hooks.runtime_version,
        "model_config": model_config_record,
        "input_checkpoint": checkpoint_meta,
        "train_entries": train_meta,
        "sealed_entries": sealed_meta,
        "generated_opponents_sha256": object_fingerprint(opponents),
        "generated_eval_seeds_sha256": object_fingerprint(eval_seeds),
        "seed": seed_info,
        "separation": {
            "train_entry_count": len(train_entries),
            "sealed_entry_count": len(sealed_entries),
            "train_state_count": len(split_meta["train"].get("states", [])),
            "sealed_state_count": len(split_meta["sealed"].get("states", [])),
            "train_deck_count": len(split_meta["train"].get("decks", [])),
            "sealed_deck_count": len(split_meta["sealed"].get("decks", [])),
            "shared_state_count": split_meta["shared_state_count"],
            "shared_origin_count": split_meta["shared_origin_count"],
            "shared_deck_count": split_meta["shared_deck_count"],
            "eval_origin_overlap_count": len(eval_origin_overlap),
            "evaluation_uses_train_entries": False,
            "evaluation_uses_fixed_fresh_seeds": True,
            "evaluation_seeds_disjoint_from_entry_origins": not eval_origin_overlap,
            "development_opponent_split": opponent_split_meta["split"],
            "development_opponent_deck_ids": opponent_split_meta["deck_ids"],
            "development_catalog_only": split_meta["require_catalog_identity"],
            "selection_excludes_sealed_entries": True,
            "selection_excludes_reserved_eval_decks": True,
        },
        "quality_claim": (
            "smoke run; randomly initialized/lightweight synthetic protocol, no quality claim"
            if config.smoke else "controlled curve only; no automatic promotion or quality claim"
        ),
    }
    config_record = config.as_dict()
    config_record["resolved_model_config"] = model_config_record
    report = {
        "schema_version": LEARNING_CURVE_SCHEMA_VERSION,
        "version": LEARNING_CURVE_VERSION,
        "config": config_record,
        "provenance": provenance,
        "protocol": {
            "training_source": config.training_source,
            "opponent_specs": _serialise(opponents),
            "eval_seeds": list(eval_seeds),
            "baseline_choice_summary": _choice_summary(baseline_choice_rows),
            "baseline_match_count": len(baseline_match_records),
            "truncation_is_not_draw": True,
            "paired_uncertainty_scope": "match-seed-clustered within this curve run",
            "checkpoint_cost_metric": (
                "cumulative_training_cost is fixed rollout_lanes * rollout_decisions "
                "per completed update"
            ),
            "reward_mode": config.reward_mode,
            "quarantine_stop": quarantine_stop,
            "api_assumptions": [
                "collect_frontier_rollout returns (rollout_batch, info)",
                "update_v4_ppo consumes that rollout batch and returns diagnostics",
                "full_match_record returns result labels win/loss/draw/truncated or winner/decisions",
                "sealed_choice_regret returns paired per-entry values when available",
            ],
        },
        "comparison_checkpoints": comparisons,
        "points": points,
        "checkpoint_selection": checkpoint_selection,
        "promotion": {"automatic": False, "selected_checkpoint": None},
    }
    report = _serialise(report)
    config_path = output_dir / "config.json"
    curve_path = output_dir / "learning_curve.json"
    provenance_path = output_dir / "provenance.json"
    config_path.write_text(canonical_json(config_record) + "\n", encoding="utf-8")
    curve_path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    provenance_path.write_text(canonical_json(provenance) + "\n", encoding="utf-8")
    report["artifacts"] = {
        "config": artifact_fingerprint(config_path),
        "curve": {"path": str(curve_path)},
        "provenance": artifact_fingerprint(provenance_path),
    }
    # Rewrite the curve once with its artifact map.  The map intentionally
    # excludes the curve's own final hash to avoid a self-referential digest.
    curve_path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    viewer_progress.update(
        status="quarantined" if quarantine_stop else "complete", phase="finished",
        report_path=str(curve_path.resolve()),
    )
    publish_progress(viewer_progress)
    return report


def run_final_evaluation(
    config: LearningCurveConfig,
    *,
    hooks: LearningCurveHooks | None = None,
    model_config: Any = None,
) -> dict[str, Any]:
    """Evaluate frozen actors on reserved evaluation decks.

    This path intentionally has no training-entry load, clone, optimizer,
    rollout, update, sealed-choice, checkpoint-save, or checkpoint-selection
    call.  The supplied checkpoint and comparison checkpoints are loaded as
    independent frozen actors and evaluated on one fixed deck/seed grid.
    """

    output_dir = _validate_output_dir(config.output_dir)
    _validate_final_input_paths(config)
    hooks = hooks or _default_hooks()
    seed_info = _set_reproducible_seed(config.seed)
    eval_seeds = config.resolved_eval_seeds()
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("final evaluation seeds must be unique")
    opponents = tuple(config.opponent_specs or _default_final_opponents())
    opponent_split_meta = _validate_opponent_specs(opponents, final=True)

    actor_paths = [config.checkpoint, *config.comparison_checkpoints]
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path in actor_paths:
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        if resolved in seen_paths:
            raise ValueError(f"final evaluation actor is listed more than once: {resolved}")
        seen_paths.add(resolved)
        resolved_paths.append(resolved)
    if not resolved_paths:
        raise ValueError("final evaluation requires at least one frozen actor")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_config = model_config if model_config is not None else config.model_config
    model_config_record = _model_config_payload(resolved_model_config)
    actor_reports: list[dict[str, Any]] = []
    actor_records: list[list[dict[str, Any]]] = []
    for actor_index, path in enumerate(resolved_paths):
        policy = _freeze_policy(
            _load_policy_for_config(
                hooks,
                config,
                model_config=resolved_model_config,
                checkpoint=path,
            )
        )
        _set_reproducible_seed(_phase_seed(config.seed, "final-evaluation", actor_index))
        records = [dict(row) for row in _invoke(
            hooks.evaluate_matches,
            policy=policy,
            opponent_specs=opponents,
            seeds=eval_seeds,
            config=config,
        )]
        _validate_match_records(
            records,
            specs=opponents,
            seeds=eval_seeds,
            label=f"final actor {actor_index}",
            final=True,
        )
        actor_records.append(records)
        actor_reports.append({
            "index": actor_index,
            "role": "primary frozen actor" if actor_index == 0 else "comparison frozen actor",
            "artifact": artifact_fingerprint(path),
            "records": _serialise(records),
        })

    pairwise: list[dict[str, Any]] = []
    pairwise_stats: list[dict[str, Any]] = []
    for baseline_index in range(len(actor_records)):
        baseline_records = actor_records[baseline_index]
        for actor_index in range(baseline_index + 1, len(actor_records)):
            stats = paired_match_statistics(
                baseline_records,
                actor_records[actor_index],
                max_decisions=config.eval_max_decisions,
            )
            pairwise_stats.append(stats)
            pairwise.append({
                "baseline_actor": baseline_index,
                "candidate_actor": actor_index,
                "winner_claim_permitted": bool(stats["paired_inference_valid"]),
                "full_match": _serialise(stats),
            })

    winner_claims_permitted = bool(pairwise_stats) and all(
        bool(stats["paired_inference_valid"])
        and int(stats["n_censored"]) == 0
        for stats in pairwise_stats
    )

    checkpoint_meta = artifact_fingerprint(config.checkpoint) if config.checkpoint else None
    runtime = _runtime_provenance(hooks.runtime_version)
    provenance = {
        "schema_version": LEARNING_CURVE_SCHEMA_VERSION,
        "runner_version": LEARNING_CURVE_VERSION,
        "git_revision": _git_revision(),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "runtime": runtime,
        "v4_runtime_version": hooks.runtime_version,
        "model_config": model_config_record,
        "input_checkpoint": checkpoint_meta,
        "generated_opponents_sha256": object_fingerprint(opponents),
        "generated_eval_seeds_sha256": object_fingerprint(eval_seeds),
        "seed": seed_info,
        "separation": {
            "final_evaluation": True,
            "final_opponent_split": opponent_split_meta["split"],
            "final_opponent_deck_ids": opponent_split_meta["deck_ids"],
            "training_entries_loaded": False,
            "sealed_entries_loaded": False,
            "checkpoint_selection_performed": False,
        },
        "quality_claim": (
            "frozen final evaluation; complete match-seed clusters support paired actor comparison"
            if winner_claims_permitted
            else (
                "frozen final evaluation; censoring or insufficient independent match-seed "
                "clusters are present, so no winner claim is permitted"
            )
        ),
    }
    config_record = config.as_dict()
    config_record["final_evaluation"] = True
    config_record["resolved_model_config"] = model_config_record
    report = {
        "schema_version": LEARNING_CURVE_SCHEMA_VERSION,
        "version": LEARNING_CURVE_VERSION,
        "final_evaluation": True,
        "config": config_record,
        "provenance": provenance,
        "protocol": {
            "opponent_specs": _serialise(opponents),
            "eval_seeds": list(eval_seeds),
            "max_decisions": config.eval_max_decisions,
            "truncation_is_not_draw": True,
            "frozen_actors_only": True,
            "selection_uses_reserved_eval_data": False,
            "winner_claims_permitted": winner_claims_permitted,
            "censoring_policy": (
                "censored pairs expose score-delta bounds and suppress inferential SE/claims"
            ),
            "paired_uncertainty_scope": (
                "match-seed-clustered within each actor pair; training-replicate "
                "aggregation remains an outer protocol step"
            ),
        },
        "actors": actor_reports,
        "pairwise": pairwise,
        "checkpoint_selection": {
            "performed": False,
            "reason": "final evaluation consumes frozen actors only",
        },
    }
    report = _serialise(report)
    config_path = output_dir / "config.json"
    final_path = output_dir / "final_evaluation.json"
    provenance_path = output_dir / "provenance.json"
    config_path.write_text(canonical_json(config_record) + "\n", encoding="utf-8")
    final_path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    provenance_path.write_text(canonical_json(provenance) + "\n", encoding="utf-8")
    report["artifacts"] = {
        "config": artifact_fingerprint(config_path),
        "final_evaluation": {"path": str(final_path)},
        "provenance": artifact_fingerprint(provenance_path),
    }
    final_path.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DEFAULT_UPDATE_SCHEDULE",
    "ExperimentHooks",
    "LEARNING_CURVE_SCHEMA_VERSION",
    "LEARNING_CURVE_VERSION",
    "LearningCurveConfig",
    "LearningCurveHooks",
    "TrainingState",
    "artifact_fingerprint",
    "canonical_json",
    "object_fingerprint",
    "paired_match_statistics",
    "parse_update_schedule",
    "run_final_evaluation",
    "run_learning_curve",
    "sha256_file",
    "select_development_checkpoint",
    "validate_entry_split",
]
