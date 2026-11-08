"""Lossless public episodes shared by the controlled tactical-learning arms.

Collection never filters observations: inactive rows and the initial prelude
remain in their original recurrent chronology. Auxiliary targets describe the
factual executed action, not a counterfactual teacher action. The V2 public
observation has no stable track IDs, so survival outcomes remain masked.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "v4-public-tactical-episodes-v1"
INPUT_KEYS = ("raster", "global_features", "entities", "entity_mask", "hand_tokens",
              "opp_hand_probs", "opp_out_of_cycle", "opp_elixir_interval", "event_history")
PUBLIC_KEYS = (*INPUT_KEYS, "legal_play", "legal_wait")
DRIVERS = ("actor", "cycle", "plans")
LABEL_KEYS = ("targets_cycle", "targets_plans")
BOOL_INPUT_KEYS = frozenset(("entity_mask", "opp_out_of_cycle", "legal_play", "legal_wait"))
_THREADS_CONFIGURED = False


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    max_decisions: int = 1440
    decision_interval_us: int = 250_000
    horizons_seconds: tuple[float, ...] = (2.0, 6.0, 12.0)

    def __post_init__(self) -> None:
        if type(self.max_decisions) is not int or not 1 <= self.max_decisions <= 1440:
            raise ValueError("max_decisions must be between 1 and 1440")
        if self.decision_interval_us != 250_000:
            raise ValueError("tactical collection requires the native 250000us cadence")
        if tuple(self.horizons_seconds) != (2.0, 6.0, 12.0):
            raise ValueError("the controlled pilot fixes consequence horizons at 2/6/12 seconds")


@dataclass(frozen=True, slots=True)
class CollectionJob:
    job_id: str
    split: str
    driver: str
    opponent: Mapping[str, Any]
    seed: int
    policy_seed: int
    episode_index: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.job_id):
            raise ValueError("job_id must be a safe filename component")
        if self.split not in ("train", "calibration") or self.driver not in DRIVERS:
            raise ValueError("unknown collection split or driver")
        if any(type(value) is not int or value < 0 for value in (self.seed, self.policy_seed, self.episode_index)):
            raise ValueError("collection seeds and episode indices must be nonnegative integers")
        _validate_opponent(self.opponent)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8], "big")


def _validate_opponent(opponent: Mapping[str, Any]) -> None:
    from ..foundation.opponent_styles import TRAIN_STYLE_IDS
    from .real_decks import load_real_decks

    train = {row.deck_id: row for row in load_real_decks() if row.split == "train"}
    deck = train.get(str(opponent.get("deck_id", "")))
    if deck is None or tuple(opponent.get("cards", ())) != deck.cards:
        raise ValueError("collection requires an exact training-catalog deck; reserved decks are excluded")
    if opponent.get("split", "train") != "train":
        raise ValueError("opponent deck must belong to the train split")
    if opponent.get("style") not in TRAIN_STYLE_IDS:
        raise ValueError("collection requires an explicit train style; dev styles are excluded")
    if opponent.get("strategy") != deck.archetype or opponent.get("archetype") != deck.archetype:
        raise ValueError("collection strategy must match the catalog archetype")
    # Custom parameters could disguise development styles as training styles.
    if any(key in opponent for key in ("style_config", "kind", "checkpoint")):
        raise ValueError("collection uses only the pinned scripted training styles")


def build_collection_jobs(
    opponents: Sequence[Any], base_seed: int, *, train_repeats: int = 1,
    train_matches: int | None = None, calibration_matches: int | None = None,
    calibration_count: int | None = None, drivers: Sequence[str] = DRIVERS,
) -> list[CollectionJob]:
    """Default: 90 train (30 decks x 3 drivers) and 30 disjoint calibration.

    Train styles rotate across deck/driver, and calibration rotates its driver
    and style independently of train seeds. Bounds permit a two-match smoke
    without silently changing the declarations for a full pilot.
    """
    from ..foundation.opponent_styles import TRAIN_STYLE_IDS

    if type(base_seed) is not int or base_seed < 0 or type(train_repeats) is not int or train_repeats < 1:
        raise ValueError("base_seed must be nonnegative and train_repeats positive")
    drivers = tuple(drivers)
    if not drivers or len(set(drivers)) != len(drivers) or any(driver not in DRIVERS for driver in drivers):
        raise ValueError("drivers must be unique members of actor/cycle/plans")
    if not opponents:
        raise ValueError("collection requires at least one training opponent")
    if calibration_count is not None:
        if calibration_matches is not None:
            raise ValueError("use only one calibration-count argument")
        calibration_matches = calibration_count
    specs = [asdict(value) if is_dataclass(value) else dict(value) for value in opponents]
    if len({value.get("deck_id") for value in specs}) != len(specs):
        raise ValueError("duplicate opponent decks")
    jobs: list[CollectionJob] = []

    def add(spec: Mapping[str, Any], split: str, driver: str, deck_index: int, repeat: int, style_index: int) -> None:
        raw = {**spec, "cards": list(spec["cards"]), "style": TRAIN_STYLE_IDS[style_index % len(TRAIN_STYLE_IDS)]}
        raw["strategy"] = raw.get("strategy") or raw["archetype"]
        seed = _stable_seed(SCHEMA, base_seed, split, raw["deck_id"], driver, repeat)
        # Style randomness belongs to this fresh match rather than a stale
        # catalog controller seed shared accidentally across pilot episodes.
        raw["controller_seed"] = _stable_seed(seed, "opponent")
        job_id = f"{split}_{driver}_{deck_index:02d}_{repeat:02d}_{raw['deck_id']}"
        jobs.append(CollectionJob(job_id, split, driver, raw, seed,
                                  _stable_seed(seed, "native-actor") & 0xFFFFFFFF, len(jobs)))

    train_candidates = len(specs) * len(drivers) * train_repeats
    for name, value, maximum in (("train_matches", train_matches, train_candidates),
                                 ("calibration_matches", calibration_matches, len(specs))):
        if value is not None and (type(value) is not int or not 0 <= value <= maximum):
            raise ValueError(f"{name} must be between zero and {maximum}")
    for repeat in range(train_repeats):
        for index, spec in enumerate(specs):
            for driver_index, driver in enumerate(drivers):
                if train_matches is None or len(jobs) < train_matches:
                    add(spec, "train", driver, index, repeat, index + driver_index + repeat)
    for index, spec in enumerate(specs[:calibration_matches]):
        add(spec, "calibration", drivers[index % len(drivers)], index, 0, index + 1)
    if len({job.seed for job in jobs}) != len(jobs):
        raise ValueError("collection seeds collided")
    return jobs


def _job(value: CollectionJob | Mapping[str, Any]) -> CollectionJob:
    return value if isinstance(value, CollectionJob) else CollectionJob(**dict(value))


def _identity_digest(identity: Mapping[str, Any]) -> str:
    value = {key: item for key, item in identity.items() if key != "identity_sha256"}
    digest = hashlib.sha256(canonical(value).encode()).hexdigest()
    if "identity_sha256" in identity and identity["identity_sha256"] != digest:
        raise ValueError("collection identity hash does not match its contents")
    return digest


def _verify_pins(identity: Mapping[str, Any], checkpoint_record: Mapping[str, Any]) -> None:
    if sha256_file(checkpoint_record["path"]) != checkpoint_record["sha256"]:
        raise ValueError("collection checkpoint hash changed")
    # A caller can pin its entire runtime through {path: sha256} without the
    # collector guessing which dirty worktree files are part of its source.
    source = identity.get("source_files", identity.get("caller", {}).get("source_files", {}))
    if not isinstance(source, Mapping):
        raise TypeError("source_files must map paths to pinned SHA256 values")
    for path, expected in source.items():
        if sha256_file(path) != expected:
            raise ValueError(f"collection source changed: {path}")


def array_digest(arrays: Mapping[str, Any], keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        array = np.asarray(arrays[key])
        if array.dtype.hasobject:
            raise ValueError("object arrays cannot cross the dataset boundary")
        digest.update(canonical({"key": key, "dtype": array.dtype.str, "shape": list(array.shape)}).encode() + b"\n")
        # memoryview avoids duplicating a full episode's raster to hash it.
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def target_array(target: Any) -> np.ndarray:
    # Teachers retain legacy duration hints; the fixed-decision simulator
    # executes exactly one native interval for every WAIT.
    if int(target.mode) == 0:
        return np.zeros(5, dtype=np.int64)
    return np.asarray((target.mode, target.card_slot, target.row, target.col, 0), dtype=np.int64)


def validate_target(target: Sequence[int], legal_play: np.ndarray, legal_wait: bool) -> None:
    mode, slot, row, col, duration = (int(value) for value in target)
    if duration != 0:
        raise ValueError("native fixed WAIT requires duration index zero")
    if mode == 0:
        if not legal_wait or (slot, row, col) != (0, 0, 0):
            raise ValueError("illegal or noncanonical WAIT target")
    elif mode == 1:
        if not (0 <= slot < 4 and 0 <= row < 32 and 0 <= col < 18 and legal_play[slot, row, col]):
            raise ValueError("illegal PLAY target")
    else:
        raise ValueError("action mode must be WAIT=0 or PLAY=1")


def active_public_mask(inputs: Mapping[str, Any]) -> tuple[bool, bool]:
    """Visible living air near our half, or a visible living enemy X-Bow."""
    from .simulator_teacher import _entity_card, saving_teacher_context

    tokens = np.asarray(inputs["entities"])
    rows = tokens[np.asarray(inputs["entity_mask"], dtype=bool)]
    air = saving_teacher_context(tokens, np.asarray(inputs["entity_mask"], dtype=bool)).air_threats > 0
    xbow = any(row[1] > .5 and row[4] > 0 and row[9] > .5 and row[10] > .5
               and _entity_card(row).get("name") == "x-bow" for row in rows)
    return bool(air), bool(xbow)


def public_measurement(inputs: Mapping[str, Any], elapsed_us: int, tower_max_hp: Sequence[float],
                       *, terminated: bool = False, truncated: bool = False) -> Any:
    """Read only the actor's public HP channels; no entity UID recovery."""
    from .tactical_outcomes import PublicTacticalMeasurement

    maximum = np.asarray(tower_max_hp, dtype=np.float64)
    if maximum.shape != (3,) or not np.isfinite(maximum).all() or np.any(maximum <= 0):
        raise ValueError("public tower caps must be three positive numbers")
    channels = np.asarray(inputs["global_features"])[12:18].astype(np.float64)
    if channels.shape != (6,) or not np.isfinite(channels).all() or np.any((channels < 0) | (channels > 1)):
        raise ValueError("public normalized tower HP channels are invalid")
    values = channels.reshape(2, 3) @ maximum / maximum.sum()
    return PublicTacticalMeasurement(elapsed_us / 1_000_000.0, float(values[0]), float(values[1]),
                                    entities=(), entities_complete=False,
                                    terminated=terminated, truncated=truncated)


def validate_episode(arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    length = int(metadata["decisions"])
    if not 1 <= length <= 1440 or not metadata["terminated"] or metadata["truncated"]:
        raise ValueError("collection must contain a complete genuine terminal episode within 1440 decisions")
    if metadata["invalid_actions"][0] != 0:
        raise ValueError("rejected learner actions contaminate the trajectory")
    if metadata["requested_learner_cards"] != metadata["accepted_learner_cards"]:
        raise ValueError("requested/accepted learner card counts differ")
    if sum(metadata["accepted_learner_cards"].values()) != metadata["card_plays"][0]:
        raise ValueError("accepted learner cards differ from engine play count")
    row_keys = (*PUBLIC_KEYS, "decision", "elapsed_us", "reset_mask", "label_valid", "active",
                "air_threat", "xbow_present", "executed_actions", "opponent_actions", *LABEL_KEYS,
                "terminated_after", "truncated_after", "outcome_values", "outcome_mask")
    if any(len(arrays[key]) != length for key in row_keys):
        raise ValueError("episode row fields do not share the complete chronology")
    public_shapes = {"raster": (21, 32, 18), "global_features": (768,), "entities": (128, 32),
                     "entity_mask": (128,), "hand_tokens": (4, 16), "opp_hand_probs": (128,),
                     "opp_out_of_cycle": (128,), "opp_elixir_interval": (2,), "event_history": (16, 8),
                     "legal_play": (4, 32, 18), "legal_wait": ()}
    for key in PUBLIC_KEYS:
        value = np.asarray(arrays[key])
        if value.shape != (length, *public_shapes[key]):
            raise ValueError(f"public field {key} has an invalid contract shape")
        expected = np.dtype(bool if key in BOOL_INPUT_KEYS else np.float32)
        if value.dtype != expected or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError(f"public field {key} must preserve exact finite {expected} data")
    if not np.array_equal(arrays["decision"], np.arange(length)):
        raise ValueError("sparse or reordered rows cannot form a recurrent episode")
    elapsed = np.asarray(arrays["elapsed_us"])
    if elapsed[0] != 0 or not np.all(np.diff(elapsed) == 250_000):
        raise ValueError("episode must start at zero and retain native 250 ms chronology")
    reset = np.arange(length) == 0
    done = np.arange(length) == length - 1
    if not np.array_equal(arrays["reset_mask"], reset) or not np.array_equal(arrays["terminated_after"], done):
        raise ValueError("only the real episode start resets and only its final action terminates")
    if np.any(arrays["truncated_after"]):
        raise ValueError("truncated trajectories are excluded")
    valid = np.asarray(arrays["label_valid"], dtype=bool)
    expected_valid = np.asarray(arrays["legal_wait"], dtype=bool) | np.asarray(arrays["legal_play"]).reshape(length, -1).any(axis=1)
    if not np.array_equal(valid, expected_valid):
        raise ValueError("label validity disagrees with the public legal masks")
    if not np.array_equal(arrays["active"], arrays["air_threat"] | arrays["xbow_present"]):
        raise ValueError("active mask must be the public air/X-Bow union")
    for key in (*LABEL_KEYS, "executed_actions"):
        targets = np.asarray(arrays[key])
        if targets.shape != (length, 5) or targets.dtype.kind not in "iu":
            raise ValueError(f"invalid action target shape/dtype: {key}")
        for row in np.flatnonzero(valid):
            validate_target(targets[row], arrays["legal_play"][row], bool(arrays["legal_wait"][row]))
        if key in LABEL_KEYS and not np.all(targets[~valid] == -1):
            raise ValueError("invalid prelude rows cannot carry fabricated labels")
        if key == "executed_actions" and not np.all(targets[~valid] == 0):
            raise ValueError("inactive prelude can execute only a canonical WAIT")
    if np.asarray(arrays["outcome_values"]).shape != (length, 3, 4) or np.asarray(arrays["outcome_mask"]).shape != (length, 3, 4):
        raise ValueError("invalid consequence target dimensions")
    if not np.isfinite(arrays["outcome_values"]).all() or np.any(arrays["outcome_mask"][:, :, 2:]):
        raise ValueError("unsupported public survival targets must remain masked")
    if len(arrays["measurement_seconds"]) != length + 1 or not bool(arrays["measurement_terminated"][-1]):
        raise ValueError("the factual labels require a public post-terminal measurement")
    if not np.array_equal(arrays["requested_card"], arrays["accepted_card"]):
        raise ValueError("requested and accepted card history differ")
    if not np.array_equal(arrays["executed_actions"][:, 0] == 1, arrays["requested_card"] != ""):
        raise ValueError("executed plays disagree with the requested card history")
    if dict(Counter(str(card) for card in arrays["accepted_card"] if card)) != metadata["accepted_learner_cards"]:
        raise ValueError("accepted card history disagrees with its counters")
    from .tactical_outcomes import PublicTacticalMeasurement, build_tactical_outcome_targets
    times = np.asarray(arrays["measurement_seconds"])
    if not np.array_equal(times[:-1], elapsed / 1_000_000.0) or not 0 < times[-1] - times[-2] <= .25 + 1e-9:
        raise ValueError("public measurements disagree with the decision chronology")
    measured = [PublicTacticalMeasurement(float(seconds), float(hp[0]), float(hp[1]),
                                         terminated=bool(terminated), truncated=bool(truncated))
                for seconds, hp, terminated, truncated in zip(times, arrays["measurement_tower_health"],
                    arrays["measurement_terminated"], arrays["measurement_truncated"], strict=True)]
    expected = build_tactical_outcome_targets(measured, list(range(length)))
    if not np.array_equal(arrays["outcome_values"], expected.values) or not np.array_equal(arrays["outcome_mask"], expected.mask):
        raise ValueError("factual consequence targets disagree with their public measurements")
    if array_digest(arrays, (*PUBLIC_KEYS, "reset_mask", "elapsed_us")) != metadata["input_digest"]:
        raise ValueError("episode public-input digest mismatch")
    if array_digest(arrays, LABEL_KEYS) != metadata["label_digest"]:
        raise ValueError("episode teacher-label digest mismatch")
    all_keys = tuple(sorted(arrays))
    if array_digest(arrays, all_keys) != metadata["arrays_digest"]:
        raise ValueError("episode action/outcome/content digest mismatch")


def load_episode(path: str | Path, *, expected_identity: str | None = None,
                 expected_sha: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError("episode file hash changed")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported tactical episode schema")
    if expected_identity is not None and metadata["identity_sha256"] != expected_identity:
        raise ValueError("episode belongs to another collection identity")
    validate_episode(arrays, metadata)
    return arrays, metadata


def _torch_runtime() -> Any:
    global _THREADS_CONFIGURED
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(1)
        # The interop setting can only be changed once in a long-lived caller.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _THREADS_CONFIGURED = True
    return torch


def _decode_target(target: Sequence[int], player: int = 0) -> Any:
    from simulator.actions import PlayCardAction, WaitAction
    return WaitAction(player) if int(target[0]) == 0 else PlayCardAction(player, int(target[1]), (int(target[3]), int(target[2])))


def _encode_action(action: Any) -> np.ndarray:
    from simulator.actions import PlayCardAction, WaitAction
    if isinstance(action, WaitAction):
        return np.zeros(5, dtype=np.int64)
    if isinstance(action, PlayCardAction):
        return np.asarray((1, action.card_slot, action.cell[1], action.cell[0], 0), dtype=np.int64)
    raise ValueError("tactical collection supports only native WAIT/PLAY actions")


def collect_episode(job: CollectionJob | Mapping[str, Any], checkpoint_record: Mapping[str, Any],
                    out: str | Path, identity: Mapping[str, Any], *,
                    config: CollectionConfig | None = None) -> dict[str, Any]:
    """Write one bounded full episode atomically; never return a censored row set."""
    from simulator.actions import PlayCardAction, action_to_dict
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from .frontier import frontier_model_config
    from .frontier_eval import make_opponent
    from .model_v4 import ModelConfigV4, masks_from_legal_play
    from .simulator_teacher import TeacherTarget, cycle_teacher_label
    from .tactical_outcomes import build_tactical_outcome_targets
    from .tactical_plans import PlanTeacherLabeler
    from .v4_ppo.runtime import batch_v4_inputs, configure_v4_runtime, decode_v4_sim_actions, load_frozen_anchor, observe_v4_inputs

    job, config, out = _job(job), config or CollectionConfig(), Path(out)
    identity_sha = _identity_digest(identity)
    _verify_pins(identity, checkpoint_record)
    ruleset = load_fixed_ruleset()
    env = SimulatorEnv(BattleEngine(ruleset, validate_every_tick=False), decision_interval_us=config.decision_interval_us)
    env.reset_v2(seed=job.seed, decks=(tuple(PLAYER_DECK), tuple(job.opponent["cards"])), shuffle_decks=True)
    opponent = make_opponent(dict(job.opponent), job.seed, env=env)
    estimator = PublicStateEstimator()
    policy = hidden = torch = None
    if job.driver == "actor":
        torch = _torch_runtime()
        raw_config = checkpoint_record.get("model_config", identity.get("model_config", identity.get("caller", {}).get("model_config")))
        model_config = ModelConfigV4(**raw_config) if raw_config else frontier_model_config()
        policy = load_frozen_anchor(str(checkpoint_record["path"]), config=model_config,
                                    expected_sha256=checkpoint_record["sha256"], device="cpu")
        configure_v4_runtime(policy)
        hidden = policy.initial_hidden(1, device="cpu")
        # Actor construction consumes RNG; seed the native stochastic stream
        # afterward to make worker assignment and resume irrelevant.
        torch.manual_seed(job.policy_seed)
    plan_teacher = PlanTeacherLabeler() if job.driver == "plans" else None
    tower_max = (ruleset.towers["princess-tower"].hitpoints,
                 ruleset.towers["king-tower"].hitpoints,
                 ruleset.towers["princess-tower"].hitpoints)
    records: dict[str, list[Any]] = {key: [] for key in (
        "decision", "elapsed_us", "reset_mask", "label_valid", "active", "air_threat", "xbow_present",
        "own_elixir", "state_hash", "executed_actions", "opponent_actions", "requested_card", "accepted_card",
        "terminated_after", "truncated_after", "plan_source", "plan_id", "plan_step", *LABEL_KEYS)}
    measurements: list[Any] = []
    requested: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    invalid_actions, card_plays = [0, 0], [0, 0]
    trace = hashlib.sha256()
    terminal = truncated = False
    winner = reason = None
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    # Public tensors live on disk while only one episode's small diagnostic
    # lists and measurements occupy memory. Compression streams the memmaps.
    with tempfile.TemporaryDirectory(prefix=".tactical-episode-", dir=out) as temporary_dir:
        public: dict[str, Any] = {}
        for decision in range(config.max_decisions):
            observation, inputs, hand, elixir = observe_v4_inputs(env, estimator)
            if decision == 0:
                for key in PUBLIC_KEYS:
                    value = np.asarray(inputs[key], dtype=bool if key in BOOL_INPUT_KEYS else np.float32)
                    public[key] = np.lib.format.open_memmap(Path(temporary_dir) / f"{key}.npy", mode="w+",
                                                            dtype=value.dtype, shape=(config.max_decisions, *value.shape))
            valid = bool(inputs["legal_wait"] or inputs["legal_play"].any())
            kwargs = dict(hand_tokens=inputs["hand_tokens"], entity_tokens=inputs["entities"],
                          entity_mask=inputs["entity_mask"], legal_play=inputs["legal_play"],
                          legal_wait=bool(inputs["legal_wait"]), own_elixir=elixir)
            teacher = plan_teacher or PlanTeacherLabeler()
            if valid:
                cycle = cycle_teacher_label(**kwargs)
                plan = teacher.label(decision_step=decision, **kwargs)
                targets = {"cycle": target_array(cycle), "plans": target_array(plan)}
                for value in targets.values():
                    validate_target(value, inputs["legal_play"], bool(inputs["legal_wait"]))
            else:
                targets = {name: np.full(5, -1, dtype=np.int64) for name in ("cycle", "plans")}
            if policy is not None:
                with torch.no_grad():
                    logits, _, hidden = policy.forward_step(batch_v4_inputs(inputs, "cpu"), hidden=hidden,
                                                            reset_mask=torch.tensor([[decision == 0]], dtype=torch.bool))
                    hidden = hidden.detach()
                    if valid:
                        if not all(bool(torch.isfinite(getattr(logits, name)).all()) for name in ("mode", "wait_duration", "card", "placement")):
                            raise FloatingPointError("nonfinite actor logits during tactical collection")
                        masks = masks_from_legal_play(torch.as_tensor(inputs["legal_play"]).reshape(1, 1, 4, 32, 18), bool(inputs["legal_wait"]))
                        sampled, joint, _ = policy.sample_action(logits, masks)
                        if not bool(torch.isfinite(joint).all()):
                            raise FloatingPointError("nonfinite sampled actor action probability")
                        action = decode_v4_sim_actions(sampled, player=0)[0]
                    else:
                        action = _decode_target(np.zeros(5, dtype=np.int64))
            else:
                action = _decode_target(targets[job.driver] if valid else np.zeros(5, dtype=np.int64))
            executed = _encode_action(action)
            if valid:
                validate_target(executed, inputs["legal_play"], bool(inputs["legal_wait"]))
            elapsed_us = int(env.state.elapsed_us)
            measurements.append(public_measurement(inputs, elapsed_us, tower_max))
            air, xbow = active_public_mask(inputs)
            for key in PUBLIC_KEYS:
                public[key][decision] = inputs[key]
            for name in ("cycle", "plans"):
                records[f"targets_{name}"].append(targets[name])
            plan_decision = getattr(teacher, "last_decision", None) if valid else None
            records["plan_source"].append(getattr(teacher, "last_source", "cycle") if valid else "invalid")
            records["plan_id"].append(plan_decision.plan_id if plan_decision is not None else "")
            records["plan_step"].append(plan_decision.step_index if plan_decision is not None else -1)
            for key, value in (("decision", decision), ("elapsed_us", elapsed_us), ("reset_mask", decision == 0),
                               ("label_valid", valid), ("active", air or xbow), ("air_threat", air), ("xbow_present", xbow),
                               ("own_elixir", elixir), ("state_hash", env.state.state_hash()), ("executed_actions", executed)):
                records[key].append(value)
            opponent_action = opponent.choose_action(env.engine, env.state, 1)
            records["opponent_actions"].append(_encode_action(opponent_action))
            requested_card = str(hand[action.card_slot]) if isinstance(action, PlayCardAction) else None
            if requested_card is not None:
                requested[requested_card] += 1
            trace.update(canonical([action_to_dict(action), action_to_dict(opponent_action)]).encode() + b"\n")
            before_events = len(env.state.events)
            result = env.step_v2_for_viewer((action, opponent_action), viewer=0)
            played, rejected = [], []
            for event in env.state.events[before_events:]:
                player = event.get("player")
                if player in (0, 1) and event.kind == "action_rejected":
                    invalid_actions[player] += 1
                    if player == 0:
                        rejected.append(str(event.get("reason")))
                if player in (0, 1) and event.kind == "card_played":
                    card_plays[player] += 1
                    if player == 0:
                        played.append(str(event.get("card_id")))
            if plan_teacher is not None and valid:
                plan_teacher.acknowledge(TeacherTarget(*(int(value) for value in executed)), accepted=not rejected and played == ([requested_card] if requested_card else []))
            if rejected or played != ([requested_card] if requested_card is not None else []):
                raise RuntimeError(f"{job.job_id} decision {decision}: learner action rejected or accepted-card mismatch: {rejected}")
            accepted.update(played)
            records["requested_card"].append(requested_card or "")
            records["accepted_card"].append(played[0] if played else "")
            terminal, truncated = bool(result.terminated), bool(result.truncated)
            records["terminated_after"].append(terminal)
            records["truncated_after"].append(truncated)
            if terminal or truncated:
                winner, reason = result.info.get("winner"), result.info.get("terminal_reason")
                # The terminal observation is a measurement, never an action
                # anchor or an extra recurrent training row.
                post = result.observations[0]
                measurements.append(public_measurement({"global_features": post.global_vector}, int(env.state.elapsed_us), tower_max,
                                                       terminated=terminal, truncated=truncated))
                break
        if not terminal or truncated:
            raise RuntimeError(f"{job.job_id} did not genuinely terminate within {config.max_decisions} decisions")
        length = len(records["decision"])
        arrays = {key: value[:length] for key, value in public.items()}
        arrays.update({key: np.asarray(value) for key, value in records.items()})
        outcomes = build_tactical_outcome_targets(measurements, list(range(length)), horizons_seconds=config.horizons_seconds)
        arrays.update(outcome_values=outcomes.values, outcome_mask=outcomes.mask,
                      outcome_future_indices=outcomes.future_indices, outcome_measured_seconds=outcomes.measured_seconds,
                      measurement_seconds=np.asarray([value.match_seconds for value in measurements], dtype=np.float64),
                      measurement_tower_health=np.asarray([[value.own_tower_health, value.enemy_tower_health] for value in measurements], dtype=np.float64),
                      measurement_terminated=np.asarray([value.terminated for value in measurements], dtype=bool),
                      measurement_truncated=np.asarray([value.truncated for value in measurements], dtype=bool))
        metadata = {**asdict(job), "schema": SCHEMA, "identity_sha256": identity_sha,
                    "collection_identity": dict(identity), "checkpoint": dict(checkpoint_record), "config": asdict(config),
                    "deck_id": job.opponent["deck_id"], "family": job.opponent["archetype"], "style": job.opponent["style"],
                    "opponent_style_identity": getattr(opponent, "identity", None),
                    "decisions": length, "terminated": terminal, "truncated": truncated, "winner": winner, "terminal_reason": reason,
                    "invalid_actions": invalid_actions, "card_plays": card_plays,
                    "requested_learner_cards": dict(requested), "accepted_learner_cards": dict(accepted),
                    "action_trace_sha256": trace.hexdigest(), "native_actor_memory": "carry",
                    "teacher_wait_normalization": "legacy duration hints mapped to native duration index 0",
                    "learner_action_selection": "native-factorized-stochastic" if policy is not None else f"{job.driver}-teacher-deterministic",
                    "survival_labels": "masked: public PolicyObservationV2 has no stable track IDs",
                    "plan_labels": "executed acknowledged followup only on plans trajectories; fresh selection elsewhere",
                    "plan_counters": asdict(plan_teacher.counters) if plan_teacher is not None else None,
                    "horizons_seconds": list(config.horizons_seconds), "tower_max_hp": list(tower_max),
                    "input_digest": array_digest(arrays, (*PUBLIC_KEYS, "reset_mask", "elapsed_us")),
                    "label_digest": array_digest(arrays, LABEL_KEYS), "arrays_digest": array_digest(arrays, tuple(sorted(arrays))),
                    "active_rows": int(arrays["active"].sum()), "wall_seconds": time.perf_counter() - started}
        validate_episode(arrays, metadata)
        _verify_pins(identity, checkpoint_record)
        path = out / "episodes" / f"{job.job_id}.npz"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays, metadata_json=np.asarray(canonical(metadata)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    record = {**metadata, "path": str(path.relative_to(out)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    _atomic_json(path.with_suffix(".json"), record)
    return json.loads(canonical(record))


def validate_manifest(manifest: Mapping[str, Any], checkpoint_record: Mapping[str, Any] | None = None) -> None:
    """Reject incomplete, mixed-origin or stale datasets before any training."""
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("training requires a complete tactical collection manifest")
    identity = manifest["identity"]
    digest = _identity_digest(identity)
    if identity.get("identity_sha256") != digest or manifest.get("identity_sha256") != digest:
        raise ValueError("manifest collection identity digest mismatch")
    pinned_checkpoint = identity["checkpoint"]
    if checkpoint_record is not None and canonical(dict(checkpoint_record)) != canonical(pinned_checkpoint):
        raise ValueError("training checkpoint/model config differs from the collection origin")
    _identity_digest(identity.get("caller", {}))
    _verify_pins(identity, pinned_checkpoint)
    CollectionConfig(**identity["config"])
    if canonical(manifest["jobs"]) != canonical(identity["jobs"]):
        raise ValueError("manifest job grid differs from its pinned identity")
    jobs = [_job(value) for value in manifest["jobs"]]
    if not jobs or len({job.job_id for job in jobs}) != len(jobs) or len({job.seed for job in jobs}) != len(jobs):
        raise ValueError("manifest job IDs and actual origin seeds must be unique")
    if set(manifest["completed"]) != {job.job_id for job in jobs}:
        raise ValueError("manifest completed episodes must cover the exact declared job grid")
    for job in jobs:
        record = manifest["completed"][job.job_id]
        if any(canonical(record.get(key)) != canonical(value) for key, value in asdict(job).items()):
            raise ValueError("completed episode record differs from its declared origin job")
        if record.get("identity_sha256") != digest or canonical(record.get("collection_identity")) != canonical(identity):
            raise ValueError("completed episode belongs to another pinned collection identity")
        if canonical(record.get("checkpoint")) != canonical(pinned_checkpoint) or canonical(record.get("config")) != canonical(identity["config"]):
            raise ValueError("completed episode checkpoint/config differs from its collection")
        if record.get("schema") != SCHEMA or record.get("path") != f"episodes/{job.job_id}.npz":
            raise ValueError("completed episode schema/path is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
            raise ValueError("completed episode lacks an exact content hash")
        if not record.get("terminated") or record.get("truncated") or not 1 <= int(record.get("decisions", 0)) <= 1440:
            raise ValueError("completed episode is not a genuine terminal full match")
        if record.get("family") != job.opponent["archetype"] or record.get("deck_id") != job.opponent["deck_id"] or record.get("style") != job.opponent["style"]:
            raise ValueError("completed episode family/deck/style identity differs from its job")


def collect_dataset(jobs: Sequence[CollectionJob | Mapping[str, Any]], checkpoint_record: Mapping[str, Any],
                    out: str | Path, identity: Mapping[str, Any], *, config: CollectionConfig | None = None,
                    workers: int = 1, resume: bool = False, max_new_jobs: int | None = None) -> dict[str, Any]:
    """Collect atomically with strict source/checkpoint/job-grid resume checks."""
    config, out = config or CollectionConfig(), Path(out)
    jobs = [_job(value) for value in jobs]
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("collection requires nonempty unique jobs")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if max_new_jobs is not None and (type(max_new_jobs) is not int or max_new_jobs < 1):
        raise ValueError("max_new_jobs must be a positive integer")
    _identity_digest(identity)
    pinned = {"schema": SCHEMA, "caller": dict(identity), "checkpoint": dict(checkpoint_record),
              "config": asdict(config), "jobs": [asdict(job) for job in jobs]}
    digest = _identity_digest(pinned)
    pinned["identity_sha256"] = digest
    _verify_pins(identity, checkpoint_record)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical(manifest["identity"]) != canonical(pinned):
            raise ValueError("resume collection identity/source/checkpoint/job grid changed")
    else:
        if any(out.iterdir()):
            raise FileExistsError("collection output must be empty unless resume is requested")
        manifest = {"schema": SCHEMA, "identity": pinned, "identity_sha256": digest,
                    "jobs": [asdict(job) for job in jobs], "completed": {}, "status": "collecting"}
    (out / "episodes").mkdir(exist_ok=True)
    declared = {job.job_id for job in jobs}
    if set(manifest["completed"]) - declared:
        raise ValueError("manifest contains undeclared episodes")
    if any(path.stem not in declared for path in (out / "episodes").glob("*.npz")):
        raise ValueError("output contains undeclared episode files")
    pending = []
    for job in jobs:
        path = out / "episodes" / f"{job.job_id}.npz"
        record_path = path.with_suffix(".json")
        if path.exists() and record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            _, metadata = load_episode(path, expected_identity=digest, expected_sha=record["sha256"])
            if any(canonical(metadata[key]) != canonical(asdict(job)[key]) for key in asdict(job)):
                raise ValueError("episode metadata disagrees with its declared job")
            if record.get("path") != str(path.relative_to(out)) or any(record.get(key) != value for key, value in metadata.items()):
                raise ValueError("durable episode record and NPZ metadata disagree")
            if job.job_id in manifest["completed"] and canonical(manifest["completed"][job.job_id]) != canonical(record):
                raise ValueError("durable episode record and manifest disagree")
            manifest["completed"][job.job_id] = record
        elif path.exists() or record_path.exists() or job.job_id in manifest["completed"]:
            raise ValueError("incomplete episode artifact; resume refuses ambiguous partial work")
        else:
            pending.append(job)
    _atomic_json(manifest_path, manifest)
    pending = pending[:max_new_jobs]

    def accept(record: Mapping[str, Any]) -> None:
        _verify_pins(identity, checkpoint_record)
        manifest["completed"][record["job_id"]] = dict(record)
        _atomic_json(manifest_path, manifest)

    if workers == 1:
        for job in pending:
            accept(collect_episode(job, checkpoint_record, out, pinned, config=config))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(collect_episode, job, checkpoint_record, out, pinned, config=config) for job in pending]
            for future in as_completed(futures):
                accept(future.result())
    manifest["status"] = "complete" if len(manifest["completed"]) == len(jobs) else "collecting"
    _atomic_json(manifest_path, manifest)
    return json.loads(canonical(manifest))
