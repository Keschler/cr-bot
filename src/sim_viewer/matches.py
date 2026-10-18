"""Background simulator matches and indexed, disk-backed playback history.

Importing this module does not load Torch or construct perception models.
The viewer runs independently of training; only its exhibition worker uses
the policy runtime, with a private, restored CPU random-number stream.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
from uuid import uuid4


class MatchBusyError(RuntimeError):
    """An exhibition or reconstructed replay is already running."""


_TORCH_LOCK = threading.Lock()


@contextmanager
def _cpu_policy_runtime():
    import torch

    with _TORCH_LOCK, torch.random.fork_rng(devices=[]), torch.inference_mode():
        previous_threads = torch.get_num_threads()
        if previous_threads != 1:
            torch.set_num_threads(1)
        try:
            yield
        finally:
            if previous_threads != 1:
                torch.set_num_threads(previous_threads)


@dataclass(slots=True)
class _Match:
    id: str
    kind: str
    path: Path
    metadata: dict[str, Any]
    state: str = "running"
    error: str | None = None
    external: bool = False
    offsets: list[int] = field(default_factory=list)
    read_offset: int = 0
    ended: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: Any = field(default_factory=threading.RLock)
    thread: threading.Thread | None = None


class MatchManager:
    """Serve complete histories while allowing one simulation worker at a time.

    JSONL recordings are indexed in place, including actively written training
    samples. They consume no simulation worker and incomplete final lines are
    left unread until their writer finishes them.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._matches: dict[str, _Match] = {}
        self._active: _Match | None = None
        self._lock = threading.RLock()
        self._closed = False

    def start_replay(self, path: Path) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"match recording does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            match = _Match(uuid4().hex, "replay", path, {"source_path": str(path)}, external=True)
            with self._lock:
                self._require_open()
                self._matches[match.id] = match
            return self.status(match.id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("decisions"), list):
            raise ValueError("expected a frontier match JSON with config and decisions")
        config = raw.get("config")
        if not isinstance(config, dict) or type(config.get("match_seed")) is not int:
            raise ValueError("frontier match requires config.match_seed")
        for name in ("learner_deck", "opponent_deck"):
            _deck_shape(config.get(name), name)
        metadata = {**config, "source_path": str(path), "source_format": "frontier-json"}
        return self._start("replay", metadata, lambda match: self._replay_frontier(match, raw))

    def start_exhibition(self, config: dict[str, Any]) -> dict[str, Any]:
        resolved = _exhibition_config(config)
        return self._start("exhibition", dict(resolved), lambda match: self._exhibition(match, resolved))

    def status(self, match_id: str) -> dict[str, Any]:
        match = self._get(match_id)
        with match.lock:
            if match.external:
                self._refresh(match)
            return {
                "id": match.id,
                "kind": match.kind,
                "state": match.state,
                "frame_count": len(match.offsets),
                "metadata": dict(match.metadata),
                "error": match.error,
            }

    def frames(self, match_id: str, after: int = -1, limit: int = 100) -> list[dict[str, Any]]:
        if type(after) is not int or after < -1:
            raise ValueError("after must be an integer of at least -1")
        if type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError("limit must be between 1 and 2000")
        match = self._get(match_id)
        with match.lock:
            if match.external:
                self._refresh(match)
            offsets = list(match.offsets[after + 1:after + 1 + limit])
            if not offsets:
                return []
            result = []
            with match.path.open("rb") as handle:
                for offset in offsets:
                    handle.seek(offset)
                    frame = json.loads(handle.readline())
                    frame.pop("type", None)
                    result.append(frame)
            return result

    def stop(self, match_id: str) -> dict[str, Any]:
        match = self._get(match_id)
        with match.lock:
            if match.state == "running":
                match.stop_event.set()
                match.state = "stopped"
        return self.status(match.id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            matches = tuple(self._matches.values())
        for match in matches:
            self.stop(match.id)
        for match in matches:
            if match.thread is not None:
                match.thread.join(timeout=2)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("match manager is closed")

    def _get(self, match_id: str) -> _Match:
        with self._lock:
            return self._matches[match_id]

    def _start(self, kind: str, metadata: dict[str, Any], work: Callable[[_Match], None]) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            if self._active is not None and self._active.thread is not None and self._active.thread.is_alive():
                raise MatchBusyError("a simulator match is already running; stop it before starting another")
            match_id = uuid4().hex
            path = self.storage_dir / f"{match_id}.jsonl"
            match = _Match(match_id, kind, path, {**metadata, "recording_path": str(path)})
            match.thread = threading.Thread(target=self._worker, args=(match, work), daemon=True)
            self._matches[match_id] = match
            self._active = match
            match.thread.start()
        return self.status(match.id)

    @staticmethod
    def _worker(match: _Match, work: Callable[[_Match], None]) -> None:
        try:
            work(match)
            with match.lock:
                if match.state == "running":
                    match.state = "stopped" if match.stop_event.is_set() else "complete"
        except Exception as error:
            with match.lock:
                match.error = f"{type(error).__name__}: {error}"
                match.state = "error"

    @staticmethod
    def _refresh(match: _Match) -> None:
        if match.stop_event.is_set() or match.state == "error":
            return
        try:
            stat = match.path.stat()
            if stat.st_size < match.read_offset:
                raise ValueError("recording was truncated while it was being viewed")
            with match.path.open("rb") as handle:
                handle.seek(match.read_offset)
                while True:
                    offset = handle.tell()
                    line = handle.readline(16 * 1024 * 1024 + 1)
                    if len(line) > 16 * 1024 * 1024:
                        raise ValueError("recording line exceeds the 16 MiB frame limit")
                    if not line or not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("recording rows must be JSON objects")
                    kind = row.get("type")
                    if match.ended:
                        raise ValueError("recording contains data after its end marker")
                    if offset == 0 and kind != "header":
                        raise ValueError("recording must begin with a schema header")
                    if kind == "header":
                        if offset != 0:
                            raise ValueError("recording contains a second header")
                        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                            raise ValueError("unsupported recording schema_version; expected 1")
                        if not isinstance(row.get("source"), str) or not row["source"].strip():
                            raise ValueError("recording header requires a source")
                        if not isinstance(row.get("match_id"), str) or not row["match_id"].strip():
                            raise ValueError("recording header requires a match_id")
                        if not isinstance(row.get("metadata"), dict):
                            raise ValueError("recording header metadata must be an object")
                        match.metadata.update(row["metadata"])
                        match.metadata["source_path"] = str(match.path)
                        match.metadata["recorded_match_id"] = row.get("match_id")
                        match.metadata["source"] = row.get("source")
                    elif kind == "frame":
                        if type(row.get("index")) is not int or row["index"] != len(match.offsets):
                            raise ValueError("recording frame indices must be consecutive from zero")
                        match.offsets.append(offset)
                    elif kind == "end":
                        match.ended = True
                        reason = str(row.get("reason", "complete"))
                        match.metadata["result"] = {"winner": row.get("winner"), "reason": reason}
                        match.state = "stopped" if reason == "stopped" else "complete"
                        if reason.startswith("error"):
                            match.state, match.error = "error", reason
                    else:
                        raise ValueError(f"unknown recording row type: {kind!r}")
                    match.read_offset = handle.tell()
            match.metadata["recording_unfinished"] = not match.ended
            match.metadata["writer_active"] = not match.ended and time.time() - stat.st_mtime < 30
            match.metadata["last_modified_at"] = stat.st_mtime
        except (OSError, ValueError, TypeError) as error:
            match.state, match.error = "error", f"{type(error).__name__}: {error}"

    @staticmethod
    def _record(match: _Match, recorder: Any, frame: dict[str, Any]) -> None:
        with match.lock:
            if match.stop_event.is_set():
                return
            if frame["index"] != len(match.offsets):
                raise ValueError("internal frame index does not match recorded history")
            offset = match.path.stat().st_size
            recorder.append(frame)
            match.offsets.append(offset)

    @staticmethod
    def _recorder(match: _Match, env: Any) -> Any:
        from simulator.viewer.recording import JsonlRecorder

        with match.lock:
            match.metadata.update({
                "engine_version": env.state.engine_version,
                "ruleset_id": env.state.ruleset_id,
                "ruleset_hash": env.state.ruleset_hash,
                "tick_us": env.engine.ruleset.tick_us,
                "decision_interval_ticks": env.decision_interval_ticks,
                "decision_interval_us": env.decision_interval_ticks * env.engine.ruleset.tick_us,
            })
            return JsonlRecorder(match.path, match_id=match.id, source=match.kind, metadata=match.metadata)

    def _replay_frontier(self, match: _Match, raw: dict[str, Any]) -> None:
        from simulator.engine import BattleEngine
        from simulator.env import SimulatorEnv
        from simulator.ruleset import load_fixed_ruleset
        from simulator.viewer.projection import project_state

        config = raw["config"]
        env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
        env.reset_v2(seed=config["match_seed"], decks=(tuple(config["learner_deck"]), tuple(config["opponent_deck"])), shuffle_decks=True)
        for key in ("engine_version", "ruleset_id", "ruleset_hash"):
            pinned = raw.get(key, config.get(key))
            if pinned is not None and pinned != getattr(env.state, key):
                raise ValueError(f"recorded {key} is incompatible with the current simulator")
        recorder = self._recorder(match, env)
        events = tuple(env.state.events)
        try:
            for index, decision in enumerate(raw["decisions"]):
                if match.stop_event.is_set():
                    break
                expected = decision.get("state_hash")
                actual = env.state.state_hash()
                if not isinstance(expected, str) or expected != actual:
                    raise ValueError(f"replay diverged at decision {index}, tick {env.state.tick}: expected {expected}, got {actual}; recording requires its original engine/ruleset revision")
                if env.state.terminal:
                    raise ValueError("recording contains decisions after the match ended")
                actions = (_recorded_action(decision["learner"], 0), _recorded_action(decision["opponent"], 1))
                frame = project_state(env.state, env.engine.ruleset, index=index, events=events, actions=actions, policy=_legacy_policy(decision))
                event_start = len(env.state.events)
                rewards, _, _, _ = env._step_core(actions)
                frame["reward"] = float(rewards[0])
                self._record(match, recorder, frame)
                events = tuple(env.state.events[event_start:])
            if not match.stop_event.is_set():
                self._record(match, recorder, project_state(env.state, env.engine.ruleset, index=len(match.offsets), events=events))
            self._finish(match, recorder, env)
        except Exception as error:
            recorder.finish(f"error: {error}")
            raise

    def _exhibition(self, match: _Match, config: dict[str, Any]) -> None:
        import torch
        from simulator.engine import BattleEngine
        from simulator.env import SimulatorEnv
        from simulator.rl.foundation.opponent_pool import make_opponent_controller
        from simulator.ruleset import load_fixed_ruleset
        from simulator.viewer.projection import project_state

        with _cpu_policy_runtime():
            if match.stop_event.is_set():
                return
            env = SimulatorEnv(BattleEngine(load_fixed_ruleset(), validate_every_tick=False))
            env.reset_v2(seed=config["seed"], decks=(tuple(config["learner_deck"]), tuple(config["opponent_deck"])), shuffle_decks=True)
            learner = _PolicyPlayer(config["checkpoint"], viewer=0)
            opponent = (_PolicyPlayer(config["opponent_checkpoint"], viewer=1) if config["opponent"] == "checkpoint" else make_opponent_controller(config["archetype"], seed=config["seed"]))
            torch.random.default_generator.manual_seed(config["seed"])
            with match.lock:
                match.metadata["checkpoint_sha256"] = learner.sha256
                match.metadata["model_config"] = learner.model_config
                match.metadata["model_config_source"] = learner.model_config_source
                if isinstance(opponent, _PolicyPlayer):
                    match.metadata["opponent_checkpoint_sha256"] = opponent.sha256
            recorder = self._recorder(match, env)
            events = tuple(env.state.events)
            try:
                for index in range(config["max_decisions"]):
                    if match.stop_event.is_set() or env.state.terminal:
                        break
                    action, policy = learner.choose(env)
                    other = opponent.choose(env)[0] if isinstance(opponent, _PolicyPlayer) else opponent.choose_action(env.engine, env.state, 1)
                    actions = (action, other)
                    frame = project_state(env.state, env.engine.ruleset, index=index, events=events, actions=actions, policy=policy)
                    event_start = len(env.state.events)
                    rewards, _, _, _ = env._step_core(actions)
                    frame["reward"] = float(rewards[0])
                    self._record(match, recorder, frame)
                    events = tuple(env.state.events[event_start:])
                if not match.stop_event.is_set():
                    self._record(match, recorder, project_state(env.state, env.engine.ruleset, index=len(match.offsets), events=events))
                self._finish(match, recorder, env)
            except Exception as error:
                recorder.finish(f"error: {error}")
                raise

    @staticmethod
    def _finish(match: _Match, recorder: Any, env: Any) -> None:
        reason = "stopped" if match.stop_event.is_set() else (env.state.terminal_reason or "decision_limit")
        recorder.finish(reason, winner=env.state.winner)
        with match.lock:
            match.metadata["result"] = {"winner": env.state.winner, "reason": reason, "terminal": env.state.terminal}


def _deck_shape(value: Any, name: str) -> list[str]:
    if not isinstance(value, (tuple, list)) or len(value) != 8 or any(not isinstance(card, str) for card in value):
        raise ValueError(f"{name} must contain eight card names")
    return list(value)


def _exhibition_config(config: dict[str, Any]) -> dict[str, Any]:
    from simulator.rl.foundation.opponent_pool import OpponentPool
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset

    if not isinstance(config, dict):
        raise ValueError("exhibition config must be an object")
    seed, maximum = config.get("seed", 0), config.get("max_decisions", 1200)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1")
    if type(maximum) is not int or not 1 <= maximum <= 1440:
        raise ValueError("max_decisions must be between 1 and 1440")
    opponent = config.get("opponent", "scripted")
    if opponent not in ("scripted", "checkpoint"):
        raise ValueError("opponent must be scripted or checkpoint")
    ruleset = load_fixed_ruleset()
    archetype = config.get("archetype", "deterministic-cycle")
    scenario = OpponentPool(ruleset, seed=seed).sample(0, archetype=archetype)
    learner_deck = [ruleset.resolve_card_id(card) for card in _deck_shape(config.get("learner_deck") or list(PLAYER_DECK), "learner_deck")]
    if set(learner_deck) != set(PLAYER_DECK):
        raise ValueError("V4 learner checkpoint requires its eight-card Hog-cycle hand vocabulary")
    default_other = learner_deck if opponent == "checkpoint" else list(scenario.deck.cards)
    opponent_deck = [ruleset.resolve_card_id(card) for card in _deck_shape(config.get("opponent_deck") or default_other, "opponent_deck")]
    if len(set(opponent_deck)) != 8:
        raise ValueError("opponent_deck must contain eight different cards")
    if opponent == "checkpoint" and set(opponent_deck) != set(PLAYER_DECK):
        raise ValueError("V4 checkpoint opponent requires its eight-card Hog-cycle hand vocabulary")
    result = {
        "checkpoint": _checkpoint_path(config.get("checkpoint")),
        "opponent": opponent,
        "opponent_checkpoint": _checkpoint_path(config.get("opponent_checkpoint")) if opponent == "checkpoint" else None,
        "archetype": archetype,
        "learner_deck": learner_deck,
        "opponent_deck": opponent_deck,
        "seed": seed,
        "max_decisions": maximum,
        "sampling": "stochastic",
    }
    return result


def _checkpoint_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a checkpoint path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V4 checkpoint does not exist: {path}")
    return str(path)


def _recorded_action(raw: dict[str, Any], player: int) -> Any:
    from simulator.actions import PlayCardAction, WaitAction

    mode = raw.get("mode")
    if raw.get("kind") == "play" or mode == 1:
        slot, col, row = (raw.get(key) for key in ("slot", "col", "row"))
        if any(type(value) is not int for value in (slot, col, row)) or not 0 <= slot < 4 or not 0 <= col < 18 or not 0 <= row < 32:
            raise ValueError("recorded play has invalid slot or coordinates")
        return PlayCardAction(player, slot, (col, row))
    if raw.get("kind") == "wait" or mode == 0:
        return WaitAction(player)
    raise ValueError("recorded action must be wait or play")


def _legacy_policy(decision: dict[str, Any]) -> dict[str, Any] | None:
    mode, cards = decision.get("mode_p"), decision.get("card_p")
    if not isinstance(mode, list) or len(mode) != 2:
        return None
    result = {"wait_probability": mode[0], "play_probability": mode[1], "slot_probabilities": cards or [], "selected_slot": decision["learner"].get("slot") if decision["learner"].get("mode") == 1 else None}
    entropy = decision.get("entropy")
    result["entropy"] = entropy.get("joint") if isinstance(entropy, dict) else None
    # Historical discovery files retain only top cells; a dense heatmap would
    # misleadingly present the unrecorded cells as zero-probability actions.
    result["heatmap"] = []
    result["top_cells"] = decision.get("top_cells", [])
    result["heatmap_available"] = False
    # Frontier traces predate the masked projection: these are raw softmax
    # probabilities, not the distribution used to sample legal actions.
    result["distribution"] = "unmasked"
    return result


def _adjacent_model_config(checkpoint: Path) -> tuple[dict[str, Any], str | None]:
    names = ("config.json", "manifest.json", "meta.json", "bootstrap-meta.json", "curve.json")
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        for name in names:
            path = directory / name
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            candidates = [raw]
            for key in ("config", "metadata"):
                if isinstance(raw.get(key), dict):
                    candidates.append(raw[key])
            for candidate in candidates:
                for key in ("resolved_model_config", "model_config"):
                    value = candidate.get(key)
                    if isinstance(value, dict):
                        return value, str(path)
    return {}, None


def _load_policy(checkpoint: str) -> tuple[Any, dict[str, Any], str]:
    import torch
    from simulator.rl.v4.model_v4 import ModelConfigV4, RecurrentV4Policy
    from simulator.rl.v4.v4_ppo.runtime import configure_v4_runtime

    path = Path(checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("expected a plain V4 actor state-dict checkpoint")
    try:
        dimensions = {
            "model_dim": int(state["encoder.entities.projection.0.weight"].shape[0]),
            "spatial_channels": int(state["encoder.spatial.body.2.weight"].shape[0]),
            "fused_dim": int(state["encoder.fusion.0.weight"].shape[0]),
            "gru_hidden_dim": int(state["recurrent.gru.weight_hh_l0"].shape[1]),
            "gru_layers": sum(bool(re.fullmatch(r"recurrent\.gru\.weight_hh_l\d+", key)) for key in state),
            "transformer_layers": sum(bool(re.fullmatch(r"encoder\.entities\.transformer\.layers\.\d+\.linear1\.weight", key)) for key in state),
            "transformer_ff_dim": int(state["encoder.entities.transformer.layers.0.linear1.weight"].shape[0]),
            "spatial_head_dim": int(state["heads.spatial_key.weight"].shape[0]),
        }
    except (KeyError, IndexError) as error:
        raise ValueError("checkpoint does not have the V4 actor architecture") from error
    metadata, source = _adjacent_model_config(path)
    allowed = {item.name for item in fields(ModelConfigV4)}
    if set(metadata) - allowed:
        raise ValueError(f"unknown checkpoint model config fields: {sorted(set(metadata) - allowed)}")
    for key, value in dimensions.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"checkpoint weights disagree with model metadata field {key}")
    resolved = {**dimensions, **metadata}
    # Attention head count leaves no state-dict tensor signature. Honor run
    # metadata, otherwise use the retained V4 convention and disclose it.
    resolved.setdefault("transformer_heads", 4)
    policy = RecurrentV4Policy(ModelConfigV4(**resolved))
    incompatible = policy.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys) - {"heads.cell_bias"}
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"incompatible V4 checkpoint: missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}")
    configure_v4_runtime(policy)
    policy.eval().requires_grad_(False)
    from dataclasses import asdict

    return policy, asdict(policy.config), source or "weights; transformer_heads=4 (V4 default)"


class _PolicyPlayer:
    def __init__(self, checkpoint: str, *, viewer: int) -> None:
        from simulator.public_state_estimator import PublicStateEstimator

        self.policy, self.model_config, self.model_config_source = _load_policy(checkpoint)
        self.hidden = self.policy.initial_hidden(1, device="cpu")
        self.estimator = PublicStateEstimator()
        self.viewer = viewer
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    def choose(self, env: Any) -> tuple[Any, dict[str, Any]]:
        import numpy as np
        import torch
        from simulator.actions import PlayCardAction, WaitAction
        from simulator.rl.v4.model_v4 import masks_from_legal_play
        from simulator.rl.v4.v4_ppo.runtime import _v4_step_inputs, batch_v4_inputs
        from simulator.viewer.projection import project_policy

        observation = env.observe_v2_for_viewer(self.viewer)
        own = env.state.players[self.viewer]
        self.estimator.sync_from_env(env, viewer=self.viewer)
        packed = _v4_step_inputs(observation, own.hand[:4], own.elixir_milli / 1000.0, self.estimator.snapshot())
        legal = torch.as_tensor(np.array(observation.legal_play, copy=True)).unsqueeze(0).unsqueeze(0)
        masks = masks_from_legal_play(legal, bool(observation.legal_wait))
        logits, _, self.hidden = self.policy.forward_step(batch_v4_inputs(packed, "cpu"), hidden=self.hidden)
        actions, _, _ = self.policy.sample_action(logits, masks)
        policy = project_policy(logits, masks, actions)
        if int(actions.mode.reshape(-1)[0]) == 0:
            return WaitAction(self.viewer), policy
        row, col = (int(value) for value in actions.placement.reshape(-1, 2)[0])
        if self.viewer == 1:
            col, row = 17 - col, 31 - row
        return PlayCardAction(self.viewer, int(actions.card_slot.reshape(-1)[0]), (col, row)), policy
