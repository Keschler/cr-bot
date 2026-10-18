"""Append-only match recordings and atomic, opt-in training progress."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from itertools import count
import os
from pathlib import Path
import time
from typing import Any
import warnings


_TRAINING_EPISODES = count()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlRecorder:
    """A writer whose completed lines are immediately visible to readers.

    Each operation closes its file. Readers must tolerate an incomplete last
    line while another process is writing, or after that process is killed.
    """

    def __init__(self, path: Path | str, *, match_id: str, source: str, metadata: dict[str, Any]):
        self.path = Path(path)
        self.finished = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "header", "schema_version": 1, "match_id": match_id,
            "source": source, "created_at": utc_now(), "metadata": metadata,
        }
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(self._line(header))

    @staticmethod
    def _line(record: dict[str, Any]) -> str:
        return json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"

    def append(self, frame: dict[str, Any]) -> None:
        if self.finished:
            raise ValueError("cannot append to a finished match recording")
        line = self._line({**frame, "type": "frame"})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    def finish(self, reason: str, winner: int | None = None) -> None:
        if self.finished:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(self._line({"type": "end", "reason": reason, "winner": winner}))
        self.finished = True


def viewer_directory() -> Path | None:
    """Recording is disabled unless the caller explicitly sets the directory."""
    value = os.environ.get("CR_SIM_VIEWER_DIR")
    return Path(value).expanduser() if value else None


def recording_every() -> int:
    value = os.environ.get("CR_SIM_VIEWER_EVERY", "1")
    try:
        every = int(value)
    except ValueError as error:
        raise ValueError("CR_SIM_VIEWER_EVERY must be a positive integer") from error
    if every < 1:
        raise ValueError("CR_SIM_VIEWER_EVERY must be a positive integer")
    return every


def write_progress(directory: Path, payload: dict[str, Any]) -> None:
    """Replace one complete progress document so polling never sees a tear."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "progress.json"
    temporary = directory / f".progress-{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps({
            "schema_version": 1, "updated_at": utc_now(), **payload,
        }, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_progress(payload: dict[str, Any]) -> None:
    directory = viewer_directory()
    if directory is None:
        return
    try:
        write_progress(directory, payload)
    except (OSError, ValueError) as error:
        warnings.warn(f"simulator viewer progress unavailable: {error}", RuntimeWarning, stacklevel=2)


class TrainingRecorder:
    """Sample exact lane-zero episodes without touching any simulator RNG.

    Pre-action policy/state are saved together with the reward from that
    action. A final state without an action is appended at every boundary.
    Optional telemetry failures disable the recorder rather than training.
    """

    def __init__(self, *, seed: int, episode_cap: int):
        self.directory = viewer_directory()
        self.every = recording_every() if self.directory is not None else 1
        self.seed = seed
        self.episode_cap = episode_cap
        self.recorder: JsonlRecorder | None = None
        self.pending: dict[str, Any] | None = None
        self.index = 0
        self.event_cursor = 0

    def _disable(self, error: Exception) -> None:
        warnings.warn(f"simulator viewer recording disabled: {error}", RuntimeWarning, stacklevel=2)
        self.directory = None
        self.recorder = None
        self.pending = None

    def start_episode(self, env: Any, entry: dict[str, Any], episode: int) -> None:
        if self.directory is None:
            return
        sequence = next(_TRAINING_EPISODES)
        if sequence % self.every:
            return
        state = env.state
        match_id = f"training-{os.getpid()}-{self.seed}-{episode}-{time.time_ns()}"
        metadata = {
            "engine_version": state.engine_version,
            "ruleset_id": state.ruleset_id, "ruleset_hash": state.ruleset_hash,
            "tick_us": env.engine.ruleset.tick_us,
            "decision_interval_ticks": getattr(env, "decision_interval_ticks", None),
            "decision_interval_us": (
                env.decision_interval_ticks * env.engine.ruleset.tick_us
                if hasattr(env, "decision_interval_ticks") else None
            ),
            "seed": state.seed, "rollout_seed": self.seed, "lane": 0,
            "episode": episode, "episode_cap": self.episode_cap,
            "sample_sequence": sequence,
            "entry_id": entry.get("entry_id"), "initial_tick": state.tick,
            "initial_state_hash": state.state_hash(),
            "opponent": entry.get("opponent", {}),
            "learner_deck": list(state.players[0].deck),
            "opponent_deck": list(state.players[1].deck),
        }
        try:
            self.recorder = JsonlRecorder(
                self.directory / f"{match_id}.jsonl", match_id=match_id,
                source="training", metadata=metadata,
            )
            self.index = 0
            self.event_cursor = len(state.events)
        except (OSError, ValueError) as error:
            self._disable(error)

    def before_step(self, env: Any, actions: Any, logits: Any, masks: Any, policy_actions: Any) -> None:
        if self.recorder is None:
            return
        from .projection import project_policy, project_state

        try:
            self.pending = project_state(
                env.state, env.engine.ruleset, index=self.index,
                events=env.state.events[self.event_cursor:], actions=actions,
                policy=project_policy(logits, masks, policy_actions, lane=0),
            )
            self.event_cursor = len(env.state.events)
        except (OSError, ValueError) as error:
            self._disable(error)

    def after_step(self, env: Any, reward: float, *, reason: str | None = None) -> None:
        if self.recorder is None or self.pending is None:
            return
        from .projection import project_state

        try:
            self.pending["reward"] = reward
            self.recorder.append(self.pending)
            self.pending = None
            self.index += 1
            if reason is not None:
                self.recorder.append(project_state(
                    env.state, env.engine.ruleset, index=self.index,
                    events=env.state.events[self.event_cursor:],
                ))
                self.recorder.finish(reason, winner=env.state.winner)
                self.recorder = None
        except (OSError, ValueError) as error:
            self._disable(error)
