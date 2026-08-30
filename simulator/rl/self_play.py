"""Public-only checkpoint opponents for self-play evaluation.

The generalized evaluator already knows how to run simulator-side heuristic
controllers.  This module adds the other important regression axis: a current
actor can be evaluated against frozen earlier actor checkpoints.  The frozen
opponent receives the public V2 observation from its own viewer perspective;
it never receives the authoritative state or the current actor's hidden state.

The opponent deck defaults to the fixed Hog-cycle deck because prototype
checkpoints currently have that deck as their actor contract.  Diverse enemy
decks remain covered by :mod:`rl.generalized`'s held-out matrix.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence


if TYPE_CHECKING:
    from .evaluation_matrix import EvaluationMatrixConfig


SELF_PLAY_STRATEGY_PREFIX = "checkpoint-"


class SelfPlayConfigurationError(ValueError):
    """Raised when a frozen checkpoint opponent cannot be configured safely."""


def _same_checkpoint_artifact(left: str | Path, right: str | Path) -> bool:
    """Return whether two checkpoint paths identify the same filesystem artifact.

    ``Path.resolve`` catches relative-path and symlink aliases, while
    ``os.path.samefile`` also catches distinct hard-link names when both files
    exist.  A copied checkpoint is intentionally not treated as the same
    artifact: it is a separate frozen snapshot even if its bytes happen to be
    identical.
    """

    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    try:
        if left_path.resolve(strict=False) == right_path.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        # Fall through to samefile when path resolution cannot complete (for
        # example, because a symlink loop is present).
        pass
    try:
        return (
            left_path.is_file()
            and right_path.is_file()
            and os.path.samefile(left_path, right_path)
        )
    except (OSError, ValueError):
        return False


class PublicCheckpointController:
    """Run one frozen recurrent actor from public V2 observations only."""

    def __init__(self, checkpoint: str | Path, *, device: str | None = "auto") -> None:
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
            raise SelfPlayConfigurationError("checkpoint must be a non-empty path")
        self.checkpoint = Path(checkpoint)
        try:
            from .prototype import load_prototype_checkpoint

            # Checkpoint loading constructs a model and historically restored
            # the saved torch RNG.  A frozen opponent is often created lazily
            # inside an active learner rollout, so either behavior would alter
            # the learner's sampling stream.  Preserve both CPU and visible
            # accelerator RNG state around construction and disable checkpoint
            # RNG restoration for this inference-only controller.
            import torch

            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = (
                [value.clone() for value in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            )
            try:
                self.learner, self.stored_config, self.metadata = load_prototype_checkpoint(
                    self.checkpoint,
                    device=device,
                    restore_rng=False,
                )
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_rng)
        except Exception as error:
            raise SelfPlayConfigurationError(
                f"cannot load self-play checkpoint {self.checkpoint}: {error}"
            ) from error
        self.learner.policy.eval()
        self._state: Any | None = None
        self._reset_pending = True
        self._last_time_left: float | None = None

    def reset(self) -> None:
        """Reset recurrent state before reusing the controller for a new match."""

        self._state = self.learner.initial_rollout_state(1)
        self._reset_pending = True
        self._last_time_left = None

    def choose_public_action(self, observation: Any, *, player: int = 0) -> Any:
        """Return a local-coordinate policy action for one public observation."""

        del player  # V2 observations are already expressed in viewer-local coordinates.
        # A training lane can be reused after a match. The public clock is
        # sufficient to detect that boundary without inspecting simulator
        # state, so the frozen GRU does not carry memory across episodes.
        try:
            from cr_bot.features.channels import GLOBAL_SCALAR_IDX

            time_left = float(
                observation.global_vector[GLOBAL_SCALAR_IDX["time_left_norm"]]
            )
        except (AttributeError, KeyError, TypeError, ValueError, IndexError):
            time_left = None
        if (
            time_left is not None
            and self._last_time_left is not None
            and time_left > self._last_time_left + 1e-6
        ):
            self.reset()
        if time_left is not None:
            self._last_time_left = time_left
        if self._state is None:
            self.reset()

        from .collector import _batch_observations
        from .learner import RecurrentRolloutState
        from cr_bot.domain.game_state import Action as PolicyAction

        import torch
        raster, global_features, entities, entity_mask, masks = _batch_observations(
            [observation],
            device=self.learner.device,
            inference=True,
        )
        reset_mask = torch.tensor(
            [[self._reset_pending]],
            dtype=torch.bool,
            device=self.learner.device,
        )
        with torch.inference_mode():
            actions, final_hidden = self.learner.policy.act_deterministic(
                raster,
                global_features,
                entities,
                entity_mask,
                masks,
                reset_mask=reset_mask,
                hidden=self._state.hidden,
            )
        self._state = RecurrentRolloutState(final_hidden.detach())
        self._reset_pending = False
        if int(actions.mode[0, 0].item()) == 0:
            return PolicyAction(kind="Wait")
        row = int(actions.placement[0, 0, 0].item())
        column = int(actions.placement[0, 0, 1].item())
        return PolicyAction(
            kind="Play",
            card_idx=int(actions.card_slot[0, 0].item()),
            cell=(column, row),
        )


def checkpoint_strategy(
    checkpoint: str | Path,
    *,
    strategy_id: str,
    device: str | None = "auto",
) -> Any:
    """Build an evaluation-matrix strategy backed by a frozen actor."""

    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise SelfPlayConfigurationError("strategy_id must be a non-empty string")
    try:
        from .evaluation_matrix import OpponentStrategySpec
    except ImportError as error:  # pragma: no cover - package-layout guard
        raise SelfPlayConfigurationError("cannot import evaluation matrix types") from error

    path = Path(checkpoint)
    metadata: dict[str, str] = {"checkpoint_path": str(path)}
    if path.is_file():
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise SelfPlayConfigurationError(
                f"cannot hash checkpoint {path}: {error}"
            ) from error
        metadata["checkpoint_sha256"] = digest.hexdigest()
    return OpponentStrategySpec(
        strategy_id=strategy_id,
        factory=lambda _seed: PublicCheckpointController(path, device=device),
        description=f"frozen public actor checkpoint: {path}",
        metadata=metadata,
    )


def build_self_play_matrix_config(
    checkpoint: str | Path,
    opponent_checkpoints: Sequence[str | Path],
    *,
    player_deck: Sequence[str] | None = None,
    seeds: Sequence[int] = (10_000,),
    device: str | None = "auto",
    max_decisions: int | None = None,
    batch_size: int = 1,
    include_match_results: bool = True,
    include_replay_hashes: bool = False,
    shuffle_decks: bool = True,
    target_player: int = 0,
) -> Any:
    """Build a matrix comparing one actor against frozen prior checkpoints."""

    paths = tuple(opponent_checkpoints)
    if not paths:
        raise SelfPlayConfigurationError("opponent_checkpoints must not be empty")
    for index, opponent_checkpoint in enumerate(paths):
        if _same_checkpoint_artifact(checkpoint, opponent_checkpoint):
            try:
                resolved_checkpoint = Path(checkpoint).expanduser().resolve(strict=False)
                resolved_opponent = Path(opponent_checkpoint).expanduser().resolve(strict=False)
                location = f" ({resolved_checkpoint} == {resolved_opponent})"
            except (OSError, RuntimeError):
                location = ""
            raise SelfPlayConfigurationError(
                "checkpoint and opponent_checkpoints[{}] resolve to the same "
                "artifact{}; self-play comparison requires a distinct frozen "
                "opponent checkpoint".format(index, location)
            )
    try:
        from .evaluation_matrix import EvaluationMatrixConfig, OpponentDeckSpec
        from ..roster import PLAYER_DECK
    except ImportError:  # pragma: no cover - top-level ``rl`` layout
        from .evaluation_matrix import EvaluationMatrixConfig, OpponentDeckSpec
        from simulator.roster import PLAYER_DECK

    selected_player_deck = tuple(PLAYER_DECK) if player_deck is None else tuple(player_deck)

    strategies = tuple(
        checkpoint_strategy(
            path,
            strategy_id=f"{SELF_PLAY_STRATEGY_PREFIX}{index}",
            device=device,
        )
        for index, path in enumerate(paths)
    )
    return EvaluationMatrixConfig(
        checkpoint=checkpoint,
        opponent_decks=(
            OpponentDeckSpec(
                deck_id="fixed-player-deck",
                cards=selected_player_deck,
                tags=("self-play", "prototype-contract"),
            ),
        ),
        strategies=strategies,
        seeds=tuple(seeds),
        player_deck=selected_player_deck,
        policy_mode="actor",
        target_player=target_player,
        max_decisions=max_decisions,
        device=device,
        shuffle_decks=shuffle_decks,
        include_match_results=include_match_results,
        include_replay_hashes=include_replay_hashes,
        held_out=False,
        batch_size=batch_size,
    )


def build_side_balanced_self_play_matrix_configs(
    checkpoint: str | Path,
    opponent_checkpoints: Sequence[str | Path],
    *,
    player_deck: Sequence[str] | None = None,
    seeds: Sequence[int] = (10_000,),
    device: str | None = "auto",
    max_decisions: int | None = None,
    batch_size: int = 1,
    include_match_results: bool = True,
    include_replay_hashes: bool = False,
    shuffle_decks: bool = True,
) -> tuple["EvaluationMatrixConfig", "EvaluationMatrixConfig"]:
    """Build matched self-play configs for both target-player assignments.

    The first config evaluates the candidate as player 0 and the second as
    player 1.  They intentionally share the same checkpoint, opponent
    checkpoints, deck, strategies, and seeds, so callers can run both reports
    and attribute outcome differences to side/first-player advantage rather
    than to a changed evaluation population.  Each underlying config exposes
    ``target_player`` in its normal serialized metadata.
    """

    # Materialize one-shot iterables once so both halves of the pair receive
    # exactly the same evaluation axes.
    paths = tuple(opponent_checkpoints)
    normalized_player_deck = None if player_deck is None else tuple(player_deck)
    normalized_seeds = tuple(seeds)
    player_zero_config = build_self_play_matrix_config(
        checkpoint,
        paths,
        player_deck=normalized_player_deck,
        seeds=normalized_seeds,
        device=device,
        max_decisions=max_decisions,
        batch_size=batch_size,
        include_match_results=include_match_results,
        include_replay_hashes=include_replay_hashes,
        shuffle_decks=shuffle_decks,
        target_player=0,
    )
    player_one_config = build_self_play_matrix_config(
        checkpoint,
        paths,
        player_deck=normalized_player_deck,
        seeds=normalized_seeds,
        device=device,
        max_decisions=max_decisions,
        batch_size=batch_size,
        include_match_results=include_match_results,
        include_replay_hashes=include_replay_hashes,
        shuffle_decks=shuffle_decks,
        target_player=1,
    )
    return (
        player_zero_config,
        player_one_config,
    )


def evaluate_against_checkpoints(
    checkpoint: str | Path,
    opponent_checkpoints: Sequence[str | Path],
    *,
    seeds: Sequence[int] = (10_000,),
    device: str | None = "auto",
    max_decisions: int | None = None,
    batch_size: int = 1,
    include_match_results: bool = True,
    include_replay_hashes: bool = False,
    shuffle_decks: bool = True,
) -> dict[str, object]:
    """Evaluate a current checkpoint against frozen actor checkpoints."""

    from .evaluation_matrix import run_evaluation_matrix

    config = build_self_play_matrix_config(
        checkpoint,
        opponent_checkpoints,
        seeds=seeds,
        device=device,
        max_decisions=max_decisions,
        batch_size=batch_size,
        include_match_results=include_match_results,
        include_replay_hashes=include_replay_hashes,
        shuffle_decks=shuffle_decks,
    )
    return run_evaluation_matrix(config)


__all__ = [
    "PublicCheckpointController",
    "SELF_PLAY_STRATEGY_PREFIX",
    "SelfPlayConfigurationError",
    "build_self_play_matrix_config",
    "build_side_balanced_self_play_matrix_configs",
    "checkpoint_strategy",
    "evaluate_against_checkpoints",
]
