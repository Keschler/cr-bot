"""Balanced opponent-matchup rotation for recurrent PPO rollouts.

The parallel rollout count is a hardware/memory setting, not a data-distribution
setting.  Historically the opponent pool was truncated to one fixed deck per
environment lane (``decks[:envs]`` in spirit) and each lane kept that deck --
and its single opponent controller and single side -- for every episode reset.
More updates therefore could not expose the policy to omitted decks, the
opposite side, or the other controller strategy.

This module separates the opponent pool from the lane count.  A pool of
opponent decks, a pool of opponent strategies, and an optional pool of
frozen-checkpoint opponents define a Cartesian schedule
(``len(decks) * len(strategies) * len(checkpoints)`` entries per learner
side).  Each lane starts at a different schedule offset and advances one
entry after every completed game, so ``envs`` lanes sweep distinct matchups
in parallel and revisit the full schedule repeatedly with varying
controller seeds.  With ``mixed_sides`` both halves share the same pools,
which yields the full ``decks * sides * strategies`` coverage globally (for
example 12 * 2 * 2 = 48 combinations with 8 lanes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


_MATCHUP_SEED_DOMAIN = 0x4D415443  # "MATC"
_MATCHUP_CONTROLLER_DOMAIN = 0xC7A10E


def _mix_seed(seed: int, *parts: int) -> int:
    """Stable 64-bit mixer shared with the prototype/collector streams."""

    value = int(seed) & ((1 << 64) - 1)
    for part in parts:
        value ^= (int(part) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    return value


def _normalize_deck(deck: object, *, field: str) -> tuple[str, ...]:
    if isinstance(deck, (str, bytes)):
        raise ValueError(f"{field} must contain eight cards, not a string")
    try:
        row = tuple(deck)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field} must be a sequence of card identifiers") from error
    if len(row) != 8 or any(not isinstance(card, str) or not card.strip() for card in row):
        raise ValueError(f"{field} must contain eight non-empty card identifiers")
    if len(set(row)) != 8:
        raise ValueError(f"{field} must not contain duplicate cards")
    return row


def _normalize_strategy(strategy: object, *, field: str) -> str:
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return strategy.strip()


@dataclass(frozen=True, slots=True)
class MatchupEntry:
    """One deck/strategy/checkpoint combination in the rotation schedule."""

    deck: tuple[str, ...]
    strategy: str
    schedule_index: int
    # Frozen-checkpoint opponent path, or None for a simulator-side
    # heuristic controller built from ``strategy``.  Checkpoints rotate like
    # any other axis so league sparring partners are also met across decks,
    # sides, and seeds instead of staying pinned to fixed lanes.
    checkpoint: str | None = None


def normalize_deck_pool(decks: object, *, field: str = "opponent_deck_pool") -> tuple[tuple[str, ...], ...]:
    """Validate an opponent-deck pool without coupling it to a lane count."""

    if not isinstance(decks, (tuple, list)):
        raise ValueError(f"{field} must be a sequence of eight-card decks")
    pool = tuple(_normalize_deck(deck, field=f"{field}[{index}]") for index, deck in enumerate(decks))
    if not pool:
        raise ValueError(f"{field} must not be empty")
    return pool


def normalize_strategy_pool(
    strategies: object,
    *,
    field: str = "opponent_strategy_pool",
) -> tuple[str, ...]:
    """Validate an opponent-strategy pool without coupling it to a lane count."""

    if not isinstance(strategies, (tuple, list)):
        raise ValueError(f"{field} must be a sequence of strategy names")
    pool = tuple(_normalize_strategy(strategy, field=f"{field}[{index}]") for index, strategy in enumerate(strategies))
    if not pool:
        raise ValueError(f"{field} must not be empty")
    return pool


def checkpoint_options(
    checkpoints: Sequence[str] | None,
) -> tuple[str | None, ...]:
    """Return the checkpoint axis for a schedule.

    ``None`` (a simulator-side heuristic episode) always stays in the
    rotation so the deterministic-cycle regression anchor keeps appearing;
    configured frozen checkpoints are added alongside it.
    """

    pool = tuple(checkpoints) if checkpoints else ()
    for index, checkpoint in enumerate(pool):
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"checkpoints[{index}] must be a non-empty path or None")
    seen: set[str] = set()
    options: list[str | None] = [None]
    for checkpoint in pool:
        if checkpoint not in seen:
            seen.add(checkpoint)
            options.append(checkpoint)
    return tuple(options)


def build_matchup_schedule(
    decks: tuple[tuple[str, ...], ...],
    strategies: tuple[str, ...],
    checkpoints: tuple[str | None, ...] = (None,),
) -> tuple[MatchupEntry, ...]:
    """Build a balanced deck x strategy x checkpoint schedule.

    The deck axis cycles fastest so parallel lanes sweep distinct decks
    immediately instead of repeating a few decks with alternating controllers.
    Consecutive entries always differ in deck (when more than one is
    configured).  The strategy index advances with both the deck cycle and
    the position inside the cycle, so each deck meets every strategy over
    one schedule sweep while parallel lanes still observe controller
    diversity from the very first episode.  With more decks than lanes this
    exposes omitted decks within one schedule stride.  Pass
    ``checkpoint_options(pool)`` for ``checkpoints`` to keep simulator-side
    episodes in the rotation.
    """

    if not decks or not strategies:
        raise ValueError("matchup schedule requires non-empty deck and strategy pools")
    checkpoint_axis = tuple(checkpoints) if checkpoints else (None,)
    for index, checkpoint in enumerate(checkpoint_axis):
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ValueError(f"checkpoints[{index}] must be a non-empty path or None")
    entries: list[MatchupEntry] = []
    deck_count = len(decks)
    strategy_count = len(strategies)
    checkpoint_count = len(checkpoint_axis)
    for position in range(deck_count * strategy_count * checkpoint_count):
        deck_index = position % deck_count
        # Shift by the deck offset so every deck meets every strategy over
        # one sweep (for fixed deck d, (k + d) runs over all strategies as
        # the cycle k advances) instead of meeting the second strategy only
        # after a full deck cycle.
        strategy_index = (position // deck_count + deck_index) % strategy_count
        checkpoint_index = (position // (deck_count * strategy_count)) % checkpoint_count
        entries.append(
            MatchupEntry(
                deck=tuple(decks[deck_index]),
                strategy=strategies[strategy_index],
                schedule_index=position,
                checkpoint=checkpoint_axis[checkpoint_index],
            )
        )
    return tuple(entries)


def matchup_position(
    *,
    lane: int,
    episode: int,
    schedule_len: int,
    lane_offset: int = 0,
) -> tuple[int, int]:
    """Return ``(schedule_index, cycle)`` for one lane episode.

    Each lane starts at a distinct global-lane offset and advances one entry
    per completed episode.  Lanes therefore sweep distinct matchups in
    parallel while each lane cycles deterministically.  ``cycle`` counts how
    many times the lane has wrapped the schedule and seeds per-repetition
    variation.
    """

    if type(lane) is not int or lane < 0:
        raise ValueError("lane must be a non-negative integer")
    if type(episode) is not int or episode < 0:
        raise ValueError("episode must be a non-negative integer")
    if type(schedule_len) is not int or schedule_len <= 0:
        raise ValueError("schedule_len must be a positive integer")
    if type(lane_offset) is not int or lane_offset < 0:
        raise ValueError("lane_offset must be a non-negative integer")
    global_lane = lane + lane_offset
    position = episode + global_lane
    return position % schedule_len, position // schedule_len


def matchup_for_lane_episode(
    schedule: tuple[MatchupEntry, ...],
    *,
    lane: int,
    episode: int,
    lane_offset: int = 0,
) -> tuple[MatchupEntry, int]:
    """Return the schedule entry and cycle for one lane episode."""

    if not schedule:
        raise ValueError("matchup schedule must not be empty")
    schedule_index, cycle = matchup_position(
        lane=lane,
        episode=episode,
        schedule_len=len(schedule),
        lane_offset=lane_offset,
    )
    return schedule[schedule_index], cycle


def controller_seed_for_matchup(
    base_seed: int,
    *,
    schedule_index: int,
    cycle: int,
) -> int:
    """Derive a varying controller seed for each schedule repetition."""

    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")
    if type(schedule_index) is not int or schedule_index < 0:
        raise ValueError("schedule_index must be a non-negative integer")
    if type(cycle) is not int or cycle < 0:
        raise ValueError("cycle must be a non-negative integer")
    return _mix_seed(base_seed, _MATCHUP_CONTROLLER_DOMAIN, schedule_index, cycle)


def episode_seed_for_matchup(
    base_seed: int,
    *,
    lane: int,
    lane_offset: int,
    episode: int,
    schedule_index: int,
    cycle: int,
) -> int:
    """Derive a deterministic episode seed that varies across repetitions."""

    for name, value in (
        ("lane", lane),
        ("lane_offset", lane_offset),
        ("episode", episode),
        ("schedule_index", schedule_index),
        ("cycle", cycle),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")
    return _mix_seed(
        base_seed,
        _MATCHUP_SEED_DOMAIN,
        lane + lane_offset,
        episode,
        schedule_index,
        cycle,
    )


__all__ = [
    "MatchupEntry",
    "build_matchup_schedule",
    "checkpoint_options",
    "controller_seed_for_matchup",
    "episode_seed_for_matchup",
    "matchup_for_lane_episode",
    "matchup_position",
    "normalize_deck_pool",
    "normalize_strategy_pool",
]
