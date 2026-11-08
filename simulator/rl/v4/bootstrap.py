"""Deterministic from-scratch V4 bootstrap data from real simulator matches.

This is the only supervised bootstrap path retained by V4.  It runs ordinary
simulator environments, records the public actor observation, and applies the
small rules teacher used by frontier auditing.  There is no synthetic feature
generator and no search or timing curriculum hidden behind this API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
import hashlib
import random
import copy
from typing import Any, Protocol

import numpy as np

try:
    from ...actions import PlayCardAction, WaitAction
    from ...engine import BattleEngine
    from ...env import SimulatorEnv
    from ...public_state_estimator import PublicStateEstimator
    from ...roster import PLAYER_DECK
    from ...ruleset import load_fixed_ruleset
    from ..foundation.opponent_pool import (
        OpponentDeckSpec,
        OpponentPool,
        OpponentScenario,
    )
    from .hand_features import build_hand_tokens
    from .real_decks import load_real_decks
    from .simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )
except ImportError:  # pragma: no cover - top-level ``rl`` imports
    from simulator.actions import PlayCardAction, WaitAction
    from simulator.engine import BattleEngine
    from simulator.env import SimulatorEnv
    from simulator.public_state_estimator import PublicStateEstimator
    from simulator.roster import PLAYER_DECK
    from simulator.ruleset import load_fixed_ruleset
    from simulator.rl.foundation.opponent_pool import (
        OpponentDeckSpec,
        OpponentPool,
        OpponentScenario,
    )
    from simulator.rl.v4.hand_features import build_hand_tokens
    from simulator.rl.v4.real_decks import load_real_decks
    from simulator.rl.v4.simulator_teacher import (
        TeacherError,
        TeacherTarget,
        soft_placement_target,
        teacher_label,
    )


BOOTSTRAP_VERSION = "v4-bootstrap-sim-v3"
BOOTSTRAP_ARCHETYPES: tuple[str, ...] = (
    "deterministic-cycle",
    "aggressive-pressure",
    "defensive-cycle",
    "beatdown",
    "air-beatdown",
    "siege-bait",
)


class BootstrapTeacher(Protocol):
    """Callable teacher contract accepted by bootstrap collection.

    The callable receives the same public-state values as the built-in
    :func:`teacher_label`.  Keeping this contract at the collection boundary
    lets an experiment compare teacher revisions over fresh trajectories
    without changing the simulator or the actor-facing observation.
    """

    def __call__(
        self,
        *,
        hand_tokens: np.ndarray,
        entity_tokens: np.ndarray,
        entity_mask: np.ndarray,
        legal_play: np.ndarray,
        legal_wait: bool,
        own_elixir: float,
    ) -> TeacherTarget:
        ...


@dataclass(frozen=True, slots=True)
class BootstrapOpponentSpec:
    """One explicit opponent identity for catalog bootstrap collection.

    ``cards`` retain the catalog's native order.  ``strategy`` and
    ``controller_seed`` are kept separate from the deck identity because a
    real catalog row can be paired with the exact scripted controller used by
    full-match evaluation.  A missing controller seed is resolved to the
    episode's match seed when the scenario is built, matching the canonical
    evaluation default.
    """

    deck_id: str
    archetype: str
    cards: tuple[str, ...]
    strategy: str | None = None
    controller_seed: int | None = None
    split: str = "train"
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.deck_id, str) or not self.deck_id:
            raise ValueError("opponent deck_id must be a non-empty string")
        if not isinstance(self.archetype, str) or not self.archetype:
            raise ValueError("opponent archetype must be a non-empty string")
        if len(self.cards) != 8 or len(set(self.cards)) != 8:
            raise ValueError("opponent cards must contain eight unique cards")
        if any(not isinstance(card, str) or not card for card in self.cards):
            raise ValueError("opponent cards must contain non-empty strings")
        if self.strategy is not None and (
            not isinstance(self.strategy, str) or not self.strategy
        ):
            raise ValueError("opponent strategy must be a non-empty string")
        if self.controller_seed is not None and type(self.controller_seed) is not int:
            raise ValueError("opponent controller_seed must be an integer")
        if self.split not in ("train", "eval"):
            raise ValueError("opponent split must be train or eval")
        if self.source_url is not None and (
            not isinstance(self.source_url, str) or not self.source_url
        ):
            raise ValueError("opponent source_url must be a non-empty string")

    @classmethod
    def from_real_deck(
        cls,
        deck: Any,
        *,
        controller_seed: int | None = None,
        strategy: str | None = None,
    ) -> "BootstrapOpponentSpec":
        """Build a spec while preserving every native ``RealDeck`` field."""

        return cls(
            deck_id=str(deck.deck_id),
            archetype=str(deck.archetype),
            cards=tuple(deck.cards),
            strategy=str(deck.archetype) if strategy is None else strategy,
            controller_seed=controller_seed,
            split=str(deck.split),
            source_url=str(deck.source_url),
        )

    def as_scenario(self, *, episode_index: int, match_seed: int) -> OpponentScenario:
        """Return the exact deck/controller scenario for one fresh episode."""

        controller_seed = (
            match_seed if self.controller_seed is None else self.controller_seed
        )
        deck = OpponentDeckSpec(
            deck_id=self.deck_id,
            archetype=self.archetype,
            cards=self.cards,
            tags=("real-deck", self.split),
            source=self.source_url or "real-catalog",
        )
        selection_seed = _stable_seed(
            match_seed, "bootstrap-opponent", self.deck_id, episode_index
        )
        return OpponentScenario(
            episode_index=episode_index,
            selection_seed=selection_seed,
            deck=deck,
            strategy=self.strategy or self.archetype,
            controller_seed=controller_seed,
            sampling_source="real-catalog" if self.source_url else "explicit",
        )


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Settings for deterministic public-state collection."""

    n_states: int
    seed: int = 0
    max_decisions: int = 1440
    samples_per_episode: int = 64
    episodes_per_archetype: int | None = None
    require_terminal: bool = False
    archetypes: tuple[str, ...] = BOOTSTRAP_ARCHETYPES
    # Opt-in explicit opponent mode.  ``None`` preserves the historical six
    # generated-family collector above; a non-empty sequence runs one or more
    # fresh episodes for each exact spec and never asks OpponentPool to invent
    # a replacement deck.
    opponent_specs: tuple[
        BootstrapOpponentSpec | OpponentScenario | Mapping[str, object], ...
    ] | None = None
    episodes_per_opponent: int | None = None

    def __post_init__(self) -> None:
        if type(self.n_states) is not int or self.n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.max_decisions) is not int or self.max_decisions <= 0:
            raise ValueError("max_decisions must be a positive integer")
        if type(self.samples_per_episode) is not int or self.samples_per_episode < 2:
            raise ValueError("samples_per_episode must be an integer of at least two")
        if self.episodes_per_archetype is not None and (
            type(self.episodes_per_archetype) is not int
            or self.episodes_per_archetype <= 0
        ):
            raise ValueError("episodes_per_archetype must be a positive integer")
        if type(self.require_terminal) is not bool:
            raise ValueError("require_terminal must be boolean")
        if not self.archetypes or any(
            not isinstance(value, str) or not value for value in self.archetypes
        ):
            raise ValueError("archetypes must contain non-empty strings")
        if len(set(self.archetypes)) != len(self.archetypes):
            raise ValueError("archetypes must not contain duplicates")
        if self.opponent_specs is not None:
            if not self.opponent_specs:
                raise ValueError("opponent_specs must not be empty")
            if isinstance(self.opponent_specs, (str, bytes)):
                raise ValueError("opponent_specs must be a sequence of specs")
            object.__setattr__(self, "opponent_specs", tuple(self.opponent_specs))
        if self.episodes_per_opponent is not None and (
            type(self.episodes_per_opponent) is not int
            or self.episodes_per_opponent <= 0
        ):
            raise ValueError("episodes_per_opponent must be a positive integer")


@dataclass(slots=True)
class BootstrapSample:
    """One public simulator observation with a rules-teacher target."""

    family: str
    raster: np.ndarray
    global_features: np.ndarray
    entity_tokens: np.ndarray
    entity_mask: np.ndarray
    hand_tokens: np.ndarray
    opp_hand_probs: np.ndarray
    opp_out_of_cycle: np.ndarray
    opp_elixir_interval: np.ndarray
    event_history: np.ndarray
    legal_play: np.ndarray
    legal_wait: bool
    target: TeacherTarget
    soft_placement: np.ndarray
    provenance: dict[str, object]
    mode_weight: float = 1.0
    retention_stratum: str = ""


@dataclass(frozen=True, slots=True)
class BootstrapEpisodeSummary:
    episode: int
    archetype: str
    opponent_deck_id: str
    opponent_deck_cards: tuple[str, ...]
    decisions: int
    terminal: bool
    eligible_wait: int
    eligible_play: int
    retained_wait: int
    retained_play: int
    opponent_strategy: str | None = None
    opponent_controller_seed: int | None = None
    opponent_source_url: str | None = None
    match_seed: int | None = None


def _target_action(target: TeacherTarget, *, player: int) -> Any:
    if target.mode == 0:
        return WaitAction(player)
    return PlayCardAction(player, target.card_slot, (target.col, target.row))


def _fallback_action(
    legal_play: np.ndarray,
    legal_wait: bool,
    *,
    player: int,
    allow_countdown_wait: bool = False,
) -> Any:
    if legal_wait:
        return WaitAction(player)
    legal = np.argwhere(np.asarray(legal_play, dtype=bool))
    if len(legal):
        slot, row, col = (int(value) for value in legal[0])
        return PlayCardAction(player, slot, (col, row))
    if allow_countdown_wait:
        # Countdown WAIT is a physical simulator action used only to advance
        # to the active phase; it is never emitted as a bootstrap label.
        return WaitAction(player)
    raise RuntimeError("active bootstrap state has no legal action")


def _catalog_controller_seed(deck_id: str, *, split: str) -> int:
    """Match the controller seed used by the catalog evaluation manifests."""

    payload = f"{split}\x1f{deck_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) & 0xFFFFFFFF


def catalog_training_opponent_specs() -> tuple[BootstrapOpponentSpec, ...]:
    """Return exact train-catalog opponent identities in catalog order.

    The helper is intentionally explicit about the split.  Callers that need
    the reserved final pool must use the final-evaluation APIs; passing those
    rows to bootstrap is rejected by :func:`generate_bootstrap_dataset`.
    """

    # Keep this order identical to ``learning_curve._default_opponents`` and
    # ``catalog_opponents()["train"]``: grouped by the canonical archetype
    # order, then by the catalog order within each archetype.
    from .real_decks import REAL_DECK_ARCHETYPES, real_decks_for

    return tuple(
        BootstrapOpponentSpec.from_real_deck(
            deck,
            strategy=deck.archetype,
            controller_seed=_catalog_controller_seed(deck.deck_id, split="real-train"),
        )
        for archetype in REAL_DECK_ARCHETYPES
        for deck in real_decks_for("train", archetype)
    )


# A descriptive alias keeps experiment scripts readable without duplicating
# the catalog construction logic.
real_catalog_training_opponent_specs = catalog_training_opponent_specs


def _coerce_opponent_spec(
    value: Any,
) -> BootstrapOpponentSpec:
    if isinstance(value, BootstrapOpponentSpec):
        return value
    if isinstance(value, OpponentScenario):
        return BootstrapOpponentSpec(
            deck_id=value.deck.deck_id,
            archetype=value.deck.archetype,
            cards=tuple(value.deck.cards),
            strategy=value.strategy,
            controller_seed=value.controller_seed,
            split="train",
            source_url=value.deck.source if value.deck.source.startswith("https://") else None,
        )
    if isinstance(value, OpponentDeckSpec):
        return BootstrapOpponentSpec(
            deck_id=value.deck_id,
            archetype=value.archetype,
            cards=tuple(value.cards),
            strategy=value.archetype,
            controller_seed=None,
            split="train",
            source_url=value.source if value.source.startswith("https://") else None,
        )
    # Accept RealDeck rows directly as a convenience for callers that already
    # loaded the pinned catalog.  The split is preserved and checked before
    # any environment is created.
    if all(hasattr(value, field) for field in ("deck_id", "archetype", "cards", "split")):
        return BootstrapOpponentSpec.from_real_deck(value)
    if not isinstance(value, Mapping):
        raise TypeError(
            "opponent_specs entries must be BootstrapOpponentSpec, "
            "OpponentScenario, or a mapping"
        )
    cards_value = value.get("cards", value.get("deck", value.get("opponent_deck")))
    if isinstance(cards_value, (str, bytes)) or cards_value is None:
        raise ValueError("opponent spec needs a sequence of eight cards")
    deck_id = value.get("deck_id", value.get("real_deck_id", value.get("opponent_deck_id")))
    if deck_id is None:
        raise ValueError("opponent spec needs deck_id")
    archetype = value.get("archetype")
    if archetype is None:
        raise ValueError("opponent spec needs archetype")
    strategy = value.get("strategy", str(archetype))
    controller_seed = value.get("controller_seed")
    if controller_seed is not None and type(controller_seed) is not int:
        raise ValueError("opponent controller_seed must be an integer")
    source_url = value.get("source_url")
    split = value.get("split", value.get("real_deck_split", "train"))
    return BootstrapOpponentSpec(
        deck_id=str(deck_id),
        archetype=str(archetype),
        cards=tuple(str(card) for card in cards_value),
        strategy=str(strategy) if strategy is not None else None,
        controller_seed=controller_seed,
        split=str(split),
        source_url=str(source_url) if source_url is not None else None,
    )


def _state_id(sample: Any, *, episode: int, decision: int) -> str:
    provenance = getattr(sample, "provenance", {})
    if isinstance(provenance, Mapping):
        for key in ("state_id", "state_hash"):
            value = provenance.get(key)
            if value is not None:
                return str(value)
    # Custom test/callable collectors may omit the simulator hash.  The
    # episode/decision fallback is still stable and never conflates two
    # records from the same trajectory.
    return f"episode:{episode}:decision:{decision}"


def _sample_stratum(sample: Any) -> str:
    provenance = getattr(sample, "provenance", {})
    if isinstance(provenance, Mapping):
        explicit = provenance.get("retention_stratum")
        if isinstance(explicit, str) and explicit:
            return explicit
    target = getattr(sample, "target", None)
    mode = int(getattr(target, "mode", -1))
    if mode == 0:
        legal_play = np.asarray(getattr(sample, "legal_play", np.zeros(0)), dtype=bool)
        return "wait:play-eligible" if bool(legal_play.any()) else "wait:no-play"
    if mode != 1:
        raise ValueError(f"unsupported bootstrap target mode: {mode}")
    card = provenance.get("target_card") if isinstance(provenance, Mapping) else None
    if card is None:
        card = provenance.get("card_id") if isinstance(provenance, Mapping) else None
    if card is None:
        slot = int(getattr(target, "card_slot", -1))
        card = f"slot-{slot}"
    return f"play:{card}"


def _driver_stratum(stratum: str, driver_name: str | None) -> str:
    if driver_name and not stratum.startswith(f"{driver_name}:"):
        return f"{driver_name}:{stratum}"
    return stratum


def _target_retention_stratum(
    target: TeacherTarget,
    *,
    legal_play: np.ndarray,
    hand: Sequence[str],
    driver_name: str | None = None,
) -> str:
    """Describe the teacher target without changing its action semantics."""

    if target.mode == 0:
        base = "wait:play-eligible" if bool(np.asarray(legal_play).any()) else "wait:no-play"
    elif target.mode == 1:
        if 0 <= int(target.card_slot) < len(hand):
            base = f"play:{hand[int(target.card_slot)]}"
        else:
            base = f"play:slot-{int(target.card_slot)}"
    else:
        raise ValueError(f"unsupported bootstrap target mode: {target.mode}")
    return f"{driver_name}:{base}" if driver_name else base


class _EpisodeStratifiedReservoir:
    """Bound one episode while retaining support for every active stratum.

    Each active stratum has an independent uniform reservoir with capacity
    ``capacity``.  At episode end, one slot is reserved for every nonempty
    stratum and the remaining slots are allocated approximately in proportion
    to each stratum's eligible population (with undersized strata capped at
    their population).  Therefore every retained state in a stratum has the
    same conditional inclusion probability, including when a rare stratum
    first appears late in the trajectory.  At most ``capacity`` states are
    retained.  If there are more nonempty strata than slots, collection fails
    explicitly because the bound and complete support cannot both hold.
    """

    def __init__(self, capacity: int, rng: random.Random) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("episode reservoir capacity must be positive")
        self.capacity = capacity
        self.rng = rng
        self._reservoirs: dict[str, list[Any]] = {}
        self._eligible_counts: dict[str, int] = {}
        self._finalized: tuple[list[Any], dict[str, int]] | None = None

    def add(self, stratum: str, sample: Any) -> None:
        if self._finalized is not None:
            raise RuntimeError("cannot add to a finalized episode reservoir")
        if stratum not in self._reservoirs and len(self._reservoirs) >= self.capacity:
            raise RuntimeError(
                "episode reservoir capacity is smaller than the observed "
                "stratum count; increase samples_per_episode to retain every stratum"
            )
        bucket = self._reservoirs.setdefault(stratum, [])
        seen = self._eligible_counts.get(stratum, 0) + 1
        self._eligible_counts[stratum] = seen
        if len(bucket) < self.capacity:
            bucket.append(sample)
            return
        # Standard reservoir sampling: the bucket is a uniform sample of
        # min(seen, capacity) records from this stratum at every point in the
        # stream, independent of other strata or their arrival times.
        replacement = self.rng.randrange(seen)
        if replacement < self.capacity:
            bucket[replacement] = sample

    def _quota_counts(self) -> dict[str, int]:
        strata = tuple(self._reservoirs)
        if not strata:
            return {}
        eligible = self._eligible_counts
        total = sum(eligible.values())
        if total <= self.capacity:
            return dict(eligible)
        # Reserve one state for every nonempty stratum.  Allocate the remaining
        # slots by largest remainder from the eligible populations after that
        # mandatory coverage.  This is count-proportional up to integer quota
        # rounding and redistributes all slots from undersized strata.
        quota = {stratum: 1 for stratum in strata}
        remaining = self.capacity - len(strata)
        available = {
            stratum: eligible[stratum] - 1
            for stratum in strata
        }
        available_total = sum(available.values())
        if remaining <= 0 or available_total <= 0:
            return quota
        floors: dict[str, int] = {}
        fractions: list[tuple[float, str]] = []
        assigned = 0
        for stratum in strata:
            exact = remaining * available[stratum] / available_total
            floor = min(available[stratum], int(exact))
            floors[stratum] = floor
            assigned += floor
            fractions.append((exact - floor, stratum))
        quota.update({stratum: 1 + floors[stratum] for stratum in strata})
        left = remaining - assigned
        for _, stratum in sorted(fractions, key=lambda item: (-item[0], item[1])):
            if left <= 0:
                break
            if quota[stratum] < eligible[stratum]:
                quota[stratum] += 1
                left -= 1
        if left:
            # The largest-remainder pass can leave slots only when a group was
            # capped by its population.  Fill them deterministically from any
            # remaining room; this path is rare and still preserves support.
            for stratum in strata:
                if left <= 0:
                    break
                room = eligible[stratum] - quota[stratum]
                take = min(room, left)
                quota[stratum] += take
                left -= take
        return quota

    def finalize(self) -> tuple[list[Any], dict[str, int]]:
        if self._finalized is not None:
            return self._finalized
        quotas = self._quota_counts()
        retained: list[Any] = []
        retained_counts: dict[str, int] = {}
        for stratum, quota in quotas.items():
            bucket = self._reservoirs[stratum]
            if quota > len(bucket):
                raise RuntimeError(
                    f"reservoir for {stratum!r} retained only {len(bucket)} "
                    f"of requested {quota} states"
                )
            if quota == len(bucket):
                selected = list(bucket)
            else:
                # The bucket is already a uniform sample of the full group;
                # choosing a uniform subset keeps the final quota uniform.
                selected = self.rng.sample(bucket, quota)
            retained.extend(selected)
            retained_counts[stratum] = quota
        self._finalized = retained, retained_counts
        return self._finalized

    def samples(self) -> list[Any]:
        return list(self.finalize()[0])

    def retained_counts(self) -> dict[str, int]:
        return dict(self.finalize()[1])


def _capture_sample(
    env: SimulatorEnv,
    estimator: PublicStateEstimator,
    *,
    archetype: str,
    episode: int,
    decision: int,
    seed: int,
    teacher: BootstrapTeacher | None = None,
    action_teacher: BootstrapTeacher | None = None,
    driver_name: str | None = None,
) -> tuple[BootstrapSample, Any] | None:
    state = env.state
    if state is None:
        raise RuntimeError("bootstrap environment has no state")
    estimator.sync_from_env(env, viewer=0)
    observation = env.observe_v2_for_viewer(0)
    player = state.players[0]
    hand = tuple(player.hand[:4])
    if len(hand) != 4:
        return None
    elixir = float(player.elixir_milli) / 1000.0
    try:
        hand_tokens = build_hand_tokens(hand, elixir)
        legal_play = np.asarray(observation.legal_play, dtype=bool)
        legal_wait = bool(observation.legal_wait)
        teacher_callable = teacher if teacher is not None else teacher_label
        target = teacher_callable(
            hand_tokens=hand_tokens,
            entity_tokens=np.asarray(observation.entity_tokens),
            entity_mask=np.asarray(observation.entity_mask),
            legal_play=legal_play,
            legal_wait=legal_wait,
            own_elixir=elixir,
        )
        action_target = target
        if action_teacher is not None:
            action_target = action_teacher(
                hand_tokens=hand_tokens,
                entity_tokens=np.asarray(observation.entity_tokens),
                entity_mask=np.asarray(observation.entity_mask),
                legal_play=legal_play,
                legal_wait=legal_wait,
                own_elixir=elixir,
            )
    except TeacherError:
        if bool(observation.legal_wait):
            raise
        # Before the active phase the observation contract intentionally
        # exposes no WAIT action; the caller advances that short prelude with
        # a physical WAIT and does not record it as training data.
        return None
    if target.mode == 1:
        soft = soft_placement_target(
            target.row, target.col, np.asarray(legal_play[target.card_slot])
        )
    else:
        soft = np.zeros(legal_play.shape[1:], dtype=np.float32)
    snapshot = estimator.snapshot()
    retention_stratum = _target_retention_stratum(
        action_target if action_teacher is not None else target,
        legal_play=legal_play,
        hand=hand,
        driver_name=driver_name,
    )
    state_id = state.state_hash()
    sample = BootstrapSample(
        family=archetype,
        raster=np.array(observation.board, dtype=np.float32, copy=True),
        global_features=np.array(observation.global_vector, dtype=np.float32, copy=True),
        entity_tokens=np.array(observation.entity_tokens, dtype=np.float32, copy=True),
        entity_mask=np.array(observation.entity_mask, dtype=bool, copy=True),
        hand_tokens=np.array(hand_tokens, dtype=np.float32, copy=True),
        opp_hand_probs=np.array(snapshot.opp_hand_probs, dtype=np.float32, copy=True),
        opp_out_of_cycle=np.array(snapshot.opp_out_of_cycle, dtype=bool, copy=True),
        opp_elixir_interval=np.array(
            [snapshot.opp_elixir_lo, snapshot.opp_elixir_hi], dtype=np.float32, copy=True
        ),
        event_history=np.array(snapshot.event_history, dtype=np.float32, copy=True),
        legal_play=np.array(legal_play, copy=True),
        legal_wait=legal_wait,
        target=target,
        soft_placement=np.asarray(soft, dtype=np.float32),
        provenance={
            "generator": BOOTSTRAP_VERSION,
            "seed": seed,
            "episode": episode,
            "decision": decision,
            "elapsed_us": state.elapsed_us,
            "phase": state.phase,
            "archetype": archetype,
            "state_hash": state_id,
            "state_id": state_id,
            "own_elixir": elixir,
            "target_card": (
                str(hand[target.card_slot])
                if target.mode == 1 and 0 <= target.card_slot < len(hand)
                else None
            ),
            "driver_target_card": (
                str(hand[action_target.card_slot])
                if action_target.mode == 1 and 0 <= action_target.card_slot < len(hand)
                else None
            ),
            "retention_stratum": retention_stratum,
        },
        retention_stratum=retention_stratum,
    )
    if driver_name is not None:
        sample.provenance["teacher_driver"] = driver_name
    if action_target != target:
        sample.provenance["action_target"] = {
            "mode": int(action_target.mode),
            "card_slot": int(action_target.card_slot),
            "row": int(action_target.row),
            "col": int(action_target.col),
            "wait_duration_idx": int(action_target.wait_duration_idx),
        }
    return sample, _target_action(action_target, player=0)


def _invoke_capture_sample(
    env: SimulatorEnv,
    estimator: PublicStateEstimator,
    *,
    archetype: str,
    episode: int,
    decision: int,
    seed: int,
    teacher: BootstrapTeacher | None,
    action_teacher: BootstrapTeacher | None,
    driver_name: str | None,
) -> tuple[BootstrapSample, Any] | None:
    """Call the capture hook without widening the legacy monkeypatch shape."""

    kwargs: dict[str, object] = {
        "archetype": archetype,
        "episode": episode,
        "decision": decision,
        "seed": seed,
    }
    if teacher is not None:
        kwargs["teacher"] = teacher
    if action_teacher is not None:
        kwargs["action_teacher"] = action_teacher
    if driver_name is not None:
        kwargs["driver_name"] = driver_name
    return _capture_sample(env, estimator, **kwargs)  # type: ignore[arg-type]


def _annotate_catalog_sample(
    sample: Any,
    *,
    spec: BootstrapOpponentSpec,
    scenario: OpponentScenario,
    episode: int,
    terminal: bool,
    decisions: int,
    match_seed: int,
    eligible_count: int,
    retained_count: int,
) -> None:
    """Attach exact catalog/controller and inclusion provenance in-place."""

    provenance = getattr(sample, "provenance", None)
    if not isinstance(provenance, dict):
        provenance = {}
        sample.provenance = provenance
    provenance.update(
        {
            "opponent_deck_id": spec.deck_id,
            "opponent_deck_cards": tuple(spec.cards),
            "opponent_archetype": spec.archetype,
            "opponent_strategy": scenario.strategy,
            "opponent_controller_seed": int(scenario.controller_seed),
            "opponent_split": spec.split,
            "opponent_source_url": spec.source_url,
            "episode": episode,
            "episode_terminal": terminal,
            "episode_decisions": decisions,
            "match_seed": match_seed,
            "eligible_stratum_count": eligible_count,
            "retained_stratum_count": retained_count,
        }
    )


def _validate_catalog_specs(
    specs: Sequence[BootstrapOpponentSpec],
    *,
    reserved_ids: set[str],
    reserved_cards: set[frozenset[str]],
) -> None:
    seen_ids: set[str] = set()
    seen_cards: set[frozenset[str]] = set()
    for spec in specs:
        cards = frozenset(spec.cards)
        if spec.deck_id in reserved_ids or cards in reserved_cards:
            raise ValueError(
                "bootstrap opponent deck overlaps reserved final evaluation: "
                f"{spec.deck_id}"
            )
        if spec.split != "train":
            raise ValueError(
                "catalog bootstrap accepts train opponent specs only; "
                f"got split={spec.split!r} for {spec.deck_id!r}"
            )
        if spec.deck_id in seen_ids:
            raise ValueError(f"duplicate bootstrap opponent deck_id: {spec.deck_id}")
        if cards in seen_cards:
            raise ValueError(f"duplicate bootstrap opponent deck cards: {spec.deck_id}")
        seen_ids.add(spec.deck_id)
        seen_cards.add(cards)


def _generate_catalog_bootstrap_dataset(
    config: BootstrapConfig,
    *,
    specs: Sequence[BootstrapOpponentSpec],
    teacher: BootstrapTeacher | None,
    action_teacher: BootstrapTeacher | None,
    driver_name: str | None,
    episode_summaries: list[BootstrapEpisodeSummary] | None,
) -> list[BootstrapSample]:
    """Collect explicit real-deck episodes with driver/card/WAIT strata.

    Explicit mode intentionally runs the declared ``specs`` × ``repeats``
    episode grid.  ``BootstrapConfig.n_states`` remains required for
    backwards-compatible construction but does not truncate this fixed
    catalog grid; ``samples_per_episode`` is the per-episode retention bound.
    """

    reserved = [deck for deck in load_real_decks() if deck.split == "eval"]
    _validate_catalog_specs(
        specs,
        reserved_ids={str(deck.deck_id) for deck in reserved},
        reserved_cards={frozenset(deck.cards) for deck in reserved},
    )
    repeats = config.episodes_per_opponent
    if repeats is None:
        # Preserve the old convenience knob for callers migrating from one
        # episode per generated family, while keeping one episode per
        # explicit spec as the catalog default.
        repeats = config.episodes_per_archetype or 1
    ruleset = load_fixed_ruleset()
    all_samples: list[BootstrapSample] = []
    seen_state_ids: set[str] = set()
    episode = 0
    for spec in specs:
        for repeat in range(repeats):
            episode_seed = _stable_seed(
                config.seed, "bootstrap-catalog-episode", spec.deck_id, repeat
            )
            scenario = spec.as_scenario(
                episode_index=episode, match_seed=episode_seed
            )
            env = SimulatorEnv(
                BattleEngine(ruleset, validate_every_tick=False),
                decision_interval_us=250_000,
            )
            env.reset_v2(
                seed=episode_seed,
                decks=(tuple(PLAYER_DECK), tuple(spec.cards)),
                shuffle_decks=True,
            )
            estimator = PublicStateEstimator()
            opponent = scenario.build_controller()
            rng = random.Random(_stable_seed(config.seed, "catalog-reservoir", episode))
            # Keep every observed stratum within the declared hard bound;
            # a budget too small for support fails explicitly.
            reservoir = _EpisodeStratifiedReservoir(config.samples_per_episode, rng)
            eligible_counts: dict[str, int] = {}
            eligible_mode_counts = {0: 0, 1: 0}
            decisions_run = 0
            for decision in range(config.max_decisions):
                if env.state is None or env.state.terminal:
                    break
                captured = _invoke_capture_sample(
                    env,
                    estimator,
                    archetype=spec.archetype,
                    episode=episode,
                    decision=decision,
                    seed=config.seed,
                    teacher=teacher,
                    action_teacher=action_teacher,
                    driver_name=driver_name,
                )
                if captured is None:
                    observation = env.observe_v2_for_viewer(0)
                    learner_action = _fallback_action(
                        np.asarray(observation.legal_play, dtype=bool),
                        bool(observation.legal_wait),
                        player=0,
                        allow_countdown_wait=(env.state.phase == "countdown"),
                    )
                else:
                    sample, learner_action = captured
                    state_id = _state_id(sample, episode=episode, decision=decision)
                    # State hashes are the public-state identity contract.  A
                    # duplicate is an input-population error: silently
                    # dropping it would make eligible counts and inverse
                    # inclusion weights depend on an undocumented dedupe.
                    if state_id in seen_state_ids:
                        raise RuntimeError(
                            "catalog bootstrap observed duplicate state ID: "
                            f"{state_id}"
                        )
                    seen_state_ids.add(state_id)
                    if isinstance(getattr(sample, "provenance", None), dict):
                        sample.provenance["state_id"] = state_id
                    stratum = _driver_stratum(
                        _sample_stratum(sample), driver_name
                    )
                    if isinstance(getattr(sample, "provenance", None), dict):
                        sample.provenance["retention_stratum"] = stratum
                    try:
                        sample.retention_stratum = stratum
                    except (AttributeError, TypeError):
                        pass
                    eligible_counts[stratum] = eligible_counts.get(stratum, 0) + 1
                    mode = int(sample.target.mode)
                    eligible_mode_counts[mode] += 1
                    reservoir.add(stratum, sample)
                opponent_action = opponent.choose_action(env.engine, env.state, 1)
                env.step_v2((learner_action, opponent_action))
                decisions_run += 1
            terminal = bool(env.state is not None and env.state.terminal)
            if config.require_terminal and not terminal:
                raise RuntimeError(
                    f"bootstrap catalog episode {episode} reached the "
                    f"{config.max_decisions}-decision cap before termination"
                )
            retained = reservoir.samples()
            retained_counts = reservoir.retained_counts()
            if set(eligible_counts) - set(retained_counts):
                raise RuntimeError("catalog bootstrap lost an observed stratum")
            retained_mode_counts = {
                mode: sum(int(sample.target.mode) == mode for sample in retained)
                for mode in (0, 1)
            }
            for sample in retained:
                stratum = _sample_stratum(sample)
                _annotate_catalog_sample(
                    sample,
                    spec=spec,
                    scenario=scenario,
                    episode=episode,
                    terminal=terminal,
                    decisions=decisions_run,
                    match_seed=episode_seed,
                    eligible_count=eligible_counts[stratum],
                    retained_count=retained_counts[stratum],
                )
                sample.provenance["eligible_mode_count"] = eligible_mode_counts[
                    int(sample.target.mode)
                ]
                sample.provenance["retained_mode_count"] = retained_mode_counts[
                    int(sample.target.mode)
                ]
                sample.provenance["driver_stratum"] = stratum
                if driver_name is not None:
                    sample.provenance["teacher_driver"] = driver_name
                sample.provenance["state_id"] = _state_id(
                    sample, episode=episode, decision=int(sample.provenance.get("decision", 0))
                )
                try:
                    sample.retention_stratum = stratum
                except (AttributeError, TypeError):
                    pass
                sample.mode_weight = (
                    eligible_counts[stratum] / retained_counts[stratum]
                )
            retained.sort(key=lambda value: int(value.provenance.get("decision", 0)))
            if episode_summaries is not None:
                episode_summaries.append(
                    BootstrapEpisodeSummary(
                        episode=episode,
                        archetype=spec.archetype,
                        opponent_deck_id=spec.deck_id,
                        opponent_deck_cards=tuple(spec.cards),
                        decisions=decisions_run,
                        terminal=terminal,
                        eligible_wait=eligible_mode_counts[0],
                        eligible_play=eligible_mode_counts[1],
                        retained_wait=retained_mode_counts[0],
                        retained_play=retained_mode_counts[1],
                        opponent_strategy=scenario.strategy,
                        opponent_controller_seed=scenario.controller_seed,
                        opponent_source_url=spec.source_url,
                        match_seed=episode_seed,
                    )
                )
            all_samples.extend(retained)
            episode += 1
    total_weight = sum(sample.mode_weight for sample in all_samples)
    if not total_weight:
        raise RuntimeError("bootstrap retained no labeled states")
    scale = len(all_samples) / total_weight
    for sample in all_samples:
        sample.mode_weight *= scale
    if len({_state_id(sample, episode=0, decision=0) for sample in all_samples}) != len(
        all_samples
    ):
        raise RuntimeError("catalog bootstrap retained duplicate state IDs")
    return all_samples


def relabel_bootstrap_samples(
    samples: Sequence[BootstrapSample],
    teacher: BootstrapTeacher,
    *,
    teacher_name: str | None = None,
) -> list[BootstrapSample]:
    """Apply another public teacher to a retained input population.

    This is intended for a fixed-input teacher comparison: the returned
    records share every observation, state identity, retention stratum, and
    inclusion weight with ``samples``.  In particular, ``mode_weight`` is
    copied verbatim; relabeling must not silently change the population's
    natural-frequency correction.  The original driver/action provenance is
    retained alongside the new target.
    """

    relabeled: list[BootstrapSample] = []
    for sample in samples:
        provenance = dict(sample.provenance)
        own_elixir = provenance.get("own_elixir")
        if own_elixir is None:
            raise ValueError("retained bootstrap sample lacks own_elixir provenance")
        target = teacher(
            hand_tokens=np.asarray(sample.hand_tokens),
            entity_tokens=np.asarray(sample.entity_tokens),
            entity_mask=np.asarray(sample.entity_mask),
            legal_play=np.asarray(sample.legal_play, dtype=bool),
            legal_wait=bool(sample.legal_wait),
            own_elixir=float(own_elixir),
        )
        if target.mode == 1:
            soft = soft_placement_target(
                target.row,
                target.col,
                np.asarray(sample.legal_play[target.card_slot]),
            )
        else:
            soft = np.zeros(sample.legal_play.shape[1:], dtype=np.float32)
        provenance["relabel_teacher"] = teacher_name or getattr(
            teacher, "__name__", teacher.__class__.__name__
        )
        if is_dataclass(sample):
            relabeled.append(
                replace(
                    sample,
                    target=target,
                    soft_placement=np.asarray(soft, dtype=np.float32),
                    provenance=provenance,
                    mode_weight=sample.mode_weight,
                    retention_stratum=sample.retention_stratum,
                )
            )
        else:
            # Lightweight test collectors sometimes use namespace records;
            # keep this helper useful for those adapters without weakening the
            # typed BootstrapSample path.
            clone = copy.copy(sample)
            clone.target = target
            clone.soft_placement = np.asarray(soft, dtype=np.float32)
            clone.provenance = provenance
            clone.mode_weight = sample.mode_weight
            clone.retention_stratum = sample.retention_stratum
            relabeled.append(clone)
    return relabeled


def generate_bootstrap_dataset(
    config: BootstrapConfig,
    *,
    episode_summaries: list[BootstrapEpisodeSummary] | None = None,
    teacher: BootstrapTeacher | None = None,
    action_teacher: BootstrapTeacher | None = None,
    driver_name: str | None = None,
) -> list[BootstrapSample]:
    """Sample whole trajectories with mode coverage and bounded match influence.

    Independent per-mode reservoirs sample uniformly over each trajectory.
    Quotas prefer balanced retained records, never truncate a trajectory or
    discard every record from a nonempty episode/mode group. Each retained
    label receives the inverse sampling probability for its own group.  When
    ``config.opponent_specs`` is provided, explicit catalog identities and
    card/WAIT strata are used instead; ``teacher`` labels the public state and
    ``action_teacher`` may drive the trajectory independently.
    """

    if config.opponent_specs is not None:
        specs = tuple(_coerce_opponent_spec(value) for value in config.opponent_specs)
        return _generate_catalog_bootstrap_dataset(
            config,
            specs=specs,
            teacher=teacher,
            action_teacher=action_teacher,
            driver_name=driver_name,
            episode_summaries=episode_summaries,
        )

    ruleset = load_fixed_ruleset()
    pool = OpponentPool(ruleset, seed=config.seed)
    reserved_decks = {
        frozenset(deck.cards) for deck in load_real_decks() if deck.split == "eval"
    }
    samples: list[BootstrapSample] = []
    quotas = {
        archetype: config.n_states // len(config.archetypes)
        + (index < config.n_states % len(config.archetypes))
        for index, archetype in enumerate(config.archetypes)
    }
    mode_cap = config.samples_per_episode // 2
    remaining = {
        archetype: {
            1: (quota + (index % 2 == 0)) // 2,
            0: quota - (quota + (index % 2 == 0)) // 2,
        }
        for index, (archetype, quota) in enumerate(quotas.items())
    }
    episode_limits = {
        archetype: max(8, 8 * ((max(targets.values()) + mode_cap - 1) // mode_cap))
        for archetype, targets in remaining.items()
    }
    family_episodes = dict.fromkeys(config.archetypes, 0)
    episode = 0
    fixed_episode_count = (
        None if config.episodes_per_archetype is None
        else config.episodes_per_archetype * len(config.archetypes)
    )
    while (
        episode < fixed_episode_count
        if fixed_episode_count is not None
        else any(any(counts.values()) for counts in remaining.values())
    ):
        archetype = config.archetypes[episode % len(config.archetypes)]
        if fixed_episode_count is None and not any(remaining[archetype].values()):
            episode += 1
            continue
        if (
            fixed_episode_count is None
            and family_episodes[archetype] >= episode_limits[archetype]
        ):
            raise RuntimeError(
                f"bootstrap could not fill PLAY/WAIT quotas for {archetype} "
                f"after {family_episodes[archetype]} episodes: "
                f"missing {remaining[archetype]}; increase max_decisions or "
                "inspect teacher action coverage"
            )
        family_episodes[archetype] += 1
        # An observed group must retain support even if its preferred quota is
        # already filled while the other mode still needs examples.
        capacities = (
            {0: mode_cap, 1: mode_cap}
            if fixed_episode_count is not None
            else {
                mode: min(mode_cap, max(1, count))
                for mode, count in remaining[archetype].items()
            }
        )
        reservoirs: dict[int, list[BootstrapSample]] = {0: [], 1: []}
        seen = {0: 0, 1: 0}
        rng = random.Random(_stable_seed(config.seed, "bootstrap-reservoir", episode))
        episode_seed = _stable_seed(config.seed, "bootstrap-episode", episode)
        scenario = pool.sample(
            episode,
            archetype=archetype,
            strategy="deterministic-cycle",
            allow_variants=False,
        )
        if frozenset(scenario.deck.cards) in reserved_decks:
            raise RuntimeError(
                f"bootstrap opponent deck overlaps final evaluation: {scenario.deck.deck_id}"
            )
        env = SimulatorEnv(
            BattleEngine(ruleset, validate_every_tick=False),
            decision_interval_us=250_000,
        )
        decks = (tuple(PLAYER_DECK), tuple(scenario.deck.cards))
        env.reset_v2(seed=episode_seed, decks=decks, shuffle_decks=True)
        estimator = PublicStateEstimator()
        opponent = scenario.build_controller()
        decisions_run = 0
        for decision in range(config.max_decisions):
            if env.state is None or env.state.terminal:
                break
            captured = _invoke_capture_sample(
                env,
                estimator,
                archetype=archetype,
                episode=episode,
                decision=decision,
                seed=config.seed,
                teacher=teacher,
                action_teacher=action_teacher,
                driver_name=driver_name,
            )
            if captured is None:
                observation = env.observe_v2_for_viewer(0)
                learner_action = _fallback_action(
                    np.asarray(observation.legal_play, dtype=bool),
                    bool(observation.legal_wait),
                    player=0,
                    allow_countdown_wait=(env.state.phase == "countdown"),
                )
            else:
                sample, learner_action = captured
                sample.provenance["opponent_deck_id"] = scenario.deck.deck_id
                sample.provenance["opponent_deck_cards"] = tuple(scenario.deck.cards)
                mode = sample.target.mode
                seen[mode] += 1
                bucket = reservoirs[mode]
                if len(bucket) < capacities[mode]:
                    bucket.append(sample)
                elif capacities[mode]:
                    replacement = rng.randrange(seen[mode])
                    if replacement < capacities[mode]:
                        bucket[replacement] = sample
            opponent_action = opponent.choose_action(env.engine, env.state, 1)
            env.step_v2((learner_action, opponent_action))
            decisions_run += 1
        terminal = bool(env.state is not None and env.state.terminal)
        if config.require_terminal and not terminal:
            raise RuntimeError(
                f"bootstrap {archetype} episode {episode} reached the "
                f"{config.max_decisions}-decision cap before termination"
            )
        retained = reservoirs[0] + reservoirs[1]
        retained.sort(key=lambda sample: int(sample.provenance["decision"]))
        if episode_summaries is not None:
            episode_summaries.append(BootstrapEpisodeSummary(
                episode=episode,
                archetype=archetype,
                opponent_deck_id=scenario.deck.deck_id,
                opponent_deck_cards=tuple(scenario.deck.cards),
                decisions=decisions_run,
                terminal=terminal,
                eligible_wait=seen[0],
                eligible_play=seen[1],
                retained_wait=len(reservoirs[0]),
                retained_play=len(reservoirs[1]),
            ))
        for mode, bucket in reservoirs.items():
            if seen[mode] and not bucket:
                raise RuntimeError("bootstrap lost an observed episode/mode group")
            for sample in bucket:
                sample.mode_weight = seen[mode] / len(bucket)
                sample.provenance["eligible_mode_count"] = seen[mode]
                sample.provenance["retained_mode_count"] = len(bucket)
                sample.provenance["episode_terminal"] = terminal
                sample.provenance["episode_decisions"] = decisions_run
        samples.extend(retained)
        for mode in (0, 1):
            remaining[archetype][mode] = max(
                0, remaining[archetype][mode] - len(reservoirs[mode])
            )
        episode += 1
    # Normalize once for the complete dataset. Minibatches then use the mean
    # of weighted per-record losses without rebalancing each minibatch.
    total_weight = sum(sample.mode_weight for sample in samples)
    if not total_weight:
        raise RuntimeError("bootstrap retained no labeled states")
    scale = len(samples) / total_weight
    for sample in samples:
        sample.mode_weight *= scale
    if fixed_episode_count is not None and {sample.target.mode for sample in samples} != {0, 1}:
        raise RuntimeError("fixed bootstrap match batch did not observe both PLAY and WAIT")
    return samples


def to_torch_batch(samples: Sequence[BootstrapSample]) -> dict[str, object]:
    """Convert samples to one-step V4 actor tensors."""

    if not samples:
        raise ValueError("cannot batch an empty bootstrap sample sequence")
    import torch

    from .model_v4 import V4ActionBatch, masks_from_legal_play

    def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(getattr(sample, name), dtype=dtype) for sample in samples]
        ).unsqueeze(1)

    legal_play = torch.stack(
        [torch.as_tensor(sample.legal_play, dtype=torch.bool) for sample in samples]
    ).unsqueeze(1)
    mode = torch.tensor(
        [sample.target.mode for sample in samples], dtype=torch.long
    ).reshape(-1, 1)
    masks = masks_from_legal_play(
        legal_play,
        torch.tensor([sample.legal_wait for sample in samples], dtype=torch.bool).reshape(-1, 1),
    )
    actions = V4ActionBatch(
        mode=mode,
        card_slot=torch.tensor(
            [sample.target.card_slot for sample in samples], dtype=torch.long
        ).reshape(-1, 1),
        placement=torch.tensor(
            [[sample.target.row, sample.target.col] for sample in samples], dtype=torch.long
        ).reshape(-1, 1, 2),
        wait_duration=torch.tensor(
            [sample.target.wait_duration_idx for sample in samples], dtype=torch.long
        ).reshape(-1, 1),
    )
    return {
        "raster": stack("raster", torch.float32),
        "global_features": stack("global_features", torch.float32),
        "entities": stack("entity_tokens", torch.float32),
        "entity_mask": stack("entity_mask", torch.bool),
        "hand_tokens": stack("hand_tokens", torch.float32),
        "opp_hand_probs": stack("opp_hand_probs", torch.float32),
        "opp_out_of_cycle": stack("opp_out_of_cycle", torch.bool),
        "opp_elixir_interval": stack("opp_elixir_interval", torch.float32),
        "event_history": stack("event_history", torch.float32),
        "legal_play": legal_play,
        "masks": masks,
        "actions": actions,
        "soft_placement": torch.stack(
            [torch.as_tensor(sample.soft_placement, dtype=torch.float32) for sample in samples]
        ).unsqueeze(1),
        "mode_weights": torch.tensor(
            [sample.mode_weight for sample in samples], dtype=torch.float32
        ).reshape(-1, 1),
        "reset_mask": torch.ones(len(samples), 1, dtype=torch.bool),
        "families": [sample.family for sample in samples],
    }


def move_batch_to_device(batch: dict[str, object], device: object) -> dict[str, object]:
    """Move tensor and validated dataclass fields to ``device``."""

    import dataclasses
    import torch

    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            moved[key] = dataclasses.replace(
                value,
                **{
                    field.name: (
                        getattr(value, field.name).to(device)
                        if isinstance(getattr(value, field.name), torch.Tensor)
                        else getattr(value, field.name)
                    )
                    for field in dataclasses.fields(value)
                },
            )
        else:
            moved[key] = value
    return moved


__all__ = [
    "BOOTSTRAP_ARCHETYPES",
    "BOOTSTRAP_VERSION",
    "BootstrapOpponentSpec",
    "BootstrapConfig",
    "BootstrapEpisodeSummary",
    "BootstrapSample",
    "BootstrapTeacher",
    "catalog_training_opponent_specs",
    "generate_bootstrap_dataset",
    "move_batch_to_device",
    "real_catalog_training_opponent_specs",
    "relabel_bootstrap_samples",
    "to_torch_batch",
]
