from __future__ import annotations

from pathlib import Path

import pytest

from simulator.actions import WaitAction
from simulator.engine import BattleEngine
from simulator.events import SimEvent
from simulator.packed_batch import (
    CapacityError,
    PackLimits,
    PackedBatch,
    WireFormatError,
    pack_batch,
    pack_state,
    unpack_batch,
    unpack_state,
)
from simulator.ruleset import load_ruleset
from simulator.state import StatusState


def _state(seed: int):
    engine = BattleEngine(load_ruleset("v1"), validate_every_tick=True)
    state = engine.new_battle(seed=seed, shuffle_decks=False)
    engine.step(state, (WaitAction(0), WaitAction(1)))

    # Exercise tuple-valued fields, nested status records, and event values in
    # addition to the ordinary towers and player state created by the engine.
    tower = min(state.entities.values(), key=lambda entity: entity.uid)
    tower.navigation_waypoints = [(101, 202), (-303, 404)]
    tower.statuses = [
        StatusState(
            kind="test-status",
            remaining_us=125_000,
            magnitude_permille=875,
            on_death_spawn_card_id="skeletons",
            on_death_spawn_count=2,
            on_death_spawn_owner=1,
            hit_speed_magnitude_permille=950,
        )
    ]
    state.events.append(
        SimEvent.create(
            state.tick,
            state.event_sequence + 1,
            "packed_batch_test",
            text="é",
            number=-17,
            enabled=True,
            missing=None,
        )
    )
    state.event_sequence += 1
    return state


def _hashes(state):
    return state.state_hash(), state.event_log_hash(), state.replay_hash()


def test_pack_unpack_preserves_canonical_state_and_replay_hashes() -> None:
    state = _state(17)
    packed = pack_state(state)

    assert packed.batch_size == 1
    assert packed.to_bytes() == pack_state(state).to_bytes()

    restored = unpack_state(packed)

    assert restored.to_primitive(include_events=True) == state.to_primitive(include_events=True)
    assert restored.canonical_json() == state.canonical_json()
    assert _hashes(restored) == _hashes(state)


def test_fixed_stride_batch_save_reload_preserves_each_lane(tmp_path: Path) -> None:
    states = (_state(21), _state(22))
    packed = pack_batch(states)

    assert packed.slot_bytes == 2 * 1024 * 1024
    assert len(packed.buffer) == packed.batch_size * packed.slot_bytes

    path = tmp_path / "packed-batch.crpb"
    packed.save(path)
    reloaded = PackedBatch.load(path)

    assert reloaded.to_bytes() == packed.to_bytes()
    restored = unpack_batch(reloaded)
    assert [_hashes(state) for state in restored] == [_hashes(state) for state in states]


def test_capacity_limits_reject_without_truncation() -> None:
    state = _state(31)

    with pytest.raises(CapacityError, match="encoded state"):
        pack_state(state, limits=PackLimits(max_state_bytes=1))

    with pytest.raises(CapacityError, match="entities"):
        pack_state(state, limits=PackLimits(max_entities=5))

    with pytest.raises(CapacityError, match="batch has"):
        pack_batch((state, _state(32)), limits=PackLimits(max_batch_size=1))


def test_wire_rejects_nonzero_padding_and_multi_lane_unpack_state() -> None:
    packed = pack_state(_state(41))
    encoded = bytearray(packed.to_bytes())
    encoded[-1] = 1

    with pytest.raises(WireFormatError, match="padding"):
        PackedBatch.from_bytes(encoded)

    multi = pack_batch((_state(42), _state(43)))
    with pytest.raises(ValueError, match="one lane"):
        unpack_state(multi)
