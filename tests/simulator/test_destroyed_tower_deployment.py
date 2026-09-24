from __future__ import annotations

import pytest

from simulator.engine import BattleEngine
from simulator.geometry import GRID_COLS, GRID_ROWS, is_ground_cell


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("role", ["left", "right"])
@pytest.mark.parametrize("card_id", ["hog-rider", "cannon"])
def test_destroyed_friendly_princess_footprint_becomes_deployable(player, role, card_id):
    engine = BattleEngine()
    state = engine.new_battle(seed=0, shuffle_decks=False)
    tower = engine._tower(state, player, role)
    card = engine.ruleset.card(card_id)
    cell = (tower.x_mtile // 1000, tower.y_mtile // 1000)
    assert not is_ground_cell(cell)  # The immutable map includes live tower holes.
    assert not engine._legal_deployment(state, player, card, cell)
    assert cell not in engine._legal_cells_for_card(state, player, card)

    tower.alive = False
    tower.hp = 0
    assert engine._legal_deployment(state, player, card, cell)
    assert cell in engine._legal_cells_for_card(state, player, card)

    # Compare actual legal placements, including full building footprints,
    # across the entire board so both scalar and batched masks agree.
    expected = {
        (col, row)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
        if engine._legal_deployment(state, player, card, (col, row))
    }
    assert set(engine._legal_cells_for_card(state, player, card)) == expected
    assert not engine._territory_cell(state, player, (-1, cell[1]))
    assert not engine._territory_cell(state, player, (8, 15))  # River.
    king = engine._tower(state, player, "king")
    king.alive = False
    king.hp = 0
    king_cell = (king.x_mtile // 1000, king.y_mtile // 1000)
    assert not engine._territory_cell(state, player, king_cell)
    assert king_cell not in engine._deployment_territory_cells(state, player)
