from __future__ import annotations

import numpy as np

from cr_bot.features.action_space import get_card_deploy_mask
from cr_bot.domain.game_state import GameState


def get_action_mask(card_name: str, state: GameState) -> np.ndarray:
    towers = state.hud.princess_towers
    return get_card_deploy_mask(card_name, **towers.as_deploy_kwargs())
