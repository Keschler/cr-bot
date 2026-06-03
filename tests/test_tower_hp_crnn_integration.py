import pytest

pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from cr_bot.vision import tower_hp  # noqa: E402
from cr_bot.vision.tower_hp_ocr import TowerHPOCRPrediction  # noqa: E402


class FakeTowerHPOCR:
    def predict_batch(self, crops_by_tower, *, debug_steps_by_tower=None):
        predictions = {}
        for tower_name, crop in crops_by_tower.items():
            if debug_steps_by_tower is not None:
                debug_steps_by_tower.setdefault(tower_name, {})["raw"] = crop
            predictions[tower_name] = TowerHPOCRPrediction(
                value=1234,
                text="1234",
                readable_prob=0.99,
                char_confidence=0.99,
            )
        return predictions


def test_live_tower_hp_uses_crnn_batch(monkeypatch):
    monkeypatch.setattr(tower_hp, "get_tower_hp_ocr", lambda: FakeTowerHPOCR())
    monkeypatch.setattr(
        tower_hp,
        "detect_if_king_tower_activated",
        lambda _frame: {"own_king_activated": True, "enemy_king_activated": True},
    )
    monkeypatch.setattr(
        tower_hp,
        "detect_if_support_tower_alive",
        lambda _frame: {
            "support_left_activated": True,
            "support_right_activated": True,
            "enemy_support_left_activated": True,
            "enemy_support_right_activated": True,
        },
    )
    frame = np.zeros((2400, 1080, 3), dtype=np.uint8)
    debug = {}

    values = tower_hp.extract_tower_hp(frame, debug_steps_by_tower=debug)

    assert values == {
        "enemy_king": 1234,
        "own_king": 1234,
        "enemy_support_left": 1234,
        "enemy_support_right": 1234,
        "own_support_left": 1234,
        "own_support_right": 1234,
    }
    assert set(debug) == set(values)
