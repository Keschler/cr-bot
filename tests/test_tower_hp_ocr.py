import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from cr_bot.vision.tower_hp_ocr import (  # noqa: E402
    TowerHPCRNN,
    TowerHPOCR,
    decode_ctc_logits,
    normalize_tower_hp_crop,
    validate_tower_hp_prediction,
)


def test_crnn_forward_shapes():
    model = TowerHPCRNN()
    logits, readable_logits = model(torch.zeros(2, 1, 32, 128))

    assert logits.shape[0] == 2
    assert logits.shape[2] == 11
    assert readable_logits.shape == (2,)


def test_decode_ctc_collapses_repeats_and_blanks():
    logits = torch.full((8, 11), -10.0)
    sequence = [0, 4, 4, 0, 4, 2, 0, 4]
    for idx, cls in enumerate(sequence):
        logits[idx, cls] = 10.0

    text, confidence = decode_ctc_logits(logits)

    assert text == "3313"
    assert confidence > 0.99


def test_normalize_tower_hp_crop_returns_model_input():
    image = np.zeros((12, 40, 3), dtype=np.uint8)

    tensor, normalized = normalize_tower_hp_crop(image)

    assert tensor.shape == (1, 32, 128)
    assert normalized.shape == (32, 128)


def test_validate_prediction_rejects_unreadable_and_out_of_range():
    assert validate_tower_hp_prediction(
        text="4424",
        readable_prob=0.2,
        char_confidence=0.9,
        tower_name="enemy_support_left",
        min_readable_prob=0.65,
        min_char_confidence=0.45,
    ) == (None, "unreadable")
    assert validate_tower_hp_prediction(
        text="9999",
        readable_prob=0.9,
        char_confidence=0.9,
        tower_name="enemy_support_left",
        min_readable_prob=0.65,
        min_char_confidence=0.45,
    ) == (None, "above_max_hp")


def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Tower HP CRNN checkpoint is missing"):
        TowerHPOCR(tmp_path / "missing.pt")
