"""Exact-input and bounded-output-transfer regressions for card classifiers."""

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from cr_bot.vision import image_utils


def _reference_tensor(image, roi=None):
    x, y, width, height = roi or (0, 0, image.shape[1], image.shape[0])
    rgb = cv2.cvtColor(image[y:y + height, x:x + width], cv2.COLOR_BGR2RGB)
    return image_utils.CLASSIFIER_TRANSFORM(Image.fromarray(rgb))


def test_numpy_card_normalization_matches_every_byte_value_exactly():
    # Each channel contains all 256 input byte values. At the native transform
    # size, resizing is an identity so every normalization case is exercised.
    image = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)
    actual = image_utils._slot_image_tensor(image)
    expected = _reference_tensor(image)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32
    assert actual.shape == (3, 128, 128)
    assert actual.is_contiguous()


@pytest.mark.parametrize("shape,roi,strided", [
    ((300, 220, 3), image_utils.HAND_CARD_ART_ROI, False),
    ((100, 120, 3), None, False),
    ((60, 60, 3), image_utils.HAND_CARD_ART_ROI, False),
    ((300, 240, 3), (3, 7, 80, 90), True),
    ((100, 120, 4), None, False),
])
def test_card_preprocessing_preserves_resize_crop_channels_and_layout(shape, roi, strided):
    image = np.random.default_rng(17).integers(0, 256, shape, dtype=np.uint8)
    if strided:
        image = image[::2, ::2]
        assert not image.flags.c_contiguous
    actual = image_utils._slot_image_tensor(image, crop_roi=roi)
    assert torch.equal(actual, _reference_tensor(image, roi))
    assert actual.is_contiguous()


def test_nondefault_tensor_dtype_keeps_original_transform_behavior():
    image = np.random.default_rng(23).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = image_utils._slot_image_tensor(image)
        assert actual.dtype == torch.float64
        assert torch.equal(actual, _reference_tensor(image))
    finally:
        torch.set_default_dtype(previous_dtype)


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("count", [1, 4, 12])
def test_classifier_outputs_match_scalar_reference_without_per_slot_scalar_reads(monkeypatch, device, count):
    images = [np.full((100, 120, 3), index * 11, dtype=np.uint8) for index in range(count)]
    reference_batch = torch.stack([_reference_tensor(image) for image in images]).to(device)
    logits = torch.randn(count, 8, generator=torch.Generator().manual_seed(9)).to(device)
    # Equal logits retain the same first-index argmax semantics.
    logits[0] = 0.0
    classes = ["hog-rider", "the-log", "knight", "cannon", "fireball", "skeletons", "musketeer", "ice-spirit"]
    confidence, indices = torch.softmax(logits, dim=1).max(dim=1)
    expected = [(classes[index.item()], float(score.item() * 100.0)) for index, score in zip(indices, confidence)]

    class Model:
        def __call__(self, batch):
            assert torch.equal(batch, reference_batch)
            return logits

    monkeypatch.setattr(image_utils, "load_card_classifier", lambda _: SimpleNamespace(model=Model(), classes=classes, device=torch.device(device)))
    scalar_reads = []
    original_item = torch.Tensor.item
    original_tolist = torch.Tensor.tolist

    def counted_item(tensor, *args, **kwargs):
        scalar_reads.append(tensor.device.type)
        return original_item(tensor, *args, **kwargs)

    def cpu_tolist(tensor, *args, **kwargs):
        assert tensor.device.type == "cpu"
        return original_tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "tolist", cpu_tolist)
    if count == 1:
        actual = [image_utils._classify_card_with_model(images[0], "checkpoint")]
    else:
        actual = image_utils._classify_cards_with_model(images, "checkpoint")
    assert actual == expected
    assert scalar_reads == []


def test_missing_classifier_and_empty_batch_preserve_results(monkeypatch):
    monkeypatch.setattr(image_utils, "load_card_classifier", lambda _: None)
    assert image_utils._classify_card_with_model(None, "missing") == (None, None)
    assert image_utils._classify_cards_with_model([None] * 4, "missing") == [(None, None)] * 4
    assert image_utils._classify_cards_with_model([], "missing") == []
    monkeypatch.setattr(image_utils, "load_card_classifier", lambda _: SimpleNamespace())
    assert image_utils._classify_cards_with_model([], "checkpoint") == []


@pytest.mark.parametrize("score", [0.0, 0.001, 1.0, 99.5])
def test_low_confidence_next_card_does_not_trigger_orb_fallback(monkeypatch, score):
    monkeypatch.setattr(image_utils, "_is_evolved_slot", lambda _: (False, False))
    monkeypatch.setattr(image_utils, "_classify_card_with_model", lambda *args: ("the-log", score))

    def unexpected_fallback(*args):
        raise AssertionError("low confidence must not change the fallback policy")

    monkeypatch.setattr(image_utils, "_feature_match_card_template", unexpected_fallback)
    assert image_utils.classify_card_for_slot(None, {}, {}, "next_card") == ("log", score)


@pytest.mark.parametrize("evolved", [False, True])
def test_missing_next_classifier_keeps_template_selection_and_score(monkeypatch, evolved):
    base, evo = {"base": object()}, {"evo": object()}
    monkeypatch.setattr(image_utils, "_is_evolved_slot", lambda _: (evolved, evolved))
    monkeypatch.setattr(image_utils, "_classify_card_with_model", lambda *args: (None, None))
    templates_used = []

    def fallback(image, templates):
        templates_used.append(templates)
        return "hog-rider-ev1", 27

    monkeypatch.setattr(image_utils, "_feature_match_card_template", fallback)
    assert image_utils.classify_card_for_slot(None, base, evo, "next_card") == (
        "evo-hog-rider" if evolved else "hog-rider", 27,
    )
    assert templates_used == [evo if evolved else base]
