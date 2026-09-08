from __future__ import annotations

from typing import Any


INFERENCE_DEVICES = ("auto", "cpu", "cuda")


def _cuda_available() -> bool:
    """Whether this environment can run CUDA inference (never raises)."""
    try:
        import torch  # lazy: keeps module import light
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_inference_devices(device: Any) -> tuple[str, str]:
    """Map a UI/API device choice to ``(yolo_device, torch_device)``.

    Accepted names (case-insensitive): ``"auto"``, ``"cpu"``, ``"cuda"``.
    ``None``/empty means ``"auto"``. Explicit ``"cpu"``
    and ``"cuda"`` override the ``YOLO_DEVICE``/``CARD_CLASSIFIER_DEVICE``
    environment; ``"auto"`` keeps the existing auto-selection. Raises
    ``ValueError`` for unknown names or when ``"cuda"`` is requested but
    unavailable.
    """

    if device is not None and not isinstance(device, str):
        raise ValueError(
            f"device must be one of {', '.join(INFERENCE_DEVICES)}, "
            f"got {device!r}"
        )
    name = (device or "").strip().lower() or "auto"
    if name not in INFERENCE_DEVICES:
        raise ValueError(
            f"device must be one of {', '.join(INFERENCE_DEVICES)}, "
            f"got {device!r}"
        )
    if name == "cpu":
        return "cpu", "cpu"
    cuda = _cuda_available()
    if name == "cuda":
        if not cuda:
            raise ValueError(
                "device 'cuda' requested but CUDA is not available "
                "in this environment"
            )
        return "cuda", "cuda"
    try:
        from cr_bot.vision.model_loader import yolo_device
    except ImportError:
        return ("cuda" if cuda else "cpu"), ("cuda" if cuda else "cpu")
    return yolo_device(), ("cuda" if cuda else "cpu")
