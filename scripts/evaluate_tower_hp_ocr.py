from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.vision.tower_hp_ocr import (  # noqa: E402
    TOWER_HP_OCR_PATH,
    TowerHPCRNN,
    decode_ctc_logits,
    normalize_tower_hp_crop,
)


@dataclass
class EvalSample:
    row_index: int
    row: dict[str, str]
    image_path: Path
    readable: bool
    label: str


def parse_bool(value: str) -> bool | None:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "readable"}:
        return True
    if text in {"0", "false", "no", "n", "unreadable"}:
        return False
    return None


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def load_samples(csv_path: Path) -> list[EvalSample]:
    samples = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            readable = parse_bool(row.get("readable", ""))
            if readable is None:
                continue
            label = (row.get("label") or "").strip()
            if readable and (not label.isdigit() or len(label) > 4):
                continue
            samples.append(
                EvalSample(
                    row_index=row_index,
                    row=dict(row),
                    image_path=resolve_path(row["image_path"]),
                    readable=readable,
                    label=label,
                )
            )
    return samples


def validation_subset(samples: list[EvalSample], val_fraction: float, seed: int) -> list[EvalSample]:
    val_size = max(1, int(round(len(samples) * val_fraction)))
    train_size = len(samples) - val_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(samples), generator=generator).tolist()
    val_indices = permutation[train_size:]
    return [samples[idx] for idx in val_indices]


@torch.inference_mode()
def predict(model: TowerHPCRNN, sample: EvalSample, device: torch.device) -> tuple[str, float]:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(sample.image_path)
    tensor, _normalized = normalize_tower_hp_crop(image)
    logits, readable_logits = model(tensor.unsqueeze(0).to(device))
    text, confidence = decode_ctc_logits(logits[0].cpu())
    readable_prob = float(torch.sigmoid(readable_logits[0]).cpu().item())
    return text, min(confidence, readable_prob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List tower HP CRNN OCR mistakes.")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "outputs/tower_hp_ocr_crops/labels.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/models/tower_hp_crnn_candidate.pt")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs/tower_hp_ocr_eval_errors.csv")
    parser.add_argument("--split", choices=("val", "all"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.labels_csv)
    eval_samples = (
        validation_subset(samples, args.val_fraction, args.seed)
        if args.split == "val"
        else samples
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint if args.checkpoint.exists() else TOWER_HP_OCR_PATH
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TowerHPCRNN().to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    errors = []
    readable_total = 0
    readable_wrong = 0
    unreadable_total = 0
    unreadable_wrong = 0
    for sample in eval_samples:
        predicted_text, confidence = predict(model, sample, device)
        predicted_readable = bool(predicted_text)
        if sample.readable:
            readable_total += 1
            wrong = predicted_text != sample.label
            if wrong:
                readable_wrong += 1
        else:
            unreadable_total += 1
            wrong = predicted_readable
            if wrong:
                unreadable_wrong += 1
        if not wrong:
            continue

        errors.append(
            {
                "row_index": sample.row_index,
                "image_path": sample.row["image_path"],
                "tower_name": sample.row.get("tower_name", ""),
                "frame_index": sample.row.get("frame_index", ""),
                "video_time_s": sample.row.get("video_time_s", ""),
                "crop_mode": sample.row.get("crop_mode", ""),
                "expected_readable": str(sample.readable).lower(),
                "expected_label": sample.label,
                "predicted_label": predicted_text,
                "confidence": f"{confidence:.4f}",
                "model_ocr": predicted_text,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "image_path",
        "tower_name",
        "frame_index",
        "video_time_s",
        "crop_mode",
        "expected_readable",
        "expected_label",
        "predicted_label",
        "confidence",
        "model_ocr",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    print(f"checkpoint: {checkpoint_path}")
    print(f"split: {args.split} samples={len(eval_samples)}")
    print(f"readable wrong: {readable_wrong}/{readable_total}")
    print(f"unreadable wrong: {unreadable_wrong}/{unreadable_total}")
    print(f"errors: {len(errors)}")
    print(f"wrote: {args.output_csv}")
    for error in errors[:30]:
        print(
            f"{error['image_path']} expected={error['expected_label'] or 'unreadable'} "
            f"pred={error['predicted_label'] or 'unreadable'} "
            f"tower={error['tower_name']} frame={error['frame_index']}"
        )


if __name__ == "__main__":
    main()
