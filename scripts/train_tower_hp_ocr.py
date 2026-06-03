from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import sys

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.vision.tower_hp_ocr import (  # noqa: E402
    TOWER_HP_OCR_PATH,
    TowerHPCRNN,
    decode_ctc_logits,
    encode_text,
    normalize_tower_hp_crop,
)


@dataclass
class Sample:
    image_path: Path
    readable: bool
    label: str


class TowerHPOCRDataset(Dataset):
    def __init__(self, samples: list[Sample], *, augment: bool = False) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.image_path)
        if self.augment:
            image = augment_image(image)
        tensor, _normalized = normalize_tower_hp_crop(image)
        target = encode_text(sample.label) if sample.readable else []
        return {
            "image": tensor,
            "readable": torch.tensor(1.0 if sample.readable else 0.0),
            "target": torch.tensor(target, dtype=torch.long),
            "text": sample.label,
        }


def augment_image(image):
    if random.random() < 0.35:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-20, 20)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if random.random() < 0.20:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if random.random() < 0.20:
        noise = torch.normal(0, 4, size=image.shape).numpy()
        image = (image.astype("float32") + noise).clip(0, 255).astype("uint8")
    return image


def parse_bool(value: str) -> bool | None:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "readable"}:
        return True
    if text in {"0", "false", "no", "n", "unreadable"}:
        return False
    return None


def load_samples(csv_path: Path) -> list[Sample]:
    samples = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            readable = parse_bool(row.get("readable", ""))
            if readable is None:
                continue
            label = (row.get("label") or "").strip()
            if readable and (not label.isdigit() or len(label) > 4):
                continue
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            samples.append(Sample(image_path=image_path, readable=readable, label=label))
    return samples


def collate_batch(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    readable = torch.stack([item["readable"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    non_empty_targets = [target for target in targets if len(target) > 0]
    if non_empty_targets:
        flat_targets = torch.cat(non_empty_targets, dim=0)
    else:
        flat_targets = torch.empty(0, dtype=torch.long)
    texts = [item["text"] for item in batch]
    return images, readable, flat_targets, target_lengths, texts


def split_dataset(samples: list[Sample], val_fraction: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(round(len(samples) * val_fraction)))
    train_size = len(samples) - val_size
    base = TowerHPOCRDataset(samples)
    train_subset, val_subset = random_split(base, [train_size, val_size], generator=generator)
    train_samples = [samples[idx] for idx in train_subset.indices]
    val_samples = [samples[idx] for idx in val_subset.indices]
    return TowerHPOCRDataset(train_samples, augment=True), TowerHPOCRDataset(val_samples)


def batch_loss(model, batch, ctc_loss, readable_loss, device):
    images, readable, flat_targets, target_lengths, _texts = batch
    images = images.to(device)
    readable = readable.to(device)
    flat_targets = flat_targets.to(device)
    target_lengths = target_lengths.to(device)

    logits, readable_logits = model(images)
    bce = readable_loss(readable_logits, readable)
    readable_indices = torch.where(readable > 0.5)[0]
    if len(readable_indices) == 0:
        ctc = torch.tensor(0.0, device=device)
    else:
        readable_logits_seq = logits[readable_indices]
        input_lengths = torch.full(
            (len(readable_indices),),
            readable_logits_seq.shape[1],
            dtype=torch.long,
            device=device,
        )
        readable_target_lengths = target_lengths[readable_indices]
        ctc = ctc_loss(
            readable_logits_seq.log_softmax(dim=-1).permute(1, 0, 2),
            flat_targets,
            input_lengths,
            readable_target_lengths,
        )
    return ctc + bce, ctc.detach(), bce.detach()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    readable_total = 0
    readable_exact = 0
    unreadable_tp = 0
    unreadable_fp = 0
    unreadable_fn = 0

    for images, readable, _flat_targets, _target_lengths, texts in loader:
        images = images.to(device)
        readable = readable.to(device)
        logits, readable_logits = model(images)
        readable_probs = torch.sigmoid(readable_logits)
        for idx, expected_text in enumerate(texts):
            expected_readable = bool(readable[idx].item() > 0.5)
            predicted_readable = bool(readable_probs[idx].item() >= 0.5)
            if expected_readable:
                readable_total += 1
                decoded, _confidence = decode_ctc_logits(logits[idx].cpu())
                if predicted_readable and decoded == expected_text:
                    readable_exact += 1
            else:
                if not predicted_readable:
                    unreadable_tp += 1
                else:
                    unreadable_fn += 1
            if predicted_readable and not expected_readable:
                unreadable_fp += 1

    exact_acc = readable_exact / readable_total if readable_total else 0.0
    precision = unreadable_tp / (unreadable_tp + unreadable_fp) if unreadable_tp + unreadable_fp else 0.0
    recall = unreadable_tp / (unreadable_tp + unreadable_fn) if unreadable_tp + unreadable_fn else 0.0
    unreadable_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return exact_acc, unreadable_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tower HP CRNN OCR model.")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "outputs/tower_hp_ocr_crops/labels.csv")
    parser.add_argument("--candidate-output", type=Path, default=ROOT / "outputs/models/tower_hp_crnn_candidate.pt")
    parser.add_argument("--promote-output", type=Path, default=TOWER_HP_OCR_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-exact-acc", type=float, default=0.99)
    parser.add_argument("--min-unreadable-f1", type=float, default=0.98)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.labels_csv)
    if len(samples) < 10:
        raise RuntimeError(f"not enough reviewed samples in {args.labels_csv}: {len(samples)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, val_ds = split_dataset(samples, args.val_fraction, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = TowerHPCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    readable_loss = nn.BCEWithLogitsLoss()

    best_score = -1.0
    promoted = False
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.promote_output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ctc, _bce = batch_loss(model, batch, ctc_loss, readable_loss, device)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        exact_acc, unreadable_f1 = evaluate(model, val_loader, device)
        score = exact_acc + unreadable_f1
        print(
            f"epoch={epoch:03d} loss={total_loss / max(1, len(train_loader)):.4f} "
            f"val_exact_acc={exact_acc:.4f} val_unreadable_f1={unreadable_f1:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_exact_acc": exact_acc,
                    "val_unreadable_f1": unreadable_f1,
                },
                args.candidate_output,
            )
        if exact_acc >= args.min_exact_acc and unreadable_f1 >= args.min_unreadable_f1:
            shutil.copyfile(args.candidate_output, args.promote_output)
            promoted = True

    if promoted:
        print(f"promoted checkpoint: {args.promote_output}")
    else:
        print(f"candidate checkpoint: {args.candidate_output}")
        print(
            "not promoted because validation did not reach "
            f"exact_acc>={args.min_exact_acc:.3f} and unreadable_f1>={args.min_unreadable_f1:.3f}"
        )


if __name__ == "__main__":
    main()
