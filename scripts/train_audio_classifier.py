from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.audio.dataset import (
    NO_EVENT_CLASS,
    GameplayBackground,
    MixedSFXCardDataset,
    build_real_event_windows,
    collect_sfx_files,
    feature_config_to_dict,
    split_sfx_samples,
)
from cr_bot.audio.features import AudioFeatureConfig
from cr_bot.audio.model import AudioCardCNN
from cr_bot.domain.card_metadata import CARD_METADATA


DEFAULT_BACKGROUND = ROOT / "dataset_generation/data/video_clips/2.6Hog_Cycle_broken.wav"
DEFAULT_GROUND_TRUTH = ROOT / "data/eval/ground_truth/2hog_cycle_champion.json"
DEFAULT_CHECKPOINT = ROOT / "assets/models/audio_card_classifier_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-card Clash Royale audio classifier.")
    parser.add_argument("--raw-sfx-dir", type=Path, default=ROOT / "data/audio_classifier/raw_sfx")
    parser.add_argument("--background-audio", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--samples-per-sfx", type=int, default=8)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-all-sfx", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = AudioFeatureConfig()
    known_cards = set(CARD_METADATA)
    samples, _, _ = collect_sfx_files(
        args.raw_sfx_dir,
        deploy_only=not args.include_all_sfx,
        known_cards=known_cards,
    )
    samples = limit_samples_per_class(samples, args.max_samples_per_class)
    if not samples:
        raise SystemExit(f"No SFX samples found in {args.raw_sfx_dir}")

    train_samples, val_samples = split_sfx_samples(samples, seed=args.seed)
    classes = [NO_EVENT_CLASS] + sorted(CARD_METADATA)

    background_paths = [args.background_audio] if args.background_audio.exists() else []
    background = None
    if background_paths:
        background = GameplayBackground(
            background_paths,
            config,
            ground_truth_path=args.ground_truth if args.ground_truth.exists() else None,
        )

    train_dataset = MixedSFXCardDataset(
        train_samples,
        classes,
        config,
        background=background,
        samples_per_sfx=args.samples_per_sfx,
        seed=args.seed,
    )
    val_dataset = MixedSFXCardDataset(
        val_samples,
        classes,
        config,
        background=background,
        samples_per_sfx=max(1, args.samples_per_sfx // 2),
        seed=args.seed + 100_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCardCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1.0

    print(f"classes={len(classes)} train_sfx={len(train_samples)} val_sfx={len(val_samples)}")
    print(f"background={'yes' if background is not None and background.available else 'no'}")
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "feature_config": feature_config_to_dict(config),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "raw_sfx_dir": str(args.raw_sfx_dir),
                },
                args.output,
            )
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    if args.ground_truth.exists() and args.background_audio.exists():
        evaluate_real_ground_truth(args.output, args.background_audio, args.ground_truth, classes, config, device)


def run_epoch(model, loader, loss_fn, device, *, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_real_ground_truth(
    checkpoint_path: Path,
    audio_path: Path,
    ground_truth_path: Path,
    classes: list[str],
    config: AudioFeatureConfig,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AudioCardCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    windows = build_real_event_windows(audio_path, ground_truth_path, classes, config)
    if not windows:
        print("real_gt_acc=skipped no matching ground-truth events")
        return
    correct = 0
    with torch.no_grad():
        for features, label, _event in windows:
            logits = model(features.unsqueeze(0).to(device))
            correct += int(logits.argmax(dim=1).item() == label)
    print(f"real_gt_acc={correct / len(windows):.4f} real_gt_events={len(windows)}")


def limit_samples_per_class(
    samples: list[tuple[str, Path]],
    max_samples_per_class: int | None,
) -> list[tuple[str, Path]]:
    if max_samples_per_class is None:
        return samples
    counts: dict[str, int] = {}
    limited = []
    for card, path in samples:
        count = counts.get(card, 0)
        if count >= max_samples_per_class:
            continue
        counts[card] = count + 1
        limited.append((card, path))
    return limited


if __name__ == "__main__":
    main()
