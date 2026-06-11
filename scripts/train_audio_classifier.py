from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.audio.dataset import (
    NO_EVENT_CLASS,
    GameplayBackground,
    MixedSFXCardDataset,
    collect_sfx_files,
    extract_mono_wav_from_video,
    feature_config_to_dict,
    load_ground_truth_events,
    split_sfx_samples,
)
from cr_bot.audio.features import AudioFeatureConfig, load_audio_window, waveform_to_log_mel
from cr_bot.audio.labels import audio_card_classes, normalize_card_key
from cr_bot.audio.manifest_dataset import ManifestAudioDataset
from cr_bot.audio.metrics import summarize_predictions
from cr_bot.audio.model import AudioCardCNN
from cr_bot.domain.card_metadata import CARD_METADATA


DEFAULT_BACKGROUND = ROOT / "dataset_generation/data/video_clips/2.6Hog_Cycle_broken.wav"
DEFAULT_GROUND_TRUTH = ROOT / "data/eval/ground_truth/2hog_cycle_champion.json"
DEFAULT_CHECKPOINT = ROOT / "assets/models/audio_card_classifier_best.pt"
DEFAULT_MINED_ROOT = ROOT / "data/audio_classifier/mined"
DEFAULT_EVAL_WAVS = {
    ROOT / "data/eval/ground_truth/2hog_cycle_champion.json": ROOT / "dataset_generation/data/video_clips/2.6Hog_Cycle_broken.wav",
    ROOT / "data/eval/ground_truth/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].json": ROOT / "dataset_generation/data/video_clips/downloaded_videos/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].wav",
}
DEFAULT_EVAL_VIDEOS = {
    ROOT / "data/eval/ground_truth/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].json": ROOT / "dataset_generation/data/video_clips/downloaded_videos/HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4",
    ROOT / "data/eval/ground_truth/2hog_cycle_champion.json": ROOT / "dataset_generation/data/video_clips/10_fps_2.6HogCycle.mp4",
}
OVERTIME_BACKGROUND_RANGES_S = [
    (2 * 60 + 54, 3 * 60 + 16),
    (6 * 60 + 15, 6 * 60 + 31),
    (12 * 60 + 16, 13 * 60 + 30),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a per-card Clash Royale audio classifier.")
    parser.add_argument("--raw-sfx-dir", type=Path, default=ROOT / "data/audio_classifier/raw_sfx")
    parser.add_argument("--background-audio", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--real-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--samples-per-sfx", type=int, default=8)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-all-sfx", action="store_true")
    parser.add_argument("--real-train-manifest", type=Path, default=DEFAULT_MINED_ROOT / "manifests/train.jsonl")
    parser.add_argument("--real-val-manifest", type=Path, default=DEFAULT_MINED_ROOT / "manifests/val.jsonl")
    parser.add_argument("--real-test-manifest", type=Path, default=DEFAULT_MINED_ROOT / "manifests/test.jsonl")
    parser.add_argument("--real-quality-tiers", default="gold,silver,bronze")
    parser.add_argument("--real-val-quality-tiers", default="gold,silver")
    parser.add_argument("--mode", choices=("synthetic", "real", "hybrid"), default="hybrid")
    parser.add_argument("--mix-real-and-synthetic", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = AudioFeatureConfig()
    classes = [NO_EVENT_CLASS] + audio_card_classes(CARD_METADATA, raw_sfx_dir=args.raw_sfx_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCardCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    synthetic_loaders = None
    if args.mode in {"synthetic", "hybrid"}:
        synthetic_loaders = build_synthetic_loaders(args, classes, config)
        train_model(
            model,
            synthetic_loaders["train"],
            synthetic_loaders["val"],
            loss_fn,
            optimizer,
            device,
            epochs=args.epochs,
            output_path=args.output,
            checkpoint_extra={
                "classes": classes,
                "feature_config": feature_config_to_dict(config),
                "raw_sfx_dir": str(args.raw_sfx_dir),
            },
        )

    if args.mode in {"real", "hybrid"} and args.real_train_manifest.exists():
        real_loaders = build_real_loaders(args, classes, config, synthetic_train_loader=synthetic_loaders["train"] if synthetic_loaders else None)
        train_model(
            model,
            real_loaders["train"],
            real_loaders["val"],
            loss_fn,
            optimizer,
            device,
            epochs=args.real_epochs,
            output_path=args.output,
            checkpoint_extra={
                "classes": classes,
                "feature_config": feature_config_to_dict(config),
                "real_train_manifest": str(args.real_train_manifest),
            },
            stage_name="real",
        )
        if args.real_test_manifest.exists():
            metrics = evaluate_manifest(model, args.real_test_manifest, classes, config, device)
            print(f"real_manifest_test_acc={metrics['accuracy']:.4f} real_manifest_test_events={metrics['event_count']}")

    if args.ground_truth.exists() and args.background_audio.exists():
        metrics = evaluate_real_ground_truth(model, args.background_audio, args.ground_truth, classes, config, device)
        print(f"default_real_gt_acc={metrics['accuracy']:.4f} default_real_gt_events={metrics['event_count']}")

    evaluate_known_ground_truths(model, classes, config, device)


def build_synthetic_loaders(args, classes: list[str], config: AudioFeatureConfig) -> dict[str, DataLoader]:
    known_cards = set(classes)
    samples, _, _ = collect_sfx_files(
        args.raw_sfx_dir,
        deploy_only=not args.include_all_sfx,
        known_cards=known_cards,
    )
    samples = limit_samples_per_class(samples, args.max_samples_per_class)
    if not samples:
        raise SystemExit(f"No SFX samples found in {args.raw_sfx_dir}")

    train_samples, val_samples = split_sfx_samples(
        samples,
        samples_per_sfx=args.samples_per_sfx,
        seed=args.seed,
    )
    val_no_event_count = int(round(len(samples) * 0.2))
    train_no_event_count = len(samples) - val_no_event_count
    background_paths = [args.background_audio] if args.background_audio.exists() else []
    background = None
    if background_paths:
        background = GameplayBackground(
            background_paths,
            config,
            ground_truth_path=args.ground_truth if args.ground_truth.exists() else None,
            overtime_ranges_s=OVERTIME_BACKGROUND_RANGES_S,
        )
    train_dataset = MixedSFXCardDataset(
        samples,
        classes,
        config,
        background=background,
        samples_per_sfx=args.samples_per_sfx,
        positive_samples=train_samples,
        no_event_count=train_no_event_count,
        seed=args.seed,
    )
    val_dataset = MixedSFXCardDataset(
        samples,
        classes,
        config,
        background=background,
        samples_per_sfx=args.samples_per_sfx,
        positive_samples=val_samples,
        no_event_count=val_no_event_count,
        seed=args.seed + 100_000,
    )
    print(
        f"synthetic_classes={len(classes)} source_sfx={len(samples)} "
        f"train_generated={len(train_samples)} val_generated={len(val_samples)} "
        f"background={'yes' if background is not None and background.available else 'no'}"
    )
    return {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }


def build_real_loaders(args, classes: list[str], config: AudioFeatureConfig, *, synthetic_train_loader: DataLoader | None) -> dict[str, DataLoader]:
    train_quality_tiers = parse_quality_tiers(args.real_quality_tiers)
    val_quality_tiers = parse_quality_tiers(args.real_val_quality_tiers)
    train_dataset = ManifestAudioDataset(args.real_train_manifest, classes, config, quality_tiers=train_quality_tiers)
    val_path = args.real_val_manifest if args.real_val_manifest.exists() else args.real_train_manifest
    val_dataset = ManifestAudioDataset(val_path, classes, config, quality_tiers=val_quality_tiers)
    if args.mix_real_and_synthetic and synthetic_train_loader is not None:
        train_dataset = ConcatDataset([synthetic_train_loader.dataset, train_dataset])
    print(
        f"real_train_rows={len(train_dataset)} real_val_rows={len(val_dataset)} "
        f"train_quality_tiers={sorted(train_quality_tiers)} val_quality_tiers={sorted(val_quality_tiers)}"
    )
    return {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }


def parse_quality_tiers(value: str) -> set[str]:
    allowed = {"gold", "silver", "bronze"}
    tiers = {
        part.strip().lower()
        for part in str(value).split(",")
        if part.strip()
    }
    if not tiers:
        raise SystemExit("--real-quality-tiers must include at least one of gold,silver,bronze")
    invalid = sorted(tiers - allowed)
    if invalid:
        raise SystemExit(
            f"Unsupported quality tiers: {', '.join(invalid)}. Expected a comma-separated subset of gold,silver,bronze."
        )
    return tiers


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    *,
    epochs: int,
    output_path: Path,
    checkpoint_extra: dict,
    stage_name: str = "synthetic",
) -> None:
    best_val_acc = -1.0
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "stage": stage_name,
                    **checkpoint_extra,
                },
                output_path,
            )
        print(
            f"{stage_name}_epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


def run_epoch(model, loader, loss_fn, device, *, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if len(batch) == 3:
                features, labels, weights = batch
            else:
                features, labels = batch
                weights = torch.ones_like(labels, dtype=torch.float32)
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device=device, dtype=torch.float32)
            logits = model(features)
            losses = loss_fn(logits, labels)
            loss = (losses * weights).sum() / torch.clamp(weights.sum(), min=1.0)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_manifest(model, manifest_path: Path, classes: list[str], config: AudioFeatureConfig, device: torch.device) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    dataset = ManifestAudioDataset(manifest_path, classes, config)
    records = []
    model.eval()
    with torch.no_grad():
        for row in dataset.rows:
            card = normalize_card_key(str(row["card"]))
            if card not in class_to_idx:
                continue
            features = waveform_to_log_mel(load_audio_window(row["wav_path"], config), config).unsqueeze(0).to(device)
            predicted_idx = int(model(features).argmax(dim=1).item())
            records.append(
                {
                    "expected_card": card,
                    "predicted_card": idx_to_class[predicted_idx],
                    "match_phase": row.get("match_phase", "normal"),
                    "quality_tier": row.get("quality_tier", "unknown"),
                }
            )
    return summarize_predictions(records)


def evaluate_real_ground_truth(
    model,
    audio_path: Path,
    ground_truth_path: Path,
    classes: list[str],
    config: AudioFeatureConfig,
    device: torch.device,
) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    events = load_ground_truth_events(ground_truth_path, fps=_ground_truth_fps(ground_truth_path), side="enemy")
    records = []
    model.eval()
    with torch.no_grad():
        for event in events:
            card = normalize_card_key(event["card"])
            if card not in class_to_idx:
                continue
            waveform = load_audio_window(audio_path, config, start_s=max(0.0, event["time_s"] - 0.3))
            features = waveform_to_log_mel(waveform, config).unsqueeze(0).to(device)
            predicted_idx = int(model(features).argmax(dim=1).item())
            records.append(
                {
                    "expected_card": card,
                    "predicted_card": idx_to_class[predicted_idx],
                    "match_phase": event["match_phase"],
                    "quality_tier": "ground_truth",
                }
            )
    return summarize_predictions(records)


def evaluate_known_ground_truths(model, classes: list[str], config: AudioFeatureConfig, device: torch.device) -> None:
    for ground_truth_path, preferred_wav in DEFAULT_EVAL_WAVS.items():
        if not ground_truth_path.exists():
            continue
        wav_path = preferred_wav
        if not wav_path.exists():
            video_path = DEFAULT_EVAL_VIDEOS.get(ground_truth_path)
            if video_path is None or not video_path.exists():
                print(f"real_gt_eval_skipped path={ground_truth_path.name} reason=missing_audio")
                continue
            wav_path = extract_mono_wav_from_video(video_path, preferred_wav)
        metrics = evaluate_real_ground_truth(model, wav_path, ground_truth_path, classes, config, device)
        print(json.dumps({"ground_truth": ground_truth_path.name, **metrics}, ensure_ascii=False))


def _ground_truth_fps(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("fps", 10.0))


def limit_samples_per_class(samples: list[tuple[str, Path]], max_samples_per_class: int | None) -> list[tuple[str, Path]]:
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
