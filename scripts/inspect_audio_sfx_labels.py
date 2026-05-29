from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.audio.dataset import collect_sfx_files, is_deploy_like
from cr_bot.audio.labels import folder_to_card_keys
from cr_bot.domain.card_metadata import CARD_METADATA


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Clash Royale SFX folder label mapping.")
    parser.add_argument(
        "--raw-sfx-dir",
        type=Path,
        default=ROOT / "data/audio_classifier/raw_sfx",
    )
    args = parser.parse_args()

    known_cards = set(CARD_METADATA)
    samples, skipped, unmapped = collect_sfx_files(
        args.raw_sfx_dir,
        deploy_only=True,
        known_cards=known_cards,
    )
    all_counts = Counter(card for card, _ in samples)
    deploy_counts = defaultdict(int)

    for folder in sorted(args.raw_sfx_dir.iterdir()):
        if not folder.is_dir():
            continue
        cards = [card for card in folder_to_card_keys(folder) if card in known_cards]
        if not cards:
            continue
        deploy_like = sum(1 for path in folder.glob("*.wav") if is_deploy_like(path))
        for card in cards:
            deploy_counts[card] += deploy_like

    print(f"mapped_classes={len(all_counts)}")
    print(f"metadata_classes={len(known_cards)}")
    print(f"missing_metadata_classes={len(known_cards - set(all_counts))}")
    print(f"selected_wavs={sum(all_counts.values())}")
    print(f"unmapped_or_tower_folders={len(unmapped)}")
    print(f"unknown_card_folders={sum(len(paths) for paths in skipped.values())}")
    print()
    print("classes:")
    for card, count in sorted(all_counts.items()):
        print(f"  {card}: selected={count} deploy_like={deploy_counts[card]}")

    missing = sorted(known_cards - set(all_counts))
    if missing:
        print()
        print("metadata cards without source wavs:")
        for card in missing:
            print(f"  {card}")

    if skipped:
        print()
        print("unknown mapped cards:")
        for card, folders in sorted(skipped.items()):
            names = ", ".join(path.name for path in folders)
            print(f"  {card}: {names}")


if __name__ == "__main__":
    main()
