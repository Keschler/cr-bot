#!/usr/bin/env python3
"""Fetch RoyaleAPI Clash Royale static card data into a local cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://royaleapi.github.io/cr-api-data/json"
DEFAULT_FILES = (
    "cards.json",
    "cards_stats.json",
    "cards_stats_troop.json",
    "cards_stats_building.json",
    "cards_stats_spell.json",
    "cards_stats_projectile.json",
    "cards_stats_characters.json",
)


def fetch_json(filename: str) -> object:
    url = f"{BASE_URL}/{filename}"
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RoyaleAPI static Clash Royale JSON data."
    )
    parser.add_argument(
        "--out-dir",
        default=".cache/royaleapi",
        help="Directory for downloaded JSON files. Defaults to .cache/royaleapi.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Specific RoyaleAPI JSON filenames to fetch.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.files:
        data = fetch_json(filename)
        out_path = out_dir / filename
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        count = len(data) if hasattr(data, "__len__") else "unknown"
        print(f"wrote {out_path} ({count} top-level entries)")


if __name__ == "__main__":
    main()
