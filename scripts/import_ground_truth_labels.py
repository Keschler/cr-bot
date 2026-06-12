from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_DIR = ROOT / "data/eval/ground_truth/human_labels"
LABEL_LINE_RE = re.compile(
    r"^(?P<card>\S+)\s+(?P<frame_index>\d+)"
    r"(?:\s+(?P<cell_row>\d+),(?P<cell_column>\d+))?"
    r"(?:\s+(?P<played_via>mirror))?$"
)
LABEL_FILE_RE = re.compile(r"^(?P<video_stem>.+) (?P<side>own|enemy)$")
CARD_CORRECTIONS = {
    "ice-spiirit": "ice-spirit",
}


def parse_labels(path: Path, *, side: str) -> tuple[list[dict], list[tuple[str, str]]]:
    events = []
    corrections = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = LABEL_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected '<card> <frame_index>', got {raw_line!r}")

        card = match.group("card")
        corrected_card = CARD_CORRECTIONS.get(card, card)
        if corrected_card != card:
            corrections.append((card, corrected_card))
        event = {
            "side": side,
            "card": corrected_card,
            "frame_index": int(match.group("frame_index")),
        }
        if match.group("cell_row") is not None:
            event["cell"] = [
                int(match.group("cell_row")),
                int(match.group("cell_column")),
            ]
        if match.group("played_via") is not None:
            event["played_via"] = match.group("played_via")
        events.append(event)
    return events, corrections


def infer_label_metadata(path: Path, *, side_override: str | None = None) -> tuple[str, str]:
    match = LABEL_FILE_RE.fullmatch(path.stem)
    if match is None:
        if side_override is not None:
            return path.stem, side_override
        raise ValueError(
            f"cannot infer side from {path.name!r}; use a filename ending in ' own.txt' or ' enemy.txt', "
            "or pass --side"
        )
    return match.group("video_stem"), side_override or match.group("side")


def default_output_path(labels_path: Path, video_stem: str) -> Path:
    output_dir = labels_path.parent.parent if labels_path.parent.name == "human_labels" else labels_path.parent
    return output_dir / f"{video_stem}.json"


def update_ground_truth(
    output_path: Path,
    *,
    video_name: str,
    fps: float,
    side: str,
    imported_events: list[dict],
    labels_path: Path,
) -> tuple[dict, bool]:
    existed = output_path.exists()
    if existed:
        ground_truth = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(ground_truth, dict):
            raise ValueError(f"{output_path}: expected a JSON object")
    else:
        ground_truth = {
            "video": video_name,
            "fps": fps,
            "notes": f"Action labels imported from {labels_path.name}.",
            "events": [],
        }

    existing_events = ground_truth.get("events", [])
    if not isinstance(existing_events, list):
        raise ValueError(f"{output_path}: 'events' must be a list")

    existing_by_key = {
        (event.get("side"), event.get("card"), event.get("frame_index")): event
        for event in existing_events
        if isinstance(event, dict)
    }
    merged_imported_events = []
    for event in imported_events:
        previous = existing_by_key.get((event["side"], event["card"], event["frame_index"]))
        if previous is None:
            merged_imported_events.append(event)
            continue
        merged_event = {**previous, **event}
        if "played_via" not in event:
            merged_event.pop("played_via", None)
        merged_imported_events.append(merged_event)

    ground_truth["video"] = video_name
    ground_truth["fps"] = fps
    ground_truth["events"] = [
        event
        for event in existing_events
        if not isinstance(event, dict) or event.get("side") != side
    ] + merged_imported_events
    return ground_truth, existed


def write_ground_truth(path: Path, ground_truth: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import '<card> <frame_index> [<cell_row>,<cell_column>] [mirror]' human labels "
            "into an eval ground-truth JSON file."
        )
    )
    parser.add_argument("labels", type=Path, help="Input text file, normally ending in ' own.txt' or ' enemy.txt'.")
    parser.add_argument("--output", type=Path, help="Ground-truth JSON to create or update.")
    parser.add_argument("--side", choices=["own", "enemy"], help="Override the side inferred from the filename.")
    parser.add_argument("--video", help="Override the source video filename stored in the JSON.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS used by frame_index labels. Default: 10.")
    args = parser.parse_args()

    inferred_video_stem, side = infer_label_metadata(args.labels, side_override=args.side)
    video_name = args.video or f"{inferred_video_stem}.mp4"
    output_path = args.output or default_output_path(args.labels, inferred_video_stem)
    events, corrections = parse_labels(args.labels, side=side)
    ground_truth, existed = update_ground_truth(
        output_path,
        video_name=video_name,
        fps=args.fps,
        side=side,
        imported_events=events,
        labels_path=args.labels,
    )
    write_ground_truth(output_path, ground_truth)

    verb = "updated" if existed else "created"
    print(f"{verb} {output_path} with {len(events)} {side} events")
    for original, corrected in sorted(set(corrections)):
        print(f"corrected card label: {original} -> {corrected}")


if __name__ == "__main__":
    main()
