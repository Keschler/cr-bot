from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.eval.action_eval import CARD_ALIASES


def _card(value: str) -> str:
    normalized = value.replace("_", "-")
    if normalized.startswith("evo-"):
        normalized = normalized[4:]
    if normalized == "the-log":
        normalized = "log"
    return CARD_ALIASES.get(normalized, normalized)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _cluster_rows(
    rows: list[dict[str, Any]],
    *,
    max_span_frames: int = 20,
    cross_candidate_span_frames: int = 5,
) -> list[list[dict[str, Any]]]:
    """Group likely duplicate proposals without hiding a later real release.

    A long held preview and its later release usually belong to different HUD
    candidates. Keep those proposals distinct once they are more than the
    evaluation tolerance apart so the independent release gate can reject the
    preview without deleting the real play. Wider disagreements are still
    merged when both workers attached them to the same candidate.
    """
    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        cluster = clusters[-1] if clusters else []
        span = (
            row["event_frame_index"] - cluster[0]["event_frame_index"]
            if cluster
            else 0
        )
        same_candidate = any(
            candidate.get("candidate_id") == row.get("candidate_id")
            for candidate in cluster
        )
        if (
            cluster
            and cluster[0]["card"] == row["card"]
            and span <= max_span_frames
            and (same_candidate or span <= cross_candidate_span_frames)
        ):
            cluster.append(row)
        else:
            clusters.append([row])
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create anonymous medium/low own-event disagreement packages."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=400)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    discovery = {
        row["candidate_id"]: row
        for row in _read(run_dir / "own_discovery_index.json")["candidates"]
    }
    primary = _read(run_dir / "own_semantics.json")["events"]
    sweep = [
        row
        for path in sorted(
            (run_dir / "worker_outputs").glob("own-complete-*.json")
        )
        for row in _read(path).get("events", [])
    ]
    rows = []
    for row in [*primary, *sweep]:
        normalized = dict(row)
        normalized["card"] = _card(normalized["card"])
        rows.append(normalized)
    rows.sort(key=lambda row: (row["card"], row["event_frame_index"]))
    clusters = _cluster_rows(rows)
    proposals = []
    for index, cluster in enumerate(clusters):
        candidates = sorted({row["candidate_id"] for row in cluster})
        artifacts = sorted(
            {
                artifact
                for row in cluster
                for artifact in (
                    row["verification_artifacts"]
                    + row["confirmation_artifacts"]
                )
            }
        )
        frames = sorted({row["event_frame_index"] for row in cluster})
        center = round(sum(frames) / len(frames))
        proposals.append(
            {
                "proposal_id": f"own-proposal-{index:04d}",
                "card": cluster[0]["card"],
                "proposed_frames": frames,
                "candidate_ids": candidates,
                "discovery_artifacts": [
                    f"reviews/discover-{candidate.replace(':', '-')}.jpg"
                    for candidate in candidates
                ],
                "candidate_evidence": [
                    {
                        "candidate_id": candidate,
                        "discovery_artifact": discovery[candidate]["artifact"],
                        "discovery_frame_indices": discovery[candidate][
                            "sampled_frame_indices"
                        ],
                    }
                    for candidate in candidates
                ],
                "exact_artifacts": artifacts,
                "candidate_rows": cluster,
                "center_frame": center,
            }
        )
    output_dir = run_dir / "work_packages"
    segment = manifest["segment"]
    summaries = []
    for start in range(
        segment["start_frame"],
        segment["end_frame_exclusive"],
        args.chunk_frames,
    ):
        end = min(segment["end_frame_exclusive"], start + args.chunk_frames)
        selected = [
            row for row in proposals if start <= row["center_frame"] < end
        ]
        path = output_dir / f"own-adjudicate-{start:06d}-{end:06d}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "fps": manifest["fps"],
                    "segment": segment,
                    "target_range": [start, end],
                    "proposals": selected,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append({"range": [start, end], "proposals": len(selected)})
    print(json.dumps({"packages": summaries, "total": len(proposals)}))


if __name__ == "__main__":
    main()
