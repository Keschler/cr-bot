import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/import_ground_truth_labels.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_creates_ground_truth_next_to_human_labels_parent(tmp_path: Path):
    labels_dir = tmp_path / "ground_truth/human_labels"
    labels_dir.mkdir(parents=True)
    labels = labels_dir / "match enemy.txt"
    labels.write_text("musketeer 53\nice-spiirit 2417\n", encoding="utf-8")

    result = _run(str(labels))

    output = tmp_path / "ground_truth/match.json"
    ground_truth = json.loads(output.read_text(encoding="utf-8"))
    assert "created" in result.stdout
    assert "ice-spiirit -> ice-spirit" in result.stdout
    assert ground_truth["video"] == "match.mp4"
    assert ground_truth["fps"] == 10.0
    assert ground_truth["events"] == [
        {"side": "enemy", "card": "musketeer", "frame_index": 53},
        {"side": "enemy", "card": "ice-spirit", "frame_index": 2417},
    ]


def test_updates_imported_side_and_preserves_matching_fields(tmp_path: Path):
    labels = tmp_path / "match enemy.txt"
    labels.write_text("musketeer 53\nice-spirit 60\n", encoding="utf-8")
    output = tmp_path / "match.json"
    output.write_text(
        json.dumps(
            {
                "video": "old.mp4",
                "fps": 30,
                "notes": "keep me",
                "events": [
                    {"side": "own", "card": "hog-rider", "frame_index": 40, "cell": [1, 17]},
                    {"side": "enemy", "card": "musketeer", "frame_index": 53, "cell": [9, 6]},
                    {"side": "enemy", "card": "fireball", "frame_index": 55},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run(str(labels), "--output", str(output), "--fps", "10")

    ground_truth = json.loads(output.read_text(encoding="utf-8"))
    assert "updated" in result.stdout
    assert ground_truth["notes"] == "keep me"
    assert ground_truth["events"] == [
        {"side": "own", "card": "hog-rider", "frame_index": 40, "cell": [1, 17]},
        {"side": "enemy", "card": "musketeer", "frame_index": 53, "cell": [9, 6]},
        {"side": "enemy", "card": "ice-spirit", "frame_index": 60},
    ]


def test_side_override_supports_filename_without_suffix(tmp_path: Path):
    labels = tmp_path / "labels.txt"
    labels.write_text("cannon 20\n", encoding="utf-8")
    output = tmp_path / "ground_truth.json"

    _run(str(labels), "--side", "own", "--output", str(output), "--video", "source.mp4")

    ground_truth = json.loads(output.read_text(encoding="utf-8"))
    assert ground_truth["video"] == "source.mp4"
    assert ground_truth["events"] == [
        {"side": "own", "card": "cannon", "frame_index": 20},
    ]
