from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from cr_bot.annotation_pipeline import validate_own_slot_interval_decisions
from scripts.codex_annotation.prepare_own_slot_interval_packages import (
    _render_card_identity,
    is_empty_card_art,
    merge_empty_frame_intervals,
    same_card_return_score,
)


def _package() -> dict[str, object]:
    return {
        "run_id": "test-run",
        "target_range": [0, 200],
        "intervals": [
            {
                "interval_id": "own-slot:1:000010-000013",
                "slot": 1,
                "empty_range": [10, 13],
                "sampled_frame_indices": list(range(8, 20)),
                "artifact": "reviews/own-slot-1-000010-000013.jpg",
                "candidate_id": "own:000010",
                "return_evidence": {
                    "outcome_constraint": "released",
                },
            },
            {
                "interval_id": "own-slot:3:000050-000052",
                "slot": 3,
                "empty_range": [50, 52],
                "sampled_frame_indices": list(range(48, 59)),
                "artifact": "reviews/own-slot-3-000050-000052.jpg",
                "candidate_id": "own:000051",
                "return_evidence": {
                    "outcome_constraint": "canceled",
                },
            },
        ],
    }


def _valid_document() -> dict[str, object]:
    return {
        "run_id": "test-run",
        "stage": "own_slot_intervals_chunk",
        "target_range": [0, 200],
        "decisions": [
            {
                "interval_id": "own-slot:1:000010-000013",
                "decision": "released",
                "card": "hog-rider",
                "event_frame_index": 10,
                "confirmation_frame_index": 17,
                "artifact": "reviews/own-slot-1-000010-000013.jpg",
                "reason": "slot cycles and the Hog persists in the arena",
            },
            {
                "interval_id": "own-slot:3:000050-000052",
                "decision": "canceled",
                "card": None,
                "event_frame_index": None,
                "confirmation_frame_index": None,
                "artifact": "reviews/own-slot-3-000050-000052.jpg",
                "reason": "the same card returns with no persistent spend",
            },
        ],
    }


def test_merge_empty_intervals_bridges_at_most_three_missing_frames() -> None:
    frames = [1, 2, 6, 7, 12, 14, 15, 30]

    assert merge_empty_frame_intervals(frames) == [(1, 7), (12, 15)]


def test_empty_card_art_rule_requires_saturation_and_low_edges() -> None:
    empty_hsv = np.full((80, 80, 3), (150, 250, 220), dtype=np.uint8)
    empty_bgr = cv2.cvtColor(empty_hsv, cv2.COLOR_HSV2BGR)
    assert is_empty_card_art(empty_bgr)

    textured = empty_bgr.copy()
    textured[:, ::2] = 0
    assert not is_empty_card_art(textured)

    gray = np.full((80, 80, 3), 180, dtype=np.uint8)
    assert not is_empty_card_art(gray)


def test_same_card_return_score_separates_same_and_different_art() -> None:
    first_hsv = np.full((80, 80, 3), (20, 240, 220), dtype=np.uint8)
    same_hsv = first_hsv.copy()
    same_hsv[:, :10, 2] = 180
    different_hsv = np.full((80, 80, 3), (100, 240, 220), dtype=np.uint8)

    first = cv2.cvtColor(first_hsv, cv2.COLOR_HSV2BGR)
    same = cv2.cvtColor(same_hsv, cv2.COLOR_HSV2BGR)
    different = cv2.cvtColor(different_hsv, cv2.COLOR_HSV2BGR)

    assert same_card_return_score([first], [same]) > 0.9
    assert same_card_return_score([first], [different]) < 0.5


def test_left_truncated_empty_interval_has_no_card_identity(
    tmp_path,
) -> None:
    manifest = {"segment": {"start_frame": 0}}

    assert _render_card_identity(
        run_dir=tmp_path,
        manifest=manifest,
        frames_by_index={},
        slot=1,
        interval_start=0,
        interval_end=28,
    ) is None


def test_own_slot_validator_accepts_exact_complete_decisions() -> None:
    validate_own_slot_interval_decisions(_valid_document(), _package())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda document: document["decisions"].pop(),
            "cover every interval exactly once",
        ),
        (
            lambda document: document["decisions"][0].update(
                {"artifact": "reviews/wrong.jpg"}
            ),
            "artifact changed",
        ),
        (
            lambda document: document["decisions"][0].update(
                {"card": "the-log"}
            ),
            "canonical slug",
        ),
        (
            lambda document: document["decisions"][0].update(
                {"confirmation_frame_index": 14}
            ),
            "5-15 frames later",
        ),
        (
            lambda document: document["decisions"][1].update(
                {"event_frame_index": 51}
            ),
            "timing must be null",
        ),
        (
            lambda document: document["decisions"][0].update(
                {
                    "decision": "canceled",
                    "card": None,
                    "event_frame_index": None,
                    "confirmation_frame_index": None,
                }
            ),
            "conflicts with deterministic",
        ),
    ],
)
def test_own_slot_validator_fails_closed(
    mutation,
    message: str,
) -> None:
    document = copy.deepcopy(_valid_document())
    mutation(document)

    with pytest.raises(ValueError, match=message):
        validate_own_slot_interval_decisions(document, _package())
