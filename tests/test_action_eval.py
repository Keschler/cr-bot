from pathlib import Path

from cr_bot.eval.action_eval import (
    ActionEvent,
    evaluate,
    load_ground_truth,
    parse_predictions_txt,
)


def test_parse_predictions_txt_keeps_first_added_time_and_latest_cell(tmp_path: Path):
    output = tmp_path / "output.txt"
    output.write_text(
        "\n".join(
            [
                "frame 1 video_time=10.00s",
                "enemy plays:",
                "  card=dart-goblin          cost=3 time_left=250 track_id=7 cell=(6, 6)",
                "own plays:",
                "  card=ice-golem            slot=1 cell=(7, 19) time_left=160 ",
                "",
                "frame 2 video_time=10.10s",
                "enemy plays:",
                "  card=dart-goblin          cost=3 time_left=250 track_id=7 cell=(8, 9)",
                "own plays:",
                "  card=ice-golem            slot=1 cell=(7, 19) time_left=160 ",
            ]
        )
    )

    events = parse_predictions_txt(output)

    assert len(events) == 2
    assert events[0].side == "enemy"
    assert events[0].cell == (8, 9)
    assert events[0].video_time_s == 10.0
    assert events[1].side == "own"
    assert events[1].cell == (7, 19)
    assert events[1].video_time_s == 10.0


def test_parse_predictions_txt_uses_action_video_time_when_present(tmp_path: Path):
    output = tmp_path / "output.txt"
    output.write_text(
        "\n".join(
            [
                "frame 20 video_time=20.00s",
                "enemy plays:",
                "own plays:",
                "  card=hog-rider            slot=0 cell=(1, 17) video_time=18.70 time_left=289 ",
            ]
        )
    )

    events = parse_predictions_txt(output)

    assert len(events) == 1
    assert events[0].side == "own"
    assert events[0].video_time_s == 18.7


def test_load_ground_truth_converts_frame_index_with_fps(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    ground_truth.write_text(
        """
        {
          "fps": 10,
          "events": [
            {"side": "own", "card": "fireball", "frame_index": 1572}
          ]
        }
        """
    )

    events = load_ground_truth(ground_truth)

    assert len(events) == 1
    assert events[0].frame_index == 1572
    assert events[0].video_time_s == 157.2
    assert events[0].cell is None


def test_evaluate_reports_time_and_added_video_time_errors():
    expected = [
        ActionEvent(
            side="own",
            card="ice-golem",
            time_left_s=159.0,
            video_time_s=133.5,
            cell=(7, 19),
        )
    ]
    predicted = [
        ActionEvent(
            side="own",
            card="ice-golem",
            time_left_s=158.5,
            video_time_s=134.2,
            cell=(8, 19),
        )
    ]

    result = evaluate(
        expected,
        predicted,
        side="own",
        time_left_tolerance_s=2.0,
        video_time_tolerance_s=2.0,
        cell_tolerance=1,
        strict_evolution=False,
    )

    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.time_left_error_s == -0.5
    assert round(match.added_video_time_error_s, 6) == 0.7
    assert match.cell_distance == 1
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_evaluate_treats_old_musketeer_as_musketeer():
    expected = [
        ActionEvent(
            side="own",
            card="old-musketeer",
            video_time_s=10.0,
        )
    ]
    predicted = [
        ActionEvent(
            side="own",
            card="musketeer",
            video_time_s=10.0,
        )
    ]

    result = evaluate(
        expected,
        predicted,
        side="own",
        time_left_tolerance_s=2.0,
        video_time_tolerance_s=2.0,
        cell_tolerance=1,
        strict_evolution=False,
    )

    assert len(result.matches) == 1
