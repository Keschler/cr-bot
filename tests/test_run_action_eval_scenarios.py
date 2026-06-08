import json
from pathlib import Path

from cr_bot.eval.run_action_eval_scenarios import (
    Scenario,
    aggregate_results,
    build_parser,
    evaluate_scenario,
    render_report,
)


def _write_ground_truth(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "side": "own",
                        "card": "ice-golem",
                        "time_left_s": 160.0,
                        "video_time_s": 10.0,
                        "cell": [7, 19],
                    },
                    {
                        "side": "enemy",
                        "card": "musketeer",
                        "time_left_s": 250.0,
                        "video_time_s": 10.0,
                        "cell": [6, 6],
                    },
                ]
            }
        )
    )


def _write_predictions(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "frame 1 video_time=10.00s",
                "enemy plays:",
                "  card=musketeer            cost=4 time_left=250 track_id=7 cell=(6, 6)",
                "own plays:",
                "  card=ice-golem            slot=1 cell=(7, 19) time_left=160",
            ]
        )
    )


def test_evaluate_scenario_returns_report(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    predictions = tmp_path / "predictions.txt"
    _write_ground_truth(ground_truth)
    _write_predictions(predictions)
    scenario = Scenario(
        key="sample",
        label="sample",
        video=tmp_path / "video.mp4",
        ground_truth=ground_truth,
        predictions=predictions,
        capture_args=(),
    )
    args = build_parser().parse_args(["--scenario", "all"])

    results, report_text = evaluate_scenario(args, scenario)

    assert [result.side for result in results] == ["own", "enemy"]
    assert "own actions:" in report_text
    assert "enemy actions:" in report_text
    assert "precision=1.000" in report_text


def test_aggregate_results_combines_sides(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    predictions = tmp_path / "predictions.txt"
    _write_ground_truth(ground_truth)
    _write_predictions(predictions)
    scenario = Scenario(
        key="sample",
        label="sample",
        video=tmp_path / "video.mp4",
        ground_truth=ground_truth,
        predictions=predictions,
        capture_args=(),
    )
    args = build_parser().parse_args(["--scenario", "all", "--side", "both"])

    results, _ = evaluate_scenario(args, scenario)
    summary = aggregate_results(results)

    own = next(item for item in summary if item.label == "own")
    enemy = next(item for item in summary if item.label == "enemy")
    overall = next(item for item in summary if item.label == "overall")
    assert (own.matched, own.missed, own.false_positive) == (1, 0, 0)
    assert (enemy.matched, enemy.missed, enemy.false_positive) == (1, 0, 0)
    assert (overall.matched, overall.missed, overall.false_positive) == (2, 0, 0)


def test_render_report_is_stable_for_single_side(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    predictions = tmp_path / "predictions.txt"
    _write_ground_truth(ground_truth)
    _write_predictions(predictions)
    scenario = Scenario(
        key="sample",
        label="sample",
        video=tmp_path / "video.mp4",
        ground_truth=ground_truth,
        predictions=predictions,
        capture_args=(),
    )
    args = build_parser().parse_args(["--scenario", "all", "--side", "own"])

    results, _ = evaluate_scenario(args, scenario)
    report = render_report(results)

    assert "own actions:" in report
    assert "enemy actions:" not in report
