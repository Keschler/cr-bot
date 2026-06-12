import json
import io
from pathlib import Path
from contextlib import redirect_stdout

from cr_bot.eval.action_eval import ActionEvent, EvalResult
from cr_bot.eval.run_action_eval_scenarios import (
    SCENARIOS,
    FocusWindow,
    Scenario,
    aggregate_results,
    build_focus_windows,
    build_parser,
    evaluate_scenario,
    filter_events,
    find_cell_mismatches,
    print_summary,
    replay_cache_path,
    render_report,
)


def test_spell_scenarios_use_matching_videos_and_ground_truth():
    for key in ("spell", "spell2", "spell3"):
        scenario = SCENARIOS[key]
        assert scenario.video.name == f"{key}.mp4"
        assert scenario.ground_truth.name == f"{key}.json"
        assert scenario.predictions.name == f"{key}.txt"
        assert scenario.video.exists()
        assert scenario.ground_truth.exists()
        assert scenario.capture_args[
            scenario.capture_args.index("--video-sample-interval") + 1
        ] == "0.1"


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


def test_build_focus_windows_for_enemy_fireball():
    events = [
        type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 183.7})(),
        type("Event", (), {"side": "enemy", "canonical_card": "hog-rider", "video_time_s": 200.0})(),
    ]

    windows = build_focus_windows(
        events,
        card="fireball",
        side="enemy",
        before_s=5.0,
        after_s=10.0,
    )

    assert windows == [FocusWindow(start_s=178.7, end_s=193.7)]


def test_build_focus_windows_can_select_nearest_occurrence():
    events = [
        type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 7.4})(),
        type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 183.7})(),
        type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 287.3})(),
    ]

    windows = build_focus_windows(
        events,
        card="fireball",
        side="enemy",
        before_s=5.0,
        after_s=10.0,
        focus_video_time_s=183.6,
    )

    assert windows == [FocusWindow(start_s=178.7, end_s=193.7)]


def test_filter_events_keeps_only_focused_card_and_window():
    matching = type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 183.7})()
    wrong_card = type("Event", (), {"side": "enemy", "canonical_card": "hog-rider", "video_time_s": 183.7})()
    outside_window = type("Event", (), {"side": "enemy", "canonical_card": "fireball", "video_time_s": 210.0})()

    filtered = filter_events(
        [matching, wrong_card, outside_window],
        card="fireball",
        side="enemy",
        windows=[FocusWindow(start_s=178.7, end_s=193.7)],
    )

    assert filtered == [matching]


def test_build_parser_accepts_focus_window_args():
    args = build_parser().parse_args(
        [
            "--scenario",
            "all",
            "--focus-card",
            "fireball",
            "--focus-side",
            "enemy",
            "--focus-window-before",
            "5",
            "--focus-window-after",
            "10",
            "--focus-video-time",
            "183.7",
        ]
    )

    assert args.focus_card == "fireball"
    assert args.focus_side == "enemy"
    assert args.focus_window_before == 5.0
    assert args.focus_window_after == 10.0
    assert args.focus_video_time == 183.7


def test_build_parser_accepts_replay_modes():
    build_args = build_parser().parse_args(["--build-replay-cache"])
    replay_args = build_parser().parse_args(["--replay-cache"])

    assert build_args.build_replay_cache is True
    assert replay_args.replay_cache is True


def test_replay_cache_path_is_scenario_specific(tmp_path: Path):
    scenario = Scenario(
        key="sample",
        label="sample",
        video=tmp_path / "video.mp4",
        ground_truth=tmp_path / "ground-truth.json",
        predictions=tmp_path / "predictions.txt",
        capture_args=(),
    )

    assert replay_cache_path(scenario).name == "sample.pkl.gz"


def test_summary_ranks_false_positives_and_misses_by_card():
    results = [
        EvalResult(
            side="enemy",
            matches=[],
            misses=[
                ActionEvent(side="enemy", card="fireball"),
                ActionEvent(side="enemy", card="evo-fireball"),
                ActionEvent(side="enemy", card="arrows"),
            ],
            false_positives=[
                ActionEvent(side="enemy", card="zap"),
                ActionEvent(side="enemy", card="arrows"),
                ActionEvent(side="enemy", card="zap"),
            ],
        )
    ]
    output = io.StringIO()

    with redirect_stdout(output):
        print_summary(
            aggregate_results(results),
            scenario_count=1,
            results=results,
        )

    text = output.getvalue()
    assert "false positives by card:\n  zap: 2\n  arrows: 1" in text
    assert "misses by card:\n  fireball: 2\n  arrows: 1" in text


def test_find_cell_mismatches_pairs_same_card_with_matching_time():
    miss = ActionEvent(
        side="enemy",
        card="fireball",
        time_left_s=120.0,
        video_time_s=30.0,
        cell=(5, 8),
    )
    different_cell = ActionEvent(
        side="enemy",
        card="fireball",
        time_left_s=119.5,
        video_time_s=30.4,
        cell=(9, 8),
    )
    wrong_time = ActionEvent(
        side="enemy",
        card="fireball",
        time_left_s=100.0,
        video_time_s=50.0,
        cell=(6, 8),
    )
    results = [
        EvalResult(
            side="enemy",
            matches=[],
            misses=[miss],
            false_positives=[wrong_time, different_cell],
        )
    ]

    mismatches = find_cell_mismatches(
        results,
        time_left_tolerance_s=2.0,
        video_time_tolerance_s=2.0,
    )

    assert len(mismatches) == 1
    assert mismatches[0].missed is miss
    assert mismatches[0].false_positive is different_cell
    assert mismatches[0].cell_distance == 4
