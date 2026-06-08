from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.eval.action_eval import EvalResult, evaluate, load_ground_truth, parse_predictions_txt, print_report


CAPTURE_SCRIPT = REPO_ROOT / "capture" / "capture.py"
EXPLAIN_SCRIPT = REPO_ROOT / "src" / "cr_bot" / "eval" / "explain_action_misses.py"


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    video: Path
    ground_truth: Path
    predictions: Path
    capture_args: tuple[str, ...]


@dataclass(frozen=True)
class AggregateCounts:
    label: str
    matched: int
    missed: int
    false_positive: int

    @property
    def precision(self) -> float:
        denom = self.matched + self.false_positive
        return self.matched / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.matched + self.missed
        return self.matched / denom if denom else 1.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


SCENARIOS: dict[str, Scenario] = {
    "2hog-cycle": Scenario(
        key="2hog-cycle",
        label="2hog_cycle_champion",
        video=REPO_ROOT / "dataset_generation" / "data" / "video_clips" / "10_fps_2.6HogCycle.mp4",
        ground_truth=REPO_ROOT / "data" / "eval" / "ground_truth" / "2hog_cycle_champion.json",
        predictions=REPO_ROOT / "outputs" / "video" / "capture" / "2hog_cycle_champion.txt",
        capture_args=(
            "--yolo-detections",
            "--video-duration",
            "296",
        ),
    ),
    "3400ladder": Scenario(
        key="3400ladder",
        label="3400Ladder",
        video=(
            REPO_ROOT
            / "dataset_generation"
            / "data"
            / "video_clips"
            / "downloaded_videos"
            / "HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].mp4"
        ),
        ground_truth=(
            REPO_ROOT
            / "data"
            / "eval"
            / "ground_truth"
            / "HOG 2.6 LADDER 🏆+3400 [tvA-OvUUHmw].json"
        ),
        predictions=REPO_ROOT / "outputs" / "video" / "capture" / "3400Ladder.txt",
        capture_args=(
            "--alternative-rois",
            "--yolo-detections",
            "--frame-stride",
            "6",
            "--video-duration",
            "296",
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or both built-in action-eval scenarios, optionally regenerate "
            "capture outputs, and print a combined accuracy summary."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="Choose one built-in scenario or evaluate all of them.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "summary"],
        default="full",
        help="Print per-run eval reports plus summary, or only the final summary.",
    )
    parser.add_argument(
        "--run-detection",
        action="store_true",
        help="Regenerate the capture txt for each selected scenario before evaluation.",
    )
    parser.add_argument(
        "--side",
        choices=["own", "enemy", "both"],
        default="both",
        help="Forwarded to action_eval matching logic.",
    )
    parser.add_argument("--time-left-tolerance", type=float, default=2.0)
    parser.add_argument("--video-time-tolerance", type=float, default=2.0)
    parser.add_argument("--cell-tolerance", type=int, default=1)
    parser.add_argument("--strict-evolution", action="store_true")
    parser.add_argument(
        "--explain-misses",
        action="store_true",
        help="Run explain_action_misses.py for each selected scenario after evaluation.",
    )
    parser.add_argument(
        "--explain-side",
        choices=["own", "enemy", "both"],
        default="both",
        help="Side filter passed to explain_action_misses.py.",
    )
    parser.add_argument("--explain-limit", type=int, default=None)
    parser.add_argument("--explain-card", default=None)
    parser.add_argument("--window-before", type=float, default=2.5)
    parser.add_argument("--window-after", type=float, default=4.0)
    parser.add_argument("--nearby-prediction-window", type=float, default=8.0)
    return parser


def selected_scenarios(name: str) -> list[Scenario]:
    if name == "all":
        return [SCENARIOS[key] for key in SCENARIOS]
    return [SCENARIOS[name]]


def render_report(results: list[EvalResult]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_report(results)
    return buffer.getvalue().rstrip()


def evaluate_scenario(args: argparse.Namespace, scenario: Scenario) -> tuple[list[EvalResult], str]:
    if args.run_detection:
        run_detection(scenario)
    elif not scenario.predictions.exists():
        raise FileNotFoundError(
            f"prediction file not found for {scenario.key}: {scenario.predictions}. "
            "Run with --run-detection to regenerate it."
        )

    expected = load_ground_truth(scenario.ground_truth)
    predicted = parse_predictions_txt(scenario.predictions)
    sides = ["own", "enemy"] if args.side == "both" else [args.side]
    results = [
        evaluate(
            expected,
            predicted,
            side=side,
            time_left_tolerance_s=args.time_left_tolerance,
            video_time_tolerance_s=args.video_time_tolerance,
            cell_tolerance=args.cell_tolerance,
            strict_evolution=args.strict_evolution,
        )
        for side in sides
    ]
    return results, render_report(results)


def run_detection(scenario: Scenario) -> None:
    scenario.predictions.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DEBUG_OWN_ACTIONS"] = "1"
    cmd = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        *scenario.capture_args,
        "--video",
        str(scenario.video),
    ]
    with scenario.predictions.open("w", encoding="utf-8") as handle:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            check=True,
        )


def print_scenario_report(scenario: Scenario, report_text: str) -> None:
    print(f"=== {scenario.key} ===")
    print(f"ground_truth: {scenario.ground_truth}")
    print(f"predictions:  {scenario.predictions}")
    if report_text:
        print(report_text)
    print()


def aggregate_results(results: list[EvalResult]) -> list[AggregateCounts]:
    grouped: dict[str, list[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.side, []).append(result)

    summary: list[AggregateCounts] = []
    total_matched = 0
    total_missed = 0
    total_false_positive = 0
    for side in ("own", "enemy"):
        side_results = grouped.get(side)
        if not side_results:
            continue
        matched = sum(len(item.matches) for item in side_results)
        missed = sum(len(item.misses) for item in side_results)
        false_positive = sum(len(item.false_positives) for item in side_results)
        summary.append(
            AggregateCounts(
                label=side,
                matched=matched,
                missed=missed,
                false_positive=false_positive,
            )
        )
        total_matched += matched
        total_missed += missed
        total_false_positive += false_positive

    summary.append(
        AggregateCounts(
            label="overall",
            matched=total_matched,
            missed=total_missed,
            false_positive=total_false_positive,
        )
    )
    return summary


def print_summary(summary: list[AggregateCounts], *, scenario_count: int) -> None:
    print("=== summary ===")
    print(f"runs: {scenario_count}")
    for counts in summary:
        print(
            f"{counts.label}: "
            f"precision={counts.precision:.3f} "
            f"recall={counts.recall:.3f} "
            f"f1={counts.f1:.3f} "
            f"matched={counts.matched} "
            f"missed={counts.missed} "
            f"false_positive={counts.false_positive}"
        )


def run_explainer(args: argparse.Namespace, scenario: Scenario, report_text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(report_text)
        handle.write("\n")
    try:
        cmd = [
            sys.executable,
            str(EXPLAIN_SCRIPT),
            "--eval-output",
            str(temp_path),
            "--predictions",
            str(scenario.predictions),
            "--side",
            args.explain_side,
            "--window-before",
            str(args.window_before),
            "--window-after",
            str(args.window_after),
            "--nearby-prediction-window",
            str(args.nearby_prediction_window),
        ]
        if args.explain_limit is not None:
            cmd.extend(["--limit", str(args.explain_limit)])
        if args.explain_card:
            cmd.extend(["--card", args.explain_card])
        print(f"=== explain {scenario.key} ===")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        print()
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scenarios = selected_scenarios(args.scenario)
    all_results: list[EvalResult] = []

    for scenario in scenarios:
        results, report_text = evaluate_scenario(args, scenario)
        all_results.extend(results)
        if args.mode == "full":
            print_scenario_report(scenario, report_text)
        if args.explain_misses:
            run_explainer(args, scenario, report_text)

    print_summary(aggregate_results(all_results), scenario_count=len(scenarios))


if __name__ == "__main__":
    main()
