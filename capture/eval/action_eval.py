from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics
from typing import Any


FRAME_RE = re.compile(r"^frame\s+\d+\s+video_time=(?P<video_time>[0-9.]+)s")
ENEMY_RE = re.compile(
    r"^\s+card=(?P<card>\S+)\s+cost=(?P<cost>\d+)\s+"
    r"time_left=(?P<time_left>[0-9.]+)\s+track_id=(?P<track_id>-?\d+)"
)
OWN_RE = re.compile(
    r"^\s+card=(?P<card>\S+)\s+slot=(?P<slot>\d+)\s+"
    r"cell=(?:(?:\((?P<cell_x>-?\d+),\s*(?P<cell_y>-?\d+)\))|unknown)\s+"
    r"(?:video_time=(?P<video_time>[0-9.]+)\s+)?"
    r"time_left=(?P<time_left>[0-9.]+)"
)
MAX_PREDICTED_TIME_LEFT_S = 300.0
CARD_ALIASES = {
    "old-musketeer": "musketeer",
}


@dataclass(frozen=True)
class ActionEvent:
    side: str
    card: str
    time_left_s: float | None = None
    video_time_s: float | None = None
    frame_index: int | None = None
    cell: tuple[int, int] | None = None
    slot: int | None = None
    track_id: int | None = None
    source: str = ""

    @property
    def canonical_card(self) -> str:
        card = self.card.replace("_", "-")
        if card.startswith("evo-"):
            return card[4:]
        return CARD_ALIASES.get(card, card)


@dataclass(frozen=True)
class Match:
    expected: ActionEvent
    predicted: ActionEvent
    time_left_error_s: float | None
    added_video_time_error_s: float | None
    cell_distance: int | None


@dataclass(frozen=True)
class EvalResult:
    side: str
    matches: list[Match]
    misses: list[ActionEvent]
    false_positives: list[ActionEvent]

    @property
    def precision(self) -> float:
        denom = len(self.matches) + len(self.false_positives)
        return len(self.matches) / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = len(self.matches) + len(self.misses)
        return len(self.matches) / denom if denom else 1.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        denom = precision + recall
        return 2 * precision * recall / denom if denom else 0.0


def load_ground_truth(path: Path) -> list[ActionEvent]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        items = data.get("events", [])
        fps = _optional_float(data.get("fps"))
    elif isinstance(data, list):
        items = data
        fps = None
    else:
        raise ValueError("ground truth must be a JSON object with events or a list")
    return [_event_from_json(item, source=str(path), fps=fps) for item in items]


def parse_predictions_txt(path: Path) -> list[ActionEvent]:
    events: list[ActionEvent] = []
    seen: set[tuple[Any, ...]] = set()
    current_video_time: float | None = None
    section: str | None = None

    for raw_line in path.read_text().splitlines():
        frame_match = FRAME_RE.match(raw_line)
        if frame_match:
            current_video_time = float(frame_match.group("video_time"))
            section = None
            continue

        line = raw_line.rstrip()
        if not line.strip():
            section = None
            continue
        if line == "enemy plays:":
            section = "enemy"
            continue
        if line == "own plays:":
            section = "own"
            continue
        if not line.startswith("  card="):
            continue

        if section == "enemy":
            match = ENEMY_RE.match(line)
            if not match:
                continue
            event = ActionEvent(
                side="enemy",
                card=match.group("card"),
                time_left_s=float(match.group("time_left")),
                video_time_s=current_video_time,
                track_id=int(match.group("track_id")),
                source=str(path),
            )
            if event.time_left_s is not None and event.time_left_s > MAX_PREDICTED_TIME_LEFT_S:
                continue
            key = (
                event.side,
                event.card,
                round(event.time_left_s or -1.0, 3),
                event.track_id,
            )
        elif section == "own":
            match = OWN_RE.match(line)
            if not match:
                continue
            cell = None
            if match.group("cell_x") is not None:
                cell = (int(match.group("cell_x")), int(match.group("cell_y")))
            event = ActionEvent(
                side="own",
                card=match.group("card"),
                time_left_s=float(match.group("time_left")),
                video_time_s=(
                    float(match.group("video_time"))
                    if match.group("video_time") is not None
                    else current_video_time
                ),
                cell=cell,
                slot=int(match.group("slot")),
                source=str(path),
            )
            if event.time_left_s is not None and event.time_left_s > MAX_PREDICTED_TIME_LEFT_S:
                continue
            key = (
                event.side,
                event.card,
                round(event.time_left_s or -1.0, 3),
                event.slot,
                event.cell,
            )
        else:
            continue

        if key in seen:
            continue
        seen.add(key)
        events.append(event)

    return events


def evaluate(
    expected: list[ActionEvent],
    predicted: list[ActionEvent],
    *,
    side: str,
    time_left_tolerance_s: float,
    video_time_tolerance_s: float,
    cell_tolerance: int,
    strict_evolution: bool,
) -> EvalResult:
    expected_side = [event for event in expected if event.side == side]
    predicted_side = [event for event in predicted if event.side == side]
    unmatched_predicted = set(range(len(predicted_side)))
    matches: list[Match] = []
    misses: list[ActionEvent] = []

    for expected_event in expected_side:
        candidates = []
        for idx in unmatched_predicted:
            predicted_event = predicted_side[idx]
            score = _match_score(
                expected_event,
                predicted_event,
                time_left_tolerance_s=time_left_tolerance_s,
                video_time_tolerance_s=video_time_tolerance_s,
                cell_tolerance=cell_tolerance,
                strict_evolution=strict_evolution,
            )
            if score is not None:
                candidates.append((score, idx, predicted_event))

        if not candidates:
            misses.append(expected_event)
            continue

        _, idx, predicted_event = min(candidates, key=lambda item: item[0])
        unmatched_predicted.remove(idx)
        matches.append(_build_match(expected_event, predicted_event))

    false_positives = [predicted_side[idx] for idx in sorted(unmatched_predicted)]
    return EvalResult(side=side, matches=matches, misses=misses, false_positives=false_positives)


def print_report(results: list[EvalResult]) -> None:
    for result in results:
        print(f"{result.side} actions:")
        print(
            f"  precision={result.precision:.3f} "
            f"recall={result.recall:.3f} "
            f"f1={result.f1:.3f} "
            f"matched={len(result.matches)} "
            f"missed={len(result.misses)} "
            f"false_positive={len(result.false_positives)}"
        )

        time_left_errors = [
            match.time_left_error_s
            for match in result.matches
            if match.time_left_error_s is not None
        ]
        added_time_errors = [
            match.added_video_time_error_s
            for match in result.matches
            if match.added_video_time_error_s is not None
        ]
        if time_left_errors:
            print(f"  time_left_error_s: {_format_error_stats(time_left_errors)}")
        if added_time_errors:
            print(f"  added_video_time_error_s: {_format_error_stats(added_time_errors)}")

        print("  matches:")
        for match in result.matches:
            expected = match.expected
            predicted = match.predicted
            print(
                "    "
                f"{expected.card} expected_tl={_fmt(expected.time_left_s)} "
                f"pred_tl={_fmt(predicted.time_left_s)} "
                f"tl_err={_fmt_signed(match.time_left_error_s)} "
                f"expected_video={_fmt(expected.video_time_s)} "
                f"added_video={_fmt(predicted.video_time_s)} "
                f"added_err={_fmt_signed(match.added_video_time_error_s)} "
                f"expected_cell={expected.cell} pred_cell={predicted.cell} "
                f"cell_dist={match.cell_distance}"
            )

        if result.misses:
            print("  misses:")
            for event in result.misses:
                print(
                    "    "
                    f"{event.card} expected_tl={_fmt(event.time_left_s)} "
                    f"expected_video={_fmt(event.video_time_s)} cell={event.cell}"
                )

        if result.false_positives:
            print("  false positives:")
            for event in result.false_positives:
                print(
                    "    "
                    f"{event.card} pred_tl={_fmt(event.time_left_s)} "
                    f"added_video={_fmt(event.video_time_s)} cell={event.cell}"
                )
        print()


def _event_from_json(item: dict[str, Any], *, source: str, fps: float | None = None) -> ActionEvent:
    cell = item.get("cell")
    if cell is not None:
        if len(cell) != 2:
            raise ValueError(f"cell must contain two values: {item}")
        cell = (int(cell[0]), int(cell[1]))

    video_time_s = _optional_float(item.get("video_time_s"))
    frame_index = item.get("frame_index")
    if video_time_s is None and frame_index is not None:
        if fps is None:
            raise ValueError(f"frame_index requires top-level fps or video_time_s: {item}")
        video_time_s = int(frame_index) / fps

    return ActionEvent(
        side=str(item["side"]),
        card=str(item["card"]).replace("_", "-"),
        time_left_s=_optional_float(item.get("time_left_s")),
        video_time_s=video_time_s,
        frame_index=int(frame_index) if frame_index is not None else None,
        cell=cell,
        slot=item.get("slot"),
        track_id=item.get("track_id"),
        source=source,
    )


def _match_score(
    expected: ActionEvent,
    predicted: ActionEvent,
    *,
    time_left_tolerance_s: float,
    video_time_tolerance_s: float,
    cell_tolerance: int,
    strict_evolution: bool,
) -> float | None:
    if expected.side != predicted.side:
        return None
    if strict_evolution:
        if expected.card != predicted.card:
            return None
    elif expected.canonical_card != predicted.canonical_card:
        return None

    score = 0.0
    has_time_constraint = False
    if expected.time_left_s is not None and predicted.time_left_s is not None:
        delta = abs(predicted.time_left_s - expected.time_left_s)
        if delta > time_left_tolerance_s:
            return None
        score += delta
        has_time_constraint = True

    if expected.video_time_s is not None and predicted.video_time_s is not None:
        delta = abs(predicted.video_time_s - expected.video_time_s)
        if delta > video_time_tolerance_s:
            return None
        score += delta
        has_time_constraint = True

    if not has_time_constraint:
        return None

    cell_distance = _cell_distance(expected.cell, predicted.cell)
    if cell_distance is not None:
        if cell_distance > cell_tolerance:
            return None
        score += cell_distance * 0.25

    return score


def _build_match(expected: ActionEvent, predicted: ActionEvent) -> Match:
    time_left_error = None
    if expected.time_left_s is not None and predicted.time_left_s is not None:
        time_left_error = predicted.time_left_s - expected.time_left_s

    added_video_time_error = None
    if expected.video_time_s is not None and predicted.video_time_s is not None:
        added_video_time_error = predicted.video_time_s - expected.video_time_s

    return Match(
        expected=expected,
        predicted=predicted,
        time_left_error_s=time_left_error,
        added_video_time_error_s=added_video_time_error,
        cell_distance=_cell_distance(expected.cell, predicted.cell),
    )


def _cell_distance(left: tuple[int, int] | None, right: tuple[int, int] | None) -> int | None:
    if left is None or right is None:
        return None
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _format_error_stats(values: list[float]) -> str:
    mean = statistics.fmean(values)
    median = statistics.median(values)
    abs_values = [abs(value) for value in values]
    return (
        f"mean={mean:+.2f} median={median:+.2f} "
        f"mean_abs={statistics.fmean(abs_values):.2f} max_abs={max(abs_values):.2f}"
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt_signed(value: float | None) -> str:
    return "-" if value is None else f"{value:+.2f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate own/enemy action detections against ground truth.")
    parser.add_argument("--ground-truth", required=True, type=Path, help="JSON file with labeled action events.")
    parser.add_argument("--predictions", required=True, type=Path, help="Txt output generated by capture/main.py.")
    parser.add_argument("--side", choices=["own", "enemy", "both"], default="both")
    parser.add_argument("--time-left-tolerance", type=float, default=2.0)
    parser.add_argument("--video-time-tolerance", type=float, default=2.0)
    parser.add_argument("--cell-tolerance", type=int, default=1)
    parser.add_argument("--strict-evolution", action="store_true")
    args = parser.parse_args()

    expected = load_ground_truth(args.ground_truth)
    predicted = parse_predictions_txt(args.predictions)
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
    print_report(results)


if __name__ == "__main__":
    main()
