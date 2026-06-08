from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cr_bot.eval.action_eval import ActionEvent, parse_predictions_txt
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD


FRAME_RE = re.compile(r"^frame\s+(?P<frame>\d+)\s+video_time=(?P<video_time>[0-9.]+)s")
MISS_RE = re.compile(
    r"^\s+(?P<card>\S+)\s+expected_tl=(?P<time_left>-|[0-9.]+)\s+"
    r"expected_video=(?P<video_time>-|[0-9.]+)\s+cell=(?P<cell>None|unknown|\([^)]+\))"
)
MATCH_RE = re.compile(
    r"^\s+troop=(?P<class>\S+)\s+team=(?P<team>\S+)\s+conf=(?P<conf>[0-9.]+)"
)
PENDING_CHECK_RE = re.compile(
    r"pending check card=(?P<card>\S+).*?"
    r"elixir_drop=(?P<drop>None|-?[0-9.eE+-]+).*?"
    r"(?:pending_elixir_drop=(?P<pending_drop>None|-?[0-9.eE+-]+).*?)?"
    r"required_drop=(?P<required>None|-?[0-9.eE+-]+).*?"
    r"elixir_confirms=(?P<confirms>True|False).*?"
    r"placed_cell=(?P<cell>None|\([^)]+\))"
)
YOLO_ITEM_RE = re.compile(r"(?P<class>[a-z0-9-]+):(?P<team>ally|enemy|neutral)\((?P<count>\d+)\)")
OWN_DEBUG_PREFIX = "[own_actions]"
CARD_ALIASES = {
    "old-musketeer": "musketeer",
}
DISPLAY_CARD_TO_UNIT_CLASSES = {
    "evo-skeletons": {"skeleton-evolution"},
    "skeletons": {"skeleton"},
}
SPELL_CARDS = {
    "arrows",
    "barbarian-barrel",
    "clone",
    "earthquake",
    "fireball",
    "freeze",
    "giant-snowball",
    "goblin-barrel",
    "graveyard",
    "lightning",
    "log",
    "poison",
    "rage",
    "rocket",
    "royal-delivery",
    "tornado",
    "zap",
}


@dataclass(frozen=True)
class FrameBlock:
    frame_index: int
    video_time_s: float
    start_line: int
    lines: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class Condition:
    name: str
    status: str
    detail: str
    evidence: tuple[str, ...] = ()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explain action_eval misses by checking nearby capture-log evidence "
            "against own/enemy action conditions."
        )
    )
    parser.add_argument("--eval-output", required=True, type=Path, help="Text output from action_eval.py.")
    parser.add_argument("--predictions", required=True, type=Path, help="Capture/debug txt used as predictions.")
    parser.add_argument("--side", choices=["own", "enemy", "both"], default="both")
    parser.add_argument("--window-before", type=float, default=2.5)
    parser.add_argument("--window-after", type=float, default=4.0)
    parser.add_argument("--nearby-prediction-window", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--card", help="Only explain misses for this card.")
    args = parser.parse_args()

    sides = {"own", "enemy"} if args.side == "both" else {args.side}
    misses = [
        miss
        for miss in parse_misses(args.eval_output)
        if miss.side in sides and (args.card is None or cards_match(miss.card, args.card))
    ]
    if args.limit is not None:
        misses = misses[: args.limit]

    frames = parse_frame_blocks(args.predictions)
    predictions = parse_predictions_txt(args.predictions)

    if not misses:
        print("No matching misses found.")
        return

    print(f"explaining {len(misses)} miss(es)")
    print()
    for idx, miss in enumerate(misses, start=1):
        print(f"{idx}. {miss.side} {miss.card} expected_video={fmt(miss.video_time_s)} cell={miss.cell}")
        window = frame_window(
            frames,
            miss.video_time_s,
            before_s=args.window_before,
            after_s=args.window_after,
        )
        nearby_predictions = nearby_events(
            predictions,
            miss,
            window_s=args.nearby_prediction_window,
        )
        print_nearby_predictions(nearby_predictions, miss)
        conditions = (
            explain_own_miss(miss, window)
            if miss.side == "own"
            else explain_enemy_miss(miss, window, nearby_predictions)
        )
        for condition in conditions:
            print(f"  {condition.status:<4} {condition.name}: {condition.detail}")
            evidence_limit = 8 if condition.name == "enemy tracker gate/debug reason" else 3
            for item in condition.evidence[:evidence_limit]:
                print(f"       {item}")
        print()


def parse_misses(path: Path) -> list[ActionEvent]:
    misses: list[ActionEvent] = []
    side: str | None = None
    in_misses = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if line == "own actions:":
            side = "own"
            in_misses = False
            continue
        if line == "enemy actions:":
            side = "enemy"
            in_misses = False
            continue
        if line.strip() == "misses:":
            in_misses = True
            continue
        if line.strip() in {"matches:", "false positives:"}:
            in_misses = False
            continue
        if not in_misses or side is None:
            continue

        match = MISS_RE.match(line)
        if not match:
            continue
        misses.append(
            ActionEvent(
                side=side,
                card=match.group("card"),
                time_left_s=parse_optional_float(match.group("time_left")),
                video_time_s=parse_optional_float(match.group("video_time")),
                cell=parse_cell(match.group("cell")),
                source=str(path),
            )
        )

    return misses


def parse_frame_blocks(path: Path) -> list[FrameBlock]:
    blocks: list[FrameBlock] = []
    current_frame: int | None = None
    current_time: float | None = None
    current_start = 0
    current_lines: list[tuple[int, str]] = []

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        frame_match = FRAME_RE.match(raw_line)
        if frame_match:
            if current_frame is not None and current_time is not None:
                blocks.append(
                    FrameBlock(
                        frame_index=current_frame,
                        video_time_s=current_time,
                        start_line=current_start,
                        lines=tuple(current_lines),
                    )
                )
            current_frame = int(frame_match.group("frame"))
            current_time = float(frame_match.group("video_time"))
            current_start = line_no
            current_lines = [(line_no, raw_line.rstrip())]
            continue
        if current_frame is not None:
            current_lines.append((line_no, raw_line.rstrip()))

    if current_frame is not None and current_time is not None:
        blocks.append(
            FrameBlock(
                frame_index=current_frame,
                video_time_s=current_time,
                start_line=current_start,
                lines=tuple(current_lines),
            )
        )
    return blocks


def frame_window(
    frames: Iterable[FrameBlock],
    video_time_s: float | None,
    *,
    before_s: float,
    after_s: float,
) -> list[FrameBlock]:
    if video_time_s is None:
        return []
    start = video_time_s - before_s
    end = video_time_s + after_s
    return [frame for frame in frames if start <= frame.video_time_s <= end]


def nearby_events(
    predictions: Iterable[ActionEvent],
    miss: ActionEvent,
    *,
    window_s: float,
) -> list[ActionEvent]:
    if miss.video_time_s is None:
        return []
    events = [
        event
        for event in predictions
        if event.side == miss.side
        and event.video_time_s is not None
        and abs(event.video_time_s - miss.video_time_s) <= window_s
    ]
    return sorted(events, key=lambda event: abs((event.video_time_s or 0.0) - miss.video_time_s))


def explain_own_miss(miss: ActionEvent, frames: list[FrameBlock]) -> list[Condition]:
    lines = flatten_lines(frames)
    card_lines = [(line_no, line) for line_no, line in lines if line_mentions_card(line, miss.card)]
    expected_classes = unit_classes_for_card(miss.card)
    is_spell = canonical_card(miss.card) in SPELL_CARDS

    conditions = [
        has_any(
            "HUD/card-drop candidate",
            card_lines,
            include=("drop detected", "rolling spell change detected"),
            pass_detail="card left the hand / rolling spell change was detected",
            fail_detail="no nearby slot drop or rolling-spell change for this card",
        ),
        own_elixir_condition(card_lines, miss.card),
    ]

    if is_spell:
        conditions.append(
            has_any(
                "spell visual confirmation",
                card_lines,
                include=("spell release marker visible", "spell aim ellipse visible", "first visible own rolling spell detected"),
                pass_detail="spell aim/release or rolling-spell visual was seen",
                fail_detail="no nearby spell aim/release visual for this card",
            )
        )
    else:
        unit_evidence = matching_unit_lines(lines, expected_classes, team="ally")
        conditions.append(
            Condition(
                "matching ally unit",
                "PASS" if unit_evidence else "FAIL",
                (
                    f"saw matching ally detector class: {', '.join(sorted(expected_classes))}"
                    if unit_evidence
                    else f"no nearby matching ally detector class: {', '.join(sorted(expected_classes)) or 'unknown'}"
                ),
                tuple(unit_evidence),
            )
        )
        conditions.append(own_deploy_clock_condition(lines, miss.card, expected_classes))

    conditions.append(
        has_any(
            "confirmed own action",
            card_lines,
            include=("confirmed own action",),
            pass_detail="tracker added an own action; eval miss is likely matching/timing/cell related",
            fail_detail="tracker never emitted a confirmed own action for this card in the window",
        )
    )

    blockers = own_blocker_lines(lines, miss.card, expected_classes)
    if blockers:
        conditions.append(
            Condition(
                "blocker/debug reason",
                "INFO",
                "nearby tracker line explains why the pending action was not added",
                tuple(blockers),
            )
        )
    return conditions


def explain_enemy_miss(
    miss: ActionEvent,
    frames: list[FrameBlock],
    nearby_predictions: list[ActionEvent],
) -> list[Condition]:
    lines = flatten_lines(frames)
    expected_classes = unit_classes_for_card(miss.card)
    enemy_units = matching_unit_lines(lines, expected_classes, team="enemy")
    enemy_clocks = clock_yolo_lines(lines, team="enemy")
    same_card_prediction = [
        event
        for event in nearby_predictions
        if cards_match(event.card, miss.card)
    ]

    conditions = [
        Condition(
            "matching enemy unit/spell",
            "PASS" if enemy_units else "FAIL",
            (
                f"saw matching enemy detector class: {', '.join(sorted(expected_classes))}"
                if enemy_units
                else f"no nearby matching enemy detector class: {', '.join(sorted(expected_classes)) or 'unknown'}"
            ),
            tuple(enemy_units),
        ),
        Condition(
            "enemy deploy clock",
            "PASS" if enemy_clocks else "FAIL",
            (
                "enemy clock detection appeared nearby"
                if enemy_clocks
                else "no nearby enemy deploy-clock summary line"
            ),
            tuple(enemy_clocks),
        ),
        Condition(
            "recorded enemy play",
            "PASS" if same_card_prediction else "FAIL",
            (
                "same-card enemy play exists nearby; eval miss is likely timing/cell/card matching"
                if same_card_prediction
                else "tracker never printed a same-card enemy play near the expected time"
            ),
            tuple(format_event_evidence(event, miss) for event in same_card_prediction),
        ),
    ]

    tracker_debug = enemy_tracker_debug_lines(lines, miss.card, expected_classes)
    if tracker_debug:
        conditions.append(
            Condition(
                "enemy tracker gate/debug reason",
                "INFO",
                "nearby enemy-card tracker lines mention this card/class",
                tuple(tracker_debug),
            )
        )

    return conditions


def own_deploy_clock_condition(
    lines: list[tuple[int, str]],
    card: str,
    expected_classes: set[str],
) -> Condition:
    positive: list[str] = []
    inferred_none: list[str] = []
    other_negative: list[str] = []
    for line_no, line in lines:
        if "inferred cell from ally clock" in line:
            class_match = re.search(r"class=(?P<class>\S+):", line)
            if class_match and class_match.group("class") not in expected_classes:
                continue
            if line.rstrip().endswith(": None"):
                inferred_none.append(format_evidence(line_no, line))
            else:
                positive.append(format_evidence(line_no, line))
        elif (
            line_mentions_card(line, card)
            and ("no matching ally deploy clock" in line or "no matching recent ally track" in line)
        ):
            other_negative.append(format_evidence(line_no, line))

    if positive:
        return Condition(
            "deploy clock cell",
            "PASS",
            "matched ally deploy clock mapped to an action-grid cell",
            tuple(positive),
        )
    if inferred_none:
        return Condition(
            "deploy clock cell",
            "FAIL",
            "matched ally deploy clock was found, but it mapped outside the action grid (cell=None)",
            tuple(inferred_none + other_negative),
        )
    if other_negative:
        return Condition(
            "deploy clock cell",
            "FAIL",
            "deploy clock/track condition failed before the action could be added",
            tuple(other_negative),
        )
    return Condition(
        "deploy clock cell",
        "FAIL",
        "no nearby matched ally deploy-clock cell for this troop/building",
    )


def own_elixir_condition(card_lines: list[tuple[int, str]], card: str) -> Condition:
    pending_checks: list[tuple[int, str, float | None, float | None, float | None, bool, str]] = []
    evidence_lines: list[str] = []
    for line_no, line in card_lines:
        pending_match = PENDING_CHECK_RE.search(line)
        if pending_match and cards_match(pending_match.group("card"), card):
            drop = parse_optional_debug_float(pending_match.group("drop"))
            pending_drop = parse_optional_debug_float(pending_match.group("pending_drop"))
            required = parse_optional_debug_float(pending_match.group("required"))
            confirms = pending_match.group("confirms") == "True"
            cell = pending_match.group("cell")
            pending_checks.append((line_no, line, drop, pending_drop, required, confirms, cell))
            continue
        if "latched numeric elixir drop" in line or "attached elixir-change time" in line:
            evidence_lines.append(format_evidence(line_no, line))

    confirmed_checks = [item for item in pending_checks if item[5]]
    positive_checks = [
        item
        for item in pending_checks
        if item[2] is not None and item[2] > 0
    ]
    positive_pending_checks = [
        item
        for item in pending_checks
        if item[3] is not None and item[3] > 0
    ]
    max_drop_check = (
        max(positive_checks, key=lambda item: item[2] or float("-inf"))
        if positive_checks
        else None
    )
    max_pending_drop_check = (
        max(positive_pending_checks, key=lambda item: item[3] or float("-inf"))
        if positive_pending_checks
        else None
    )
    required_values = [
        item[4]
        for item in pending_checks
        if item[4] is not None
    ]
    required_text = fmt(max(required_values)) if required_values else "-"
    max_drop_text = fmt(max_drop_check[2]) if max_drop_check is not None else "-"
    max_pending_drop_text = (
        fmt(max_pending_drop_check[3]) if max_pending_drop_check is not None else "-"
    )

    evidence: list[str] = []
    for item in confirmed_checks:
        evidence.append(
            f"{format_evidence(item[0], item[1])} "
            f"[global decrease={fmt(item[2])}, "
            f"after-card decrease={fmt(item[3])}, required={fmt(item[4])}, "
            f"cell={item[6]}]"
        )
    if max_drop_check is not None and max_drop_check not in confirmed_checks:
        evidence.append(
            f"{format_evidence(max_drop_check[0], max_drop_check[1])} "
            f"[max observed global decrease={fmt(max_drop_check[2])}, "
            f"after-card decrease={fmt(max_drop_check[3])}, "
            f"required={fmt(max_drop_check[4])}, cell={max_drop_check[6]}]"
        )
    if (
        max_pending_drop_check is not None
        and max_pending_drop_check not in confirmed_checks
        and max_pending_drop_check is not max_drop_check
    ):
        evidence.append(
            f"{format_evidence(max_pending_drop_check[0], max_pending_drop_check[1])} "
            f"[global decrease={fmt(max_pending_drop_check[2])}, "
            f"max after-card decrease={fmt(max_pending_drop_check[3])}, "
            f"required={fmt(max_pending_drop_check[4])}, cell={max_pending_drop_check[6]}]"
        )
    evidence.extend(evidence_lines)

    has_confirmation = bool(confirmed_checks or evidence_lines)
    if has_confirmation:
        confirmed_drop_text = (
            fmt(max((item[2] for item in confirmed_checks if item[2] is not None), default=None))
            if confirmed_checks
            else "-"
        )
        return Condition(
            "elixir confirmation",
            "PASS",
            (
                "elixir evidence was attached to the pending action; "
                f"max global decrease={max_drop_text}, "
                f"max after-card decrease={max_pending_drop_text}, "
                f"confirmed decrease={confirmed_drop_text}, required={required_text}"
            ),
            tuple(unique_preserve_order(evidence)),
        )

    if pending_checks:
        return Condition(
            "elixir confirmation",
            "FAIL",
            (
                "no nearby numeric/estimated elixir confirmation for this card; "
                f"max global decrease={max_drop_text}, "
                f"max after-card decrease={max_pending_drop_text}, required={required_text}"
            ),
            tuple(unique_preserve_order(evidence)),
        )

    return Condition(
        "elixir confirmation",
        "FAIL",
        "no nearby numeric/estimated elixir confirmation for this card; no pending elixir-drop measurements found",
    )


def own_blocker_lines(
    lines: list[tuple[int, str]],
    card: str,
    expected_classes: set[str],
) -> list[str]:
    blockers: list[str] = []
    for line_no, line in lines:
        if not any(token in line for token in ("blocked", "cancelled", "ignored", "not falling back", "not using")):
            continue
        if line_mentions_card(line, card):
            blockers.append(format_evidence(line_no, line))
            continue
        class_match = re.search(r"class=(?P<class>\S+)", line)
        if class_match and class_match.group("class").rstrip(":") in expected_classes:
            blockers.append(format_evidence(line_no, line))
    return blockers


def enemy_tracker_debug_lines(
    lines: list[tuple[int, str]],
    card: str,
    expected_classes: set[str],
) -> list[str]:
    clock_candidates: list[str] = []
    primary: list[str] = []
    skipped: list[str] = []
    candidates = card_candidates(card) | expected_classes
    for line_no, line in lines:
        if "[enemy_cards]" not in line:
            continue
        if not enemy_debug_mentions_any(line, candidates):
            continue
        evidence = format_evidence(line_no, line)
        if "clock candidate" in line:
            clock_candidates.append(evidence)
        elif "skip class=" in line:
            skipped.append(evidence)
        else:
            primary.append(evidence)
    return unique_preserve_order(clock_candidates + primary + skipped)


def enemy_debug_mentions_any(line: str, candidates: set[str]) -> bool:
    for candidate in candidates:
        if not candidate:
            continue
        escaped = re.escape(candidate)
        if re.search(rf"\b(?:card|class)={escaped}(?=\s|:|$)", line):
            return True
        if re.search(rf"\benemy {escaped}(?=\s|:|$)", line):
            return True
    return False


def has_any(
    name: str,
    lines: list[tuple[int, str]],
    *,
    include: tuple[str, ...],
    pass_detail: str,
    fail_detail: str,
) -> Condition:
    evidence = [
        format_evidence(line_no, line)
        for line_no, line in lines
        if any(token in line for token in include)
    ]
    return Condition(
        name,
        "PASS" if evidence else "FAIL",
        pass_detail if evidence else fail_detail,
        tuple(evidence),
    )


def matching_unit_lines(
    lines: list[tuple[int, str]],
    expected_classes: set[str],
    *,
    team: str,
) -> list[str]:
    evidence: list[str] = []
    for line_no, line in lines:
        match = MATCH_RE.match(line)
        if match and match.group("team") == team and match.group("class") in expected_classes:
            evidence.append(format_evidence(line_no, line))
            continue
        if team == "ally" and "new ally track" in line:
            class_match = re.search(r"class=(?P<class>\S+)", line)
            if class_match and class_match.group("class") in expected_classes:
                evidence.append(format_evidence(line_no, line))
    return evidence


def clock_yolo_lines(lines: list[tuple[int, str]], *, team: str) -> list[str]:
    evidence: list[str] = []
    needle = f"clock:{team} x"
    for line_no, line in lines:
        if line.startswith("yolo:") and needle in line:
            evidence.append(format_evidence(line_no, line))
    return evidence


def flatten_lines(frames: Iterable[FrameBlock]) -> list[tuple[int, str]]:
    return [line for frame in frames for line in frame.lines]


def line_mentions_card(line: str, card: str) -> bool:
    candidates = card_candidates(card)
    return any(candidate in line for candidate in candidates)


def card_candidates(card: str) -> set[str]:
    canonical = canonical_card(card)
    candidates = {card, canonical, card.replace("-", "_"), canonical.replace("-", "_")}
    candidates.update(unit_classes_for_card(card))
    return {candidate for candidate in candidates if candidate}


def unit_classes_for_card(card: str) -> set[str]:
    card = card.replace("_", "-")
    canonical = canonical_card(card)
    classes = set(DISPLAY_CARD_TO_UNIT_CLASSES.get(card, set()))
    classes.update(DISPLAY_CARD_TO_UNIT_CLASSES.get(canonical, set()))
    for unit_class, mapped_card in DIRECT_UNIT_TO_CARD.items():
        if mapped_card is not None and canonical_card(mapped_card) == canonical:
            classes.add(unit_class)
    if not classes:
        classes.add(canonical)
    return classes


def cards_match(left: str, right: str) -> bool:
    return canonical_card(left) == canonical_card(right)


def canonical_card(card: str) -> str:
    card = card.replace("_", "-")
    if card.startswith("evo-"):
        card = card[4:]
    return CARD_ALIASES.get(card, card)


def print_nearby_predictions(events: list[ActionEvent], miss: ActionEvent) -> None:
    same_card = [event for event in events if cards_match(event.card, miss.card)]
    source = same_card or events[:3]
    if not source:
        print("  nearby predictions: none")
        return
    label = "same-card" if same_card else "nearest"
    formatted = ", ".join(format_event_delta(event, miss) for event in source[:3])
    print(f"  nearby predictions ({label}): {formatted}")


def format_event_delta(event: ActionEvent, miss: ActionEvent) -> str:
    delta = None
    if event.video_time_s is not None and miss.video_time_s is not None:
        delta = event.video_time_s - miss.video_time_s
    delta_text = "" if delta is None else f" ({delta:+.2f}s)"
    return f"{event.card}@{fmt(event.video_time_s)}{delta_text} cell={event.cell}"


def format_event_evidence(event: ActionEvent, miss: ActionEvent) -> str:
    return f"prediction {format_event_delta(event, miss)}"


def format_evidence(line_no: int, line: str) -> str:
    return f"L{line_no}: {line}"


def parse_optional_float(value: str) -> float | None:
    if value == "-":
        return None
    return float(value)


def parse_optional_debug_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    return float(value)


def parse_cell(value: str) -> tuple[int, int] | None:
    if value in {"None", "unknown"}:
        return None
    left, right = value.strip("()").split(",")
    return (int(left.strip()), int(right.strip()))


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    main()
