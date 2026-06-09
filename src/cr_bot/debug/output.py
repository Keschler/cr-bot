import cv2
import numpy as np

from cr_bot.features.action_space import ACTION_GRID
from cr_bot.vision.yolo_runtime import summarize_detections


def render_debug_panel(img: np.ndarray | None, label: str, tile_w: int, tile_h: int) -> np.ndarray:
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if img is None or img.size == 0:
        cv2.putText(tile, "missing", (8, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return tile

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = min((tile_w - 10) / img.shape[1], (tile_h - 26) / img.shape[0])
    resized = cv2.resize(
        img,
        (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    y0 = 22 + (tile_h - 22 - resized.shape[0]) // 2
    x0 = (tile_w - resized.shape[1]) // 2
    tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return tile


def render_tower_hp_debug(steps_by_tower: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    order = [
        "enemy_king",
        "enemy_support_left",
        "enemy_support_right",
        "own_king",
        "own_support_left",
        "own_support_right",
    ]
    step_order = ["raw", "normalized", "binary", "boxes", "digits"]
    cell_w = 180
    cell_h = 90
    rows = []

    for tower_name in order:
        row_tiles = []
        steps = steps_by_tower.get(tower_name) or {}
        for step_name in step_order:
            row_tiles.append(render_debug_panel(steps.get(step_name), f"{tower_name}:{step_name}", cell_w, cell_h))
        rows.append(np.hstack(row_tiles))

    return np.vstack(rows)


def render_timer_debug(steps: dict[str, np.ndarray]) -> np.ndarray:
    step_order = ["raw", "binary", "boxes", "digits"]
    cell_w = 220
    cell_h = 110
    tiles = [render_debug_panel(steps.get(step_name), f"timer:{step_name}", cell_w, cell_h) for step_name in step_order]
    return np.hstack(tiles)


def crop_detection(frame: np.ndarray, detection, pad: int = 6) -> np.ndarray | None:
    if detection is None:
        return None

    h, w = frame.shape[:2]
    x1 = max(0, int(detection.x1) - pad)
    y1 = max(0, int(detection.y1) - pad)
    x2 = min(w, int(detection.x2) + pad)
    y2 = min(h, int(detection.y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def render_match_debug(frame: np.ndarray, matches) -> np.ndarray:
    cell_w = 180
    cell_h = 110
    rows = []

    for idx, match in enumerate(matches):
        troop_label = f"{idx}:{match.troop.class_name}:{match.troop.team}"
        bar_label = f"bar:{match.bar.team}" if match.bar is not None else "bar:missing"
        troop_crop = crop_detection(frame, match.troop, pad=12)
        bar_crop = crop_detection(frame, match.bar, pad=6)
        row = np.hstack([
            render_debug_panel(troop_crop, troop_label, cell_w, cell_h),
            render_debug_panel(bar_crop, bar_label, cell_w, cell_h),
        ])
        rows.append(row)

    if not rows:
        return np.hstack([
            render_debug_panel(None, "troop", cell_w, cell_h),
            render_debug_panel(None, "bar", cell_w, cell_h),
        ])

    return np.vstack(rows)


def format_tower_ocr_debug(steps):
    value = steps.get("ocr_value")
    rejected_reason = steps.get("ocr_rejected_reason")
    missing_reason = steps.get("ocr_missing_reason")
    scores = steps.get("digit_scores") or []
    if value is None and not scores:
        reason_text = ""
        if rejected_reason:
            reason_text += f" rejected={rejected_reason}"
        if missing_reason:
            reason_text += f" reason={missing_reason}"
        return f"ocr=missing{reason_text}"

    score_parts = []
    for item in scores:
        char = item.get("char")
        score = item.get("score")
        if score is None:
            score_parts.append(str(char))
        else:
            score_parts.append(f"{char}:{score:.3f}")

    reason_text = f" rejected={rejected_reason}" if rejected_reason else ""
    if missing_reason:
        reason_text += f" reason={missing_reason}"
    return f"ocr={value or 'none'}{reason_text} scores=[{', '.join(score_parts)}]"


def match_cell(match, arena_px):
    if arena_px is None:
        return None
    return ACTION_GRID.pixel_to_cell(
        match.troop.center_x,
        match.troop.center_y,
        arena_px,
    )


def print_frame_result(result, enemy_card_tracker, own_action_tracker=None):
    if enemy_card_tracker.elixir_enemy_est is None:
        print("enemy elixir is undefined")
    else:
        print(f"enemy elixir est: {enemy_card_tracker.elixir_enemy_est:.2f}")
    print(f"seen enemy cards: {sorted(enemy_card_tracker.confirmed_seen_cards)}")
    print("enemy plays:")
    for play in enemy_card_tracker.detected_card_plays:
        print(
            f"  card={play['card']:<20} "
            f"cost={play['cost']} "
            f"time_left={play['time_left_s']} "
            f"track_id={play['track_id']} "
            f"cell={play.get('cell')}"
        )
    if own_action_tracker is not None:
        print("own plays:")
        for action in own_action_tracker.actions:
            video_time = action.get("video_time_s")
            video_time_text = (
                f"video_time={video_time:.2f} "
                if video_time is not None
                else ""
            )
            print(
                f"  card={action['card']:<20} "
                f"slot={action['slot_idx']} "
                f"cell={action['cell']} "
                f"{video_time_text}"
                f"time_left={action['time_left_s']} "
            )
    print()

    elixir = result.elixir
    detection_summary = summarize_detections(result.yolo_boxes)
    print(f"time:   {result.time_left_s}")
    print(f"elixir: {elixir['estimated_value'] + elixir['displayed_digit']}")
    print(f"yolo:   {detection_summary}")

    print("towers:")
    tower_debug_steps = result.tower_hp_debug_steps or {}
    for name, hp in result.towers_hp.items():
        debug_text = format_tower_ocr_debug(tower_debug_steps.get(name) or {})
        print(f"{name}: {hp} ({debug_text})")

    print("state:")
    for slot, value in result.hand_state.items():
        print(f"  {slot}: {value}")
    print("matches:")
    for match in result.matches:
        cell = match_cell(match, result.arena_px)
        print(
            f"  troop={match.troop.class_name:<18} "
            f"team={match.troop.team:<5} "
            f"conf={match.troop.confidence:.3f} "
            f"hp={match.troop.estimated_hp} "
            f"cell={cell}"
        )
    print()


def print_debug_frame_result(result, enemy_card_tracker, own_action_tracker):
    elixir = result.elixir
    print(f"Estimated elixir {elixir['estimated_value'] + elixir['displayed_digit']}")
    print(f"Overtime {result.overtime}")

    detection_summary = summarize_detections(result.yolo_boxes)
    print(f"time:   {result.time} time_left {result.total_remaining_s}")
    print(f"yolo:   {detection_summary}")

    print("towers:")
    tower_debug_steps = result.tower_hp_debug_steps or {}
    for name, hp in result.towers_hp.items():
        debug_text = format_tower_ocr_debug(tower_debug_steps.get(name) or {})
        print(f"{name}: {hp} ({debug_text})")

    print("state:")
    for slot, value in result.hand_state.items():
        print(f"  {slot}: {value}")

    print("matches:")
    for match in result.matches:
        cell = match_cell(match, result.arena_px)
        print(
            f"  troop={match.troop.class_name:<18} "
            f"team={match.troop.team:<5} "
            f"conf={match.troop.confidence:.3f} "
            f"hp={match.troop.estimated_hp} "
            f"cell={cell}"
        )

    print("enemy plays:")
    for play in enemy_card_tracker.detected_card_plays:
        print(
            f"  card={play['card']:<20} "
            f"cost={play['cost']} "
            f"time_left={play['time_left_s']} "
            f"track_id={play['track_id']} "
            f"cell={play.get('cell')}"
        )
    print("own plays:")
    for action in own_action_tracker.actions:
        print(
            f"  card={action['card']:<20} "
            f"slot={action['slot_idx']} "
            f"cell={action['cell']} "
            f"time_left={action['time_left_s']}"
        )
    print()
