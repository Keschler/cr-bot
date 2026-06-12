from cr_bot.domain.constants import (
    ENEMY_CLOCK_FIRST_SEEN_MIN_CONF,
    ENEMY_RECENT_CLOCK_CONFIRM_SECONDS,
    FRAME_CONFIRM_TROOPS,
)
from cr_bot.trackers.direct_unit_to_card import DIRECT_UNIT_TO_CARD

from .models import FRAME_CONFIRM_SPELL_CLASSES, RecentEnemyClock


class EnemyClockTracker:
    def __init__(self, debug):
        self.recent: list[RecentEnemyClock] = []
        self._debug = debug

    def remember(self, clock_boxes, now_s):
        if now_s is None:
            self.recent = []
            return
        self.recent = [
            clock
            for clock in self.recent
            if clock.seen_at_s is not None
            and now_s - clock.seen_at_s <= ENEMY_RECENT_CLOCK_CONFIRM_SECONDS
        ]
        for clock in clock_boxes:
            if clock["confidence"] < 0.5 or clock["team"] != "enemy":
                continue
            remembered = self._find(clock)
            if remembered is None:
                self.recent.append(
                    RecentEnemyClock(
                        seen_at_s=now_s,
                        center_x=clock["center_x"],
                        center_y=clock["center_y"],
                        track_id=clock.get("track_id"),
                    )
                )
                continue
            remembered.seen_at_s = now_s
            remembered.center_x = clock["center_x"]
            remembered.center_y = clock["center_y"]
            if remembered.track_id is None:
                remembered.track_id = clock.get("track_id")

    def confirm(self, memory, troop, clock_boxes, now_s=None):
        saw_enemy_clock = False
        for clock in clock_boxes:
            if clock["confidence"] < 0.5:
                self._debug_candidate(
                    memory, troop, clock, source="current", status="skipped",
                    reject_reason=f"clock confidence {clock['confidence']:.3f} < 0.500",
                )
                continue
            if clock["team"] != "enemy":
                self._debug_candidate(
                    memory, troop, clock, source="current", status="skipped",
                    reject_reason=f"clock team {clock['team']} != enemy",
                )
                continue
            saw_enemy_clock = True
            if self._claim_current(memory, troop, clock):
                return True

        card_name = DIRECT_UNIT_TO_CARD.get(troop.class_name)
        if (
            now_s is None
            or memory.first_seen_now_s is None
            or card_name is None
            or troop.class_name in FRAME_CONFIRM_SPELL_CLASSES
            or troop.class_name in FRAME_CONFIRM_TROOPS
            or now_s > memory.first_seen_now_s + ENEMY_RECENT_CLOCK_CONFIRM_SECONDS
        ):
            if not saw_enemy_clock and memory.last_clock_reject_reason is None:
                memory.last_clock_reject_reason = "no current enemy clock box"
            elif now_s is None:
                memory.last_clock_reject_reason = "no monotonic time for remembered-clock lookup"
            elif memory.first_seen_now_s is None:
                memory.last_clock_reject_reason = "track has no first-seen monotonic time"
            elif card_name is None:
                memory.last_clock_reject_reason = f"class {troop.class_name} does not map to a card"
            elif troop.class_name in FRAME_CONFIRM_SPELL_CLASSES | FRAME_CONFIRM_TROOPS:
                memory.last_clock_reject_reason = f"class {troop.class_name} uses frame confirmation"
            else:
                memory.last_clock_reject_reason = "remembered-clock window expired"
            return False

        saw_recent_clock = False
        for clock in self.recent:
            if clock.seen_at_s is None or now_s - clock.seen_at_s > ENEMY_RECENT_CLOCK_CONFIRM_SECONDS:
                continue
            saw_recent_clock = True
            reject_reason = self._position_reject_reason(clock.center_x, clock.center_y, troop)
            if reject_reason is None:
                reject_reason = self._claim_reject_reason(memory, troop)
            if reject_reason is not None:
                memory.last_clock_reject_reason = reject_reason
                self._debug_candidate(
                    memory,
                    troop,
                    clock,
                    source="recent",
                    status="rejected",
                    reject_reason=reject_reason,
                )
                continue
            consumed_by = clock.consumed_by_track_id
            if consumed_by is not None and consumed_by != memory.track_id:
                reject_reason = f"enemy clock already consumed by track {consumed_by}"
                memory.last_clock_reject_reason = reject_reason
                self._debug_candidate(
                    memory,
                    troop,
                    clock,
                    source="recent",
                    status="consumed",
                    reject_reason=reject_reason,
                )
                continue
            if self._claim(memory, clock):
                self._debug_candidate(memory, troop, clock, source="recent", status="accepted")
                return True
        if memory.last_clock_reject_reason is None:
            memory.last_clock_reject_reason = (
                "no recent enemy clock box" if not saw_recent_clock else "enemy clock already consumed"
            )
        return False

    def _find(self, clock):
        track_id = clock.get("track_id")
        for remembered in self.recent:
            if track_id is not None and remembered.track_id == track_id:
                return remembered
            if (
                abs(remembered.center_x - clock["center_x"]) <= 12
                and abs(remembered.center_y - clock["center_y"]) <= 12
            ):
                return remembered
        return None

    def _claim_current(self, memory, troop, clock):
        reject_reason = self._position_reject_reason(
            clock["center_x"],
            clock["center_y"],
            troop,
        )
        if reject_reason is None:
            reject_reason = self._claim_reject_reason(memory, troop)
        if reject_reason is not None:
            memory.last_clock_reject_reason = reject_reason
            self._debug_candidate(
                memory, troop, clock, source="current", status="rejected",
                reject_reason=reject_reason,
            )
            return False
        remembered = self._find(clock)
        if remembered is None:
            remembered = RecentEnemyClock(
                seen_at_s=None,
                center_x=clock["center_x"],
                center_y=clock["center_y"],
                track_id=clock.get("track_id"),
            )
            self.recent.append(remembered)
        consumed_by = remembered.consumed_by_track_id
        if consumed_by is not None and consumed_by != memory.track_id:
            reject_reason = f"enemy clock already consumed by track {consumed_by}"
            memory.last_clock_reject_reason = reject_reason
            self._debug_candidate(
                memory, troop, clock, source="current", status="consumed",
                reject_reason=reject_reason, consumed_by=consumed_by,
            )
            return False
        claimed = self._claim(memory, remembered)
        self._debug_candidate(
            memory,
            troop,
            clock,
            source="current",
            status="accepted" if claimed else "rejected",
            reject_reason=None if claimed else "enemy clock claim failed",
            consumed_by=remembered.consumed_by_track_id,
        )
        return claimed

    @staticmethod
    def _claim(memory, clock):
        if (
            clock.consumed_by_track_id is not None
            and clock.consumed_by_track_id != memory.track_id
        ):
            return False
        clock.consumed_by_track_id = memory.track_id
        memory.deploy_clock_center_x = clock.center_x
        memory.deploy_clock_center_y = clock.center_y
        return True

    @staticmethod
    def _claim_reject_reason(memory, troop):
        if memory.seen_frames <= 1 and troop.confidence < ENEMY_CLOCK_FIRST_SEEN_MIN_CONF:
            return (
                f"first-seen confidence {troop.confidence:.3f} < "
                f"{ENEMY_CLOCK_FIRST_SEEN_MIN_CONF:.3f}"
            )
        return None

    @staticmethod
    def _position_reject_reason(clock_center_x, clock_center_y, troop):
        horizontal_gap = abs(clock_center_x - troop.center_x)
        vertical_gap = clock_center_y - troop.center_y
        if horizontal_gap > 90:
            return f"clock horizontal gap {horizontal_gap:.1f} > 90"
        if vertical_gap < 10:
            return f"clock vertical gap {vertical_gap:.1f} < 10"
        if vertical_gap > 140:
            return f"clock vertical gap {vertical_gap:.1f} > 140"
        return None

    def _debug_candidate(
        self,
        memory,
        troop,
        clock,
        *,
        source,
        status,
        reject_reason=None,
        consumed_by=None,
    ):
        clock_x = clock.center_x if isinstance(clock, RecentEnemyClock) else clock["center_x"]
        clock_y = clock.center_y if isinstance(clock, RecentEnemyClock) else clock["center_y"]
        clock_team = None if isinstance(clock, RecentEnemyClock) else clock.get("team")
        clock_track = clock.track_id if isinstance(clock, RecentEnemyClock) else clock.get("track_id")
        clock_confidence = None if isinstance(clock, RecentEnemyClock) else clock.get("confidence")
        if consumed_by is None and isinstance(clock, RecentEnemyClock):
            consumed_by = clock.consumed_by_track_id
        clock_conf = f"{clock_confidence:.3f}" if clock_confidence is not None else "-"
        self._debug(
            f"clock candidate track={memory.track_id} "
            f"class={memory.best_class or troop.class_name} "
            f"source={source} status={status} "
            f"troop_center=({troop.center_x:.1f},{troop.center_y:.1f}) "
            f"clock_center=({clock_x:.1f},{clock_y:.1f}) "
            f"clock_team={clock_team or '-'} "
            f"clock_track={clock_track if clock_track is not None else '-'} "
            f"clock_conf={clock_conf} dx={abs(clock_x - troop.center_x):.1f} "
            f"dy={clock_y - troop.center_y:.1f} "
            f"reject={reject_reason or '-'} "
            f"consumed_by={consumed_by if consumed_by is not None else '-'}"
        )
