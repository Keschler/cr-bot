from collections import deque

from cr_bot.domain.constants import FULL_TOWER_HP


class TowerHPFilter:
    def __init__(
        self,
        history_size: int = 5,
        confirm_count: int = 3,
        close_tolerance: int = 10,
        correction_tolerance: int = 100,
        large_drop_threshold: int = 300,
        large_drop_confirm_count: int = 10,
    ) -> None:
        self.history_size = history_size
        self.confirm_count = confirm_count
        self.close_tolerance = close_tolerance
        self.correction_tolerance = correction_tolerance
        self.large_drop_threshold = large_drop_threshold
        self.large_drop_confirm_count = large_drop_confirm_count
        self.history = {
            tower_name: deque(maxlen=history_size)
            for tower_name in FULL_TOWER_HP
        }
        self.candidates = {
            tower_name: deque(maxlen=max(history_size, large_drop_confirm_count))
            for tower_name in FULL_TOWER_HP
        }

    def reset(self) -> None:
        for values in self.history.values():
            values.clear()
        for values in self.candidates.values():
            values.clear()

    def update(self, towers_hp):
        filtered = {}

        for tower_name, raw_value in towers_hp.items():
            max_hp = FULL_TOWER_HP[tower_name]
            recent_values = self.history[tower_name]
            recent_hp = min(recent_values) if recent_values else max_hp

            value = self._repair_value(raw_value, max_hp)

            if value is None:
                filtered[tower_name] = recent_hp
                continue

            if value > recent_hp:
                if value - recent_hp <= self.correction_tolerance:
                    self.candidates[tower_name].append(value)
                    confirmed = self._confirmed_candidate(self.candidates[tower_name])
                    if confirmed is not None:
                        filtered[tower_name] = confirmed
                        recent_values.clear()
                        recent_values.append(confirmed)
                        self.candidates[tower_name].clear()
                        continue
                filtered[tower_name] = recent_hp
                continue

            if value < recent_hp:
                if self._looks_like_dropped_leading_digit(value, recent_hp):
                    filtered[tower_name] = recent_hp
                    continue
                self.candidates[tower_name].append(value)
                confirm_count = (
                    self.large_drop_confirm_count
                    if value != 0 and recent_hp - value >= self.large_drop_threshold
                    else self.confirm_count
                )
                confirmed = self._confirmed_candidate(
                    self.candidates[tower_name],
                    confirm_count=confirm_count,
                )
                if confirmed is None:
                    filtered[tower_name] = recent_hp
                    continue

                filtered[tower_name] = confirmed
                recent_values.append(confirmed)
                self.candidates[tower_name].clear()
                continue

            filtered[tower_name] = value
            recent_values.append(value)
            self.candidates[tower_name].clear()

        return filtered

    def _confirmed_candidate(self, candidates, confirm_count=None):
        if not candidates:
            return None

        confirm_count = confirm_count or self.confirm_count
        for value in sorted(set(candidates)):
            close_values = [
                candidate
                for candidate in candidates
                if abs(candidate - value) <= self.close_tolerance
            ]
            if len(close_values) >= confirm_count:
                close_values.sort()
                return close_values[len(close_values) // 2]

        return None

    def _repair_value(self, value, max_hp):
        if value is None:
            return None

        try:
            value = int(value)
        except (TypeError, ValueError):
            return None

        if 0 <= value <= max_hp:
            return value

        return None

    def _looks_like_dropped_leading_digit(self, value, recent_hp):
        if value == 0:
            return False
        if len(str(value)) >= len(str(recent_hp)):
            return False
        return recent_hp - value > 500
