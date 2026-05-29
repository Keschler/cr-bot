from collections import deque

from cr_bot.domain.constants import FULL_TOWER_HP


class TowerHPFilter:
    def __init__(
        self,
        history_size: int = 5,
        confirm_count: int = 3,
        close_tolerance: int = 10,
    ) -> None:
        self.history_size = history_size
        self.confirm_count = confirm_count
        self.close_tolerance = close_tolerance
        self.history = {
            tower_name: deque(maxlen=history_size)
            for tower_name in FULL_TOWER_HP
        }
        self.candidates = {
            tower_name: deque(maxlen=history_size)
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
                filtered[tower_name] = recent_hp
                continue

            if value < recent_hp:
                self.candidates[tower_name].append(value)
                confirmed = self._confirmed_candidate(self.candidates[tower_name])
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

    def _confirmed_candidate(self, candidates):
        if not candidates:
            return None

        for value in sorted(set(candidates)):
            close_values = [
                candidate
                for candidate in candidates
                if abs(candidate - value) <= self.close_tolerance
            ]
            if len(close_values) >= self.confirm_count:
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

        text = str(value)
        if len(text) <= len(str(max_hp)):
            return None

        for idx in range(1, len(text)):
            candidate = int(text[idx:])
            if 0 <= candidate <= max_hp:
                return candidate

        return None
