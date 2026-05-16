from collections import deque

from constants import FULL_TOWER_HP


class TowerHPFilter:
    def __init__(self, history_size: int = 5) -> None:
        self.history_size = history_size
        self.history = {
            tower_name: deque(maxlen=history_size)
            for tower_name in FULL_TOWER_HP
        }

    def reset(self) -> None:
        for values in self.history.values():
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

            filtered[tower_name] = value
            recent_values.append(value)

        return filtered

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
