from cr_bot.domain.constants import (
    ELIXIR_PER_SECOND_DOUBLE,
    ELIXIR_PER_SECOND_NORMAL,
    ELIXIR_PER_SECOND_TRIPLE,
    MAX_ELIXIR,
    STARTING_ELIXIR_EST,
)


class EnemyElixirEstimator:
    def __init__(self):
        self.value: float | None = None
        self.last_time_left_s: float | None = None
        self.last_update_monotonic_s: float | None = None

    def start_match(self, time_left_s, total_remaining_s, now_s=None):
        opening_elapsed = max(0.0, 180.0 - time_left_s)
        self.value = min(
            MAX_ELIXIR,
            STARTING_ELIXIR_EST + opening_elapsed * ELIXIR_PER_SECOND_NORMAL,
        )
        self.last_time_left_s = total_remaining_s
        self.last_update_monotonic_s = now_s

    def update(self, time_left_s, now_s=None):
        if self.last_time_left_s is None:
            self.last_time_left_s = time_left_s
            self.last_update_monotonic_s = now_s
            return
        if now_s is not None:
            if self.last_update_monotonic_s is None:
                self.last_update_monotonic_s = now_s
                self.last_time_left_s = time_left_s
                return
            elapsed = now_s - self.last_update_monotonic_s
            if elapsed <= 0 or elapsed > 2.0:
                self.last_update_monotonic_s = now_s
                self.last_time_left_s = time_left_s
                return
            self._add(elapsed * self._rate(time_left_s))
            self.last_update_monotonic_s = now_s
            self.last_time_left_s = time_left_s
            return
        elapsed = self.last_time_left_s - time_left_s
        if elapsed <= 0 or elapsed > 2.0:
            self.last_time_left_s = time_left_s
            return
        self._add(elapsed * self._rate(time_left_s))
        self.last_time_left_s = time_left_s

    def spend(self, cost):
        self.value = max(0.0, self.value - cost)

    def refund(self, amount):
        self.value = min(MAX_ELIXIR, self.value + amount)

    def _add(self, amount):
        self.value = min(MAX_ELIXIR, self.value + amount)

    @staticmethod
    def _rate(time_left_s):
        if time_left_s <= 60:
            return ELIXIR_PER_SECOND_TRIPLE
        if time_left_s <= 180:
            return ELIXIR_PER_SECOND_DOUBLE
        return ELIXIR_PER_SECOND_NORMAL
