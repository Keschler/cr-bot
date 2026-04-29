class MatchClockFilter:
    def __init__(self) -> None:
        self.last_time_left_s = None
        self.last_seen_monotonic_s = None
    def update(self, detected_time_left_s, now_s):
        if detected_time_left_s is None:
            return self.last_time_left_s

        if self.last_time_left_s is None:
            self.last_time_left_s = detected_time_left_s
            self.last_seen_monotonic_s = now_s
            return detected_time_left_s

        wall_elapsed = now_s - self.last_seen_monotonic_s
        expected_drop = wall_elapsed

        observed_drop = self.last_time_left_s - detected_time_left_s
        
        if -1.0 <= observed_drop <= expected_drop + 1.5:
            self.last_time_left_s = detected_time_left_s
            self.last_seen_monotonic_s = now_s
        
        return self.last_time_left_s
