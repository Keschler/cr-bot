class MatchClockFilter:
    def __init__(self) -> None:
        self.last_time_left_s = None
        self.last_seen_monotonic_s = None
        self.initialised = False
        self.initial_seen_values = []
        self.last_overtime = False

    def initialise(self, detected_time_left_s, now_s) -> None:
        if not self.initialised and detected_time_left_s is not None:
            self.initial_seen_values.append(detected_time_left_s)
            if len(self.initial_seen_values) >= 10: 
                self.last_seen_monotonic_s = now_s
                self.last_time_left_s = max(self.initial_seen_values, key=self.initial_seen_values.count)
                self.initialised = True
        
        
    def update(self, detected_time_left_s, now_s, overtime):
        entering_overtime = overtime and not self.last_overtime
        self.last_overtime = overtime
        if self.last_time_left_s is None or self.last_seen_monotonic_s is None:
            if detected_time_left_s is not None:
                self.last_time_left_s = detected_time_left_s
                self.last_seen_monotonic_s = now_s
            return detected_time_left_s

        wall_elapsed = now_s - self.last_seen_monotonic_s
        predicted_time_left_s = max(0.0, self.last_time_left_s - wall_elapsed)

        if detected_time_left_s is None:
            return predicted_time_left_s

        if abs(detected_time_left_s - predicted_time_left_s) <= 1.5 or (entering_overtime and detected_time_left_s > predicted_time_left_s + 30):
            self.last_time_left_s = detected_time_left_s
            self.last_seen_monotonic_s = now_s
            return detected_time_left_s
        
        return predicted_time_left_s
