class HandStateFilter:
    hand_slots = ("card_1", "card_2", "card_3", "card_4")
    full_hand_confirmation_frames = 3

    def __init__(self):
        self.last_state = None
        self.full_hand_candidate = None
        self.full_hand_candidate_frames = 0

    def update(self, state):
        if self.last_state is None:
            self.last_state = state.copy()
            return state

        if self._all_hand_slots_changed(self.last_state, state):
            if self._same_hand(self.full_hand_candidate, state):
                self.full_hand_candidate_frames += 1
            else:
                self.full_hand_candidate = state.copy()
                self.full_hand_candidate_frames = 1
            if self.full_hand_candidate_frames >= self.full_hand_confirmation_frames:
                self.last_state = state.copy()
                self._clear_full_hand_candidate()
                return state

            filtered = state.copy()
            for slot in self.hand_slots:
                filtered[slot] = self.last_state[slot]
            return filtered

        self._clear_full_hand_candidate()
        self.last_state = state.copy()
        return state

    def reset(self):
        self.last_state = None
        self._clear_full_hand_candidate()

    def _all_hand_slots_changed(self, previous, current):
        return all(
            self._card_name(previous.get(slot)) is not None
            and self._card_name(previous.get(slot)) != self._card_name(current.get(slot))
            for slot in self.hand_slots
        )

    def _card_name(self, card):
        if isinstance(card, (tuple, list)) and len(card) >= 1:
            card = card[0]
        if isinstance(card, str) and card.lower() == "none":
            return None
        return card

    def _same_hand(self, first, second):
        if first is None:
            return False
        return all(
            self._card_name(first.get(slot)) == self._card_name(second.get(slot))
            for slot in self.hand_slots
        )

    def _clear_full_hand_candidate(self):
        self.full_hand_candidate = None
        self.full_hand_candidate_frames = 0
