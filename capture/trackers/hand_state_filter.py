class HandStateFilter:
    hand_slots = ("card_1", "card_2", "card_3", "card_4")

    def __init__(self):
        self.last_state = None

    def update(self, state):
        if self.last_state is None:
            self.last_state = state.copy()
            return state

        if self._all_hand_slots_changed(self.last_state, state):
            filtered = state.copy()
            for slot in self.hand_slots:
                filtered[slot] = self.last_state[slot]
            self.last_state = filtered.copy()
            return filtered

        self.last_state = state.copy()
        return state

    def reset(self):
        self.last_state = None

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
