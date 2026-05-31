from cr_bot.domain.rois import ELIXIR_SLOT_ROIS, ROIS


ALTERNATIVE_VIDEO_ROIS = {
    "elixir_bar": (240, 2250, 805, 120),
    "elixir_digit": (295, 2250, 53, 53),
    "elixir_fill_slot_1": (270, 2280, 100, 50),
    "elixir_fill_slot_2": (348, 2280, 100, 50),
    "elixir_fill_slot_3": (426, 2280, 100, 50),
    "elixir_fill_slot_4": (504, 2280, 100, 50),
    "elixir_fill_slot_5": (582, 2280, 100, 50),
    "elixir_fill_slot_6": (660, 2280, 100, 50),
    "elixir_fill_slot_7": (738, 2280, 100, 50),
    "elixir_fill_slot_8": (816, 2280, 100, 50),
    "elixir_fill_slot_9": (894, 2280, 100, 50),
    "elixir_fill_slot_10": (972, 2280, 68, 50),
    "hand_card_slot_1": (230, 1960, 220, 300),
    "hand_card_slot_2": (430, 1960, 220, 300),
    "hand_card_slot_3": (630, 1960, 220, 300),
    "hand_card_slot_4": (840, 1960, 220, 300),
    "next_card_slot": (40, 2200, 120, 125),
}


def activate_alternative_video_rois() -> None:
    ROIS.update(ALTERNATIVE_VIDEO_ROIS)
    ELIXIR_SLOT_ROIS[:] = [
        ROIS[f"elixir_fill_slot_{slot}"]
        for slot in range(1, 11)
    ]
