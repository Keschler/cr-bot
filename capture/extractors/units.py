from game_state import Detection, Match
from troop_hp_level16 import get_troop_hp_level16


def find_corresponding_bar(troop, bars):
    corresponding_bar = None
    best_score = 100000  # the lower the better

    for bar in bars:
        x_distance = abs(bar["center_x"] - troop["center_x"])
        vertical_gap = troop["center_y"] - bar["center_y"]

        if vertical_gap < 0 or bar["team"] != troop["team"]:
            continue

        score = x_distance + 2 * vertical_gap
        if score < best_score:
            best_score = score
            corresponding_bar = bar

    return corresponding_bar


def match_troops_to_bars(troops, bars):
    matches = []
    available_bars = bars.copy()

    for troop in troops:
        corresponding_bar = find_corresponding_bar(troop, available_bars)
        matches.append({"troop": troop, "bar": corresponding_bar})
        if corresponding_bar is not None:
            available_bars.remove(corresponding_bar)

    return matches


def detection_from_dict(d: dict) -> Detection:
    return Detection(
        track_id=d.get("track_id"),
        class_name=d["class_name"],
        team=d["team"],
        confidence=d["confidence"],
        x1=d["x1"],
        y1=d["y1"],
        x2=d["x2"],
        y2=d["y2"],
        center_x=d["center_x"],
        center_y=d["center_y"],
        estimated_hp=d.get("estimated_hp"),
    )


def match_from_dict(d: dict) -> Match:
    troop = detection_from_dict(d["troop"])
    bar = detection_from_dict(d["bar"]) if d["bar"] is not None else None

    if bar is not None:
        troop_hp = get_troop_hp_level16(troop.class_name)
        if troop_hp:
            print(f"Troop hp{troop_hp}, BAR_ESTIMATED_HP {bar.estimated_hp}")
            troop.estimated_hp = bar.estimated_hp * troop_hp
            print("CALCULATED", troop.estimated_hp)
    else:
        troop.estimated_hp = get_troop_hp_level16(troop.class_name)

    return Match(
        troop=troop,
        bar=bar,
    )
