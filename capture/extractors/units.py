from game_state import Detection, Match
from troop_hp_level16 import get_unit_hp_level16


def pair_score(troop, bar):
    x_distance = abs(bar["center_x"] - troop["center_x"])
    vertical_gap = troop["center_y"] - bar["center_y"]

    if vertical_gap < 0 or bar["team"] != troop["team"]:
        return None
    if x_distance > 120 or vertical_gap > 220:
        return None

    return x_distance + 2 * vertical_gap

def match_troops_to_bars(troops, bars):
    candidates = []
    
    for troop_idx, troop in enumerate(troops):
        for bar_idx, bar in enumerate(bars):
            score = pair_score(troop, bar)
            if score is None:
                continue
            candidates.append((score, troop_idx, bar_idx))
    candidates.sort(key=lambda x: x[0]) # Best match first

    used_troops = set()
    used_bars = set()
    assigned = {}

    for score, troop_idx, bar_idx in candidates:
        if troop_idx in used_troops or bar_idx in used_bars:
            continue
        assigned[troop_idx] = bars[bar_idx]
        used_troops.add(troop_idx)
        used_bars.add(bar_idx)
    matches = []
    for troop_idx, troop in enumerate(troops):
        bar = assigned.get(troop_idx)
        matches.append({"troop": troop, "bar": bar})

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
        troop_hp = get_unit_hp_level16(troop.class_name)
        if troop_hp:
            print(f"Troop hp{troop_hp}, BAR_ESTIMATED_HP {bar.estimated_hp}")
            troop.estimated_hp = bar.estimated_hp * troop_hp
            print("CALCULATED", troop.estimated_hp)
    else:
        troop.estimated_hp = get_unit_hp_level16(troop.class_name)

    return Match(
        troop=troop,
        bar=bar,
    )
