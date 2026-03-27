from image_utils import extract_health_bar

def get_valid_troop_bars(frame, matched):
    valid_troop_bars = []
    for match in matched:
        bar = match["bar"]
        if bar is None:
            continue
        print(match["troop"]["class_name"])
        if extract_health_bar(frame, bar):
           valid_troop_bars.append([match["troop"], bar]) 
    return valid_troop_bars

def filter_real_bars(frame, bars):
    return [bar for bar in bars if extract_health_bar(frame, bar)]
