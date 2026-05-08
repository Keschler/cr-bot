import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "assets/templates/cr-api-assets/cards-150"
OUTPUT_FILE = Path("seed_labels/cvat_cards_150_labels.json")
PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
]


def build_label(name: str, color: str) -> dict:
    return {
        "name": name,
        "color": color,
        "type": "any",
        "attributes": [
            {
                "name": "team",
                "input_type": "select",
                "mutable": False,
                "values": ["blue", "red"],
                "default_value": "blue",
            }
        ],
    }


def main() -> None:
    names = sorted(
        path.stem
        for path in CARDS_DIR.glob("*.png")
        if "hero-ev" not in path.stem
    )

    labels = [
        build_label(name, PALETTE[index % len(PALETTE)])
        for index, name in enumerate(names)
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(labels)} labels to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
