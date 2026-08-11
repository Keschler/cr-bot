from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cr_bot.temporal_spells.config import TARGET_CLASSES, TemporalSpellConfig
from cr_bot.temporal_spells.dataset import TemporalSpellDataset
from cr_bot.temporal_spells.metrics import classification_metrics, localization_metrics
from cr_bot.temporal_spells.model import TemporalSpellCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a temporal spell checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = list(checkpoint["classes"])
    config = TemporalSpellConfig(**checkpoint["input_config"])
    dataset = TemporalSpellDataset(args.manifest, config, include_ownership=True)
    model = TemporalSpellCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    targets, predictions = [], []
    ownerships = []
    predicted_cells, target_cells = [], []
    with torch.inference_mode():
        for inputs, labels, heatmaps, has_target, batch_ownerships in DataLoader(
            dataset,
            batch_size=args.batch_size,
        ):
            event_logits, heatmap_logits = model(inputs.to(device))
            batch_predictions = event_logits.argmax(dim=1).cpu()
            targets.extend(labels.tolist())
            predictions.extend(batch_predictions.tolist())
            ownerships.extend(batch_ownerships)
            for index in range(len(labels)):
                if not bool(has_target[index]) or int(labels[index]) == 0:
                    continue
                class_index = int(labels[index]) - 1
                predicted_flat = int(heatmap_logits[index, class_index].argmax())
                target_flat = int(heatmaps[index].argmax())
                predicted_cells.append((predicted_flat % 18, predicted_flat // 18))
                target_cells.append((target_flat % 18, target_flat // 18))
    result = classification_metrics(targets, predictions, classes)
    result["by_ownership"] = {
        ownership: classification_metrics(
            [target for target, owner in zip(targets, ownerships) if owner == ownership],
            [
                prediction
                for prediction, owner in zip(predictions, ownerships)
                if owner == ownership
            ],
            classes,
        )
        for ownership in ("own", "enemy")
        if ownership in ownerships
    }
    result["localization"] = localization_metrics(predicted_cells, target_cells)
    result["thresholds"] = checkpoint.get(
        "thresholds",
        {card: 0.5 for card in TARGET_CLASSES},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
