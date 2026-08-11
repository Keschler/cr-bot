from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cr_bot.temporal_spells.config import SPELL_CLASSES, TARGET_CLASSES, TemporalSpellConfig
from cr_bot.temporal_spells.dataset import TemporalSpellDataset
from cr_bot.temporal_spells.model import TemporalSpellCNN
from cr_bot.temporal_spells.training import focal_cross_entropy, spatial_soft_target_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal spell classifier.")
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/models/temporal_spell_classifier_best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = TemporalSpellConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TemporalSpellDataset(args.train_manifest, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TemporalSpellCNN(len(SPELL_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for inputs, labels, heatmaps, has_target in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            heatmaps, has_target = heatmaps.to(device), has_target.to(device)
            event_logits, heatmap_logits = model(inputs)
            loss = focal_cross_entropy(event_logits, labels)
            loss += spatial_soft_target_loss(heatmap_logits, labels, heatmaps, has_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classes": list(SPELL_CLASSES),
            "input_config": config.as_dict(),
            "thresholds": {card: 0.5 for card in TARGET_CLASSES},
            "model_state": model.state_dict(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
