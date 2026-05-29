from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .features import AudioFeatureConfig, load_audio_window, waveform_to_log_mel
from .model import AudioCardCNN


class AudioCardPredictor:
    def __init__(self, checkpoint_path: str | Path, *, device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = list(checkpoint["classes"])
        self.config = AudioFeatureConfig(**checkpoint.get("feature_config", {}))
        self.model = AudioCardCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_file(self, audio_path: str | Path, *, top_k: int = 5) -> dict:
        waveform = load_audio_window(audio_path, self.config)
        features = waveform_to_log_mel(waveform, self.config).unsqueeze(0).to(self.device)
        logits = self.model(features)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(top_k, len(self.classes)))
        top = [
            {"card": self.classes[int(idx)], "confidence": float(value)}
            for value, idx in zip(values.cpu(), indices.cpu())
        ]
        return {
            "card": top[0]["card"],
            "confidence": top[0]["confidence"],
            "top": top,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a Clash Royale card from an audio clip.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    args = parser.parse_args()

    predictor = AudioCardPredictor(args.checkpoint)
    result = predictor.predict_file(args.audio, top_k=args.top_k)
    for item in result["top"]:
        print(f"{item['card']}: {item['confidence']:.4f}")


if __name__ == "__main__":
    main()

