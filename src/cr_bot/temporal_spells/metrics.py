from __future__ import annotations

import numpy as np


def classification_metrics(targets, predictions, classes: list[str]) -> dict:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class = {}
    for index, name in enumerate(classes):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
        }
    return {"per_class": per_class, "confusion_matrix": confusion.tolist()}


def localization_metrics(predicted_cells, target_cells) -> dict:
    errors = [
        abs(predicted[0] - target[0]) + abs(predicted[1] - target[1])
        for predicted, target in zip(predicted_cells, target_cells)
        if predicted is not None and target is not None
    ]
    return {
        "mean_manhattan_error": float(np.mean(errors)) if errors else None,
        "within_distance_2": float(np.mean(np.asarray(errors) <= 2)) if errors else None,
    }
