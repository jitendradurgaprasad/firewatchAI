"""Classification metrics with explicit class ordering."""

from .model import LABEL_ORDER


def classification_metrics(actual: list[str], predicted: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must have the same non-zero length.")
    if any(value not in LABEL_ORDER for value in actual + predicted):
        raise ValueError("Metrics accept only the two supported training labels.")
    matrix = {truth: {guess: 0 for guess in LABEL_ORDER} for truth in LABEL_ORDER}
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    per_class = {}
    for label in LABEL_ORDER:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABEL_ORDER if other != label)
        fn = sum(matrix[label][other] for other in LABEL_ORDER if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"accuracy": sum(matrix[label][label] for label in LABEL_ORDER) / len(actual),
            "confusion_matrix": matrix, "per_class": per_class}
