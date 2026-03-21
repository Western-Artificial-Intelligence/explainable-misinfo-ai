"""
Basic metrics for classification:
- Accuracy
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix
- Expected Calibration Error (ECE)
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score


def compute_basic_metrics(preds: np.ndarray, labels: np.ndarray):
    """
    Compute basic classification metrics.

    Args:
        preds: np.array of shape [N] with predicted labels (e.g., 0/1/2)
        labels: np.array of shape [N] with true labels

    Returns:
        dict with keys:
            - accuracy
            - macro_f1
            - per_class: dict with precision/recall/f1 lists
            - confusion_matrix: 2D list
    """
    mask = labels != -100  # only evaluate valid labels
    if mask.sum() == 0:
        # nothing to compute
        return {
            "accuracy": None,
            "macro_f1": None,
            "per_class": None,
            "confusion_matrix": None,
        }

    p, r, f, _ = precision_recall_fscore_support(
        labels[mask], preds[mask], labels=[0, 1], zero_division=0
    )
    macro_f1 = np.mean(f)
    acc = accuracy_score(labels[mask], preds[mask])
    cm = confusion_matrix(labels[mask], preds[mask], labels=[0, 1])
    per_class = {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}

    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm.tolist()}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Compute simple Expected Calibration Error (ECE).

    Args:
        probs: np.array of shape [N, num_classes], predicted probabilities
        labels: np.array of shape [N], true labels (0..K-1 or -100)
        n_bins: number of bins to use

    Returns:
        float ECE value or None if no valid labels
    """
    mask = labels != -100
    if mask.sum() == 0:
        return None

    probs = probs[mask]
    labels = labels[mask]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    N = len(confidences)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)