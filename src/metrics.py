"""Evaluation metrics: accuracy, precision/recall/F1, ROC-AUC, EER, NLL, Brier, ECE, optional min-tDCF."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

EPS = 1e-12


def equal_error_rate(spoof_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Compute EER given spoof-probability scores in [0, 1].

    Returns (eer, threshold). Labels: 1 = spoof, 0 = bonafide.
    """
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1 - tpr
    # EER = point where FPR == FNR.
    abs_diff = np.abs(fpr - fnr)
    idx = int(np.argmin(abs_diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thr[idx])


def negative_log_likelihood(probs_spoof: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs_spoof, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def brier_score(probs_spoof: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs_spoof - labels) ** 2))


def expected_calibration_error(
    probs_spoof: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE for binary classification using probability of the predicted class."""
    preds = (probs_spoof >= 0.5).astype(int)
    confidence = np.where(preds == 1, probs_spoof, 1 - probs_spoof)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs_spoof)
    for i in range(n_bins):
        in_bin = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if not in_bin.any():
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def classification_metrics(
    spoof_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    preds = (spoof_scores >= threshold).astype(int)
    eer, eer_thr = equal_error_rate(spoof_scores, labels)
    try:
        auc = float(roc_auc_score(labels, spoof_scores))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_spoof": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_spoof": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_spoof": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "decision_threshold": threshold,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "support": {"bonafide": int((labels == 0).sum()), "spoof": int((labels == 1).sum())},
    }


def min_tdcf(
    spoof_scores: np.ndarray,
    labels: np.ndarray,
    cost_miss: float = 1.0,
    cost_fa: float = 10.0,
    p_target: float = 0.05,
) -> float | None:
    """Simplified min t-DCF (no ASV system component). Optional.

    Returns None if computation is degenerate (single class). The full t-DCF
    metric from the ASVspoof toolkit additionally couples with an ASV system
    score; we omit that here so the project has no hard external dependency.
    """
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, spoof_scores, pos_label=1)
    fnr = 1 - tpr
    dcf = cost_miss * p_target * fnr + cost_fa * (1 - p_target) * fpr
    return float(np.min(dcf))
