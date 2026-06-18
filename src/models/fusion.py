"""Late-fusion strategies between WavLM and Wav2Vec2 spoof-probability scores.

All scores are expected as `P(spoof)` ∈ [0, 1]. The fusion API also accepts
calibrated logits for logistic-regression fusion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..metrics import equal_error_rate


# --------------------------- simple strategies --------------------------- #

def fuse_average(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return 0.5 * s1 + 0.5 * s2


def fuse_weighted(s1: np.ndarray, s2: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * s1 + (1.0 - alpha) * s2


def sweep_alpha(
    s1_dev: np.ndarray,
    s2_dev: np.ndarray,
    labels_dev: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid search alpha ∈ [0, 1] on the dev set; return (best_alpha, dev_eer)."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_alpha, best_eer = 0.5, float("inf")
    for a in grid:
        fused = fuse_weighted(s1_dev, s2_dev, float(a))
        eer, _ = equal_error_rate(fused, labels_dev)
        if not np.isnan(eer) and eer < best_eer:
            best_eer = eer
            best_alpha = float(a)
    return best_alpha, best_eer


# --------------------------- logistic-regression fusion --------------------------- #

@dataclass
class LogRegFusion:
    coef: list[float]
    intercept: float

    def predict_proba_spoof(self, feats: np.ndarray) -> np.ndarray:
        z = feats @ np.array(self.coef, dtype=np.float64) + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def fit_logreg_fusion(
    wavlm_calibrated_logits: np.ndarray,
    wav2vec2_calibrated_logits: np.ndarray,
    labels_dev: np.ndarray,
) -> LogRegFusion:
    """
    Train a 2-feature logistic regression on calibrated spoof-logits.
    Features: log(P_spoof / P_bonafide) per model = logit difference (z_spoof - z_bonafide).
    """
    f1 = wavlm_calibrated_logits[:, 1] - wavlm_calibrated_logits[:, 0]
    f2 = wav2vec2_calibrated_logits[:, 1] - wav2vec2_calibrated_logits[:, 0]
    X = np.stack([f1, f2], axis=1)
    y = labels_dev.astype(int)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X, y)
    return LogRegFusion(coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]))


def logreg_features(
    wavlm_calibrated_logits: np.ndarray,
    wav2vec2_calibrated_logits: np.ndarray,
) -> np.ndarray:
    f1 = wavlm_calibrated_logits[:, 1] - wavlm_calibrated_logits[:, 0]
    f2 = wav2vec2_calibrated_logits[:, 1] - wav2vec2_calibrated_logits[:, 0]
    return np.stack([f1, f2], axis=1)
