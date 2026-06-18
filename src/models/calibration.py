"""Per-model temperature scaling for confidence calibration.

Fit a single scalar T on the development set such that softmax(logits / T) is
better calibrated than the raw softmax. The evaluation set must NOT be used
for fitting T.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # Parametrise as log(T) so T stays positive.
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(init_T))))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).detach().cpu().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temperature)

    @torch.no_grad()
    def calibrate_logits_np(self, logits: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(logits.astype(np.float32))
        return self.forward(t).cpu().numpy()


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.05,
) -> TemperatureScaler:
    """Fit T by minimising NLL via L-BFGS on the dev set."""
    scaler = TemperatureScaler()
    logits_t = torch.from_numpy(logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.int64))

    optimiser = torch.optim.LBFGS(
        scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return scaler


def softmax_spoof_prob(logits: np.ndarray) -> np.ndarray:
    """P(spoof) under softmax over [bonafide, spoof] logits."""
    t = torch.from_numpy(logits.astype(np.float32))
    return F.softmax(t, dim=-1)[:, 1].cpu().numpy()
