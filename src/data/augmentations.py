"""Lightweight waveform-domain augmentations for spoof detection training.

All augmentations operate on 1-D float numpy arrays sampled at the model's
working sample rate (default 16 kHz). They are applied only on the training
partition; dev/eval pipelines skip augmentation entirely.
"""
from __future__ import annotations

import numpy as np


class WaveformAugment:
    def __init__(
        self,
        sample_rate: int = 16000,
        gaussian_noise_prob: float = 0.5,
        gaussian_noise_std: float = 0.005,
        random_gain_prob: float = 0.5,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        time_shift_prob: float = 0.5,
        time_shift_max_ratio: float = 0.1,
        reverb_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.sample_rate = sample_rate
        self.p_noise = gaussian_noise_prob
        self.noise_std = gaussian_noise_std
        self.p_gain = random_gain_prob
        self.gain_db = gain_db_range
        self.p_shift = time_shift_prob
        self.shift_max = time_shift_max_ratio
        self.p_reverb = reverb_prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.rng.random() < self.p_noise:
            x = x + self.rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        if self.rng.random() < self.p_gain:
            db = float(self.rng.uniform(*self.gain_db))
            x = x * (10.0 ** (db / 20.0))
        if self.rng.random() < self.p_shift and x.size > 1:
            max_shift = int(x.size * self.shift_max)
            if max_shift > 0:
                shift = int(self.rng.integers(-max_shift, max_shift + 1))
                x = np.roll(x, shift)
                if shift > 0:
                    x[:shift] = 0.0
                elif shift < 0:
                    x[shift:] = 0.0
        if self.p_reverb > 0.0 and self.rng.random() < self.p_reverb:
            x = _cheap_reverb(x, rng=self.rng)
        # Clip to avoid any overflow after gain.
        np.clip(x, -1.0, 1.0, out=x)
        return x


def _cheap_reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic single-tap reverb with exponential decay — no SciPy dependency."""
    decay = float(rng.uniform(0.2, 0.6))
    delay = int(rng.integers(800, 2400))  # 50–150 ms @ 16 kHz
    if delay >= x.size:
        return x
    y = x.copy()
    y[delay:] += decay * x[:-delay]
    return np.clip(y, -1.0, 1.0)
