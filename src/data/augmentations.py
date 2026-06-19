"""Lightweight waveform-domain augmentations for spoof detection training.

All augmentations operate on 1-D float numpy arrays sampled at the model's
working sample rate (default 16 kHz). They are applied only on the training
partition; dev/eval pipelines skip augmentation entirely.

The phone-band and mu-law codec augmentations are added specifically to
close the gap between ASVspoof's clean studio recordings and real-world
browser/microphone captures, which is the main OOD failure mode.
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
        bandpass_prob: float = 0.0,
        bandpass_range_hz: tuple[float, float, float, float] = (200.0, 400.0, 3000.0, 3800.0),
        mulaw_prob: float = 0.0,
        mulaw_bits_choices: tuple[int, ...] = (8, 8, 10, 12),
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
        self.p_bandpass = bandpass_prob
        self.bandpass_range = bandpass_range_hz
        self.p_mulaw = mulaw_prob
        self.mulaw_bits = mulaw_bits_choices
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
        if self.p_bandpass > 0.0 and self.rng.random() < self.p_bandpass:
            x = _phone_band(x, self.sample_rate, self.bandpass_range, self.rng)
        if self.p_mulaw > 0.0 and self.rng.random() < self.p_mulaw:
            x = _mulaw_codec(x, self.mulaw_bits, self.rng)
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


def _phone_band(
    x: np.ndarray,
    sr: int,
    band: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Telephone-band (~300–3400 Hz) bandpass via cascaded biquads.

    Mimics what WebRTC, codec-relayed microphones, and phone capture do to
    bonafide speech — narrows the spectrum and removes the high-frequency
    information SSL encoders sometimes latch onto when separating clean
    studio speech from TTS.
    """
    try:
        import torch
        import torchaudio.functional as AF
    except Exception:
        return x
    low_min, low_max, high_min, high_max = band
    low_cut = float(rng.uniform(low_min, low_max))
    high_cut = float(rng.uniform(high_min, high_max))
    t = torch.from_numpy(x).unsqueeze(0)
    t = AF.highpass_biquad(t, sr, cutoff_freq=low_cut)
    t = AF.lowpass_biquad(t, sr, cutoff_freq=high_cut)
    return t.squeeze(0).numpy().astype(np.float32, copy=False)


def _mulaw_codec(
    x: np.ndarray,
    bit_choices: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Lossy μ-law re-encoding (G.711-style). Inflicts the kind of
    quantisation artefacts a phone/VoIP path leaves on bonafide speech,
    so the model stops treating them as "spoof tells"."""
    try:
        import torch
        import torchaudio.functional as AF
    except Exception:
        return x
    bits = int(rng.choice(np.asarray(bit_choices)))
    qc = 2 ** bits
    t = torch.from_numpy(x).clamp(-1.0, 1.0)
    enc = AF.mu_law_encoding(t, quantization_channels=qc)
    dec = AF.mu_law_decoding(enc, quantization_channels=qc)
    return dec.numpy().astype(np.float32, copy=False)
