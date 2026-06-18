"""Waveform Dataset for ASVspoof2019 LA.

Returns 4-second mono float32 waveforms at 16 kHz. Train uses random crop +
augmentation; dev/eval use center crop. Short clips are zero-padded.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import WaveformAugment
from .protocol import ProtocolItem, BONAFIDE, SPOOF, load_partition


def _read_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Read audio as mono float32 resampled to target_sr."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file missing: {path}")
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return data.astype(np.float32, copy=False)


def _resample(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Lightweight resampling via torchaudio (avoids librosa heavyweight import)."""
    import torchaudio.functional as AF
    t = torch.from_numpy(x).unsqueeze(0)
    y = AF.resample(t, orig_freq=src_sr, new_freq=tgt_sr)
    return y.squeeze(0).numpy().astype(np.float32, copy=False)


def _trim_silence(x: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Cheap top-dB based trimming — falls back to librosa only when present."""
    try:
        import librosa
        y, _ = librosa.effects.trim(x, top_db=top_db)
        return y.astype(np.float32, copy=False) if y.size > 0 else x
    except Exception:
        return x


def _fix_length(
    x: np.ndarray,
    target_len: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pad with zeros if shorter, crop (random/center) if longer."""
    if x.size == target_len:
        return x
    if x.size < target_len:
        pad = np.zeros(target_len, dtype=np.float32)
        pad[: x.size] = x
        return pad
    excess = x.size - target_len
    if mode == "random":
        start = int(rng.integers(0, excess + 1))
    else:  # center
        start = excess // 2
    return x[start : start + target_len].astype(np.float32, copy=False)


def _peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        return (x / peak).astype(np.float32, copy=False)
    return x


class ASVspoofDataset(Dataset):
    """Returns dict with `waveform` (float32 tensor) and `label` (long tensor)."""

    def __init__(
        self,
        items: list[ProtocolItem],
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        partition: str = "train",         # 'train' → random crop + aug, else center crop
        augment: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        trim_silence: bool = False,
        normalize_waveform: bool = True,
        seed: int = 42,
    ):
        self.items = items
        self.sample_rate = sample_rate
        self.target_len = int(round(duration_seconds * sample_rate))
        self.partition = partition
        self.augment = augment if (augment and partition == "train") else None
        self.trim_silence = trim_silence
        self.normalize_waveform = normalize_waveform
        self._rng = np.random.default_rng(seed + (0 if partition == "train" else 1))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        x = _read_audio(item.audio_path, self.sample_rate)
        if self.trim_silence:
            x = _trim_silence(x)
        crop_mode = "random" if self.partition == "train" else "center"
        x = _fix_length(x, self.target_len, crop_mode, self._rng)
        if self.augment is not None:
            x = self.augment(x)
        if self.normalize_waveform:
            x = _peak_normalize(x)
        return {
            "waveform": torch.from_numpy(x),
            "label": torch.tensor(item.label, dtype=torch.long),
            "file_id": item.file_id,
            "system_id": item.system_id,
            "speaker_id": item.speaker_id,
        }


def collate_batch(batch: list[dict]) -> dict:
    waveforms = torch.stack([b["waveform"] for b in batch], dim=0)  # [B, T]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    file_ids = [b["file_id"] for b in batch]
    system_ids = [b["system_id"] for b in batch]
    speaker_ids = [b["speaker_id"] for b in batch]
    return {
        "waveform": waveforms,
        "label": labels,
        "file_id": file_ids,
        "system_id": system_ids,
        "speaker_id": speaker_ids,
    }


# ------------------------- helpers ------------------------- #

def subsample(items: list[ProtocolItem], max_samples: int | None, seed: int = 42) -> list[ProtocolItem]:
    """Subsample preserving class balance for quick experiments."""
    if max_samples is None or max_samples >= len(items):
        return items
    rng = np.random.default_rng(seed)
    bonafide = [x for x in items if x.label == BONAFIDE]
    spoof = [x for x in items if x.label == SPOOF]
    half = max_samples // 2
    # Sample at most half from each class, then fill the remainder from the larger pool.
    bona_n = min(half, len(bonafide))
    spoof_n = min(max_samples - bona_n, len(spoof))
    if bona_n + spoof_n < max_samples:
        # Top up bonafide if spoof exhausted.
        bona_n = min(len(bonafide), max_samples - spoof_n)
    bona_idx = rng.choice(len(bonafide), size=bona_n, replace=False)
    spoof_idx = rng.choice(len(spoof), size=spoof_n, replace=False)
    out = [bonafide[i] for i in bona_idx] + [spoof[i] for i in spoof_idx]
    rng.shuffle(out)
    return out


def build_dataset(
    items: list[ProtocolItem],
    cfg_data: dict,
    cfg_aug: dict,
    partition: str,
    seed: int = 42,
) -> ASVspoofDataset:
    aug = None
    if partition == "train" and cfg_aug.get("enabled", False):
        aug = WaveformAugment(
            sample_rate=cfg_data["sample_rate"],
            gaussian_noise_prob=cfg_aug.get("gaussian_noise_prob", 0.5),
            gaussian_noise_std=cfg_aug.get("gaussian_noise_std", 0.005),
            random_gain_prob=cfg_aug.get("random_gain_prob", 0.5),
            gain_db_range=tuple(cfg_aug.get("gain_db_range", (-6.0, 6.0))),
            time_shift_prob=cfg_aug.get("time_shift_prob", 0.5),
            time_shift_max_ratio=cfg_aug.get("time_shift_max_ratio", 0.1),
            reverb_prob=cfg_aug.get("reverb_prob", 0.0),
            rng=np.random.default_rng(seed),
        )
    return ASVspoofDataset(
        items=items,
        sample_rate=cfg_data["sample_rate"],
        duration_seconds=cfg_data["duration_seconds"],
        partition=partition,
        augment=aug,
        trim_silence=cfg_data.get("trim_silence", False),
        normalize_waveform=cfg_data.get("normalize_waveform", True),
        seed=seed,
    )


def make_balanced_sampler(items: list[ProtocolItem]) -> WeightedRandomSampler:
    """Per-class inverse-frequency sampler so spoof/bonafide appear equally often."""
    n_bona = sum(1 for x in items if x.label == BONAFIDE) or 1
    n_spoof = sum(1 for x in items if x.label == SPOOF) or 1
    weights = [
        1.0 / n_bona if x.label == BONAFIDE else 1.0 / n_spoof for x in items
    ]
    return WeightedRandomSampler(weights, num_samples=len(items), replacement=True)


def load_items_for_partition(
    dataset_root: str,
    partition: str,
    max_samples: int | None,
    seed: int,
    require_audio: bool = False,
) -> list[ProtocolItem]:
    items = load_partition(dataset_root, partition, require_audio=require_audio)
    items = subsample(items, max_samples, seed=seed)
    return items
