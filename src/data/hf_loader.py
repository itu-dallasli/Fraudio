"""Hugging Face Datasets loader for ASVspoof 2019 LA.

Streams or caches `Bisher/ASVspoof_2019_LA` (or any compatible repo with the
same column schema: speaker_id / audio_file_name / audio / system_id / key)
through the same waveform pre-processing pipeline used by the file-based
loader, so the rest of the codebase doesn't need to care which source is
active.

Two modes:
  * cached (`streaming: false`) — full ~7.5 GB download to the HF cache; supports
    len(), random crop, weighted sampler, dev EER best-checkpoint tracking, etc.
  * streaming (`streaming: true`) — no download to disk beyond a few hundred MB
    of parquet shards at a time. Returns a torch IterableDataset; balanced
    sampler is unavailable in this mode, so use class-weighted CE instead.

HF split mapping (Bisher/ASVspoof_2019_LA): train→train, validation→dev, test→eval.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .augmentations import WaveformAugment
from .dataset import _fix_length, _peak_normalize, _resample, _trim_silence
from .protocol import LABEL_TO_ID

DEFAULT_HF_REPO = "Bisher/ASVspoof_2019_LA"

# Bisher/ASVspoof_2019_LA uses HF-standard split names; map to our partitions.
PARTITION_TO_HF_SPLIT = {
    "train": "train",
    "dev":   "validation",
    "eval":  "test",
}


def _normalise_label(key) -> int:
    """Bisher/ASVspoof_2019_LA may expose `key` as str ('bonafide'/'spoof') or int."""
    if isinstance(key, str):
        return LABEL_TO_ID[key.lower()]
    if isinstance(key, (int, np.integer)):
        # If it's a HF ClassLabel int we need the names. We don't have access to
        # the feature spec here, but Bisher's schema labels: 0=bonafide, 1=spoof.
        return int(key)
    raise ValueError(f"Unrecognised key value: {key!r}")


def _decode_audio(audio_field, target_sr: int) -> np.ndarray:
    """HF `Audio` feature is decoded to {'array': np.ndarray, 'sampling_rate': int}."""
    if isinstance(audio_field, dict):
        arr = np.asarray(audio_field.get("array"))
        sr = int(audio_field.get("sampling_rate", target_sr))
    elif hasattr(audio_field, "get_all_samples"):  # newer datasets returns AudioDecoder
        samples = audio_field.get_all_samples()
        arr = np.asarray(samples.data).astype(np.float32)
        sr = int(samples.sample_rate)
    else:
        # Bytes fallback — read via soundfile.
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32", always_2d=False)
        arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    arr = arr.astype(np.float32, copy=False)
    if sr != target_sr:
        arr = _resample(arr, sr, target_sr)
    return arr


def _to_sample(
    row: dict,
    target_sr: int,
    target_len: int,
    crop_mode: str,
    rng: np.random.Generator,
    augment: Optional[callable],
    trim: bool,
    normalize: bool,
) -> dict:
    audio = _decode_audio(row["audio"], target_sr)
    if trim:
        audio = _trim_silence(audio)
    audio = _fix_length(audio, target_len, crop_mode, rng)
    if augment is not None:
        audio = augment(audio)
    if normalize:
        audio = _peak_normalize(audio)
    return {
        "waveform": torch.from_numpy(audio),
        "label": torch.tensor(_normalise_label(row["key"]), dtype=torch.long),
        "file_id": str(row.get("audio_file_name", "")),
        "system_id": str(row.get("system_id", "")),
        "speaker_id": str(row.get("speaker_id", "")),
    }


# --------------------------------------------------------------------- #
# cached (map-style) dataset
# --------------------------------------------------------------------- #

class HFASVspoofDataset(Dataset):
    """Map-style wrapper around a downloaded HF dataset split."""

    def __init__(
        self,
        hf_dataset,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        partition: str = "train",
        augment=None,
        trim_silence: bool = False,
        normalize_waveform: bool = True,
        seed: int = 42,
    ):
        self._ds = hf_dataset
        self.sample_rate = sample_rate
        self.target_len = int(round(duration_seconds * sample_rate))
        self.partition = partition
        self.augment = augment if (augment and partition == "train") else None
        self.trim_silence = trim_silence
        self.normalize_waveform = normalize_waveform
        self._rng = np.random.default_rng(seed + (0 if partition == "train" else 1))

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[int(idx)]
        crop_mode = "random" if self.partition == "train" else "center"
        return _to_sample(
            row, self.sample_rate, self.target_len, crop_mode,
            self._rng, self.augment, self.trim_silence, self.normalize_waveform,
        )

    @property
    def labels(self) -> list[int]:
        return [_normalise_label(k) for k in self._ds["key"]]


# --------------------------------------------------------------------- #
# streaming (iterable) dataset
# --------------------------------------------------------------------- #

class HFASVspoofIterable(IterableDataset):
    """Iterable wrapper around a streaming HF dataset (no `len()`)."""

    def __init__(
        self,
        hf_streaming_dataset,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        partition: str = "train",
        augment=None,
        trim_silence: bool = False,
        normalize_waveform: bool = True,
        seed: int = 42,
        max_samples: int | None = None,
    ):
        self._ds = hf_streaming_dataset
        self.sample_rate = sample_rate
        self.target_len = int(round(duration_seconds * sample_rate))
        self.partition = partition
        self.augment = augment if (augment and partition == "train") else None
        self.trim_silence = trim_silence
        self.normalize_waveform = normalize_waveform
        self._seed = seed
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self._seed + worker_id + (0 if self.partition == "train" else 1))
        crop_mode = "random" if self.partition == "train" else "center"
        n = 0
        for row in self._ds:
            yield _to_sample(
                row, self.sample_rate, self.target_len, crop_mode,
                rng, self.augment, self.trim_silence, self.normalize_waveform,
            )
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


# --------------------------------------------------------------------- #
# top-level builders
# --------------------------------------------------------------------- #

def build_hf_dataset(
    cfg_data: dict,
    cfg_aug: dict,
    partition: str,                # 'train' / 'dev' / 'eval'
    seed: int = 42,
):
    """Build either HFASVspoofDataset or HFASVspoofIterable based on cfg.

    Required cfg_data keys:
      hf_repo (default Bisher/ASVspoof_2019_LA)
      streaming (bool, default False)
      sample_rate, duration_seconds, trim_silence, normalize_waveform
      max_train_samples / max_dev_samples / max_eval_samples (optional)
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face `datasets` package is required for source=huggingface. "
            "Install via `pip install datasets`."
        ) from e

    repo = cfg_data.get("hf_repo", DEFAULT_HF_REPO)
    streaming = bool(cfg_data.get("streaming", False))
    hf_split = PARTITION_TO_HF_SPLIT[partition]
    cap = cfg_data.get({
        "train": "max_train_samples",
        "dev":   "max_dev_samples",
        "eval":  "max_eval_samples",
    }[partition])

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

    ds = load_dataset(repo, split=hf_split, streaming=streaming)
    # Force in-decoder resampling so decode hands us 16 kHz audio directly.
    try:
        ds = ds.cast_column("audio", Audio(sampling_rate=cfg_data["sample_rate"]))
    except Exception:
        # Some streaming variants don't expose cast_column; we then resample by hand.
        pass

    common_kwargs = dict(
        sample_rate=cfg_data["sample_rate"],
        duration_seconds=cfg_data["duration_seconds"],
        partition=partition,
        augment=aug,
        trim_silence=cfg_data.get("trim_silence", False),
        normalize_waveform=cfg_data.get("normalize_waveform", True),
        seed=seed,
    )

    if streaming:
        if cap is not None:
            ds = ds.take(int(cap))
        return HFASVspoofIterable(ds, max_samples=cap, **common_kwargs)
    else:
        if cap is not None and cap < len(ds):
            # Class-balanced subsample using HF's filter.
            indices = _balanced_subsample_indices(ds["key"], int(cap), seed)
            ds = ds.select(indices)
        return HFASVspoofDataset(ds, **common_kwargs)


def _balanced_subsample_indices(keys: list, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    keys_arr = np.array([_normalise_label(k) for k in keys])
    bona = np.where(keys_arr == 0)[0]
    spoof = np.where(keys_arr == 1)[0]
    half = n // 2
    bn = min(half, len(bona))
    sn = min(n - bn, len(spoof))
    if bn + sn < n:
        bn = min(len(bona), n - sn)
    sel = np.concatenate(
        [
            rng.choice(bona, size=bn, replace=False),
            rng.choice(spoof, size=sn, replace=False),
        ]
    )
    rng.shuffle(sel)
    return sel.tolist()


def estimate_class_counts(hf_dataset, max_probe: int = 5000) -> tuple[int, int]:
    """For streaming mode where we can't `len()`: probe up to `max_probe` rows
    to estimate class balance. Used to derive class weights for CE loss.
    """
    bona = spoof = 0
    for i, row in enumerate(hf_dataset):
        if _normalise_label(row["key"]) == 0:
            bona += 1
        else:
            spoof += 1
        if i + 1 >= max_probe:
            break
    return bona, spoof
