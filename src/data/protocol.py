"""ASVspoof 2019 Logical Access protocol parser.

Expected directory layout (resmî ASVspoof2019 LA paketi):

    LA/
      ASVspoof2019_LA_train/flac/<FILE_ID>.flac
      ASVspoof2019_LA_dev/flac/<FILE_ID>.flac
      ASVspoof2019_LA_eval/flac/<FILE_ID>.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Protocol line format:
    SPEAKER_ID FILE_ID - SYSTEM_ID KEY
    (KEY ∈ {bonafide, spoof})
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BONAFIDE = 0
SPOOF = 1
LABEL_TO_ID = {"bonafide": BONAFIDE, "spoof": SPOOF}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

_PARTITION_FILES = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}


@dataclass
class ProtocolItem:
    speaker_id: str
    file_id: str
    system_id: str
    label: int      # 0=bonafide, 1=spoof
    audio_path: str


def _resolve_audio_path(audio_dir: Path, file_id: str) -> str:
    # ASVspoof19 ships .flac; some mirrors re-encode to .wav.
    for ext in (".flac", ".wav"):
        p = audio_dir / "flac" / f"{file_id}{ext}"
        if p.exists():
            return str(p)
        p_flat = audio_dir / f"{file_id}{ext}"
        if p_flat.exists():
            return str(p_flat)
    # Return canonical .flac path even if missing — loader will surface a clear error.
    return str(audio_dir / "flac" / f"{file_id}.flac")


def load_partition(
    dataset_root: str | os.PathLike,
    partition: str,
    require_audio: bool = False,
) -> list[ProtocolItem]:
    """Read a partition protocol file and return one ProtocolItem per line."""
    if partition not in _PARTITION_FILES:
        raise ValueError(f"Unknown partition '{partition}'. Use one of {list(_PARTITION_FILES)}.")

    root = Path(dataset_root)
    audio_subdir, protocol_name = _PARTITION_FILES[partition]
    audio_dir = root / audio_subdir
    protocol_path = root / "ASVspoof2019_LA_cm_protocols" / protocol_name
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}. "
            "Make sure dataset_root points to the LA folder containing "
            "ASVspoof2019_LA_cm_protocols/."
        )

    items: list[ProtocolItem] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                # Some mirrors omit the dash; pad defensively.
                continue
            speaker_id, file_id, _dash, system_id, key = parts[0], parts[1], parts[2], parts[3], parts[4]
            if key not in LABEL_TO_ID:
                continue
            audio_path = _resolve_audio_path(audio_dir, file_id)
            if require_audio and not os.path.exists(audio_path):
                continue
            items.append(
                ProtocolItem(
                    speaker_id=speaker_id,
                    file_id=file_id,
                    system_id=system_id,
                    label=LABEL_TO_ID[key],
                    audio_path=audio_path,
                )
            )
    return items


def summarise(items: Iterable[ProtocolItem]) -> dict:
    items = list(items)
    n_bonafide = sum(1 for x in items if x.label == BONAFIDE)
    n_spoof = sum(1 for x in items if x.label == SPOOF)
    speakers = sorted({x.speaker_id for x in items})
    systems = sorted({x.system_id for x in items})
    return {
        "total": len(items),
        "bonafide": n_bonafide,
        "spoof": n_spoof,
        "num_speakers": len(speakers),
        "systems": systems,
    }
