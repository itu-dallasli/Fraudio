"""Smoke tests using synthetic audio + a tiny mock SSL encoder.

These do NOT require ASVspoof data or transformers downloads. They exercise:
  - protocol parsing (with a synthetic mini-protocol),
  - dataset/augmentation/collate pipeline,
  - pooling math and head shapes (via a mock encoder substituting for HF model),
  - metrics, temperature scaling, fusion strategies.

Run:
    python -m tests.test_smoke
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------- #
# 1. Synthetic dataset on disk
# --------------------------------------------------------------------- #

def _make_synthetic_la(root: Path, n_per_partition: int = 8, sr: int = 16000) -> Path:
    """Create a fake ASVspoof2019 LA layout under `root`."""
    la = root / "LA"
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    parts = {
        "train": (la / "ASVspoof2019_LA_train" / "flac", "ASVspoof2019.LA.cm.train.trn.txt"),
        "dev":   (la / "ASVspoof2019_LA_dev"   / "flac", "ASVspoof2019.LA.cm.dev.trl.txt"),
        "eval":  (la / "ASVspoof2019_LA_eval"  / "flac", "ASVspoof2019.LA.cm.eval.trl.txt"),
    }
    for part, (audio_dir, proto_name) in parts.items():
        audio_dir.mkdir(parents=True, exist_ok=True)
        with open(proto_dir / proto_name, "w", encoding="utf-8") as f:
            for i in range(n_per_partition):
                label = "bonafide" if i % 2 == 0 else "spoof"
                file_id = f"{part}_{i:04d}"
                # Random short (~1s) waveform.
                wav = rng.standard_normal(int(sr * 1.0)).astype(np.float32) * 0.05
                if label == "spoof":
                    # Add a deterministic high-freq buzz for "spoof" class so a trainable
                    # classifier can in principle separate them.
                    t = np.arange(wav.size) / sr
                    wav = wav + 0.2 * np.sin(2 * np.pi * 3500 * t).astype(np.float32)
                sf.write(audio_dir / f"{file_id}.flac", wav, sr, format="FLAC")
                speaker_id = f"SPK_{i % 4:02d}"
                system_id = "-" if label == "bonafide" else "A07"
                f.write(f"{speaker_id} {file_id} - {system_id} {label}\n")
    return la


# --------------------------------------------------------------------- #
# 2. Mock encoder (mimics WavLM / Wav2Vec2 interface used by SSLForSpoofDetection)
# --------------------------------------------------------------------- #

class _MockEncoderConfig:
    hidden_size = 32
    mask_time_prob = 0.0
    mask_feature_prob = 0.0


class _MockEncoderInner(nn.Module):
    """Mimics encoder.encoder.layers nesting expected by apply_freezing."""

    def __init__(self, n_layers=4, hidden=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )


class MockSSLEncoder(nn.Module):
    """Replaces HF AutoModel during smoke tests."""

    def __init__(self):
        super().__init__()
        self.config = _MockEncoderConfig()
        self.feature_extractor = nn.Conv1d(1, _MockEncoderConfig.hidden_size, kernel_size=320, stride=160)
        self.feature_projection = nn.Linear(_MockEncoderConfig.hidden_size, _MockEncoderConfig.hidden_size)
        self.encoder = _MockEncoderInner(n_layers=4, hidden=_MockEncoderConfig.hidden_size)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # Match conv1d stride.
        return ((input_lengths - 320) // 160 + 1).clamp(min=1)

    def forward(self, input_values: torch.Tensor, attention_mask=None, return_dict=True):
        # [B, T] → [B, 1, T] → conv1d → [B, H, T'] → transpose → [B, T', H]
        x = input_values.unsqueeze(1)
        x = self.feature_extractor(x).transpose(1, 2)
        x = self.feature_projection(x)
        for layer in self.encoder.layers:
            x = layer(x) + x  # residual
        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = x
        return out


# --------------------------------------------------------------------- #
# 3. Tests
# --------------------------------------------------------------------- #

def test_protocol_and_dataset():
    print("[smoke] protocol + dataset…")
    with tempfile.TemporaryDirectory() as tmp:
        la = _make_synthetic_la(Path(tmp), n_per_partition=8)

        from src.data.dataset import build_dataset, collate_batch, load_items_for_partition

        items = load_items_for_partition(str(la), "train", max_samples=None, seed=0)
        assert len(items) == 8, f"expected 8 train items got {len(items)}"
        cfg_data = {"sample_rate": 16000, "duration_seconds": 1.0,
                    "trim_silence": False, "normalize_waveform": True}
        cfg_aug = {"enabled": True, "gaussian_noise_prob": 1.0, "gaussian_noise_std": 0.001,
                   "random_gain_prob": 0.5, "gain_db_range": (-3.0, 3.0),
                   "time_shift_prob": 0.5, "time_shift_max_ratio": 0.05, "reverb_prob": 0.0}
        ds = build_dataset(items, cfg_data, cfg_aug, partition="train", seed=0)
        batch = collate_batch([ds[0], ds[1], ds[2]])
        assert batch["waveform"].shape == (3, 16000)
        assert batch["label"].shape == (3,)
        print("  ok — batch shape", tuple(batch["waveform"].shape))


def test_pooling_and_head():
    print("[smoke] masked pooling + head shapes…")
    from src.models.ssl_classifier import SSLForSpoofDetection
    # Patch transformers loader to use our mock.
    import src.models.ssl_classifier as m
    original_from_pretrained = m.AutoModel.from_pretrained
    original_config = m.AutoConfig.from_pretrained
    try:
        m.AutoConfig.from_pretrained = lambda *a, **kw: _MockEncoderConfig()
        m.AutoModel.from_pretrained = lambda *a, **kw: MockSSLEncoder()

        model = SSLForSpoofDetection(
            encoder_name="mock",
            hidden_size=_MockEncoderConfig.hidden_size,
            proj_dim=16,
            num_classes=2,
            dropout=0.0,
            freeze_encoder=False,
            unfreeze_last_n_layers=2,
        )
        wave = torch.randn(2, 16000)
        out = model(wave)
        assert out["logits"].shape == (2, 2), out["logits"].shape
        # Backprop sanity check.
        loss = out["logits"].sum()
        loss.backward()
        # Some params must have gradients (head).
        n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert n_grad > 0
        print("  ok — logits", tuple(out["logits"].shape), "params w/ grad:", n_grad)
    finally:
        m.AutoModel.from_pretrained = original_from_pretrained
        m.AutoConfig.from_pretrained = original_config


def test_metrics_calibration_fusion():
    print("[smoke] metrics / calibration / fusion…")
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(int)
    # Logits that correlate with labels.
    z = rng.standard_normal((n, 2)) * 0.5
    z[labels == 1, 1] += 1.5
    z[labels == 0, 0] += 1.5

    from src.metrics import (
        brier_score, classification_metrics, equal_error_rate,
        expected_calibration_error, min_tdcf, negative_log_likelihood,
    )
    from src.models.calibration import fit_temperature, softmax_spoof_prob
    from src.models.fusion import (
        fit_logreg_fusion, fuse_average, fuse_weighted, sweep_alpha,
    )

    p = softmax_spoof_prob(z)
    metrics = classification_metrics(p, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0
    eer, _ = equal_error_rate(p, labels)
    assert not np.isnan(eer)
    tdcf = min_tdcf(p, labels)
    assert tdcf is not None
    nll = negative_log_likelihood(p, labels)
    brier = brier_score(p, labels)
    ece = expected_calibration_error(p, labels)
    assert nll > 0 and brier >= 0 and ece >= 0

    # Calibration: temperature must be finite, positive.
    scaler = fit_temperature(z, labels)
    assert scaler.temperature > 0.0
    z_cal = scaler.calibrate_logits_np(z)
    assert z_cal.shape == z.shape

    # Fusion strategies.
    s1 = p
    s2 = softmax_spoof_prob(z + 0.3 * rng.standard_normal(z.shape))
    avg = fuse_average(s1, s2)
    w = fuse_weighted(s1, s2, 0.3)
    alpha, dev_eer = sweep_alpha(s1, s2, labels)
    assert 0.0 <= alpha <= 1.0
    lr = fit_logreg_fusion(z, z + 0.1, labels)
    assert len(lr.coef) == 2
    print(f"  ok — accuracy={metrics['accuracy']:.3f} EER={eer:.3f} T={scaler.temperature:.3f} alpha={alpha:.2f}")


def test_app_pipeline_with_mocks():
    """Build a SpoofPipeline using mocked encoders and run analyse() on synthetic audio."""
    print("[smoke] app.SpoofPipeline (mocked)…")
    import src.models.ssl_classifier as m
    from src.models.ssl_classifier import build_model_from_cfg
    original_from_pretrained = m.AutoModel.from_pretrained
    original_config = m.AutoConfig.from_pretrained
    m.AutoConfig.from_pretrained = lambda *a, **kw: _MockEncoderConfig()
    m.AutoModel.from_pretrained = lambda *a, **kw: MockSSLEncoder()
    try:
        cfg = {
            "model": {
                "encoder_name": "mock",
                "hidden_size": _MockEncoderConfig.hidden_size,
                "proj_dim": 16,
                "num_classes": 2,
                "dropout": 0.0,
                "freeze_encoder": False,
                "unfreeze_last_n_layers": 2,
            }
        }
        model_a = build_model_from_cfg(cfg["model"])
        model_b = build_model_from_cfg(cfg["model"])
        with tempfile.TemporaryDirectory() as tmp:
            pa = Path(tmp) / "a.pt"
            pb = Path(tmp) / "b.pt"
            torch.save({"model_state": model_a.state_dict(), "cfg": cfg}, pa)
            torch.save({"model_state": model_b.state_dict(), "cfg": cfg}, pb)

            from app import SpoofPipeline, FusionBundle
            pipe = SpoofPipeline(
                str(pa), str(pb),
                fusion=FusionBundle(method="weighted", alpha=0.6, wavlm_T=1.2, wav2vec2_T=0.9),
            )
            # 6 s of audio at 16 kHz → triggers sliding-window branch.
            audio = np.random.randn(int(16000 * 6.0)).astype(np.float32) * 0.05
            out = pipe.analyse(audio, sr=16000)
            assert out["decision"] in {"SPOOF", "BONAFIDE", "UNCERTAIN"}
            assert out["windows"] is not None and len(out["windows"]) >= 1
            # Also test the short-audio branch.
            short = audio[:16000]
            out_short = pipe.analyse(short, sr=16000)
            assert out_short["windows"] is None
            print(f"  ok — long decision={out['decision']} short decision={out_short['decision']}")
    finally:
        m.AutoModel.from_pretrained = original_from_pretrained
        m.AutoConfig.from_pretrained = original_config


def main():
    test_protocol_and_dataset()
    test_pooling_and_head()
    test_metrics_calibration_fusion()
    test_app_pipeline_with_mocks()
    print("\n[smoke] ALL TESTS PASSED")


if __name__ == "__main__":
    main()
