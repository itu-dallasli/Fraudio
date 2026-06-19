"""Gradio demo for the two-model spoof detector with fusion + uncertainty.

Run:
    python app.py \
        --wavlm-checkpoint   checkpoints/wavlm/best.pt \
        --wav2vec2-checkpoint checkpoints/wav2vec2/best.pt \
        --fusion-bundle      outputs/evaluation/fusion/fusion_bundle.json

If --fusion-bundle is omitted, a default 0.5/0.5 average is used.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from src.models.calibration import TemperatureScaler
from src.models.ssl_classifier import build_model_from_cfg


# ----------------------------- example clip provisioning ----------------------------- #

def ensure_example_clips(
    out_dir: str = "examples",
    hf_repo: str = "Bisher/ASVspoof_2019_LA",
    hf_split: str = "test",
    max_probe: int = 200,
) -> dict[str, Optional[str]]:
    """Make sure one bonafide and one spoof example exist on disk.

    On first run, streams the HF dataset and writes the first matching
    sample of each class to `out_dir/bonafide.flac` and `out_dir/spoof.flac`.
    Subsequent runs are no-ops. If HF is unreachable or the package is
    missing, returns None for any clip we couldn't produce — Gradio just
    skips that example.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "bonafide": os.path.join(out_dir, "bonafide.flac"),
        "spoof":    os.path.join(out_dir, "spoof.flac"),
    }
    missing = {k: p for k, p in paths.items() if not os.path.exists(p)}
    if not missing:
        return paths

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print("[examples] `datasets` not available; skipping auto-fetch.")
        return {k: (p if os.path.exists(p) else None) for k, p in paths.items()}

    try:
        ds = load_dataset(hf_repo, split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"[examples] HF stream failed ({e}); skipping auto-fetch.")
        return {k: (p if os.path.exists(p) else None) for k, p in paths.items()}

    found: dict[str, dict] = {}
    target_int = {0: "bonafide", 1: "spoof"}
    target_str = {"bonafide": "bonafide", "spoof": "spoof"}
    for i, row in enumerate(ds):
        if len(found) == len(missing):
            break
        if i >= max_probe:
            break
        key = row.get("key")
        if isinstance(key, str):
            label = target_str.get(key.lower())
        elif isinstance(key, int):
            label = target_int.get(int(key))
        else:
            label = None
        if label and label in missing and label not in found:
            found[label] = row

    for label, row in found.items():
        aud = row["audio"]
        if isinstance(aud, dict):
            arr, sr = aud["array"], int(aud["sampling_rate"])
        else:
            s = aud.get_all_samples()
            arr, sr = s.data, int(s.sample_rate)
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        sf.write(missing[label], arr, sr, format="FLAC")
        print(f"[examples] wrote {missing[label]} ({arr.shape[0]/sr:.2f}s, sr={sr})")

    return {k: (p if os.path.exists(p) else None) for k, p in paths.items()}


# ----------------------------- inference engine ----------------------------- #

@dataclass
class FusionBundle:
    method: str = "average"
    alpha: float = 0.5
    logreg_coef: list[float] | None = None
    logreg_intercept: float = 0.0
    wavlm_T: float = 1.0
    wav2vec2_T: float = 1.0

    @staticmethod
    def from_json(path: str) -> "FusionBundle":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FusionBundle(
            method=d.get("best_method", "average"),
            alpha=float(d.get("alpha", 0.5)),
            logreg_coef=d.get("logreg_coef"),
            logreg_intercept=float(d.get("logreg_intercept", 0.0)),
            wavlm_T=float(d.get("wavlm_temperature", 1.0)),
            wav2vec2_T=float(d.get("wav2vec2_temperature", 1.0)),
        )


class SpoofPipeline:
    SAMPLE_RATE = 16000
    WINDOW_SECONDS = 4.0
    STRIDE_SECONDS = 2.0

    def __init__(
        self,
        wavlm_ckpt: str,
        wav2vec2_ckpt: str,
        fusion: FusionBundle | None = None,
        decision_threshold: float = 0.5,
        uncertainty_margin: float = 0.10,
        min_confidence: float = 0.55,
        disagreement_margin: float = 0.30,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.wavlm, _ = self._load(wavlm_ckpt)
        self.w2v, _ = self._load(wav2vec2_ckpt)
        self.fusion = fusion or FusionBundle()
        self.decision_threshold = decision_threshold
        self.uncertainty_margin = uncertainty_margin
        self.min_confidence = min_confidence
        self.disagreement_margin = disagreement_margin

    def _load(self, ckpt_path: str):
        obj = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = obj["cfg"]
        model = build_model_from_cfg(cfg["model"]).to(self.device).eval()
        model.load_state_dict(obj["model_state"])
        return model, cfg

    # -------------------------- audio handling -------------------------- #

    def _preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32, copy=False)
        if sr != self.SAMPLE_RATE:
            import torchaudio.functional as AF
            t = torch.from_numpy(audio).unsqueeze(0)
            audio = AF.resample(t, sr, self.SAMPLE_RATE).squeeze(0).numpy().astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = audio / peak
        return audio

    def _fixed_window(self, audio: np.ndarray) -> np.ndarray:
        target = int(self.WINDOW_SECONDS * self.SAMPLE_RATE)
        if audio.size >= target:
            start = (audio.size - target) // 2
            return audio[start : start + target]
        out = np.zeros(target, dtype=np.float32)
        out[: audio.size] = audio
        return out

    # -------------------------- inference -------------------------- #

    @torch.no_grad()
    def _scores_for_chunk(self, chunk: np.ndarray) -> dict:
        x = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
        w_out = self.wavlm(x)["logits"].float().cpu().numpy()
        v_out = self.w2v(x)["logits"].float().cpu().numpy()
        w_cal = w_out / max(self.fusion.wavlm_T, 1e-6)
        v_cal = v_out / max(self.fusion.wav2vec2_T, 1e-6)
        w_p = float(F.softmax(torch.from_numpy(w_cal), dim=-1)[0, 1])
        v_p = float(F.softmax(torch.from_numpy(v_cal), dim=-1)[0, 1])
        return {
            "wavlm_logits": w_cal[0].tolist(),
            "wav2vec2_logits": v_cal[0].tolist(),
            "wavlm_spoof_prob": w_p,
            "wav2vec2_spoof_prob": v_p,
        }

    def _fuse(self, w_p: float, v_p: float, w_logits: list[float], v_logits: list[float]) -> float:
        if self.fusion.method == "average":
            return 0.5 * w_p + 0.5 * v_p
        if self.fusion.method == "weighted":
            a = self.fusion.alpha
            return a * w_p + (1 - a) * v_p
        if self.fusion.method == "logreg" and self.fusion.logreg_coef:
            f1 = w_logits[1] - w_logits[0]
            f2 = v_logits[1] - v_logits[0]
            z = self.fusion.logreg_coef[0] * f1 + self.fusion.logreg_coef[1] * f2 + self.fusion.logreg_intercept
            return float(1.0 / (1.0 + np.exp(-z)))
        return 0.5 * w_p + 0.5 * v_p

    def _decide(self, fused_score: float, w_p: float, v_p: float) -> tuple[str, float, str]:
        thr = self.decision_threshold
        # Confidence is distance from the decision boundary, mapped to [0, 1].
        confidence = float(min(1.0, abs(fused_score - thr) * 2.0))
        w_pred = w_p >= thr
        v_pred = v_p >= thr
        disagree = w_pred != v_pred
        score_diff = abs(w_p - v_p)
        if disagree and score_diff >= self.disagreement_margin:
            return "UNCERTAIN", confidence, "models disagree with margin"
        if abs(fused_score - thr) < self.uncertainty_margin:
            return "UNCERTAIN", confidence, "fusion score near decision boundary"
        if confidence < self.min_confidence:
            return "UNCERTAIN", confidence, "confidence below minimum threshold"
        label = "SPOOF" if fused_score >= thr else "BONAFIDE"
        return label, confidence, "high-confidence agreement"

    # -------------------------- public API -------------------------- #

    def analyse(
        self,
        audio: np.ndarray,
        sr: int,
        decision_threshold: float | None = None,
        uncertainty_margin: float | None = None,
    ) -> dict:
        # Allow per-call overrides so the demo can dial thresholds live without
        # rebuilding the pipeline.
        if decision_threshold is not None:
            saved_thr, self.decision_threshold = self.decision_threshold, float(decision_threshold)
        else:
            saved_thr = None
        if uncertainty_margin is not None:
            saved_unc, self.uncertainty_margin = self.uncertainty_margin, float(uncertainty_margin)
        else:
            saved_unc = None
        try:
            return self._analyse_impl(audio, sr)
        finally:
            if saved_thr is not None:
                self.decision_threshold = saved_thr
            if saved_unc is not None:
                self.uncertainty_margin = saved_unc

    def _analyse_impl(self, audio: np.ndarray, sr: int) -> dict:
        audio = self._preprocess(audio, sr)
        duration_s = audio.size / self.SAMPLE_RATE

        if duration_s < self.WINDOW_SECONDS:
            chunk = self._fixed_window(audio)
            scores = self._scores_for_chunk(chunk)
            fused = self._fuse(
                scores["wavlm_spoof_prob"],
                scores["wav2vec2_spoof_prob"],
                scores["wavlm_logits"],
                scores["wav2vec2_logits"],
            )
            decision, confidence, reason = self._decide(
                fused, scores["wavlm_spoof_prob"], scores["wav2vec2_spoof_prob"]
            )
            return {
                "duration_s": duration_s,
                "wavlm_spoof_prob": scores["wavlm_spoof_prob"],
                "wav2vec2_spoof_prob": scores["wav2vec2_spoof_prob"],
                "fusion_score": fused,
                "decision": decision,
                "confidence": confidence,
                "reason": reason,
                "fusion_method": self.fusion.method,
                "windows": None,
            }

        # Sliding window analysis.
        win = int(self.WINDOW_SECONDS * self.SAMPLE_RATE)
        stride = int(self.STRIDE_SECONDS * self.SAMPLE_RATE)
        starts = list(range(0, max(1, audio.size - win + 1), stride))
        if not starts:
            starts = [0]
        per_window = []
        w_ps, v_ps, fused_scores = [], [], []
        for s in starts:
            chunk = audio[s : s + win]
            if chunk.size < win:
                pad = np.zeros(win, dtype=np.float32)
                pad[: chunk.size] = chunk
                chunk = pad
            sc = self._scores_for_chunk(chunk)
            fused = self._fuse(
                sc["wavlm_spoof_prob"], sc["wav2vec2_spoof_prob"],
                sc["wavlm_logits"], sc["wav2vec2_logits"],
            )
            per_window.append({
                "t_start": s / self.SAMPLE_RATE,
                "t_end": (s + win) / self.SAMPLE_RATE,
                "wavlm_spoof_prob": sc["wavlm_spoof_prob"],
                "wav2vec2_spoof_prob": sc["wav2vec2_spoof_prob"],
                "fusion_score": fused,
            })
            w_ps.append(sc["wavlm_spoof_prob"])
            v_ps.append(sc["wav2vec2_spoof_prob"])
            fused_scores.append(fused)

        # Aggregate by mean over windows.
        w_mean = float(np.mean(w_ps))
        v_mean = float(np.mean(v_ps))
        fused_mean = float(np.mean(fused_scores))
        decision, confidence, reason = self._decide(fused_mean, w_mean, v_mean)
        return {
            "duration_s": duration_s,
            "wavlm_spoof_prob": w_mean,
            "wav2vec2_spoof_prob": v_mean,
            "fusion_score": fused_mean,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "fusion_method": self.fusion.method,
            "windows": per_window,
        }


# ----------------------------- Gradio UI ----------------------------- #

def build_interface(pipeline: SpoofPipeline, example_paths: dict[str, Optional[str]] | None = None):
    import gradio as gr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DISCLAIMER = (
        "⚠️ Bu sistem akademik bir prototiptir; güvenlik açısından kesin kanıt üretmez."
    )

    def _plot_waveform(audio: np.ndarray, sr: int):
        t = np.arange(audio.size) / sr
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.plot(t, audio, linewidth=0.6)
        ax.set_xlabel("seconds")
        ax.set_ylabel("amplitude")
        ax.set_title("Waveform")
        fig.tight_layout()
        return fig

    def _plot_windows(windows: list[dict] | None, thr: float):
        if not windows:
            return None
        fig, ax = plt.subplots(figsize=(7, 2.4))
        t = [(w["t_start"] + w["t_end"]) / 2 for w in windows]
        ax.plot(t, [w["fusion_score"] for w in windows], "o-", label="fusion")
        ax.plot(t, [w["wavlm_spoof_prob"] for w in windows], "s--", alpha=0.6, label="WavLM")
        ax.plot(t, [w["wav2vec2_spoof_prob"] for w in windows], "^--", alpha=0.6, label="Wav2Vec2")
        ax.axhline(thr, color="k", linestyle=":", alpha=0.4, label=f"threshold={thr:.2f}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("spoof probability")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Per-window spoof probability")
        fig.tight_layout()
        return fig

    def infer(audio_tuple, decision_threshold, uncertainty_margin):
        if audio_tuple is None:
            return None, "No audio.", None
        sr, audio = audio_tuple
        if audio is None or len(audio) == 0:
            return None, "No audio.", None
        # Normalise int16 inputs (gradio sometimes returns ints).
        if audio.dtype.kind in {"i", "u"}:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        result = pipeline.analyse(
            audio, sr,
            decision_threshold=decision_threshold,
            uncertainty_margin=uncertainty_margin,
        )

        thr = float(decision_threshold)
        agree = ((result["wavlm_spoof_prob"] >= thr) == (result["wav2vec2_spoof_prob"] >= thr))
        agreement = "agree ✅" if agree else "disagree ⚠️"
        text = (
            f"### Final decision: **{result['decision']}**\n"
            f"- Final confidence: **{result['confidence']:.3f}**\n"
            f"- Fusion score (P spoof): **{result['fusion_score']:.3f}**  "
            f"({result['fusion_method']}, threshold={thr:.2f})\n"
            f"- WavLM spoof probability: **{result['wavlm_spoof_prob']:.3f}**\n"
            f"- Wav2Vec2 spoof probability: **{result['wav2vec2_spoof_prob']:.3f}**\n"
            f"- Model agreement: **{agreement}**\n"
            f"- Reason: {result['reason']}\n\n"
            f"_Duration: {result['duration_s']:.2f}s_\n\n"
            f"{DISCLAIMER}"
        )
        wf_fig = _plot_waveform(audio.astype(np.float32), sr)
        win_fig = _plot_windows(result["windows"], thr)
        return wf_fig, text, win_fig

    with gr.Blocks(title="Voice Spoof Detection (PoC)") as demo:
        gr.Markdown(
            "## Voice Spoof Detection (WavLM + Wav2Vec2, late fusion)\n"
            f"{DISCLAIMER}"
        )
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio (mic/upload)")
        with gr.Accordion("Decision controls (advanced)", open=False):
            threshold_slider = gr.Slider(
                minimum=0.30, maximum=0.95, step=0.01,
                value=pipeline.decision_threshold,
                label="Decision threshold (P spoof above this → SPOOF)",
            )
            margin_slider = gr.Slider(
                minimum=0.00, maximum=0.30, step=0.01,
                value=pipeline.uncertainty_margin,
                label="Uncertainty margin (|fused − threshold| < margin → UNCERTAIN)",
            )
        run_btn = gr.Button("Analyse", variant="primary")

        # Pre-bundled examples (one bonafide, one spoof from ASVspoof19 LA test).
        ex_rows = []
        if example_paths:
            for label_name in ("bonafide", "spoof"):
                p = example_paths.get(label_name)
                if p and os.path.exists(p):
                    ex_rows.append([p, pipeline.decision_threshold, pipeline.uncertainty_margin])
        if ex_rows:
            gr.Markdown(
                "### Hazır örnekler\n"
                "Aşağıdaki iki klip ASVspoof 2019 LA **test split**'inden çekildi — "
                "model bu örnekleri eğitimde görmedi.  \n"
                "Tıkla → yukarıdaki **Analyse** butonuna bas. Sonra mikrofon sekmesinden kendi sesini kaydet ve karşılaştır."
            )
            gr.Examples(
                examples=ex_rows,
                inputs=[audio_in, threshold_slider, margin_slider],
                label="bonafide.flac ↔ spoof.flac",
                cache_examples=False,
            )

        with gr.Row():
            waveform_plot = gr.Plot(label="Waveform")
        result_md = gr.Markdown()
        windows_plot = gr.Plot(label="Sliding-window analysis (if audio > 4 s)")
        run_btn.click(
            infer,
            inputs=[audio_in, threshold_slider, margin_slider],
            outputs=[waveform_plot, result_md, windows_plot],
        )
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wavlm-checkpoint", required=True)
    p.add_argument("--wav2vec2-checkpoint", required=True)
    p.add_argument("--fusion-bundle", default=None)
    p.add_argument("--decision-threshold", type=float, default=0.65,
                   help="Initial decision threshold. The UI slider can override per-call.")
    p.add_argument("--uncertainty-margin", type=float, default=0.12)
    p.add_argument("--min-confidence", type=float, default=0.45)
    p.add_argument("--disagreement-margin", type=float, default=0.30)
    p.add_argument("--share", action="store_true")
    p.add_argument("--server-name", default=None)
    p.add_argument("--examples-dir", default="examples",
                   help="Local folder for the bonafide.flac / spoof.flac example clips.")
    p.add_argument("--examples-hf-repo", default="Bisher/ASVspoof_2019_LA",
                   help="HF dataset to pull example clips from on first launch.")
    p.add_argument("--no-fetch-examples", action="store_true",
                   help="Skip auto-fetching example clips; only use what's already on disk.")
    args = p.parse_args()

    fusion = FusionBundle.from_json(args.fusion_bundle) if args.fusion_bundle else None
    pipe = SpoofPipeline(
        args.wavlm_checkpoint, args.wav2vec2_checkpoint, fusion=fusion,
        decision_threshold=args.decision_threshold,
        uncertainty_margin=args.uncertainty_margin,
        min_confidence=args.min_confidence,
        disagreement_margin=args.disagreement_margin,
    )
    if args.no_fetch_examples:
        examples = {
            "bonafide": (os.path.join(args.examples_dir, "bonafide.flac")
                         if os.path.exists(os.path.join(args.examples_dir, "bonafide.flac")) else None),
            "spoof":    (os.path.join(args.examples_dir, "spoof.flac")
                         if os.path.exists(os.path.join(args.examples_dir, "spoof.flac")) else None),
        }
    else:
        examples = ensure_example_clips(out_dir=args.examples_dir, hf_repo=args.examples_hf_repo)
    demo = build_interface(pipe, example_paths=examples)
    demo.launch(share=args.share, server_name=args.server_name)


if __name__ == "__main__":
    main()
