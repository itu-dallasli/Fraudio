"""Evaluate trained model(s) and optionally fuse them.

Usage:
    python -m src.evaluate \
        --wavlm-checkpoint   checkpoints/wavlm/best.pt \
        --wav2vec2-checkpoint checkpoints/wav2vec2/best.pt \
        --fusion
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import build_dataset, collate_batch, load_items_for_partition
from .metrics import (
    brier_score,
    classification_metrics,
    equal_error_rate,
    expected_calibration_error,
    min_tdcf,
    negative_log_likelihood,
)
from .models.calibration import fit_temperature, softmax_spoof_prob
from .models.fusion import (
    fit_logreg_fusion,
    fuse_average,
    fuse_weighted,
    logreg_features,
    sweep_alpha,
)
from .models.ssl_classifier import build_model_from_cfg
from .utils import get_logger, save_json, select_device, set_seed

LOG = get_logger("evaluate")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def _collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    logits, labels, files, systems = [], [], [], []
    for batch in tqdm(loader, desc="collect", leave=False):
        batch = _move_batch(batch, device)
        out = model(batch["waveform"])
        logits.append(out["logits"].float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        files.extend(batch["file_id"])
        systems.extend(batch["system_id"])
    return (
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0),
        files,
        systems,
    )


def _load_checkpoint(ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = obj["cfg"]
    model = build_model_from_cfg(cfg["model"]).to(device)
    model.load_state_dict(obj["model_state"])
    return model, cfg


def _build_loaders(cfg: dict, dataset_root_override: str | None, seed: int):
    cfg_data = dict(cfg["data"])
    if dataset_root_override:
        cfg_data["dataset_root"] = dataset_root_override
    cfg_aug = cfg.get("augmentation", {})
    source = (cfg_data.get("source") or "file").lower()

    if source in ("file", "asvspoof", "asvspoof_la"):
        dev_items = load_items_for_partition(
            cfg_data["dataset_root"], "dev", cfg_data.get("max_dev_samples"), seed
        )
        eval_items = load_items_for_partition(
            cfg_data["dataset_root"], "eval", cfg_data.get("max_eval_samples"), seed
        )
        dev_ds = build_dataset(dev_items, cfg_data, cfg_aug, partition="dev", seed=seed)
        eval_ds = build_dataset(eval_items, cfg_data, cfg_aug, partition="eval", seed=seed)
    elif source in ("huggingface", "hf"):
        from .data.hf_loader import build_hf_dataset
        dev_ds = build_hf_dataset(cfg_data, cfg_aug, partition="dev", seed=seed)
        eval_ds = build_hf_dataset(cfg_data, cfg_aug, partition="eval", seed=seed)
        dev_items = eval_items = None
    else:
        raise ValueError(f"Unknown data.source: {source}")

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg_data.get("num_workers", 2),
        collate_fn=collate_batch,
    )
    return DataLoader(dev_ds, **loader_kwargs), DataLoader(eval_ds, **loader_kwargs), dev_items, eval_items


# ---------------------------------------------------------------------- #
# plotting helpers
# ---------------------------------------------------------------------- #

def _plot_confusion(cm: list[list[int]], path: str, title: str):
    if not HAVE_MPL:
        return
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["bonafide", "spoof"])
    ax.set_yticklabels(["bonafide", "spoof"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_roc(scores: np.ndarray, labels: np.ndarray, path: str, title: str):
    if not HAVE_MPL or len(np.unique(labels)) < 2:
        return
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_history(history_path: str, out_path: str):
    if not HAVE_MPL or not os.path.exists(history_path):
        return
    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    epochs = range(1, len(h.get("train_loss", [])) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, h.get("train_loss", []), "b-o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, h.get("dev_eer", []), "r-s", label="dev EER")
    ax2.set_ylabel("EER", color="r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------- #

def evaluate(
    wavlm_ckpt: str | None,
    wav2vec2_ckpt: str | None,
    dataset_root: str | None,
    output_dir: str = "outputs/evaluation",
    do_fusion: bool = True,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = select_device()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    per_model: dict[str, dict] = {}
    for tag, ckpt in (("wavlm", wavlm_ckpt), ("wav2vec2", wav2vec2_ckpt)):
        if not ckpt:
            continue
        LOG.info(f"[{tag}] loading {ckpt}")
        model, cfg = _load_checkpoint(ckpt, device)
        dev_loader, eval_loader, _, _ = _build_loaders(cfg, dataset_root, seed)

        dev_logits, dev_labels, dev_files, dev_systems = _collect_logits(model, dev_loader, device)
        eval_logits, eval_labels, eval_files, eval_systems = _collect_logits(model, eval_loader, device)

        # Raw (uncalibrated) spoof probabilities.
        dev_p = softmax_spoof_prob(dev_logits)
        eval_p = softmax_spoof_prob(eval_logits)

        # Temperature scaling fitted on dev.
        scaler = fit_temperature(dev_logits, dev_labels)
        T = scaler.temperature
        cal_dev_logits = scaler.calibrate_logits_np(dev_logits)
        cal_eval_logits = scaler.calibrate_logits_np(eval_logits)
        cal_dev_p = softmax_spoof_prob(cal_dev_logits)
        cal_eval_p = softmax_spoof_prob(cal_eval_logits)

        # Metrics on raw scores (eval).
        metrics_eval = classification_metrics(eval_p, eval_labels)
        tdcf = min_tdcf(eval_p, eval_labels)
        if tdcf is not None:
            metrics_eval["min_tdcf"] = tdcf

        calib = {
            "temperature": T,
            "before": {
                "nll": negative_log_likelihood(dev_p, dev_labels),
                "brier": brier_score(dev_p, dev_labels),
                "ece": expected_calibration_error(dev_p, dev_labels),
            },
            "after": {
                "nll": negative_log_likelihood(cal_dev_p, dev_labels),
                "brier": brier_score(cal_dev_p, dev_labels),
                "ece": expected_calibration_error(cal_dev_p, dev_labels),
            },
        }
        metrics_eval["calibration"] = calib
        LOG.info(
            f"[{tag}] eval EER={metrics_eval['eer']:.4f}  AUC={metrics_eval['roc_auc']:.4f}  T={T:.3f}"
        )

        # Save per-model outputs.
        model_out = out_dir / tag
        model_out.mkdir(parents=True, exist_ok=True)
        save_json(metrics_eval, model_out / "metrics.json")

        df = pd.DataFrame(
            {
                "file_id": eval_files,
                "system_id": eval_systems,
                "label": eval_labels,
                "spoof_prob": eval_p,
                "spoof_prob_calibrated": cal_eval_p,
            }
        )
        df.to_csv(model_out / "predictions.csv", index=False)

        _plot_confusion(metrics_eval["confusion_matrix"], str(model_out / "confusion_matrix.png"), f"{tag} confusion")
        _plot_roc(eval_p, eval_labels, str(model_out / "roc.png"), f"{tag} ROC")
        # Loss curve from training run, if available.
        cfg_out = cfg["training"]["output_dir"]
        _plot_history(os.path.join(cfg_out, "history.json"), str(model_out / "training_history.png"))

        per_model[tag] = {
            "ckpt": ckpt,
            "dev_logits": dev_logits,
            "dev_labels": dev_labels,
            "dev_p": dev_p,
            "dev_files": dev_files,
            "cal_dev_logits": cal_dev_logits,
            "cal_dev_p": cal_dev_p,
            "eval_logits": eval_logits,
            "eval_labels": eval_labels,
            "eval_p": eval_p,
            "eval_files": eval_files,
            "eval_systems": eval_systems,
            "cal_eval_logits": cal_eval_logits,
            "cal_eval_p": cal_eval_p,
            "temperature": T,
            "metrics": metrics_eval,
        }
        summary[tag] = {k: v for k, v in metrics_eval.items() if k != "confusion_matrix"}
        # release GPU memory between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------- fusion --------------------------- #
    if do_fusion and len(per_model) == 2:
        wavlm, w2v = per_model["wavlm"], per_model["wav2vec2"]

        # Align by file_id (in case orders differ — shouldn't, but defensive).
        assert wavlm["eval_files"] == w2v["eval_files"], "Eval ordering mismatch between models"
        labels_dev = wavlm["dev_labels"]
        labels_eval = wavlm["eval_labels"]

        # 1) simple average
        avg_eval = fuse_average(wavlm["cal_eval_p"], w2v["cal_eval_p"])
        avg_dev = fuse_average(wavlm["cal_dev_p"], w2v["cal_dev_p"])

        # 2) weighted average — sweep alpha on dev
        best_alpha, dev_eer_w = sweep_alpha(wavlm["cal_dev_p"], w2v["cal_dev_p"], labels_dev)
        weighted_eval = fuse_weighted(wavlm["cal_eval_p"], w2v["cal_eval_p"], best_alpha)
        weighted_dev = fuse_weighted(wavlm["cal_dev_p"], w2v["cal_dev_p"], best_alpha)

        # 3) logistic regression on calibrated logit differences
        lr_fusion = fit_logreg_fusion(wavlm["cal_dev_logits"], w2v["cal_dev_logits"], labels_dev)
        feats_dev = logreg_features(wavlm["cal_dev_logits"], w2v["cal_dev_logits"])
        feats_eval = logreg_features(wavlm["cal_eval_logits"], w2v["cal_eval_logits"])
        lr_dev = lr_fusion.predict_proba_spoof(feats_dev)
        lr_eval = lr_fusion.predict_proba_spoof(feats_eval)

        fusion_methods = {
            "average":  {"dev": avg_dev,       "eval": avg_eval},
            "weighted": {"dev": weighted_dev,  "eval": weighted_eval,
                         "alpha": best_alpha},
            "logreg":   {"dev": lr_dev,        "eval": lr_eval,
                         "coef": lr_fusion.coef, "intercept": lr_fusion.intercept},
        }

        fusion_summary = {}
        best_name, best_dev_eer = None, float("inf")
        for name, d in fusion_methods.items():
            dev_eer, _ = equal_error_rate(d["dev"], labels_dev)
            eval_m = classification_metrics(d["eval"], labels_eval)
            entry = {"dev_eer": dev_eer, "eval_metrics": {k: v for k, v in eval_m.items() if k != "confusion_matrix"}}
            if "alpha" in d:
                entry["alpha"] = d["alpha"]
            if "coef" in d:
                entry["coef"] = d["coef"]
                entry["intercept"] = d["intercept"]
            fusion_summary[name] = entry
            if dev_eer < best_dev_eer:
                best_dev_eer = dev_eer
                best_name = name

        fusion_summary["best"] = best_name
        LOG.info(f"Fusion best by dev EER: {best_name} (dev EER={best_dev_eer:.4f})")

        # Save outputs
        fusion_out = out_dir / "fusion"
        fusion_out.mkdir(parents=True, exist_ok=True)
        save_json(fusion_summary, fusion_out / "metrics.json")
        df = pd.DataFrame({
            "file_id": wavlm["eval_files"],
            "system_id": wavlm["eval_systems"],
            "label": labels_eval,
            "wavlm_spoof_prob": wavlm["cal_eval_p"],
            "wav2vec2_spoof_prob": w2v["cal_eval_p"],
            "avg_spoof_prob": avg_eval,
            "weighted_spoof_prob": weighted_eval,
            "logreg_spoof_prob": lr_eval,
        })
        df.to_csv(fusion_out / "predictions.csv", index=False)
        best_fused_eval = fusion_methods[best_name]["eval"]
        cm = classification_metrics(best_fused_eval, labels_eval)["confusion_matrix"]
        _plot_confusion(cm, str(fusion_out / "confusion_matrix.png"), f"fusion ({best_name}) confusion")
        _plot_roc(best_fused_eval, labels_eval, str(fusion_out / "roc.png"), f"fusion ({best_name}) ROC")

        # Save a "best fusion" config bundle Gradio can load.
        save_json(
            {
                "best_method": best_name,
                "alpha": float(best_alpha),
                "logreg_coef": lr_fusion.coef,
                "logreg_intercept": lr_fusion.intercept,
                "wavlm_temperature": wavlm["temperature"],
                "wav2vec2_temperature": w2v["temperature"],
            },
            fusion_out / "fusion_bundle.json",
        )
        summary["fusion"] = fusion_summary

    # --------------------------- comparison table --------------------------- #
    rows = []
    for name, m in summary.items():
        if name == "fusion":
            continue
        rows.append(
            {
                "model": name,
                "EER": m.get("eer"),
                "AUC": m.get("roc_auc"),
                "F1_spoof": m.get("f1_spoof"),
                "Accuracy": m.get("accuracy"),
            }
        )
    if "fusion" in summary:
        best = summary["fusion"]["best"]
        em = summary["fusion"][best]["eval_metrics"]
        rows.append(
            {
                "model": f"fusion[{best}]",
                "EER": em.get("eer"),
                "AUC": em.get("roc_auc"),
                "F1_spoof": em.get("f1_spoof"),
                "Accuracy": em.get("accuracy"),
            }
        )
    if rows:
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(out_dir / "comparison.csv", index=False)
        save_json(rows, out_dir / "comparison.json")
        LOG.info(f"Comparison:\n{cmp_df.to_string(index=False)}")

    save_json(summary, out_dir / "summary.json")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wavlm-checkpoint", default=None)
    p.add_argument("--wav2vec2-checkpoint", default=None)
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--output_dir", default="outputs/evaluation")
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    evaluate(
        wavlm_ckpt=args.wavlm_checkpoint,
        wav2vec2_ckpt=args.wav2vec2_checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        do_fusion=args.fusion,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
