"""Train a single SSL spoof-detection model on ASVspoof 2019 LA.

Usage:
    python -m src.train --config configs/wavlm.yaml
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import (
    build_dataset,
    collate_batch,
    load_items_for_partition,
    make_balanced_sampler,
)
from .metrics import equal_error_rate, classification_metrics
from .models.ssl_classifier import build_model_from_cfg
from .utils import get_logger, load_yaml, save_json, select_device, set_seed
from torch.utils.data import WeightedRandomSampler


def _build_partition(cfg_data: dict, cfg_aug: dict, partition: str, seed: int):
    """Return (dataset, labels_list_or_None) for either file or HF source."""
    source = (cfg_data.get("source") or "file").lower()
    if source in ("file", "asvspoof", "asvspoof_la"):
        items = load_items_for_partition(
            cfg_data["dataset_root"], partition,
            cfg_data.get({"train": "max_train_samples", "dev": "max_dev_samples", "eval": "max_eval_samples"}[partition]),
            seed,
        )
        ds = build_dataset(items, cfg_data, cfg_aug, partition=partition, seed=seed)
        return ds, [x.label for x in items]
    elif source in ("huggingface", "hf"):
        from .data.hf_loader import HFASVspoofDataset, build_hf_dataset
        ds = build_hf_dataset(cfg_data, cfg_aug, partition=partition, seed=seed)
        labels = ds.labels if isinstance(ds, HFASVspoofDataset) else None
        return ds, labels
    else:
        raise ValueError(f"Unknown data.source: {source}")


def _balanced_sampler_from_labels(labels: list[int]) -> WeightedRandomSampler:
    n_bona = sum(1 for x in labels if x == 0) or 1
    n_spoof = sum(1 for x in labels if x == 1) or 1
    weights = [1.0 / n_bona if x == 0 else 1.0 / n_spoof for x in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


LOG = get_logger("train")


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #

def _class_weights(labels, mode: str) -> torch.Tensor | None:
    if mode in (None, "none") or labels is None:
        return None
    n_bona = sum(1 for x in labels if x == 0) or 1
    n_spoof = sum(1 for x in labels if x == 1) or 1
    total = n_bona + n_spoof
    if mode == "balanced":
        w0 = total / (2.0 * n_bona)
        w1 = total / (2.0 * n_spoof)
    elif mode == "sqrt":
        w0 = math.sqrt(total / n_bona)
        w1 = math.sqrt(total / n_spoof)
    else:
        return None
    return torch.tensor([w0, w1], dtype=torch.float32)


def _build_optimiser(model, cfg_train: dict):
    groups = model.trainable_parameter_groups(
        encoder_lr=cfg_train["encoder_lr"],
        head_lr=cfg_train["head_lr"],
        weight_decay=cfg_train.get("weight_decay", 1e-4),
    )
    return torch.optim.AdamW(groups)


def _build_scheduler(optimiser, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


# ---------------------------------------------------------------------- #
# eval (used inside train for dev EER tracking)
# ---------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels, all_files = [], [], []
    for batch in tqdm(loader, desc="dev-eval", leave=False):
        batch = _move_batch(batch, device)
        out = model(batch["waveform"])
        all_logits.append(out["logits"].float().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_files.extend(batch["file_id"])
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    spoof_scores = F.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    metrics = classification_metrics(spoof_scores, labels)
    eer, _ = equal_error_rate(spoof_scores, labels)
    metrics["eer"] = eer
    # Class-collapse detector: if ≥98% of dev predictions land in one class while EER
    # is roughly random, the model is shortcutting on the majority class. Warn loudly so
    # the user can stop and re-check sampler/loss/learning-rate settings.
    preds = (spoof_scores >= 0.5).astype(int)
    n = len(preds)
    spoof_frac = float((preds == 1).sum()) / max(1, n)
    metrics["dev_pred_spoof_frac"] = spoof_frac
    if n > 0 and (spoof_frac >= 0.98 or spoof_frac <= 0.02) and (eer > 0.40 or np.isnan(eer)):
        LOG.warning(
            f"[class-collapse] dev predictions are {spoof_frac*100:.1f}% spoof while EER={eer:.3f}. "
            f"Model is shortcutting on the majority class. "
            f"Try use_balanced_sampler=true, higher head_lr, more unfrozen layers, or more epochs."
        )
    return {
        "logits": logits,
        "labels": labels,
        "scores": spoof_scores,
        "file_ids": all_files,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------- #
# main training loop
# ---------------------------------------------------------------------- #

def train(cfg: dict, dataset_root_override: str | None = None) -> dict:
    set_seed(cfg.get("seed", 42))
    device = select_device()
    LOG.info(f"Device: {device}")

    cfg_data = dict(cfg["data"])
    if dataset_root_override:
        cfg_data["dataset_root"] = dataset_root_override
    cfg_aug = cfg.get("augmentation", {})
    cfg_train = cfg["training"]

    train_ds, train_labels = _build_partition(cfg_data, cfg_aug, "train", cfg["seed"])
    dev_ds, _dev_labels = _build_partition(cfg_data, cfg_aug, "dev", cfg["seed"])
    is_iterable_train = isinstance(train_ds, torch.utils.data.IterableDataset)
    if not is_iterable_train:
        LOG.info(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")
    else:
        LOG.info("Train: streaming (HF) | Dev: streaming (HF)")

    sampler = None
    if cfg_train.get("use_balanced_sampler", False):
        if is_iterable_train or train_labels is None:
            LOG.warning("Balanced sampler unavailable in streaming/HF mode; using class-weighted CE instead.")
        else:
            sampler = _balanced_sampler_from_labels(train_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_train["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None and not is_iterable_train),
        num_workers=cfg_data.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg_train["batch_size"],
        shuffle=False,
        num_workers=cfg_data.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = build_model_from_cfg(cfg["model"]).to(device)
    optimiser = _build_optimiser(model, cfg_train)
    if is_iterable_train:
        steps_per_epoch = max(1, cfg_train.get("streaming_steps_per_epoch", 500))
    else:
        steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * cfg_train["epochs"] // max(1, cfg_train.get("grad_accum_steps", 1)))
    scheduler = _build_scheduler(optimiser, total_steps, cfg_train.get("warmup_ratio", 0.05))

    class_w = _class_weights(train_labels, cfg_train.get("class_weighting", "balanced"))
    if class_w is not None:
        class_w = class_w.to(device)
        LOG.info(f"Class weights (bonafide, spoof) = {class_w.tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    use_amp = bool(cfg_train.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(cfg_train["save_dir"])
    output_dir = Path(cfg_train["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_eer = float("inf")
    history = {"train_loss": [], "dev_eer": [], "dev_acc": []}
    grad_accum = max(1, cfg_train.get("grad_accum_steps", 1))
    grad_clip = cfg_train.get("grad_clip", 1.0)

    global_step = 0
    for epoch in range(cfg_train["epochs"]):
        model.train()
        running, batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg_train['epochs']}", leave=False)
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            batch = _move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch["waveform"])
                loss = loss_fn(out["logits"], batch["label"]) / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                global_step += 1

            running += float(loss.item()) * grad_accum
            batches += 1
            if (step + 1) % cfg_train.get("log_every", 25) == 0:
                pbar.set_postfix(loss=f"{running/batches:.4f}")

        train_loss = running / max(1, batches)
        dev_out = evaluate_loader(model, dev_loader, device)
        dev_eer = dev_out["metrics"]["eer"]
        dev_acc = dev_out["metrics"]["accuracy"]
        history["train_loss"].append(train_loss)
        history["dev_eer"].append(dev_eer)
        history["dev_acc"].append(dev_acc)
        LOG.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}  dev_eer={dev_eer:.4f}  dev_acc={dev_acc:.4f}"
        )

        if not math.isnan(dev_eer) and dev_eer < best_eer:
            best_eer = dev_eer
            ckpt_path = save_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch + 1,
                    "dev_eer": dev_eer,
                },
                ckpt_path,
            )
            LOG.info(f"  ↳ saved new best to {ckpt_path}")

    save_json(history, output_dir / "history.json")
    LOG.info(f"Best dev EER: {best_eer:.4f}")
    return {"best_dev_eer": best_eer, "history": history, "save_dir": str(save_dir)}


# ---------------------------------------------------------------------- #
# entry point
# ---------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--dataset_root", default=None, help="Override data.dataset_root")
    p.add_argument("--max_samples", type=int, default=None, help="Cap each partition for quick mode")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if args.max_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_samples
        cfg["data"]["max_dev_samples"] = max(64, args.max_samples // 4)
        cfg["data"]["max_eval_samples"] = max(64, args.max_samples // 4)
    train(cfg, dataset_root_override=args.dataset_root)


if __name__ == "__main__":
    main()
