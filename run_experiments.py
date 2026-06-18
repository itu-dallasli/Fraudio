"""Train both SSL models sequentially and run fused evaluation.

Examples:
    python run_experiments.py                                 # full run
    python run_experiments.py --quick                         # 256 samples / partition
    python run_experiments.py --dataset_root /path/to/LA
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate
from src.train import train
from src.utils import get_logger, load_yaml

LOG = get_logger("experiments")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wavlm-config", default="configs/wavlm.yaml")
    p.add_argument("--wav2vec2-config", default="configs/wav2vec2.yaml")
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--quick", action="store_true", help="Cap each partition to a small subset.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir", default="outputs/evaluation")
    args = p.parse_args()

    cap = args.max_samples or (256 if args.quick else None)

    def _patch(cfg: dict) -> dict:
        if cap is not None:
            cfg["data"]["max_train_samples"] = cap
            cfg["data"]["max_dev_samples"] = max(64, cap // 4)
            cfg["data"]["max_eval_samples"] = max(64, cap // 4)
            cfg["training"]["epochs"] = 1
            cfg["training"]["batch_size"] = min(cfg["training"]["batch_size"], 8)
        return cfg

    wavlm_cfg = _patch(load_yaml(args.wavlm_config))
    w2v_cfg = _patch(load_yaml(args.wav2vec2_config))

    LOG.info("=== Training WavLM ===")
    wavlm_result = train(wavlm_cfg, dataset_root_override=args.dataset_root)
    LOG.info("=== Training Wav2Vec2 ===")
    w2v_result = train(w2v_cfg, dataset_root_override=args.dataset_root)

    wavlm_ckpt = str(Path(wavlm_result["save_dir"]) / "best.pt")
    w2v_ckpt = str(Path(w2v_result["save_dir"]) / "best.pt")

    LOG.info("=== Evaluation + Fusion ===")
    evaluate(
        wavlm_ckpt=wavlm_ckpt,
        wav2vec2_ckpt=w2v_ckpt,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        do_fusion=True,
        seed=wavlm_cfg.get("seed", 42),
    )
    LOG.info("Done.")


if __name__ == "__main__":
    main()
