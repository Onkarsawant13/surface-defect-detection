"""
train_eval.py — Main entry point
----------------------------------
Run:
    python train_eval.py --data /path/to/mvtec --category bottle

This script:
  1. Loads MVTec AD train/test splits
  2. Builds PatchCore memory bank from normal training images
  3. Scores all test images
  4. Prints AUROC + F1 metrics
  5. Saves heatmaps and score distribution to outputs/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from pathlib import Path

from data.dataset    import get_loaders
from models.patchcore import PatchCore
from utils.metrics   import (compute_metrics, print_metrics,
                              save_heatmap, save_score_distribution)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True, help="MVTec AD root folder")
    p.add_argument("--category",   default="bottle", help="MVTec category")
    p.add_argument("--output",     default="outputs", help="Output folder")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--save_n",     type=int, default=10,
                   help="Save heatmaps for first N test images")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. data ──────────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(
        args.data, args.category,
        batch_size=args.batch_size, num_workers=args.workers,
    )
    print(f"Category: {args.category}  |  "
          f"Train: {len(train_loader.dataset)}  |  "
          f"Test: {len(test_loader.dataset)}")

    # ── 2. build memory bank ─────────────────────────────────────────────────
    model = PatchCore(device=device)
    model.fit(train_loader)

    # save memory bank for reuse
    out = Path(args.output) / args.category
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "memory_bank.pt"))

    # ── 3. score test images ─────────────────────────────────────────────────
    scores, anomaly_maps, labels = model.predict(test_loader)

    # ── 4. metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(scores, labels)
    print_metrics(metrics, category=args.category)

    # ── 5. save outputs ──────────────────────────────────────────────────────
    save_score_distribution(scores, labels, str(out / "score_dist.png"))

    test_ds = test_loader.dataset
    for i in range(min(args.save_n, len(test_ds))):
        sample = test_ds[i]
        save_heatmap(
            image_tensor = sample["image"].unsqueeze(0),
            anomaly_map  = anomaly_maps[i],
            label        = labels[i],
            score        = scores[i],
            save_path    = str(out / f"heatmap_{i:03d}.png"),
            gt_mask      = sample["mask"].numpy(),
        )

    print(f"Outputs saved → {out}/")


if __name__ == "__main__":
    main()