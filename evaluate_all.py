"""
evaluate_all.py — Score all categories from saved memory banks
--------------------------------------------------------------
Run:
    python evaluate_all.py --data ./dataset

Loads each saved memory_bank.pt, rescores test images,
prints a summary table of all AUROC scores.
No retraining needed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from pathlib import Path

from data.dataset     import get_loaders
from models.patchcore import PatchCore
from utils.metrics    import compute_metrics

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True, help="MVTec AD root folder")
    p.add_argument("--output",  default="outputs", help="Folder with saved memory banks")
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    missing = []

    for cat in CATEGORIES:
        bank_path = Path(args.output) / cat / "memory_bank.pt"

        if not bank_path.exists():
            missing.append(cat)
            continue

        print(f"Scoring {cat}...", end=" ", flush=True)

        # load memory bank
        model = PatchCore(device=device)
        model.load(str(bank_path))

        # load test split only — no need to rebuild memory bank
        _, test_loader = get_loaders(
            args.data, cat,
            batch_size=1, num_workers=args.workers
        )

        scores, anomaly_maps, labels = model.predict(test_loader)
        metrics = compute_metrics(scores, labels)

        auroc = metrics["image_auroc"] * 100
        f1    = metrics["best_f1"]
        results.append((cat, auroc, f1))
        print(f"AUROC {auroc:.1f}%  F1 {f1:.3f}")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "═" * 44)
    print(f"  {'Category':<16} {'Image AUROC':>12} {'Best F1':>10}")
    print("═" * 44)

    for cat, auroc, f1 in results:
        bar    = "█" * int(auroc / 10)
        flag   = "✓" if auroc >= 95 else "~" if auroc >= 90 else "✗"
        print(f"  {flag} {cat:<14} {auroc:>10.1f}%  {f1:>9.3f}  {bar}")

    if results:
        avg_auroc = sum(a for _, a, _ in results) / len(results)
        avg_f1    = sum(f for _, _, f in results) / len(results)
        print("═" * 44)
        print(f"  {'AVERAGE':<16} {avg_auroc:>10.1f}%  {avg_f1:>9.3f}")
        print("═" * 44)

    if missing:
        print(f"\n  Skipped (no memory bank found): {', '.join(missing)}")
        print(f"  Run train_eval.py for those categories first.")


if __name__ == "__main__":
    main()