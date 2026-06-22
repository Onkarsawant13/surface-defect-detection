"""
Evaluation + Visualisation Utilities
--------------------------------------
Metrics used:
  • AUROC (image-level) : standard MVTec AD benchmark metric.
    Measures how well anomaly scores separate defective from normal images.
    Threshold-free — no need to pick a cutoff.

  • PRO (Per-Region Overlap) : pixel-level metric that weights small
    defect regions fairly. Better than pixel-AUROC for tiny defects.
    We use a simplified version here (pixel AUROC as proxy).

  • F1 @ best threshold : practical metric — the score you'd report
    when deploying with a real decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path


# ── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(scores: list[float], labels: list[int],
                    anomaly_maps: list[np.ndarray] = None,
                    masks: list[np.ndarray] = None) -> dict:
    """
    Args:
        scores      : image-level anomaly scores
        labels      : ground truth (0=normal, 1=defective)
        anomaly_maps: pixel-level heatmaps (optional, for pixel AUROC)
        masks       : ground truth binary masks (optional)

    Returns dict with image_auroc, pixel_auroc (if maps provided), f1.
    """
    scores = np.array(scores)
    labels = np.array(labels)

    image_auroc = roc_auc_score(labels, scores)

    # best F1 over all possible thresholds
    thresholds  = np.linspace(scores.min(), scores.max(), 200)
    best_f1     = max(
        f1_score(labels, scores >= t, zero_division=0)
        for t in thresholds
    )

    results = {"image_auroc": image_auroc, "best_f1": best_f1}

    # pixel-level AUROC (requires masks)
    if anomaly_maps is not None and masks is not None:
        flat_maps   = np.concatenate([m.flatten() for m in anomaly_maps])
        flat_masks  = np.concatenate([m.flatten() for m in masks])
        if flat_masks.max() > 0:   # skip if no defective pixels at all
            results["pixel_auroc"] = roc_auc_score(flat_masks, flat_maps)

    return results


def print_metrics(metrics: dict, category: str = ""):
    header = f"  Results — {category}" if category else "  Results"
    print(f"\n{'─'*40}")
    print(header)
    print(f"{'─'*40}")
    print(f"  Image AUROC : {metrics['image_auroc']*100:.2f}%")
    if "pixel_auroc" in metrics:
        print(f"  Pixel AUROC : {metrics['pixel_auroc']*100:.2f}%")
    print(f"  Best F1     : {metrics['best_f1']:.4f}")
    print(f"{'─'*40}\n")


# ── visualisation ────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """Convert normalised tensor back to H×W×3 uint8 for display."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def save_heatmap(image_tensor, anomaly_map: np.ndarray, label: int,
                 score: float, save_path: str, gt_mask: np.ndarray = None):
    """
    Save a 3-panel (or 4-panel with GT mask) inspection figure:
      Original | Heatmap overlay | Anomaly map | [GT mask]
    """
    img  = denormalize(image_tensor.squeeze(0))
    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    # panel 1 — original
    axes[0].imshow(img)
    axes[0].set_title(f"{'DEFECT' if label else 'NORMAL'}  "
                      f"score={score:.3f}", fontsize=10)
    axes[0].axis("off")

    # panel 2 — heatmap overlay
    norm_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap  = cm.jet(norm_map)[..., :3]
    overlay  = 0.55 * img / 255.0 + 0.45 * heatmap
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Heatmap overlay", fontsize=10)
    axes[1].axis("off")

    # panel 3 — raw anomaly map
    axes[2].imshow(norm_map, cmap="hot")
    axes[2].set_title("Anomaly map", fontsize=10)
    axes[2].axis("off")

    # panel 4 — ground truth mask (if available)
    if gt_mask is not None:
        axes[3].imshow(gt_mask.squeeze(), cmap="gray")
        axes[3].set_title("Ground truth", fontsize=10)
        axes[3].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_score_distribution(scores: list[float], labels: list[int],
                             save_path: str):
    """Histogram of anomaly scores split by class — useful for threshold tuning."""
    scores = np.array(scores)
    labels = np.array(labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=30, alpha=0.65,
            color="#378ADD", label="Normal")
    ax.hist(scores[labels == 1], bins=30, alpha=0.65,
            color="#D85A30", label="Defective")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution — Normal vs Defective")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Score distribution saved → {save_path}")
