"""
reporter.py — Pipeline orchestrator (no auto-detection)
--------------------------------------------------------
Flow:
  1. User selects category (dropdown in UI)
  2. PatchCore loads memory bank + scores image
  3. If anomaly → LLM explains defect
  4. Return structured result to UI

API key loaded from .env file — never hardcode it.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pathlib import Path

_here = Path(__file__).resolve().parent
for _candidate in [_here / ".env", _here.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=True)
        break

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from models.patchcore import PatchCore
from llm.explainer    import explain_defect, burn_heatmap_on_image

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def load_image(image_path: str) -> tuple:
    """Load image → (tensor for model, uint8 array for LLM)."""
    pil    = Image.open(image_path).convert("RGB")
    arr    = np.array(pil.resize((224, 224)))          # uint8 for LLM
    tensor = TRANSFORM(pil)                            # normalised for model
    return tensor, arr


def run_pipeline(
    image_path  : str,
    category    : str,
    memory_dir  : str  = "outputs",
    threshold   : float = 2.8,
    device      : str  = "cpu",
) -> dict:
    """
    Full pipeline: image + category → anomaly score → LLM report.

    API key is read automatically from .env — no need to pass it.

    Args:
        image_path : path to uploaded image
        category   : product category selected by user e.g. "bottle"
        memory_dir : folder containing memory_bank.pt files
        threshold  : anomaly score cutoff (default 0.5)
        device     : "cpu" or "cuda"

    Returns:
        dict with status, score, report, heatmap, overlay
    """

    # ── validate category ────────────────────────────────────────────────────
    if category not in CATEGORIES:
        return {
            "status" : "error",
            "message": f"Unknown category '{category}'. "
                       f"Supported: {', '.join(CATEGORIES)}"
        }

    # ── load memory bank ─────────────────────────────────────────────────────
    bank_path = Path(memory_dir) / category / "memory_bank.pt"
    if not bank_path.exists():
        return {
            "status"  : "error",
            "message" : f"No memory bank found for '{category}'. "
                        f"Run: python train_eval.py --category {category}"
        }

    model = PatchCore(device=device)
    model.load(str(bank_path))

    # ── score image ──────────────────────────────────────────────────────────
    tensor, img_array = load_image(image_path)
    score, heatmap    = model.predict_single(tensor)
    is_anomaly        = score >= threshold

    print(f"Category: {category} | Score: {score:.4f} | "
          f"{'ANOMALY' if is_anomaly else 'NORMAL'}")

    # ── normal image — return early, no LLM call ─────────────────────────────
    if not is_anomaly:
        return {
            "status"       : "normal",
            "category"     : category,
            "anomaly_score": round(score, 4),
            "threshold"    : threshold,
            "message"      : "No defect detected. Surface appears normal.",
            "heatmap"      : heatmap,
            "image_array"  : img_array,
            "overlay"      : None,
        }

    # ── anomaly — call LLM for explanation ───────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "status" : "error",
            "message": "GEMINI_API_KEY not found. "
                       "Add it to your .env file."
        }

    print("Anomaly detected — generating LLM explanation...")
    report  = explain_defect(img_array, heatmap, score, category, api_key)
    overlay = burn_heatmap_on_image(img_array, heatmap)

    return {
        "status"       : "anomaly",
        "category"     : category,
        "anomaly_score": round(score, 4),
        "threshold"    : threshold,
        "report"       : report,
        "heatmap"      : heatmap,
        "overlay"      : overlay,
        "image_array"  : img_array,
    }