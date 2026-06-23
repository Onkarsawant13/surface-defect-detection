"""
reporter.py — Pipeline orchestrator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

_here = Path(__file__).resolve().parent
for _candidate in [_here / ".env", _here.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=True)
        break

import numpy as np
import torch
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

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def _get_api_key() -> str | None:
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def load_image(image_path: str) -> tuple:
    pil    = Image.open(image_path).convert("RGB")
    arr    = np.array(pil.resize((224, 224)))
    tensor = TRANSFORM(pil)
    return tensor, arr


def run_pipeline(
    image_path  : str,
    category    : str,
    memory_dir  : str   = "outputs",
    threshold   : float = 2.8,
    device      : str   = "cpu",
) -> dict:

    if category not in CATEGORIES:
        return {
            "status" : "error",
            "message": f"Unknown category '{category}'. Supported: {', '.join(CATEGORIES)}"
        }

    bank_path = Path(memory_dir) / category / "memory_bank.pt"
    if not bank_path.exists():
        return {
            "status" : "error",
            "message": f"No memory bank found for '{category}'."
        }

    model = PatchCore(device=device)
    model.load(str(bank_path))

    tensor, img_array = load_image(image_path)
    score, heatmap    = model.predict_single(tensor)
    is_anomaly        = score >= threshold

    print(f"Category: {category} | Score: {score:.4f} | {'ANOMALY' if is_anomaly else 'NORMAL'}")

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
            "llm_failed"   : False,
            "report"       : None,
        }

    api_key = _get_api_key()
    if not api_key:
        return {
            "status" : "error",
            "message": "GEMINI_API_KEY not found. Add it to Streamlit Cloud Secrets."
        }

    print("Anomaly detected — generating LLM explanation...")
    overlay = burn_heatmap_on_image(img_array, heatmap)

    try:
        report     = explain_defect(img_array, heatmap, score, category, api_key)
        llm_failed = False
    except Exception as e:
        print(f"LLM call failed: {e}")
        report     = None
        llm_failed = True

    return {
        "status"       : "anomaly",
        "category"     : category,
        "anomaly_score": round(score, 4),
        "threshold"    : threshold,
        "report"       : report,
        "llm_failed"   : llm_failed,
        "heatmap"      : heatmap,
        "overlay"      : overlay,
        "image_array"  : img_array,
    }