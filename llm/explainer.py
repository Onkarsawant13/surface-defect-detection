"""
explainer.py — Defect explanation using Gemini vision (google-genai SDK)
---------------------------------------------------------------------------
Model: gemini-2.5-flash-lite
  - Current free-tier model as of 2026 (15 RPM, 1000 requests/day, free)
  - Gemini 2.0 models are deprecated — do not use those
"""

import json
import re
import io
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from google import genai
from google.genai import types


EXPLANATION_PROMPT = """You are an expert industrial quality control engineer.

The image shows a surface inspection result. The RED/ORANGE highlighted region 
indicates where an anomaly detection model found a defect.

Anomaly score: {score:.3f} (scale 0.0-1.0, higher = more anomalous)
Product category: {category}

Analyze the highlighted region and return ONLY a JSON object — no extra text, 
no markdown fences, just raw JSON:

{{
    "defect_class": "specific defect type e.g. crack, scratch, contamination, hole, dent, discoloration, missing_part",
    "location": "precise location e.g. bottom-left edge, center, near seam",
    "description": "2-3 sentence visual description of what you observe",
    "root_cause": "most likely manufacturing cause of this defect",
    "severity": "low or medium or high",
    "severity_reason": "one sentence explaining the severity level",
    "recommended_fix": "specific corrective action for the manufacturing process",
    "confidence": "high or medium or low based on how clearly visible the defect is"
}}"""


def burn_heatmap_on_image(
    image  : np.ndarray,
    heatmap: np.ndarray,
    alpha  : float = 0.45
) -> np.ndarray:
    """Overlay heatmap onto image. Returns H×W×3 uint8."""
    norm     = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    coloured = cm.jet(norm)[..., :3]
    blended  = (1 - alpha) * (image / 255.0) + alpha * coloured
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def array_to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL image."""
    return Image.fromarray(image_array.astype(np.uint8))


def explain_defect(
    image_array  : np.ndarray,
    heatmap      : np.ndarray,
    anomaly_score: float,
    category     : str,
    api_key      : str,
) -> dict | None:
    """
    Get full defect explanation from Gemini vision.
    Returns None if Gemini is unavailable instead of raising.
    """
    try:
        client = genai.Client(api_key=api_key)

        overlay   = burn_heatmap_on_image(image_array, heatmap)
        pil_image = array_to_pil(overlay)

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        prompt = EXPLANATION_PROMPT.format(
            score    = anomaly_score,
            category = category,
        )

        response = client.models.generate_content(
            model    = "gemini-2.5-flash-lite",
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ]
        )

        raw_text = response.text.strip()
        clean    = re.sub(r"```json|```", "", raw_text).strip()

        try:
            report = json.loads(clean)
        except json.JSONDecodeError:
            match  = re.search(r"\{.*\}", clean, re.DOTALL)
            report = json.loads(match.group()) if match else {"raw_response": raw_text}

        report["anomaly_score"] = round(anomaly_score, 4)
        report["category"]      = category
        return report

    except Exception as e:
        print(f"[explainer] Gemini call failed: {type(e).__name__}: {e}")
        return None