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
import time
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


def _call_gemini(client, image_bytes: bytes, prompt: str) -> str:
    """
    Single Gemini API call — completely isolated so ANY exception is catchable.
    Returns response text or raises.
    """
    response = client.models.generate_content(
        model    = "gemini-2.5-flash-lite",
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ]
    )
    return response.text.strip()


def explain_defect(
    image_array  : np.ndarray,
    heatmap      : np.ndarray,
    anomaly_score: float,
    category     : str,
    api_key      : str,
) -> dict | None:
    """
    Get full defect explanation from Gemini vision.
    Returns None if Gemini is unavailable — never raises.
    """
    try:
        client = genai.Client(api_key=api_key)

        overlay   = burn_heatmap_on_image(image_array, heatmap)
        pil_image = array_to_pil(overlay)

        # resize to smaller image to reduce token usage
        pil_image = pil_image.resize((112, 112))
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=75)  # JPEG uses far fewer tokens than PNG
        image_bytes = buf.getvalue()

        prompt = EXPLANATION_PROMPT.format(
            score    = anomaly_score,
            category = category,
        )

        # retry up to 2 times with delay for 503/overload errors
        raw_text = None
        last_error = None
        for attempt in range(3):
            try:
                raw_text = _call_gemini(client, image_bytes, prompt)
                break  # success
            except BaseException as e:
                last_error = e
                err_str = str(e)
                # 503 overload or 429 rate limit — wait and retry
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = (attempt + 1) * 8
                    print(f"[explainer] Gemini busy (attempt {attempt+1}), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                # any other error — don't retry
                print(f"[explainer] Gemini error: {type(e).__name__}: {err_str}")
                return None

        if raw_text is None:
            print(f"[explainer] All attempts failed: {last_error}")
            return None

        # parse JSON response
        clean = re.sub(r"```json|```", "", raw_text).strip()
        try:
            report = json.loads(clean)
        except json.JSONDecodeError:
            match  = re.search(r"\{.*\}", clean, re.DOTALL)
            report = json.loads(match.group()) if match else {"raw_response": raw_text}

        report["anomaly_score"] = round(anomaly_score, 4)
        report["category"]      = category
        return report

    except BaseException as e:
        print(f"[explainer] Unexpected error: {type(e).__name__}: {e}")
        return None