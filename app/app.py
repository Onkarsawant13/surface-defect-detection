"""
app.py — Surface Defect Inspection System (Streamlit UI)
-----------------------------------------------------------
Industrial HMI-inspired dashboard:
  - Dark charcoal theme, monospace data readouts
  - Side-by-side original vs heatmap overlay
  - Circular gauge showing anomaly score with severity zones
  - Inspection-certificate style defect report

Run:
    streamlit run app/app.py
"""
import os
from huggingface_hub import hf_hub_download

HF_REPO = "Onkarsawant/surface-defect-memory-banks"

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
] 

def ensure_memory_banks():
    for cat in CATEGORIES:
        local_path = f"outputs/{cat}/memory_bank.pt"
        if not os.path.exists(local_path):
            os.makedirs(f"outputs/{cat}", exist_ok=True)
            print(f"Downloading memory bank for {cat}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{cat}/memory_bank.pt",
                repo_type="dataset",
                local_dir="outputs"
            )
            print(f"✓ {cat} ready")

ensure_memory_banks()  # runs once at startup
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
from PIL import Image
import io
import math

from llm.reporter import run_pipeline, CATEGORIES


# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surface Defect Inspection",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── design tokens / CSS ──────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #1A1D23;
    --panel:     #23272F;
    --panel-2:   #2B313C;
    --border:    #383E4A;
    --text:      #E8E6E1;
    --text-dim:  #8A909C;
    --accent-red:   #E5483D;
    --accent-teal:  #3FA796;
    --accent-amber: #D9A441;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp { background-color: var(--bg); }

section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);
}

/* headers */
.hmi-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 0 0 4px;
}
.hmi-heading {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 28px;
    color: var(--text);
    margin: 0 0 24px;
    letter-spacing: -0.01em;
}

/* panels */
.hmi-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.hmi-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 0 0 10px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

/* status badges */
.status-pass {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(63,167,150,0.12);
    border: 1px solid var(--accent-teal);
    color: var(--accent-teal);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 6px 14px; border-radius: 4px;
}
.status-fail {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(229,72,61,0.12);
    border: 1px solid var(--accent-red);
    color: var(--accent-red);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 6px 14px; border-radius: 4px;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 0 3px currentColor33;
}

/* report fields */
.report-field {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    align-items: start;
}
.report-field:last-child { border-bottom: none; }
.report-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    padding-top: 2px;
}
.report-value {
    font-size: 14px;
    line-height: 1.55;
    color: var(--text);
}
.severity-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 3px;
}
.severity-low    { background: rgba(63,167,150,0.15); color: var(--accent-teal); }
.severity-medium { background: rgba(217,164,65,0.15); color: var(--accent-amber); }
.severity-high   { background: rgba(229,72,61,0.15); color: var(--accent-red); }

/* image caption */
.img-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    text-align: center;
    margin-top: 8px;
}

/* divider */
.hmi-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0 16px;
}

/* sidebar headings */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 20px 0 8px;
}

/* hide streamlit branding clutter (keep header for sidebar toggle) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {
    background-color: transparent;
}
header [data-testid="stHeader"] {
    background-color: transparent;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ── gauge — signature element ────────────────────────────────────────────────
def render_gauge(score: float, max_score: float = 6.0, threshold: float = 2.8) -> str:
    """
    Returns SVG markup for a circular gauge showing anomaly score.
    Needle position maps score -> angle. Color zones: teal (normal) / amber / red.
    """
    # clamp
    score = max(0, min(score, max_score))
    pct   = score / max_score

    # gauge spans 210 degrees, starting at -105deg (bottom-left) to +105deg (bottom-right)
    start_angle = -105
    span        = 210
    angle_deg   = start_angle + pct * span
    angle_rad   = math.radians(angle_deg)

    cx, cy, r = 100, 100, 78
    needle_len = 62
    nx = cx + needle_len * math.sin(angle_rad)
    ny = cy - needle_len * math.cos(angle_rad)

    # threshold marker angle
    th_pct   = threshold / max_score
    th_angle = math.radians(start_angle + th_pct * span)
    th_x1 = cx + (r-8) * math.sin(th_angle)
    th_y1 = cy - (r-8) * math.cos(th_angle)
    th_x2 = cx + (r+8) * math.sin(th_angle)
    th_y2 = cy - (r+8) * math.cos(th_angle)

    # status color
    if score < threshold:
        color = "#3FA796"
        status_text = "NORMAL"
    elif score < threshold * 1.3:
        color = "#D9A441"
        status_text = "ANOMALY"
    else:
        color = "#E5483D"
        status_text = "ANOMALY"

    # arc path (background track) — describe arc from start to end
    def polar(angle_deg, radius):
        a = math.radians(angle_deg)
        return cx + radius * math.sin(a), cy - radius * math.cos(a)

    x_start, y_start = polar(start_angle, r)
    x_end, y_end     = polar(start_angle + span, r)
    x_val, y_val     = polar(angle_deg, r)

    large_arc = 1 if span > 180 else 0

    return f"""
    <svg viewBox="0 0 200 160" width="100%" style="max-width: 280px; display:block; margin: 0 auto;">
      <!-- background track -->
      <path d="M {x_start:.2f} {y_start:.2f} A {r} {r} 0 {large_arc} 1 {x_end:.2f} {y_end:.2f}"
            fill="none" stroke="#383E4A" stroke-width="10" stroke-linecap="round"/>
      <!-- value arc -->
      <path d="M {x_start:.2f} {y_start:.2f} A {r} {r} 0 0 1 {x_val:.2f} {y_val:.2f}"
            fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
      <!-- threshold tick -->
      <line x1="{th_x1:.2f}" y1="{th_y1:.2f}" x2="{th_x2:.2f}" y2="{th_y2:.2f}"
            stroke="#8A909C" stroke-width="2" stroke-dasharray="2 2"/>
      <!-- needle -->
      <line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}"
            stroke="{color}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="5" fill="{color}"/>
      <!-- score text -->
      <text x="100" y="118" text-anchor="middle"
            font-family="IBM Plex Mono, monospace" font-size="22" font-weight="600"
            fill="{color}">{score:.3f}</text>
      <text x="100" y="138" text-anchor="middle"
            font-family="IBM Plex Mono, monospace" font-size="10" letter-spacing="2"
            fill="#8A909C">{status_text}</text>
    </svg>
    """


# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hmi-title">Inspection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="hmi-heading" style="font-size:20px;">Surface Defect Detection</p>',
                unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Product Category</p>', unsafe_allow_html=True)
    category = st.selectbox(
        "Category", CATEGORIES, index=0,
        label_visibility="collapsed"
    )

    st.markdown('<p class="sidebar-section">Sample Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload", type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed"
    )

    # ── sample image button ───────────────────────────────────────────────
    sample_path = f"samples/{category}.png"
    if not os.path.exists(sample_path):
        sample_path = f"samples/{category}.jpg"

    if os.path.exists(sample_path):
        if st.button("⚡ Try Sample Image", use_container_width=True):
            with open(sample_path, "rb") as f:
                from io import BytesIO
                buf = BytesIO(f.read())
                buf.name = f"{category}.png"
                st.session_state["sample_image"] = buf
                st.session_state["sample_category"] = category

    # load from session if no file uploaded
    if uploaded is None and "sample_image" in st.session_state:
        if st.session_state.get("sample_category") == category:
            uploaded = st.session_state["sample_image"]
        else:
            del st.session_state["sample_image"]

    st.markdown('<hr class="hmi-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section">System Info</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8A909C; line-height:1.8;">
        MODEL&nbsp;&nbsp;&nbsp;&nbsp;PatchCore (WideResNet-50)<br>
        DATASET&nbsp;&nbsp;MVTec AD — 15 categories<br>
        LLM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gemini 2.5 Flash-Lite<br>
        AVG AUROC&nbsp;98.4%
        </div>
        """,
        unsafe_allow_html=True
    )

    run_button = st.button("▸  Run Inspection", use_container_width=True, type="primary")


# ── main panel ────────────────────────────────────────────────────────────
st.markdown('<p class="hmi-title">Surface Quality Control</p>', unsafe_allow_html=True)
st.markdown('<p class="hmi-heading">Defect Detection &amp; Root Cause Analysis</p>',
            unsafe_allow_html=True)

if uploaded is None:
    st.markdown(
        """
        <div class="hmi-panel" style="text-align:center; padding: 60px 20px;">
            <p style="color:#8A909C; font-family:'IBM Plex Mono',monospace; font-size:13px; letter-spacing:0.1em;">
                AWAITING INPUT
            </p>
            <p style="color:#E8E6E1; margin-top:8px;">
                Select a product category and upload a surface image to begin inspection.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

if not run_button:
    st.markdown(
        """
        <div class="hmi-panel" style="text-align:center; padding: 40px 20px;">
            <p style="color:#E8E6E1;">Image loaded. Press <b>Run Inspection</b> to analyze.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(uploaded, width=300)
    st.stop()


# ── run pipeline ──────────────────────────────────────────────────────────
with st.spinner("Running anomaly detection..."):
    # save uploaded file to temp path
    temp_path = "temp_upload.png"
    if hasattr(uploaded, 'seek'):
        uploaded.seek(0)  # reset buffer for BytesIO (sample images)
    img = Image.open(uploaded).convert("RGB")
    img.save(temp_path)

    result = run_pipeline(
        image_path = temp_path,
        category   = category,
    )

if result["status"] == "error":
    st.error(result["message"])
    st.stop()


# ── results layout ───────────────────────────────────────────────────────
score     = result["anomaly_score"]
threshold = result["threshold"]
is_anomaly = result["status"] == "anomaly"

col_left, col_right = st.columns([2, 1])

# ── left: image comparison ───────────────────────────────────────────────
with col_left:
    with st.container(border=True):
        st.markdown('<p class="hmi-panel-label">Visual Inspection</p>', unsafe_allow_html=True)

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(result["image_array"], use_container_width=True)
            st.markdown('<p class="img-caption">Original Surface</p>', unsafe_allow_html=True)
        with img_col2:
            if result.get("overlay") is not None:
                st.image(result["overlay"], use_container_width=True)
                st.markdown('<p class="img-caption">Anomaly Heatmap</p>', unsafe_allow_html=True)
            else:
                from llm.explainer import burn_heatmap_on_image
                overlay = burn_heatmap_on_image(result["image_array"], result["heatmap"])
                st.image(overlay, use_container_width=True)
                st.markdown('<p class="img-caption">Anomaly Heatmap</p>', unsafe_allow_html=True)

    # ── defect report (only if anomaly) ──────────────────────────────────
    if is_anomaly:
        if result.get("llm_failed") or result.get("report") is None:
            st.warning(
                "⚠️ Gemini server is currently unavailable. "
                "Showing anomaly detection results only — defect report unavailable."
            )
        else:
            report = result["report"]
            sev    = report.get("severity", "medium").lower()

            with st.container(border=True):
                st.markdown('<p class="hmi-panel-label">Defect Report — Root Cause Analysis</p>',
                            unsafe_allow_html=True)

                fields = [
                    ("Defect Class", report.get("defect_class", "—")),
                    ("Location", report.get("location", "—")),
                    ("Description", report.get("description", "—")),
                    ("Root Cause", report.get("root_cause", "—")),
                    ("Severity", f'<span class="severity-pill severity-{sev}">{sev}</span> '
                                 f'— {report.get("severity_reason","")}'),
                    ("Recommended Fix", report.get("recommended_fix", "—")),
                    ("Confidence", report.get("confidence", "—")),
                ]

                rows_html = "".join(
                    f'<div class="report-field">'
                    f'<div class="report-label">{label}</div>'
                    f'<div class="report-value">{value}</div>'
                    f'</div>'
                    for label, value in fields
                )
                st.markdown(rows_html, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<p class="hmi-panel-label">Defect Report — Root Cause Analysis</p>',
                        unsafe_allow_html=True)

            fields = [
                ("Defect Class", report.get("defect_class", "—")),
                ("Location", report.get("location", "—")),
                ("Description", report.get("description", "—")),
                ("Root Cause", report.get("root_cause", "—")),
                ("Severity", f'<span class="severity-pill severity-{sev}">{sev}</span> '
                             f'— {report.get("severity_reason","")}'),
                ("Recommended Fix", report.get("recommended_fix", "—")),
                ("Confidence", report.get("confidence", "—")),
            ]

            rows_html = "".join(
                f'<div class="report-field">'
                f'<div class="report-label">{label}</div>'
                f'<div class="report-value">{value}</div>'
                f'</div>'
                for label, value in fields
            )
            st.markdown(rows_html, unsafe_allow_html=True)


# ── right: gauge + status ────────────────────────────────────────────────
with col_right:
    with st.container(border=True):
        st.markdown('<p class="hmi-panel-label">Anomaly Score</p>', unsafe_allow_html=True)

        st.markdown(render_gauge(score, max_score=6.0, threshold=threshold),
                    unsafe_allow_html=True)

        if is_anomaly:
            status_html = '<span class="status-fail"><span class="status-dot"></span>DEFECT DETECTED</span>'
        else:
            status_html = '<span class="status-pass"><span class="status-dot"></span>SURFACE NORMAL</span>'

        st.markdown(
            f'<div style="text-align:center; margin-top: 12px;">{status_html}</div>',
            unsafe_allow_html=True
        )

    # readout panel
    with st.container(border=True):
        st.markdown('<p class="hmi-panel-label">Readout</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="report-field">
                <div class="report-label">Category</div>
                <div class="report-value" style="font-family:'IBM Plex Mono',monospace;">{category}</div>
            </div>
            <div class="report-field">
                <div class="report-label">Score</div>
                <div class="report-value" style="font-family:'IBM Plex Mono',monospace;">{score:.4f}</div>
            </div>
            <div class="report-field">
                <div class="report-label">Threshold</div>
                <div class="report-value" style="font-family:'IBM Plex Mono',monospace;">{threshold:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not is_anomaly:
        with st.container(border=True):
            st.markdown(
                '<p style="color:#8A909C; font-size:13px;">'
                'No anomaly detected. LLM analysis was skipped — '
                'root cause analysis only runs on flagged images to minimize API cost.'
                '</p>',
                unsafe_allow_html=True
            )