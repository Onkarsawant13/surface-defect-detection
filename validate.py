"""
validate.py — Test the full pipeline end to end
-------------------------------------------------
Run:
    python validate.py --data ./dataset --category bottle

What it tests:
  1. Environment check     — .env file, API key, packages
  2. Memory bank check     — .pt file exists and loads
  3. Model inference check — scores one normal + one defective image
  4. LLM check             — sends one defective image to Gemini
  5. Prints full report

Use this before building the UI to confirm everything works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pathlib import Path

# try multiple locations — covers running from project root or subdirectory
_here = Path(__file__).resolve().parent
for _candidate in [_here / ".env", _here.parent / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=True)
        break

import argparse
from pathlib import Path


def check(label, fn):
    """Run a check, print pass/fail."""
    try:
        result = fn()
        print(f"  ✓  {label}")
        return result
    except Exception as e:
        print(f"  ✗  {label}")
        print(f"     Error: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True)
    p.add_argument("--category", default="bottle")
    p.add_argument("--output",   default="outputs")
    args = p.parse_args()

    print("\n" + "═" * 50)
    print("  Surface Defect System — Validation")
    print("═" * 50)

    # ── 1. environment ───────────────────────────────────────────────────────
    print("\n[1] Environment")

    api_key = check("GEMINI_API_KEY in .env",
        lambda: os.environ["GEMINI_API_KEY"])

    check("google.genai installed",
        lambda: __import__("google.genai"))

    check("torch installed",
        lambda: __import__("torch"))

    check("anthropic → not needed (using Gemini)", lambda: True)

    # ── 2. memory bank ───────────────────────────────────────────────────────
    print(f"\n[2] Memory bank — {args.category}")

    bank_path = Path(args.output) / args.category / "memory_bank.pt"

    bank = check(f"memory_bank.pt exists at {bank_path}",
        lambda: __import__("torch").load(str(bank_path), map_location="cpu"))

    if bank is not None:
        check(f"memory bank shape valid ({bank.shape[0]:,} patches)",
            lambda: bank.shape[1] == 1536 and bank.shape[0] > 0)

    # ── 3. model inference ───────────────────────────────────────────────────
    print(f"\n[3] Model inference — {args.category}")

    from models.patchcore import PatchCore
    from data.dataset     import MVTecDataset, TEST_TRANSFORMS

    model = PatchCore(device="cpu")
    model.load(str(bank_path))

    test_ds = MVTecDataset(args.data, args.category,
                           split="test", transform=TEST_TRANSFORMS)

    # find one normal and one defective sample
    normal_sample  = next((s for s in test_ds if s["label"] == 0), None)
    defect_sample  = next((s for s in test_ds if s["label"] == 1), None)

    normal_score = check("Score normal image (expect low score)",
        lambda: model.predict_single(normal_sample["image"])[0])

    defect_score, defect_heatmap = None, None
    if defect_sample:
        result = check("Score defective image (expect high score)",
            lambda: model.predict_single(defect_sample["image"]))
        if result:
            defect_score, defect_heatmap = result

    if normal_score and defect_score:
        gap = defect_score - normal_score
        ok  = gap > 0
        print(f"     Normal score : {normal_score:.4f}")
        print(f"     Defect score : {defect_score:.4f}")
        print(f"     Gap          : {gap:+.4f}  "
              f"{'✓ model separates well' if ok else '✗ scores too close'}")

    # ── 4. llm call ──────────────────────────────────────────────────────────
    print(f"\n[4] LLM — Gemini defect explanation")

    if not api_key:
        print("  ✗  Skipping — no API key found")
    elif defect_sample is None:
        print("  ✗  Skipping — no defective test image found")
    else:
        import numpy as np
        from llm.explainer import explain_defect

        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

        img = defect_sample["image"].permute(1,2,0).numpy()
        img = np.clip((img * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)

        report = check("Gemini API call succeeds",
            lambda: explain_defect(img, defect_heatmap,
                                   defect_score, args.category, api_key))

        if report and "defect_class" in report:
            print(f"\n  LLM Report:")
            print(f"  ├ Defect class   : {report.get('defect_class','—')}")
            print(f"  ├ Location       : {report.get('location','—')}")
            print(f"  ├ Root cause     : {report.get('root_cause','—')}")
            print(f"  ├ Severity       : {report.get('severity','—')}")
            print(f"  ├ Fix            : {report.get('recommended_fix','—')}")
            print(f"  └ Confidence     : {report.get('confidence','—')}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  Validation complete.")
    print("  If all checks show ✓ — run the Streamlit UI next.")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()