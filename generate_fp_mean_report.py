"""
Generate Mean AI Score (Real images) and False Positive risk table.

What this script does
- Sends every image in the dataset to POST /api/analyze
- Extracts per-model confidence (AI probability %) from result.individual_models
- Computes:
  - Mean AI score on REAL images (ground truth via filename rule: '_T' token or startswith 'TT')
  - False Positive Rate on REAL images at a chosen threshold (default 50%)

Outputs
- fp_mean_report.csv
- fp_mean_report.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASET_DEFAULT = r"backend\Dataset Senior Project"
BACKEND_DEFAULT = "http://localhost:8001"


def _is_real_image(filename: str) -> bool:
    """Ground truth rule used in this project.

    Real images:
    - contains '_T' as a token boundary (e.g. 0082_T.jpg, 116356_0_T.jpg)
    - OR starts with 'TT' (e.g. TT (1).jpg)
    """
    name = filename.lower()
    if re.search(r"(?:^|[^a-z0-9])_t(?:[^a-z0-9]|$)", name):
        return True
    if name.endswith(("_t.jpg", "_t.jpeg", "_t.png")):
        return True
    if name.startswith("tt"):
        return True
    return False


def _iter_images(dataset_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _risk_label(fp_rate_pct: float) -> str:
    """Map FP rate to a human-readable risk label."""
    # These bands are adjustable; chosen for report readability.
    if fp_rate_pct < 5.0:
        return "ต่ำ"
    if fp_rate_pct < 15.0:
        return "ปานกลาง"
    return "สูง"


@dataclass
class Record:
    filename: str
    is_real: bool
    # Confidence values are AI probabilities in percent (0..100)
    ela: Optional[float] = None
    pixel: Optional[float] = None
    frequency: Optional[float] = None
    xception: Optional[float] = None
    ensemble: Optional[float] = None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_confidences(payload: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Return (ensemble_confidence, per_model_confidence)."""
    result = payload.get("result") or {}
    ensemble_conf = _safe_float(result.get("confidence"))
    models = result.get("individual_models") or {}

    per_model = {
        "ela": _safe_float((models.get("ela") or {}).get("confidence")),
        "pixel": _safe_float((models.get("pixel") or {}).get("confidence")),
        "frequency": _safe_float((models.get("frequency") or {}).get("confidence")),
        "xception": _safe_float((models.get("xception") or {}).get("confidence")),
    }
    return ensemble_conf, per_model


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Mean AI Score + False Positive risk tables.")
    ap.add_argument("--url", default=os.environ.get("BACKEND_URL", BACKEND_DEFAULT), help="Backend base URL")
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", DATASET_DEFAULT), help="Dataset directory")
    # argparse does %-formatting on help strings in some Python builds; escape literal % as %%
    ap.add_argument("--threshold", type=float, default=50.0, help="AI threshold (%%) for FP on real images")
    ap.add_argument("--timeout", type=float, default=180.0, help="Request timeout (seconds)")
    ap.add_argument("--retries", type=int, default=2, help="Retries per image")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0=all)")
    ap.add_argument("--out-prefix", default="fp_mean_report", help="Output prefix (csv/md)")
    ap.add_argument("--save-raw", action="store_true", help="Also save per-image raw confidences to <out-prefix>_raw.csv")
    ap.add_argument("--only-real", action="store_true", help="Process only REAL images (per filename rule)")
    ap.add_argument("--only-ai", action="store_true", help="Process only AI images (per filename rule)")
    args = ap.parse_args()

    try:
        import requests  # type: ignore
    except Exception:
        print("ERROR: 'requests' is required. Install with: pip install requests", file=sys.stderr)
        return 1

    dataset_dir = Path(args.dataset).expanduser()
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    images = list(_iter_images(dataset_dir))

    if args.only_real and args.only_ai:
        print("ERROR: Choose only one of --only-real or --only-ai", file=sys.stderr)
        return 2

    if args.only_real:
        images = [p for p in images if _is_real_image(p.name)]
    elif args.only_ai:
        images = [p for p in images if not _is_real_image(p.name)]

    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: no images found under: {dataset_dir}", file=sys.stderr)
        return 3

    base = args.url.rstrip("/")
    endpoint = f"{base}/api/analyze"

    print("Generating report from /api/analyze")
    print(f"Backend : {base}")
    print(f"Dataset : {dataset_dir}")
    print(f"Images  : {len(images)}")
    print(f"FP thr  : {args.threshold:.2f}%")
    print("")

    session = requests.Session()
    records: List[Record] = []
    ok = 0
    fail = 0

    for idx, img_path in enumerate(images, start=1):
        is_real = _is_real_image(img_path.name)
        label = "REAL" if is_real else "AI"
        print(f"[{idx:>3}/{len(images)}] {label} {img_path.name}", end=" ")

        payload = None
        last_err = None
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                t0 = time.time()
                with open(img_path, "rb") as f:
                    files = {"image": (img_path.name, f, "application/octet-stream")}
                    r = session.post(endpoint, files=files, timeout=args.timeout)
                _ = time.time() - t0
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    payload = None
                else:
                    payload = r.json()
                if payload is not None:
                    break
            except Exception as e:
                last_err = str(e)
                payload = None
            if attempt < args.retries:
                time.sleep(min(2.0, 0.5 * attempt))

        if not payload or not payload.get("success"):
            print(f"-> FAIL ({last_err or 'success=false'})")
            fail += 1
            continue

        ensemble_conf, per_model = _extract_confidences(payload)
        rec = Record(
            filename=img_path.name,
            is_real=is_real,
            ela=per_model["ela"],
            pixel=per_model["pixel"],
            frequency=per_model["frequency"],
            xception=per_model["xception"],
            ensemble=ensemble_conf,
        )
        records.append(rec)
        ok += 1
        print("-> OK")

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    if not records:
        print("ERROR: no successful records to analyze", file=sys.stderr)
        return 4

    real_recs = [r for r in records if r.is_real]
    if not real_recs:
        print("ERROR: no REAL images found by filename rule", file=sys.stderr)
        return 5

    def real_vals(field: str) -> List[float]:
        out: List[float] = []
        for r in real_recs:
            v = getattr(r, field)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    # Optional: save raw per-image confidences for auditability.
    if args.save_raw:
        raw_path = Path(f"{args.out_prefix}_raw.csv")
        with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "is_real",
                    "ensemble",
                    "frequency",
                    "ela",
                    "xception",
                    "pixel",
                ],
            )
            w.writeheader()
            for r in records:
                w.writerow(
                    {
                        "filename": r.filename,
                        "is_real": "Real" if r.is_real else "AI",
                        "ensemble": f"{r.ensemble:.6f}" if isinstance(r.ensemble, (int, float)) else "",
                        "frequency": f"{r.frequency:.6f}" if isinstance(r.frequency, (int, float)) else "",
                        "ela": f"{r.ela:.6f}" if isinstance(r.ela, (int, float)) else "",
                        "xception": f"{r.xception:.6f}" if isinstance(r.xception, (int, float)) else "",
                        "pixel": f"{r.pixel:.6f}" if isinstance(r.pixel, (int, float)) else "",
                    }
                )
        print(f"Saved raw per-image file: {raw_path}")

    models = [
        ("frequency", "Frequency"),
        ("ela", "ELA"),
        ("xception", "CNN"),
        ("pixel", "Pixel-level"),
    ]

    summary_rows = []
    for key, label in models:
        vals = real_vals(key)
        mean_ai = _mean(vals)
        fp_count = sum(1 for v in vals if v >= args.threshold)
        fp_rate = (fp_count / len(vals) * 100.0) if vals else 0.0
        summary_rows.append(
            {
                "model_key": key,
                "model": label,
                "mean_ai_score_real": mean_ai,
                "false_positive_rate_real": fp_rate,
                "false_positive_risk": _risk_label(fp_rate),
                "real_samples_used": len(vals),
                "real_sum": sum(vals),
            }
        )

    out_csv = Path(f"{args.out_prefix}.csv")
    out_md = Path(f"{args.out_prefix}.md")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "mean_ai_score_real",
                "false_positive_rate_real",
                "false_positive_risk",
                "real_samples_used",
                "threshold",
                "total_images_ok",
                "total_images_fail",
                "generated_at",
            ],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow(
                {
                    "model": row["model"],
                    "mean_ai_score_real": f"{row['mean_ai_score_real']:.2f}",
                    "false_positive_rate_real": f"{row['false_positive_rate_real']:.2f}",
                    "false_positive_risk": row["false_positive_risk"],
                    "real_samples_used": row["real_samples_used"],
                    "threshold": f"{args.threshold:.2f}",
                    "total_images_ok": ok,
                    "total_images_fail": fail,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    lines: List[str] = []
    lines.append("## Mean AI Score (ภาพจริง) และความเสี่ยง False Positive")
    lines.append("")
    lines.append(f"- **Dataset**: `{dataset_dir}`")
    lines.append(f"- **Backend**: `{base}`")
    lines.append(f"- **Threshold (FP on Real)**: **{args.threshold:.2f}%**")
    lines.append(f"- **API success**: {ok} | **fail**: {fail}")
    lines.append("")
    lines.append("| โมเดล | Mean AI Score (ภาพจริง) | False Positive (ภาพจริง) | ความเสี่ยง False Positive |")
    lines.append("|---|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['mean_ai_score_real']:.2f}% | {row['false_positive_rate_real']:.2f}% | {row['false_positive_risk']} |"
        )
    lines.append("")
    lines.append("หมายเหตุ:")
    lines.append("- Mean AI Score (ภาพจริง) = ค่าเฉลี่ยของ `confidence` (AI probability %) ของโมเดลนั้น เฉพาะรูปที่เป็น Real")
    lines.append("- False Positive (ภาพจริง) = สัดส่วนรูป Real ที่โมเดลให้ `confidence >= threshold`")
    lines.append("- ระดับความเสี่ยง: ต่ำ (<5%), ปานกลาง (5–<15%), สูง (>=15%)")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("")
    # Print audit numbers for Frequency mean (most commonly cited).
    freq_row = next((r for r in summary_rows if r["model_key"] == "frequency"), None)
    if freq_row:
        n = freq_row["real_samples_used"]
        s = freq_row["real_sum"]
        mean = freq_row["mean_ai_score_real"]
        print(f"Audit (Frequency, Real): sum={s:.6f} over N={n} => mean={mean:.2f}%")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

