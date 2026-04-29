"""
Threshold Comparison Test — Senior Project Dataset
====================================================
วิธีการทดสอบ:
  1. นำรูปภาพในชุดข้อมูลทั้งหมดส่งผ่าน API /api/analyze
  2. บันทึกค่าความน่าจะเป็น AI ของแต่ละรูป
  3. ทดลองใช้ Threshold หลายค่า ได้แก่ 20%, 30%, 40%, 50%, 60%, 70%, 80%
  4. คำนวณ Metrics เปรียบเทียบแต่ละค่า Threshold

กฎการจำแนก Dataset:
  • ชื่อไฟล์มี "_T" หรือ "TT" → Real
  • อื่น ๆ                    → AI-generated
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
DATASET_DIR = Path(os.environ.get("DATASET_DIR", r"backend\Dataset Senior Project"))
TIMEOUT = 180.0
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.0  # seconds between requests (0 = ไม่หน่วง)

THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]  # % ค่า Threshold ที่จะทดลอง

# ---------------------------------------------------------------------------
# Dataset label helpers
# ---------------------------------------------------------------------------

def _is_expected_real(filename: str) -> bool:
    """กฎการจำแนก: '_T' หรือ 'TT' ในชื่อไฟล์ → Real, อื่น ๆ → AI"""
    name = filename.lower()
    # "_t" as a token boundary  (e.g. 0082_T.jpg, 116356_0_T.jpg)
    if re.search(r"(?:^|[^a-z0-9])_t(?:[^a-z0-9]|$)", name):
        return True
    if name.endswith(("_t.jpg", "_t.jpeg", "_t.png")):
        return True
    # "TT" at the start of filename (e.g. TT (1).jpg, TT (2).png)
    if name.startswith("tt"):
        return True
    return False


def _iter_images(dataset_dir: Path):
    """ค้นหารูปภาพทั้งหมดในโฟลเดอร์ Dataset"""
    exts = {".jpg", ".jpeg", ".png"}
    if not dataset_dir.exists():
        return
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _call_api(session, image_path: Path, url: str = None, timeout: float = None) -> dict | None:
    """ส่งรูปภาพไปยัง /api/analyze แล้วคืน response dict (หรือ None ถ้าล้มเหลว)"""
    base = (url or BACKEND_URL).rstrip('/')
    req_timeout = timeout or TIMEOUT
    endpoint = f"{base}/api/analyze"
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "application/octet-stream")}
                r = session.post(endpoint, files=files, timeout=req_timeout)
            elapsed = time.time() - t0

            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                if attempt < MAX_RETRIES:
                    backoff = min(10.0, 1.5 * attempt)
                    print(f"    ⚠ attempt {attempt}/{MAX_RETRIES} → {last_err} (retry in {backoff:.1f}s)")
                    time.sleep(backoff)
                    continue

            if r.status_code != 200:
                return None

            payload = r.json()
            payload["_elapsed"] = elapsed
            return payload

        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                backoff = min(10.0, 1.5 * attempt)
                print(f"    ⚠ attempt {attempt}/{MAX_RETRIES} → {last_err} (retry in {backoff:.1f}s)")
                time.sleep(backoff)
                continue

    print(f"    ✗ ล้มเหลว: {last_err}")
    return None


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def _compute_metrics(records: list[dict], threshold: float) -> dict:
    """
    คำนวณ Confusion Matrix + Metrics สำหรับ threshold ที่กำหนด
    
    Positive = AI,  Negative = Real
    TP = คาดว่า AI & ทำนาย AI       (ถูกต้อง)
    TN = คาดว่า Real & ทำนาย Real   (ถูกต้อง)
    FP = คาดว่า Real & ทำนาย AI     (False Alarm)
    FN = คาดว่า AI & ทำนาย Real     (Miss)
    """
    tp = tn = fp = fn = 0

    for rec in records:
        ai_prob = rec["ai_probability"]
        expected_ai = rec["expected_ai"]
        predicted_ai = ai_prob >= threshold

        if expected_ai and predicted_ai:
            tp += 1
        elif not expected_ai and not predicted_ai:
            tn += 1
        elif not expected_ai and predicted_ai:
            fp += 1
        else:  # expected_ai and not predicted_ai
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0      # Sensitivity / TPR
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0  # TNR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _print_threshold_table(all_metrics: list[dict]):
    """แสดงตารางเปรียบเทียบ Threshold แบบ ASCII"""
    header = f"{'Threshold':>10} │ {'Acc(%)':>8} │ {'Prec(%)':>8} │ {'Recall(%)':>9} │ {'Spec(%)':>8} │ {'F1(%)':>8} │ {'TP':>4} │ {'TN':>4} │ {'FP':>4} │ {'FN':>4}"
    sep    = "─" * len(header)

    print(sep)
    print(header)
    print(sep)
    for m in all_metrics:
        print(
            f"{m['threshold']:>9.0f}% │ "
            f"{m['accuracy']:>7.2f}% │ "
            f"{m['precision']:>7.2f}% │ "
            f"{m['recall']:>8.2f}% │ "
            f"{m['specificity']:>7.2f}% │ "
            f"{m['f1']:>7.2f}% │ "
            f"{m['tp']:>4} │ {m['tn']:>4} │ {m['fp']:>4} │ {m['fn']:>4}"
        )
    print(sep)

    # หา Threshold ที่ F1 สูงสุด
    best = max(all_metrics, key=lambda x: x["f1"])
    print(f"\n🏆 Threshold ที่ดีที่สุด (F1 สูงสุด): {best['threshold']:.0f}%  →  F1 = {best['f1']:.2f}%")

    # หา Threshold ที่ Accuracy สูงสุด
    best_acc = max(all_metrics, key=lambda x: x["accuracy"])
    print(f"🎯 Threshold ที่ Accuracy สูงสุด:     {best_acc['threshold']:.0f}%  →  Accuracy = {best_acc['accuracy']:.2f}%")


def _save_raw_csv(records: list[dict], out_path: Path):
    """บันทึกผลลัพธ์ดิบ (ต่อรูป) เป็น CSV"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename", "expected_label", "expected_ai",
        "ai_probability", "elapsed_seconds",
        "ela_prob", "pixel_prob", "frequency_prob", "xception_prob",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in fieldnames})
    print(f"📁 Raw CSV saved → {out_path}")


def _save_threshold_csv(all_metrics: list[dict], out_path: Path):
    """บันทึกตาราง Threshold Comparison เป็น CSV"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["threshold", "accuracy", "precision", "recall", "specificity", "f1", "tp", "tn", "fp", "fn"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in all_metrics:
            w.writerow(m)
    print(f"📁 Threshold CSV saved → {out_path}")


def _save_markdown_report(records: list[dict], all_metrics: list[dict], out_path: Path,
                          total_images: int, real_count: int, ai_count: int,
                          success_count: int, fail_count: int, total_time: float):
    """สร้างรายงาน Markdown"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = max(all_metrics, key=lambda x: x["f1"])
    best_acc = max(all_metrics, key=lambda x: x["accuracy"])

    lines = []
    lines.append("# Threshold Comparison Report")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## 1. Dataset Summary")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Total images | {total_images} |")
    lines.append(f"| Real images (Ground Truth) | {real_count} |")
    lines.append(f"| AI images (Ground Truth) | {ai_count} |")
    lines.append(f"| API success | {success_count} |")
    lines.append(f"| API fail | {fail_count} |")
    lines.append(f"| Total analysis time | {total_time:.1f}s |")
    lines.append(f"| Avg time per image | {total_time/max(success_count,1):.2f}s |")
    lines.append("")

    lines.append("## 2. Threshold Comparison")
    lines.append("")
    lines.append("| Threshold | Accuracy | Precision | Recall | Specificity | F1-Score | TP | TN | FP | FN |")
    lines.append("|:---------:|:--------:|:---------:|:------:|:-----------:|:--------:|:--:|:--:|:--:|:--:|")
    for m in all_metrics:
        marker = " ⭐" if m["threshold"] == best_f1["threshold"] else ""
        lines.append(
            f"| {m['threshold']:.0f}% | {m['accuracy']:.2f}% | {m['precision']:.2f}% | "
            f"{m['recall']:.2f}% | {m['specificity']:.2f}% | {m['f1']:.2f}%{marker} | "
            f"{m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} |"
        )
    lines.append("")

    lines.append("## 3. Best Threshold")
    lines.append(f"- **Best F1-Score:** Threshold = **{best_f1['threshold']:.0f}%** → F1 = {best_f1['f1']:.2f}%")
    lines.append(f"- **Best Accuracy:** Threshold = **{best_acc['threshold']:.0f}%** → Accuracy = {best_acc['accuracy']:.2f}%")
    lines.append("")

    lines.append("## 4. Metrics Definitions")
    lines.append("| Metric | Formula | Description |")
    lines.append("|--------|---------|-------------|")
    lines.append("| Accuracy | (TP+TN)/(TP+TN+FP+FN) | สัดส่วนการทำนายถูกต้องทั้งหมด |")
    lines.append("| Precision | TP/(TP+FP) | สัดส่วนที่ทำนายว่า AI แล้วถูกจริง |")
    lines.append("| Recall (Sensitivity) | TP/(TP+FN) | สัดส่วนภาพ AI ที่ถูกตรวจจับได้ |")
    lines.append("| Specificity | TN/(TN+FP) | สัดส่วนภาพ Real ที่ถูกตรวจจับได้ |")
    lines.append("| F1-Score | 2×Precision×Recall/(Precision+Recall) | ค่าเฉลี่ยฮาร์มอนิกของ Precision & Recall |")
    lines.append("")

    lines.append("## 5. Classification Rules")
    lines.append("- **Positive (AI):** ภาพที่ไม่มี `_T` หรือ `TT` ในชื่อไฟล์")
    lines.append("- **Negative (Real):** ภาพที่มี `_T` หรือ `TT` ในชื่อไฟล์")
    lines.append("- **Predicted AI:** `ensemble.confidence >= threshold`")
    lines.append("- **Predicted Real:** `ensemble.confidence < threshold`")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📁 Markdown report saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Threshold Comparison Test — Senior Project Dataset")
    ap.add_argument("--url", default=BACKEND_URL, help="Backend base URL")
    ap.add_argument("--dataset", default=str(DATASET_DIR), help="Dataset directory")
    ap.add_argument("--limit", type=int, default=0, help="Limit images (0 = all)")
    ap.add_argument("--timeout", type=float, default=TIMEOUT, help="Request timeout (seconds)")
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN, help="Sleep between requests (seconds)")
    ap.add_argument("--out-dir", default="", help="Output directory for reports")
    args = ap.parse_args()

    # Use args values as local config
    backend_url = args.url
    timeout = args.timeout
    sleep_between = args.sleep

    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is required. Install with: pip install requests", file=sys.stderr)
        sys.exit(1)

    dataset_dir = Path(args.dataset).expanduser()
    if not dataset_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ Dataset: {dataset_dir}", file=sys.stderr)
        sys.exit(2)

    all_images = list(_iter_images(dataset_dir))
    if args.limit and args.limit > 0:
        all_images = all_images[:args.limit]

    if not all_images:
        print(f"❌ ไม่พบรูปภาพในโฟลเดอร์: {dataset_dir}", file=sys.stderr)
        sys.exit(3)

    real_images = [img for img in all_images if _is_expected_real(img.name)]
    ai_images   = [img for img in all_images if not _is_expected_real(img.name)]

    # ─── Header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🔬 Threshold Comparison Test — Senior Project Dataset")
    print("=" * 80)
    print(f"  Backend URL  : {backend_url}")
    print(f"  Dataset      : {dataset_dir}")
    print(f"  Total images : {len(all_images)}")
    print(f"  Real (GT)    : {len(real_images)}")
    print(f"  AI (GT)      : {len(ai_images)}")
    print(f"  Thresholds   : {', '.join(f'{t}%' for t in THRESHOLDS)}")
    print("=" * 80 + "\n")

    # ─── Phase 1: ส่งรูปภาพทุกรูปผ่าน API ─────────────────────────────
    print("📤 Phase 1: ส่งรูปภาพผ่าน API /api/analyze ...")
    print("─" * 60)

    session = requests.Session()
    records = []
    success_count = 0
    fail_count = 0
    total_time = 0.0

    for idx, img_path in enumerate(all_images, start=1):
        expected_real = _is_expected_real(img_path.name)
        expected_ai = not expected_real
        expected_label = "Real" if expected_real else "AI"

        print(f"  [{idx:>3}/{len(all_images)}] {img_path.name} (Expected: {expected_label})", end=" ")

        payload = _call_api(session, img_path, url=backend_url, timeout=timeout)

        if payload is None:
            print("→ ✗ FAILED")
            fail_count += 1
            continue

        elapsed = payload.get("_elapsed", 0.0)
        total_time += elapsed

        # ดึงค่า ensemble confidence (= AI probability %)
        ensemble = payload.get("ensemble", payload.get("result", {}))
        ai_prob = float(ensemble.get("confidence", 0.0))

        # ดึงค่าแต่ละโมเดล
        models = payload.get("models", {})
        ela_prob = models.get("ela", {}).get("confidence", "")
        pixel_prob = models.get("pixel", {}).get("confidence", "")
        freq_prob = models.get("frequency", {}).get("confidence", "")
        xcep_prob = models.get("xception", {}).get("confidence", "")

        record = {
            "filename": img_path.name,
            "expected_label": expected_label,
            "expected_ai": expected_ai,
            "ai_probability": ai_prob,
            "elapsed_seconds": f"{elapsed:.4f}",
            "ela_prob": f"{ela_prob:.2f}" if isinstance(ela_prob, (int, float)) else str(ela_prob),
            "pixel_prob": f"{pixel_prob:.2f}" if isinstance(pixel_prob, (int, float)) else str(pixel_prob),
            "frequency_prob": f"{freq_prob:.2f}" if isinstance(freq_prob, (int, float)) else str(freq_prob),
            "xception_prob": f"{xcep_prob:.2f}" if isinstance(xcep_prob, (int, float)) else str(xcep_prob),
        }
        records.append(record)
        success_count += 1

        print(f"→ AI prob = {ai_prob:.2f}% | {elapsed:.1f}s")

        if sleep_between > 0:
            time.sleep(sleep_between)

    print("─" * 60)
    print(f"✅ Phase 1 เสร็จสิ้น: สำเร็จ {success_count}/{len(all_images)} | ล้มเหลว {fail_count}")
    print(f"⏱  เวลารวม: {total_time:.1f}s | เฉลี่ย: {total_time / max(success_count, 1):.2f}s/รูป\n")

    if not records:
        print("❌ ไม่มีผลลัพธ์ให้วิเคราะห์")
        sys.exit(4)

    # ─── Phase 2: AI Probability Distribution ──────────────────────────
    print("📊 Phase 2: การกระจายค่า AI Probability")
    print("─" * 60)

    real_probs = [r["ai_probability"] for r in records if not r["expected_ai"]]
    ai_probs   = [r["ai_probability"] for r in records if r["expected_ai"]]

    if real_probs:
        print(f"  Real images ({len(real_probs)} รูป):")
        print(f"    Min = {min(real_probs):.2f}%  |  Max = {max(real_probs):.2f}%  |  Avg = {sum(real_probs)/len(real_probs):.2f}%")
    if ai_probs:
        print(f"  AI images ({len(ai_probs)} รูป):")
        print(f"    Min = {min(ai_probs):.2f}%  |  Max = {max(ai_probs):.2f}%  |  Avg = {sum(ai_probs)/len(ai_probs):.2f}%")
    print()

    # ─── Phase 3: Threshold Comparison ─────────────────────────────────
    print("📐 Phase 3: เปรียบเทียบ Threshold")
    print("─" * 60)

    all_metrics = []
    for thr in THRESHOLDS:
        metrics = _compute_metrics(records, thr)
        all_metrics.append(metrics)

    _print_threshold_table(all_metrics)
    print()

    # ─── Phase 4: Save reports ─────────────────────────────────────────
    print("💾 Phase 4: บันทึกรายงาน")
    print("─" * 60)

    if args.out_dir:
        report_dir = Path(args.out_dir)
    else:
        report_dir = Path(r"C:\temp\uat-test\reports")

    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    _save_raw_csv(records, report_dir / f"threshold_raw_results_{ts}.csv")
    _save_threshold_csv(all_metrics, report_dir / f"threshold_comparison_{ts}.csv")
    _save_markdown_report(
        records, all_metrics,
        report_dir / f"threshold_report_{ts}.md",
        total_images=len(all_images),
        real_count=len(real_images),
        ai_count=len(ai_images),
        success_count=success_count,
        fail_count=fail_count,
        total_time=total_time,
    )

    # ─── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✅ เสร็จสิ้น — Threshold Comparison Test")
    print("=" * 80)


if __name__ == "__main__":
    main()
