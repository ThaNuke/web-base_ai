import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


def _guess_expected_ai(filename: str) -> bool:
    """
    Dataset rule (per user):
    - If filename contains "_T" (suffix marker) OR contains "TT" => Real
    - Otherwise => AI
    Returns True if expected AI, False if expected Real.
    """
    name = filename.lower()

    # "_t" marker as a token in the basename (e.g. 0082_T.jpg)
    if re.search(r"(?:^|[^a-z0-9])_t(?:[^a-z0-9]|$)", name) or name.endswith(("_t.jpg", "_t.jpeg", "_t.png")):
        return False

    # "TT" at the start of filename (e.g. TT (1).jpg)
    if name.startswith("tt"):
        return False

    return True


def _iter_images(dataset_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser(description="TruPic dataset tester (Windows-friendly).")
    ap.add_argument("--url", default=os.environ.get("BACKEND_URL", "http://localhost:8001"), help="Backend base URL")
    ap.add_argument(
        "--dataset",
        default=os.environ.get("DATASET_DIR", r"backend\Dataset Senior Project"),
        help="Dataset directory path",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    ap.add_argument("--timeout", type=float, default=180.0, help="Request timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="Retries per image")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between images")
    ap.add_argument("--out", default="", help="Output CSV path (default: C:\\temp\\uat-test\\reports\\...)")
    args = ap.parse_args()

    try:
        import requests  # type: ignore
    except Exception:
        print("ERROR: 'requests' is required. Install with: pip install requests", file=sys.stderr)
        sys.exit(1)
    try:
        from requests import exceptions as req_exc  # type: ignore
    except Exception:  # pragma: no cover
        req_exc = None

    base = args.url.rstrip("/")
    endpoint = f"{base}/api/analyze"

    dataset_dir = Path(args.dataset).expanduser()
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(2)

    images = list(_iter_images(dataset_dir))
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"ERROR: no images found in: {dataset_dir}", file=sys.stderr)
        sys.exit(3)

    if args.out:
        out_path = Path(args.out)
    else:
        report_dir = Path(r"C:\temp\uat-test\reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = report_dir / f"Dataset_Test_Results_{ts}.csv"

    print("=" * 48)
    print("TruPic Dataset Performance Test (Python)")
    print("=" * 48)
    print(f"Backend URL : {base}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Images      : {len(images)}")
    print(f"Output CSV  : {out_path}")
    print("")

    results = []
    ai_count = 0
    real_count = 0
    success_count = 0
    fail_count = 0
    total_time = 0.0

    s = requests.Session()

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] Testing: {img_path.name}")

        expected_ai = _guess_expected_ai(img_path.name)

        last_err = None
        status_code = None
        body_text = None
        elapsed = None

        for attempt in range(1, max(1, args.retries) + 1):
            try:
                t0 = time.time()
                with img_path.open("rb") as f:
                    files = {"image": (img_path.name, f, "application/octet-stream")}
                    r = s.post(endpoint, files=files, timeout=args.timeout)
                elapsed = time.time() - t0
                status_code = r.status_code
                body_text = r.text

                if r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}"
                    if attempt < args.retries:
                        backoff = min(10.0, 1.5 * attempt)
                        print(f"  WARN: attempt {attempt}/{args.retries} failed: {last_err} (retrying in {backoff:.1f}s)")
                        time.sleep(backoff)
                        continue
                break
            except Exception as e:
                # Network errors / backend restarts should not crash the whole run.
                last_err = str(e)
                if attempt < args.retries:
                    backoff = min(10.0, 1.5 * attempt)
                    print(f"  WARN: attempt {attempt}/{args.retries} failed: {last_err} (retrying in {backoff:.1f}s)")
                    time.sleep(backoff)
                    continue
                status_code = None
                body_text = None
                elapsed = elapsed if elapsed is not None else None

        if status_code != 200:
            print(f"  ERROR: Status {status_code or 'N/A'}")
            fail_count += 1
            results.append(
                {
                    "FileName": img_path.name,
                    "StatusCode": status_code or "",
                    "Error": last_err or "Unknown error",
                    "IsAI": "",
                    "Confidence": "",
                    "TimeSeconds": f"{elapsed:.4f}" if elapsed is not None else "",
                    "Expected": str(expected_ai),
                    "Match": "",
                }
            )
            continue

        try:
            payload = __import__("json").loads(body_text or "{}")
        except Exception as e:
            print(f"  ERROR: JSON parse failed: {e}")
            fail_count += 1
            results.append(
                {
                    "FileName": img_path.name,
                    "StatusCode": status_code,
                    "Error": f"JSON parse failed: {e}",
                    "IsAI": "",
                    "Confidence": "",
                    "TimeSeconds": f"{elapsed:.4f}" if elapsed is not None else "",
                    "Expected": str(expected_ai),
                    "Match": "",
                }
            )
            continue

        if not payload.get("success"):
            err_detail = payload.get("detail") or payload.get("error") or "success=false"
            print(f"  ERROR: {err_detail}")
            fail_count += 1
            results.append(
                {
                    "FileName": img_path.name,
                    "StatusCode": status_code,
                    "Error": str(err_detail),
                    "IsAI": "",
                    "Confidence": "",
                    "TimeSeconds": f"{elapsed:.4f}" if elapsed is not None else "",
                    "Expected": str(expected_ai),
                    "Match": "",
                }
            )
            continue

        result = (payload.get("result") or {}) if isinstance(payload, dict) else {}
        is_ai = bool(result.get("isAIGenerated"))
        confidence = result.get("confidence", "")
        verdict = "AI" if is_ai else "REAL"

        match = "OK" if is_ai == expected_ai else "WRONG"
        print(f"  Result: {verdict} | Confidence: {confidence}% | Time: {elapsed:.2f}s | Match: {match}")

        if is_ai:
            ai_count += 1
        else:
            real_count += 1
        success_count += 1
        total_time += float(elapsed or 0.0)

        results.append(
            {
                "FileName": img_path.name,
                "StatusCode": status_code,
                "Error": "",
                "IsAI": str(is_ai),
                "Confidence": confidence,
                "TimeSeconds": f"{elapsed:.4f}" if elapsed is not None else "",
                "Expected": str(expected_ai),
                "Match": match,
            }
        )

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    # Summary
    print("")
    print("=" * 48)
    print("TEST SUMMARY")
    print("=" * 48)
    total = len(images)
    success_rate = (success_count / total * 100.0) if total else 0.0
    ai_pct = (ai_count / success_count * 100.0) if success_count else 0.0
    real_pct = (real_count / success_count * 100.0) if success_count else 0.0
    avg_time = (total_time / success_count) if success_count else 0.0

    correct_matches = sum(1 for r in results if r.get("Match") == "OK")
    accuracy = (correct_matches / success_count * 100.0) if success_count else 0.0

    print(f"Total Images Tested: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("")
    print(f"AI Detected: {ai_count} ({ai_pct:.2f}%)")
    print(f"Real Detected: {real_count} ({real_pct:.2f}%)")
    print("")
    print(f"Average Analysis Time: {avg_time:.2f}s")
    print(f"Detection Accuracy: {accuracy:.2f}%")

    # Write CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "FileName",
            "StatusCode",
            "Error",
            "IsAI",
            "Confidence",
            "TimeSeconds",
            "Expected",
            "Match",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    print("")
    print(f"Done. Results saved to: {out_path}")


if __name__ == "__main__":
    main()

