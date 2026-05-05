"""
Test Model Importance using Dataset Senior Project
Real images: filenames containing "_T" or "TT"
AI images: all other filenames
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import requests

# Configuration
DATASET_DIR = Path("backend/Dataset Senior Project")
BACKEND_URL = "http://localhost:8001"
API_ENDPOINT = f"{BACKEND_URL}/api/test-model-importance"

# Track results
results = {
    "total_images": 0,
    "real_images": 0,
    "ai_images": 0,
    "test_results": [],
    "impacts": defaultdict(list),
    "summary": {}
}


def is_real_image(filename: str) -> bool:
    """
    Determine if image is real or AI based on filename
    Real: contains "_T" or "TT"
    AI: everything else
    """
    name_lower = filename.lower()
    # Check for _T or TT in filename
    return "_t" in name_lower or name_lower.startswith("tt")


def get_image_files():
    """Get all image files from dataset"""
    images = []
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        return images
    
    for file in DATASET_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            images.append(file)
    
    return sorted(images)


def test_image(image_path: Path) -> dict:
    """
    Send image to backend for ablation study testing
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/jpeg')}
            response = requests.post(API_ENDPOINT, files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[ERROR] HTTP {response.status_code}: {image_path.name}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to test {image_path.name}: {e}")
        return None


def calculate_impacts(test_result: dict) -> dict:
    """
    Extract impact values from test result.
    
    Returns:
      {
        "baseline": float,
        "without": {model: float},
        "impacts": {model: float},          # absolute impacts
        "signed_impacts": {model: float},   # baseline - without (keeps direction)
      }
    """
    if not test_result or "success" not in test_result:
        return None
    
    try:
        result_data = test_result.get("result", {})
        baseline = result_data.get("baseline", {})
        test_results = result_data.get("test_results", {})
        
        baseline_score = baseline.get("isAI", 0)

        without_scores = {
            "frequency": test_results.get("without_frequency", {}).get("isAI", 0),
            "pixel": test_results.get("without_pixel", {}).get("isAI", 0),
            "ela": test_results.get("without_ela", {}).get("isAI", 0),
            "xception": test_results.get("without_xception", {}).get("isAI", 0),
        }

        signed_impacts = {k: (baseline_score - v) for k, v in without_scores.items()}
        impacts = {k: abs(v) for k, v in signed_impacts.items()}

        return {
            "baseline": baseline_score,
            "without": without_scores,
            "impacts": impacts,
            "signed_impacts": signed_impacts,
        }
    except Exception as e:
        print(f"[ERROR] Error extracting impacts: {e}")
        return None


def run_ablation_study():
    """
    Run ablation study on all images in dataset
    """
    print("Starting Model Importance Test on Dataset Senior Project")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Backend: {BACKEND_URL}")
    print()
    
    images = get_image_files()
    if not images:
        print("[ERROR] No images found in dataset")
        return
    
    # Separate real and AI images
    real_images = [img for img in images if is_real_image(img.name)]
    ai_images = [img for img in images if not is_real_image(img.name)]
    
    print("Dataset Summary:")
    print(f"   Total images: {len(images)}")
    print(f"   Real images (_T or TT): {len(real_images)}")
    print(f"   AI images (others): {len(ai_images)}")
    print()
    
    results["total_images"] = len(images)
    results["real_images"] = len(real_images)
    results["ai_images"] = len(ai_images)
    
    # Test each image
    print("Testing images... (this may take a while)")
    print()
    
    for idx, image_path in enumerate(images, 1):
        is_real = is_real_image(image_path.name)
        label = "REAL" if is_real else "AI"
        
        print(f"[{idx}/{len(images)}] {label} {image_path.name}...", end=" ", flush=True)
        
        # Test image
        test_result = test_image(image_path)
        
        if test_result:
            extracted = calculate_impacts(test_result)
            if extracted:
                impacts = extracted["impacts"]
                # Store impacts (absolute)
                for model, impact in impacts.items():
                    results["impacts"][model].append(impact)

                results["test_results"].append({
                    "filename": image_path.name,
                    "is_real": is_real,
                    "baseline": extracted["baseline"],
                    "without": extracted["without"],
                    "impacts": extracted["impacts"],
                    "signed_impacts": extracted["signed_impacts"],
                })
                
                print("OK")
            else:
                print("[WARN] Could not extract impacts")
        else:
            print("[ERROR]")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Calculate averages
    avg_impacts = {}
    for model, impacts_list in results["impacts"].items():
        if impacts_list:
            avg_impact = sum(impacts_list) / len(impacts_list)
            avg_impacts[model] = avg_impact
    
    # Sort by average impact
    sorted_models = sorted(avg_impacts.items(), key=lambda x: x[1], reverse=True)
    
    print()
    print("Model Importance Ranking (by Average Impact):")
    print()
    
    total_impact = sum(avg_impacts.values())
    for rank, (model, avg_impact) in enumerate(sorted_models, 1):
        weight_pct = (avg_impact / total_impact * 100) if total_impact > 0 else 0
        print(f"{rank}. {model.upper()}")
        print(f"   Average Impact: {avg_impact:.4f}")
        print(f"   Raw Weight: {weight_pct:.2f}%")
        print()
    
    results["summary"] = {
        "average_impacts": {model: value for model, value in sorted_models},
        "total_impact": total_impact,
        "weights": {
            model: (value / total_impact * 100) if total_impact > 0 else 0
            for model, value in sorted_models
        }
    }
    
    # Save results to JSON
    output_file = Path("model_importance_results_senior_project.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert defaultdict to regular dict for JSON serialization
        json_results = {
            "total_images": results["total_images"],
            "real_images": results["real_images"],
            "ai_images": results["ai_images"],
            "test_results": results["test_results"],
            "impacts": dict(results["impacts"]),
            "summary": results["summary"]
        }
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Summary by image type
    print("=" * 70)
    print("Analysis by Image Type:")
    print()
    
    real_impacts = defaultdict(list)
    ai_impacts = defaultdict(list)
    
    for test in results["test_results"]:
        if test["is_real"]:
            for model, impact in test["impacts"].items():
                real_impacts[model].append(impact)
        else:
            for model, impact in test["impacts"].items():
                ai_impacts[model].append(impact)
    
    print("Real Images (_T or TT):")
    for model in sorted(avg_impacts.keys()):
        if real_impacts[model]:
            avg = sum(real_impacts[model]) / len(real_impacts[model])
            print(f"  {model}: {avg:.4f}")
    
    print()
    print("AI Images (others):")
    for model in sorted(avg_impacts.keys()):
        if ai_impacts[model]:
            avg = sum(ai_impacts[model]) / len(ai_impacts[model])
            print(f"  {model}: {avg:.4f}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("[ERROR] Backend is not responding properly")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot connect to backend at {BACKEND_URL}")
        print(f"   Make sure the backend is running: python backend/main.py")
        sys.exit(1)
    
    run_ablation_study()
