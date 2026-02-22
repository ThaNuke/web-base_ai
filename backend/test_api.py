#!/usr/bin/env python3
"""Test the API"""
import requests
from pathlib import Path

api_url = "http://localhost:3003/api/analyze"
test_images = [
    Path(__file__).parent / "uploads" / "test_real.jpg",
    Path(__file__).parent / "uploads" / "test_ai.jpg",
]

for img_path in test_images:
    if not img_path.exists():
        print(f"[X] {img_path} not found")
        continue
    
    print(f"\nTesting {img_path.name}...")
    with open(img_path, "rb") as f:
        files = {"image": f}
        try:
            response = requests.post(api_url, files=files, timeout=5)
            result = response.json()
            if response.status_code == 200 and result.get("success"):
                conf = result["result"]["confidence"]
                is_ai = result["result"]["isAIGenerated"]
                print(f"  Confidence: {conf:.1f}%")
                print(f"  Is AI: {is_ai}")
                print(f"  Message: {result['result']['message']}")
            else:
                print(f"  Error: {result}")
        except Exception as e:
            print(f"  Exception: {e}")
