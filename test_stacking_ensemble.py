"""
Test Script: Weighted Boosting Ensemble API
Tests the weighted boosting ensemble with fixed model weights
"""

import requests
import json
from pathlib import Path

API_URL = "http://localhost:8001/api/analyze"

# Test image paths
test_images = [
    "backend/uploads/AI.png",
    "backend/uploads/Cho_La_Pass_in_snow,_Nepal,_Himalayas.jpg"
]

print("=" * 70)
print("WEIGHTED BOOSTING ENSEMBLE - API TEST")
print("=" * 70)

for img_path in test_images:
    img_full_path = Path(img_path)
    
    if not img_full_path.exists():
        print(f"\n⚠️  Image not found: {img_path}")
        continue
    
    print(f"\n{'=' * 70}")
    print(f"Testing: {img_path}")
    print(f"{'=' * 70}")
    
    try:
        with open(img_full_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            
            # Display main prediction
            confidence = result.get('confidence', 0)
            is_ai = result.get('isAIGenerated', False)
            
            print(f"\n✓ API Response Received")
            print(f"\n  Prediction: {'🤖 AI-Generated' if is_ai else '📸 Real Image'}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Message: {result.get('message', 'N/A')}")
            
            # Display ensemble details
            details = result.get('details', {})
            print(f"\n  Ensemble Type: {details.get('ensemble_type', 'Unknown')}")
            print(f"  Ensemble Votes: {details.get('ensemble_votes', 'N/A')}")
            print(f"  Timestamp: {details.get('timestamp', 'N/A')}")
            
            # Display ensemble weights
            ensemble_weights = result.get('ensemble_weights', {})
            if ensemble_weights:
                print(f"\n  Model Weights Configuration:")
                for model_name, weight in ensemble_weights.items():
                    print(f"    - {model_name.upper()}: {weight*100:.0f}%")
            
            # Display weighted details (per-model contribution)
            weighted_details = result.get('weighted_details', {})
            if weighted_details:
                print(f"\n  Per-Model Weighted Analysis:")
                for model_name, details_info in weighted_details.items():
                    weight = details_info.get('weight', 0)
                    confidence_val = details_info.get('confidence', 0)
                    weighted_score = details_info.get('weighted_score', 0)
                    print(f"    - {model_name.upper()}:")
                    print(f"        Weight: {weight*100:.0f}%, Confidence: {confidence_val:.2f}%, Contribution: {weighted_score:.2f}%")
            
            # Display voting strength
            voting_strength = result.get('weighted_vote_strength', 0)
            print(f"\n  Voting Strength: {voting_strength:.2f} (0-1 scale)")
            
            # Display ensemble method
            ensemble_method = result.get('ensemble_method', 'unknown')
            print(f"  Ensemble Method: {ensemble_method.upper()}")
            
            # Display individual model predictions
            print(f"\n  Individual Model Predictions:")
            models = result.get('individual_models', {})
            for model_name, pred in models.items():
                if 'error' not in pred:
                    conf = pred.get('confidence', 0)
                    is_ai_model = pred.get('isAI', False)
                    print(f"    - {model_name}: {conf:.2f}% AI {'✓' if is_ai_model else '✗'}")
            
        else:
            print(f"\n✗ API Error: {response.status_code}")
            print(f"  Response: {response.text}")
    
    except Exception as e:
        print(f"\n✗ Request Failed: {e}")

print(f"\n{'=' * 70}")
print("TEST COMPLETED")
print(f"{'=' * 70}\n")
