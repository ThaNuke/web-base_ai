#!/usr/bin/env python3
"""
Test Weight Boosting Ensemble method
"""
import requests
import json
import os
from pathlib import Path

# Test image paths
test_dir = Path(__file__).parent.parent / "test_images"

# Try to find test images in common locations
test_images = [
    test_dir / "ai_generated.png",
    test_dir / "ai_sample.png",
    test_dir / "real_photo.png",
]

# Fallback - look for any images in test_images or uploads
if not any(img.exists() for img in test_images):
    if test_dir.exists():
        all_images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg"))
        test_images = all_images[:2] if all_images else test_images

API_URL = "http://localhost:8001/api/analyze"

def test_image(image_path):
    """Test a single image"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print(f"{'='*60}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display ensemble result
            if result.get("success"):
                data = result["result"]
                print(f"\n✅ ENSEMBLE RESULT:")
                print(f"   AI Generated: {data['isAIGenerated']}")
                print(f"   Confidence: {data['confidence']:.2f}%")
                print(f"   Message: {data['message']}")
                print(f"   Ensemble Type: {data['details']['ensemble_type']}")
                
                # Display individual model predictions
                print(f"\n📊 INDIVIDUAL MODEL PREDICTIONS:")
                models = data.get("individual_models", {})
                for model_name, model_result in models.items():
                    if "error" not in model_result:
                        print(f"   {model_result['name']:25} | AI: {model_result['confidence']:6.2f}% | Real: {model_result['real_prob']:6.2f}%")
                        print(f"      └─ Classification: {'🤖 AI GENERATED' if model_result['isAI'] else '📸 REAL'}")
                
                # Display ensemble weights
                print(f"\n⚖️  ENSEMBLE WEIGHTS (Weight Boosting):")
                weights = data.get("ensemble_weights", {})
                for model, weight in weights.items():
                    print(f"   {model:12}: {weight:.2f}")
                
                # Display vote strength
                print(f"\n🗳️  WEIGHTED VOTE STRENGTH: {data['weighted_vote_strength']:.2f}")
                
                return data
            else:
                print(f"❌ Error: {result}")
                return None
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("🚀 WEIGHT BOOSTING ENSEMBLE TEST")
    print("="*60)
    
    # Find available test images
    valid_images = [img for img in test_images if img.exists()]
    
    if not valid_images:
        print("⚠️  No test images found!")
        print("\nTrying to create synthetic test images...")
        
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create AI-like image
            ai_img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(ai_img)
            np.random.seed(42)
            for _ in range(100):
                x, y = np.random.randint(0, 224, 2)
                draw.ellipse([x, y, x+10, y+10], fill='red')
            ai_path = test_dir / "ai_generated.png"
            ai_img.save(ai_path)
            valid_images.append(ai_path)
            print(f"✅ Created synthetic AI image")
            
            # Create real-like image
            real_img = Image.new('RGB', (224, 224))
            real_pixels = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            real_img = Image.fromarray(real_pixels)
            real_path = test_dir / "real_photo.png"
            real_img.save(real_path)
            valid_images.append(real_path)
            print(f"✅ Created synthetic real image")
            
        except Exception as e:
            print(f"Error creating test images: {e}")
            return
    
    # Test each image
    results = []
    for image_path in valid_images[:2]:
        result = test_image(image_path)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    
    if results:
        # Check if ensemble method is weight boosting
        ensemble_types = [r['details']['ensemble_type'] for r in results]
        print(f"\nEnsemble methods used:")
        for et in set(ensemble_types):
            count = ensemble_types.count(et)
            print(f"   • {et}: {count} test(s)")
        
        # Verify weight boosting is primary
        if any("Weight Boosting" in et for et in ensemble_types):
            print(f"\n✅ Weight Boosting Ensemble is ACTIVE as primary method!")
        else:
            print(f"\n⚠️  Weight Boosting not used as primary - fallback methods may be active")

if __name__ == "__main__":
    main()
