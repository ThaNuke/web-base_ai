#!/usr/bin/env python3
"""Test weight stacking ensemble API"""

import asyncio
import json
from pathlib import Path
from main import analyze_image

async def test_weight_stacking():
    """Test weight stacking ensemble"""
    test_images = [
        ("uploads/AI.png", "AI-generated image"),
        ("uploads/Cho_La_Pass_in_snow,_Nepal,_Himalayas.jpg", "Real photograph"),
    ]
    
    for image_path, description in test_images:
        full_path = Path("backend") / image_path if not Path(image_path).exists() else Path(image_path)
        
        if not full_path.exists():
            full_path = Path(image_path)
        
        if not full_path.exists():
            print(f"[X] {full_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"File: {full_path}")
        print(f"{'='*60}")
        
        try:
            result = await analyze_image(str(full_path))
            
            ensemble = result["ensemble"]
            models = result["models"]
            
            print(f"\n✓ Ensemble Result:")
            print(f"  Type: {ensemble['ensemble_type']}")
            print(f"  Prediction: {'AI-Generated' if ensemble['isAIGenerated'] else 'Real'}")
            print(f"  Confidence: {ensemble['confidence_rounded']:.2f}%")
            print(f"  Probability: {ensemble['probability']:.4f}")
            print(f"  Votes: {ensemble['votes']}")
            
            print(f"\n✓ Individual Model Predictions:")
            for model_name, model_result in models.items():
                if "error" not in model_result:
                    print(f"  {model_name.upper()}:")
                    print(f"    - Prediction: {'AI' if model_result['isAI'] else 'Real'}")
                    print(f"    - Confidence: {model_result['confidence']:.2f}%")
                    print(f"    - Real Prob: {model_result['real_prob']:.2f}%")
                else:
                    print(f"  {model_name.upper()}: ERROR - {model_result['error']}")
            
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_weight_stacking())
