#!/usr/bin/env python3
"""
Debug script to examine the .pth file structure
"""
import torch
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model" / "best_model_ela.pth"

print(f"Opening: {MODEL_PATH}")
print(f"File exists: {MODEL_PATH.exists()}")

if MODEL_PATH.exists():
    raw = torch.load(MODEL_PATH, map_location="cpu")
    
    print(f"\n[PKG] Raw type: {type(raw).__name__}")
    
    if isinstance(raw, dict):
        print(f"[INFO] Keys in raw dict: {list(raw.keys())}")
        
        # Check for model_state_dict
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
            print(f"\n[OK] Found 'model_state_dict'")
            print(f"   Type: {type(state_dict).__name__}")
            
            if isinstance(state_dict, dict):
                print(f"   Total keys: {len(state_dict)}")
                keys = list(state_dict.keys())
                print(f"   First 10 keys: {keys[:10]}")
                
                # Check for modelELA prefix
                has_prefix = any(k.startswith("modelELA.") for k in keys)
                print(f"   Has 'modelELA.' prefix: {has_prefix}")
                
                # Count architectures
                resnet_count = sum(1 for k in keys if "layer1" in k or "layer2" in k or "layer3" in k)
                mobilenet_count = sum(1 for k in keys if "features." in k)
                
                print(f"   ResNet-like keys: {resnet_count}")
                print(f"   MobileNet-like keys: {mobilenet_count}")
                
        # Check for model_config
        if "model_config" in raw:
            config = raw["model_config"]
            print(f"\n[OK] Found 'model_config': {config}")
        else:
            print(f"\n[MISSING] No 'model_config' found")
            
    else:
        print(f"\n Raw is not a dict - trying to load as direct state_dict")
        if isinstance(raw, dict):
            keys = list(raw.keys())
            print(f"   Total keys: {len(keys)}")
            print(f"   First 10 keys: {keys[:10]}")
else:
    print(f"\n[ERROR] Model file not found!")
