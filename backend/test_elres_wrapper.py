#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as tv_models
from pathlib import Path
import warnings

# Suppress size mismatch warnings for cleaner output
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

MODEL_PATH = Path(__file__).parent.parent / "model" / "best_model_ela.pth"

print("Loading checkpoint...")
raw = torch.load(MODEL_PATH, map_location="cpu")

if "model_state_dict" in raw:
    state_dict = raw["model_state_dict"]
    print(f"[OK] Found model_state_dict with {len(state_dict)}")
    
    # Check for 2-stream
    has_ela = any(k.startswith("modelELA.") for k in state_dict.keys())
    has_rgb = any(k.startswith("modelRGB.") for k in state_dict.keys())
    print(f"  modelELA prefix: {has_ela}")
    print(f"  modelRGB prefix: {has_rgb}")
    
    if has_ela and has_rgb:
        print("\n>>> ATTEMPTING 2-STREAM LOADING...")
        
        # Create ELResWrapper
        class ELResWrapper(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.modelELA = tv_models.resnet50(num_classes=1000)
                self.modelRGB = tv_models.resnet50(num_classes=1000)
                in_features = self.modelELA.fc.in_features
                self.modelELA.fc = nn.Identity()
                self.modelRGB.fc = nn.Identity()
                self.fc = nn.Linear(in_features * 2, num_classes)
            
            def forward(self, x_ela, x_rgb):
                feat_ela = self.modelELA(x_ela)
                feat_rgb = self.modelRGB(x_rgb)
                feat_combined = torch.cat([feat_ela, feat_rgb], dim=1)
                out = self.fc(feat_combined)
                return out
       
        try:
            print("Creating ELResWrapper...")
            model = ELResWrapper(num_classes=2)
            print("Loading state_dict...")
            result = model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Loaded successfully!")
            print(f"  Missing keys: {len(result.missing_keys)}")
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
            if len(result.missing_keys) > 0:
                print(f"  First missing: {list(result.missing_keys)[:5]}")
            if len(result.unexpected_keys) > 0:
                print(f"  First unexpected: {list(result.unexpected_keys)[:5]}")
        except Exception as e:
            print(f"[FAIL] Failed: {type(e).__name__}: {e}")
else:
    print("[FAIL] No model_state_dict found")
