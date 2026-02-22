#!/usr/bin/env python3
import torch
from pathlib import Path
from collections import defaultdict

MODEL_PATH = Path(__file__).parent.parent / "model" / "best_model_ela.pth"

print(f"Opening: {MODEL_PATH}\n")

raw = torch.load(MODEL_PATH, map_location="cpu")

if "model_state_dict" in raw:
    state_dict = raw["model_state_dict"]
    keys = list(state_dict.keys())
    
    print(f"Total keys: {len(keys)}\n")
    
    # Find all prefixes (model streams)
    prefixes = defaultdict(int)
    for key in keys:
        prefix = key.split(".")[0]
        prefixes[prefix] += 1
    
    print(f"Model streams/prefixes:")
    for prefix in sorted(prefixes.keys()):
        print(f"  {prefix}: {prefixes[prefix]} keys")
    
    print(f"\nFirst 5 keys from each prefix:")
    for prefix in sorted(prefixes.keys()):
        prefix_keys = [k for k in keys if k.startswith(f"{prefix}.")][:5]
        for k in prefix_keys:
            val = state_dict[k]
            if isinstance(val, torch.Tensor):
                print(f"  {k}: {val.shape} - dtype: {val.dtype}")
            else:
                print(f"  {k}: {type(val).__name__}")
