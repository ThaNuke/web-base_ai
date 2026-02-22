#!/usr/bin/env python3
import pickle
from pathlib import Path

PKL_PATH = Path(__file__).parent.parent / "model" / "elres_model.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print("PKL file contents:")
for key in data.keys():
    val = data[key]
    if isinstance(val, dict):
        print(f" {key}: dict with keys {list(val.keys())[:5]}")
    else:
        print(f"  {key}: {type(val).__name__}")

print("\n model_config contents:")
config = data.get("model_config", {})
for key, val in config.items():
    print(f"  {key}: {val}")

print("\n model_state_dict type:", type(data.get("model_state_dict")).__name__)
print(" model_state_dict keys sample:", list(data.get("model_state_dict", {}).keys())[:5])
