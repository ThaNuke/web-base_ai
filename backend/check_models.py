#!/usr/bin/env python3
import pickle
from pathlib import Path

PKL_PATH = Path(__file__).parent.parent / "model" / "elres_model.pkl"
PTH_PATH = Path(__file__).parent.parent / "model" / "best_model_ela.pth"

print(f"Checking {PKL_PATH}")
print(f"Exists: {PKL_PATH.exists()}")
if PKL_PATH.exists():
    print(f"Size: {PKL_PATH. stat().st_size / 1024:.1f} KB")
    try:
        with open(PKL_PATH, "rb") as f:
            obj = pickle.load(f)
        print(f"Type: {type(obj).__name__}")
        if isinstance(obj, dict):
            print(f"Keys: {list(obj.keys())}")
        elif hasattr(obj, "__dict__"):
            print(f"Attributes: {[k for k in dir(obj) if not k.startswith('_')][:10]}")
    except Exception as e:
        print(f"Error: {e}")

print(f"\nChecking {PTH_PATH}")
print(f"Exists: {PTH_PATH.exists()}")
if PTH_PATH.exists():
    print(f"Size: {PTH_PATH.stat().st_size / 1024 / 1024:.1f} MB")
