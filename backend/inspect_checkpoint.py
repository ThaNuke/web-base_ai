import torch
from pathlib import Path

checkpoint_path = Path(__file__).parent.parent / "model" / "best_model_ela.pth"

print(f"Loading checkpoint from: {checkpoint_path}")
print(f"File exists: {checkpoint_path.exists()}")
print(f"File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB\n")

ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f"Checkpoint type: {type(ckpt).__name__}")

if isinstance(ckpt, dict):
    print(f"Keys in checkpoint: {list(ckpt.keys())}\n")
    
    for key in list(ckpt.keys())[:10]:
        value = ckpt[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape} - dtype: {value.dtype}")
            # Check if weights are trained (not zeros or random)
            stats = {
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.mean().item()),
                "std": float(value.std().item()) if value.numel() > 1 else 0,
            }
            print(f"    Stats: {stats}")
        elif isinstance(value, dict):
            print(f"  {key}: Dict with keys {list(value.keys())[:3]}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
else:
    print(f"Checkpoint is a {type(ckpt).__name__}")
    print(f"Content sample: {ckpt}")
