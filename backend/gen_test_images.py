#!/usr/bin/env python3
"""Generate a simple test image for API testing"""
from PIL import Image
import numpy as np
from pathlib import Path

# Create a simple test image (gradient pattern)
test_dir = Path(__file__).parent / "uploads"
test_dir.mkdir(exist_ok=True)

# Create two test images
# 1. Real-looking image (smooth gradient - low variance ELA)
img1 = Image.new("RGB", (224, 224), color=(100, 150, 200))
img1_path = test_dir / "test_real.jpg"
img1.save(img1_path, "JPEG", quality=95)
print(f"Created {img1_path}")

# 2. AI-looking image (random noise - high variance ELA)
noise_data = np.random.randint(0, 256, (224, 224, 3),  dtype=np.uint8)
img2 = Image.fromarray(noise_data)
img2_path = test_dir / "test_ai.jpg"
img2.save(img2_path, "JPEG")
print(f"Created {img2_path}")
