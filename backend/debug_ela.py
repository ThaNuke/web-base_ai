#!/usr/bin/env python3
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile

test_images = [
    Path(__file__).parent / "uploads" / "test_real.jpg",
    Path(__file__).parent / "uploads" / "test_ai.jpg",
]

for img_path in test_images:
    print(f"\n{img_path.name}:")
    image = Image.open(img_path).convert("RGB")
    image = image.resize((224, 224))
    
    # ELA calculation using ImageChops.difference
    from PIL import ImageChops, ImageEnhance
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
    
    image.save(temp_path, 'JPEG', quality=90)
    resaved = Image.open(temp_path)
    ela_image = ImageChops.difference(image, resaved)
    
    # Scale ELA to utilize full range
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    arr = np.array(ela_image) / 255.0
    ela_energy = np.mean(np.abs(arr))
    
    print(f"  ELA energy: {ela_energy:.6f}")
    print(f"  Max diff: {max_diff}")
    print(f"  Scale: {scale}")
    print(f"  Array stats: mean={arr.mean():.6f}, std={arr.std():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
    
    Path(temp_path).unlink()
