# =========================================
# File: ela_inference_folder.py
# Error Level Analysis (ELA) Deepfake Detector (batch inference)
# Output format matches xception_inference.py / pixel_inference_folder.py / freq_inference_folder.py
# =========================================

import torch, pickle, time, csv
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models

# ===== Model Definition =====
class ELRes(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        self.modelELA = models.resnext50_32x4d(weights=weights)
        self.modelRGB = models.resnext50_32x4d(weights=weights)
        self.modelELA.fc = nn.Identity()
        self.modelRGB.fc = nn.Identity()
        self.fc = nn.Linear(2048 * 2, num_classes)

    def forward(self, x_ela, x_rgb):
        feat_ela = self.modelELA(x_ela)
        feat_rgb = self.modelRGB(x_rgb)
        combined = torch.cat([feat_ela, feat_rgb], dim=1)
        return self.fc(combined)

# ===== ELA Conversion =====
def convert_to_ela_image(pil_img, quality=90):
    temp = Path("temp_ela.jpg")
    try:
        pil_img.save(temp, 'JPEG', quality=quality)
        resaved = Image.open(temp)
        ela = ImageChops.difference(pil_img, resaved)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        return ela
    finally:
        if temp.exists():
            temp.unlink()

# ===== Predict Single Image =====
def predict_image(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    ela_img = convert_to_ela_image(img)

    img = img.resize((224, 224))
    ela_img = ela_img.resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    rgb_tensor = transform(img).unsqueeze(0).to(device)
    ela_tensor = transform(ela_img).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(ela_tensor, rgb_tensor)
        prob = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
    t1 = time.time()

    real_prob = prob[0] * 100
    ai_prob = prob[1] * 100
    label = "Real" if real_prob >= ai_prob else "AI-Generated"
    elapsed = t1 - t0

    print(f"{Path(image_path).name}: {label} ({real_prob:.2f}% Real, {ai_prob:.2f}% AI) | {elapsed:.3f}s")
    return {
        "image_path": Path(image_path).name,
        "real_prob": real_prob,
        "ai_prob": ai_prob,
        "label": label,
        "time": elapsed
    }

# ===== Batch Inference + Save CSV =====
def batch_infer_and_save(image_folder='Dataset_Senior_Project',
                         checkpoint_path='elres_model.pkl',
                         output_csv='ela_results.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = script_dir / 'model' / checkpoint_path
    
    image_folder = Path(image_folder)
    if not image_folder.is_absolute():
        image_folder = script_dir / image_folder
    
    print(f"[INFO] Loading model from {checkpoint_path}...)")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Handle both direct model and checkpoint dict formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint and 'model_config' in checkpoint:
            # Reconstruct model from state dict
            config = checkpoint.get('model_config', {})
            num_classes = config.get('num_classes', 2)
            model = ELRes(num_classes=num_classes, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()

    # [INFO] Collect image files (recursive)
    exts = ['.jpg', '.jpeg', '.png']
    image_list = [str(p) for p in image_folder.rglob('*') if p.suffix.lower() in exts]
    if not image_list:
        print(f"⚠️ No images found in folder: {image_folder}")
        return

    print(f"[INFO] Found {len(image_list)} images in '{image_folder}'\n")

    results = []
    for img_path in image_list:
        try:
            result = predict_image(model, img_path, device)
            results.append(result)
        except Exception as e:
            print(f"[WARN] Skipping {img_path}: {e}")

    # [INFO] Save results to CSV (with Label)
    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = script_dir / output_csv
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Image Name','Label', '%Real', '%AI',  'Time(s)'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'Image Name': r['image_path'],
                'Label': r['label'],
                '%Real': f"{r['real_prob']:.2f}",
                '%AI': f"{r['ai_prob']:.2f}",
                'Time(s)': f"{r['time']:.3f}"
            })
    print(f"[DONE] Results saved to {output_csv}")

# ===== Run Example =====
if __name__ == "__main__":
    # Example: batch_infer_and_save(image_folder="Dataset_Senior_Project")
    # For testing, use local test images:
    batch_infer_and_save(image_folder="backend/uploads", output_csv="ela_results.csv")
