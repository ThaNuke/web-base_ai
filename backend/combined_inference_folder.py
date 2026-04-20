# =========================================
# File: combined_inference_folder.py
# Combine ELA + Pixel + Frequency + Xception Inference Results
# =========================================

import torch, time, csv
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import cv2
from PIL import ImageChops, ImageEnhance
import timm
import pickle
import io


# ==============================================================
# 🔹 Custom Unpickler for CUDA → CPU device mapping
# ==============================================================
class CPUUnpickler(pickle.Unpickler):
    """Map CUDA device references to CPU when unpickling"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_checkpoint_with_device_map(file_path, device):
    """Load checkpoint with proper CUDA→CPU device mapping"""
    try:
        # Try torch.load first with explicit CPU mapping
        return torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e1:
        try:
            # Fallback: use custom unpickler
            with open(file_path, 'rb') as f:
                unpickler = CPUUnpickler(f)
                return unpickler.load()
        except Exception as e2:
            raise Exception(f"torch.load failed: {e1}, custom unpickler failed: {e2}")

# ==============================================================
# 🔹 Shared utility
# ==============================================================
def get_image_list(image_folder):
    exts = ['.jpg', '.jpeg', '.png']
    return [str(p) for p in Path(image_folder).rglob('*') if p.suffix.lower() in exts]


# ==============================================================
# 🔹 Model 1: ELA Model (2-Stream: ELA + RGB)==============================================================

class ELRes(nn.Module):
    def __init__(self, num_classes=2):
        super(ELRes, self).__init__()
        self.modelELA = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.modelRGB = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        
        self.modelELA.fc = nn.Identity()
        self.modelRGB.fc = nn.Identity()
        
        self.fc = nn.Linear(2048*2, num_classes)
    
    def forward(self, x_ela, x_rgb):
        ela_feat = self.modelELA(x_ela)
        rgb_feat = self.modelRGB(x_rgb)
        combined = torch.cat((ela_feat, rgb_feat), dim=1)
        return self.fc(combined)
def convert_to_ela_image(img, quality=90):
    temp_path = "temp_ela_inference.jpg"
    try:
        img = img.convert('RGB')
        
        # Resize if too large (from notebook logic)
        max_size = 2000
        if img.size[0] > max_size or img.size[1] > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        img.save(temp_path, 'JPEG', quality=quality)
        resaved = Image.open(temp_path)
        
        ela_image = ImageChops.difference(img, resaved)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    except Exception as e:
        print(f"Error in ELA conversion: {e}")
        return img
    finally:
        if Path(temp_path).exists():
           try:
               Path(temp_path).unlink()
           except:
               pass

def illumination_map(img):
    # Keeping this if needed for other models, but ELA doesn't use it now.
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(g, (31,31), 0)
    blur = (blur - blur.min()) / (blur.max() + 1e-6) * 255
    return Image.fromarray(blur.astype(np.uint8)).convert("RGB")


# ==============================================================
# 🔹 Model 2: Pixel Model
# ==============================================================
class PixelRes(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        self.modelPix = models.resnext50_32x4d(weights=weights)
        
        # Modify first layer to accept 4 channels (Laplacian, Variance, SobelX, Gray)
        old_conv = self.modelPix.conv1
        self.modelPix.conv1 = nn.Conv2d(4, old_conv.out_channels, 
                                        kernel_size=old_conv.kernel_size, 
                                        stride=old_conv.stride, 
                                        padding=old_conv.padding, 
                                        bias=old_conv.bias is not None)

        self.modelRGB = models.resnext50_32x4d(weights=weights)
        self.modelPix.fc = nn.Identity()
        self.modelRGB.fc = nn.Identity()
        self.fc = nn.Linear(2048 * 2, num_classes)

    def forward(self, x_pix, x_rgb):
        feat_pix = self.modelPix(x_pix)
        feat_rgb = self.modelRGB(x_rgb)
        combined = torch.cat([feat_pix, feat_rgb], dim=1)
        return self.fc(combined)

def convert_to_pixel_map_from_pil(pil_img, ksize=7):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap = (lap / lap.max() * 255).astype(np.uint8) if lap.max() != 0 else lap
    # Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)
    if sobelx.max() != 0:
        sobelx = (sobelx / sobelx.max()) * 255.0
    sobelx = sobelx.astype(np.uint8)

    img_float = gray.astype(np.float32)
    mean = cv2.blur(img_float, (ksize, ksize))
    sqmean = cv2.blur(img_float ** 2, (ksize, ksize))
    variance = sqmean - mean ** 2
    if variance.max() != 0:
        variance = (variance / variance.max()) * 255.0
    variance = np.clip(variance, 0, 255).astype(np.uint8)

    combined = np.stack([lap, variance, sobelx, gray], axis=2)
    return Image.fromarray(combined)


# ==============================================================
# 🔹 Model 3: Frequency Domain Model
# ==============================================================
class FreqResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        # use resnet18 (default 3 channels)
        self.backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        # No need to modify conv1 for 3 channels as it is standard
        num_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

def fft_feature_map(pil_img, eps=1e-8):
    """Return PIL.Image of (Magnitude, Phase, BandPass) from a grayscale image."""
    img = pil_img.convert('L')
    arr = np.array(img).astype(np.float32)

    # 2D FFT
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    
    # 1. Magnitude Spectrum
    magnitude = np.abs(fshift)
    mag_log = np.log1p(magnitude + eps)
    mag_norm = mag_log - mag_log.min()
    if mag_norm.max() != 0:
        mag_norm = mag_norm / (mag_norm.max())
    mag_uint8 = (mag_norm * 255.0).astype(np.uint8)

    # 2. Phase Spectrum
    phase = np.angle(fshift) # [-pi, pi]
    # Normalize to [0, 255]
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_uint8 = (phase_norm * 255.0).astype(np.uint8)

    # 3. Band-Pass Filtered Magnitude
    rows, cols = arr.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    r = 30 # radius for high-pass
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
    mask[mask_area] = 0
    
    fshift_filtered = fshift * mask
    mag_filtered = np.abs(fshift_filtered)
    mag_filt_log = np.log1p(mag_filtered + eps)
    mag_filt_norm = mag_filt_log - mag_filt_log.min()
    if mag_filt_norm.max() != 0:
        mag_filt_norm = mag_filt_norm / mag_filt_norm.max()
    mag_filt_uint8 = (mag_filt_norm * 255.0).astype(np.uint8)

    # Stack to 3 channels
    combined = np.stack([mag_uint8, phase_uint8, mag_filt_uint8], axis=2)
    return Image.fromarray(combined)


# ==============================================================
# 🔹 Model 4: Xception Model
# ==============================================================
class XceptionBinary(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0, global_pool="")
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = self.gap(feats).view(feats.size(0), -1)
        logits = self.classifier(feats)
        return logits.squeeze(1)


# ==============================================================
# 🔹 Combined Inference Runner
# ==============================================================
def predict_all_models(image_path, device, models_dict):
    img = Image.open(image_path).convert('RGB')
    results = {'Image Name': Path(image_path).name}
    
    # 🔹 Determine State (Ground Truth) based on filename
    # Rule: If "T" is in filename -> Real, else -> AI-Generated (or whatever not T implies)
    # Be careful, T (99).jpg has "T".
    filename = Path(image_path).name
    if "T" in filename:
        results['State'] = "Real"
    else:
        results['State'] = "AI-Generated"

    # Shared transform
    t_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ----- Xception -----
    t_xcep = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    xcep_tensor = t_xcep(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = models_dict['xception'](xcep_tensor)
        p = torch.sigmoid(logits).item()
    results['Xcep %Real'], results['Xcep %AI'] = p*100, (1-p)*100
    results['Xcep Label'] = "Real" if p >= 0.5 else "AI-Generated"

    # ----- ELA (3-stream) -----
    ela_img = convert_to_ela_image(img)
    light_img = illumination_map(img)
    
    ela_tensor = t_rgb(ela_img).unsqueeze(0).to(device)
    rgb_tensor = t_rgb(img).unsqueeze(0).to(device)
    light_tensor = t_rgb(light_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = models_dict['ela'](ela_tensor, rgb_tensor)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    results['ELA %Real'], results['ELA %AI'] = prob[0]*100, prob[1]*100
    results['ELA Label'] = "Real" if prob[0] >= prob[1] else "AI-Generated"

    # ----- Pixel (4 channels) -----
    pix_img = convert_to_pixel_map_from_pil(img)
    
    # Specific transform for pixel map (4 channels)
    t_pix = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*4, [0.5]*4)
    ])
    
    pix_tensor = t_pix(pix_img).unsqueeze(0).to(device)
    # Re-use rgb_tensor from ELA section or create new if needed (it's same 224x224 RGB)
    # rgb_tensor is already (1, 3, 224, 224) on device
    
    with torch.no_grad():
        out = models_dict['pixel'](pix_tensor, rgb_tensor)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    results['Pixel %Real'], results['Pixel %AI'] = prob[0]*100, prob[1]*100
    results['Pixel Label'] = "Real" if prob[0] >= prob[1] else "AI-Generated"

    # ----- Frequency (3 channels) -----
    freq_img = fft_feature_map(img)
    
    # Specific transform for freq (3 channels)
    # Normalization should match training: 0.5 mean, 0.5 std
    t_freq = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    freq_tensor = t_freq(freq_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = models_dict['freq'](freq_tensor)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()
    results['Freq %Real'], results['Freq %AI'] = prob[0]*100, prob[1]*100
    results['Freq Label'] = "Real" if prob[0] >= prob[1] else "AI-Generated"

    return results


# ==============================================================
# 🔹 Main Batch Function
# ==============================================================
def batch_infer_all(image_folder="Dataset_Senior_Project"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("🔹 Loading all 4 models...")
    models_dict = {
        'ela': ELRes(num_classes=2).to(device).eval(),
        'pixel': PixelRes().to(device).eval(),
        'freq': FreqResNet().to(device).eval(),
        'xception': XceptionBinary().to(device).eval()
    }

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent.parent
    model_dir = script_dir / 'model'
    
    ckpt_paths = {
        'ela': model_dir / 'full_model_ela.pkl',
        'pixel': model_dir / 'full_model_pixelhybrid.pkl',
        'freq': model_dir / 'full_model_freq.pkl',
        'xception': model_dir / 'full_model_xception.pkl'
    }

    for key, path in ckpt_paths.items():
        if not path.exists():
            print(f"⚠️ {key} checkpoint not found at {path}, skipping...")
            continue
        
        try:
            import torch.nn as nn
            
            if str(path).endswith(('.pkl', '.pickle')):
                # Use custom device mapping for pickle files
                ckpt_raw = load_checkpoint_with_device_map(str(path), device)
                
                # Extract state_dict properly
                if isinstance(ckpt_raw, nn.Module):
                    state = ckpt_raw.state_dict() if hasattr(ckpt_raw, 'state_dict') else ckpt_raw
                elif isinstance(ckpt_raw, dict):
                    if 'model_state_dict' in ckpt_raw:
                        state = ckpt_raw.get('model_state_dict')
                    elif 'state_dict' in ckpt_raw:
                        state = ckpt_raw.get('state_dict')
                    else:
                        state = ckpt_raw
                else:
                    state = ckpt_raw
            else:
                # Load PyTorch .pth files
                ckpt = torch.load(str(path), map_location=torch.device('cpu'), weights_only=False)
                state = ckpt.get('model_state_dict', ckpt)
            
            models_dict[key].load_state_dict(state, strict=False)
            print(f"✅ Loaded {key} checkpoint from {path}")
        except Exception as e:
            print(f"⚠️ Failed to load {key} from {path}: {e}")

    images = get_image_list(image_folder)
    print(f"\n📁 Found {len(images)} images in '{image_folder}'\n")

    if len(images) == 0:
        print(f"❌ No images found in {image_folder}")
        return

    all_results = []
    for img_path in images:
        try:
            result = predict_all_models(img_path, device, models_dict)
            all_results.append(result)
            print(f"✅ {result['Image Name']}")
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    # 🔹 Save to CSV
    output_csv = "combined_results.csv"
    fieldnames = ['Image Name', 'State',
                      'Xcep Label', 'Xcep %Real', 'Xcep %AI',
                      'Freq Label', 'Freq %Real', 'Freq %AI',
                      'Pixel Label', 'Pixel %Real', 'Pixel %AI',
                      'ELA Label', 'ELA %Real', 'ELA %AI']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n✅ Combined results saved to {output_csv}")


# ==============================================================
# 🔹 Run Example
# ==============================================================
if __name__ == "__main__":
    # Resolve dataset folder path
    script_dir = Path(__file__).parent
    possible_folders = [
        script_dir / 'uploads',           # Test images folder
        script_dir.parent / 'Dataset_test',  # If exists
        Path('Dataset_test'),              # Current directory
    ]
    
    dataset = None
    for folder in possible_folders:
        if folder.exists() and list(folder.glob('*.[jJ][pP][gG]')):
            dataset = str(folder)
            break
    
    if not dataset:
        print("❌ No test image folder found. Creating sample results...")
        dataset = str(script_dir / 'uploads')
    
    print(f"Using dataset: {dataset}\n")
    batch_infer_all(image_folder=dataset)
