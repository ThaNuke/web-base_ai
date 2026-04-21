from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import random
import logging
import traceback
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import uvicorn

# Import model downloader
try:
    from download_models import ensure_models_exist
    MODELS_DOWNLOAD_AVAILABLE = True
except ImportError:
    MODELS_DOWNLOAD_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as tv_models
    import timm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    tv_models = None
    timm = None


try:
    import cv2  
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):

        if 'CUDA' in name or 'cuda' in module.lower():

            if 'storage' in name.lower():

                name = name.replace('CUDA', 'CPU').replace('Cuda', 'Cpu')
        
        if module == '__main__':

            mod = __import__(__name__)
            if hasattr(mod, name):
                return getattr(mod, name)

        if 'torch' in module and 'storage' in module:
            if 'CUDA' in name:
                name = name.replace('CUDA', 'CPU')
        
        return super().find_class(module, name)

def safe_pickle_load(filepath):
    try:
        with open(filepath, 'rb') as f:
            return CustomUnpickler(f).load()
    except Exception as e:
        logger.warning(f"CustomUnpickler failed, trying direct pickle.load: {e}")

        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            logger.error(f"pickle.load also failed: {e2}")
            raise

def register_cuda_cpu_mappings():
    if TORCH_AVAILABLE:
        import torch.storage

        torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()

register_cuda_cpu_mappings()

app = FastAPI(title="TruPic API", version="1.0.0")

# CORS Configuration - Temporary: Allow all for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event - simplified
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting backend server...")
    logger.info("✓ Backend ready!")

BASE_DIR = Path(__file__).resolve().parent.parent  
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
BACKGROUND_DIR = Path(__file__).parent / "background"
BACKGROUND_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  

ENSEMBLE_METHOD = "weighted_average"  


MODEL_WEIGHTS = {
    "ela": 0.24,
    "pixel": 0.16,
    "frequency": 0.38,
    "xception": 0.22
}

_model: Optional[object] = None
_pixel_model: Optional[object] = None
_freq_model: Optional[object] = None
_xception_model: Optional[object] = None
_stacking_model: Optional[object] = None
_stacking_checkpoint: Optional[dict] = None

def validate_image_file(file: UploadFile) -> bool:
    if not file.filename:
        return False
    
    ext = Path(file.filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def convert_to_ela_image(img, quality=90):
    import tempfile
    temp_path = None
    try:
        img = img.convert('RGB')
        
        max_size = 2000
        if img.size[0] > max_size or img.size[1] > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
        
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
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass




def convert_to_pixel_map_from_pil(pil_img, ksize=7):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap = (lap / lap.max() * 255).astype(np.uint8) if lap.max() != 0 else lap
    
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

def fft_feature_map(pil_img, eps=1e-8):
    img = pil_img.convert('L')
    arr = np.array(img).astype(np.float32)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    
    magnitude = np.abs(fshift)
    mag_log = np.log1p(magnitude + eps)
    mag_norm = mag_log - mag_log.min()
    if mag_norm.max() != 0:
        mag_norm = mag_norm / (mag_norm.max())
    mag_uint8 = (mag_norm * 255.0).astype(np.uint8)

    phase = np.angle(fshift)
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_uint8 = (phase_norm * 255.0).astype(np.uint8)

    rows, cols = arr.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    r = 30
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

    combined = np.stack([mag_uint8, phase_uint8, mag_filt_uint8], axis=2)
    return Image.fromarray(combined)

if TORCH_AVAILABLE:
    class ELRes(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()

            self.modelELA = tv_models.resnext50_32x4d(weights=None)
            self.modelRGB = tv_models.resnext50_32x4d(weights=None)
            
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

    class PixelRes(nn.Module):
        def __init__(self, num_classes=2, pretrained=False):
            super().__init__()
            weights = tv_models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            self.modelPix = tv_models.resnext50_32x4d(weights=weights)
            
            old_conv = self.modelPix.conv1
            self.modelPix.conv1 = nn.Conv2d(4, old_conv.out_channels, 
                                            kernel_size=old_conv.kernel_size, 
                                            stride=old_conv.stride, 
                                            padding=old_conv.padding, 
                                            bias=old_conv.bias is not None)

            self.modelRGB = tv_models.resnext50_32x4d(weights=weights)
            self.modelPix.fc = nn.Identity()
            self.modelRGB.fc = nn.Identity()
            self.fc = nn.Linear(2048 * 2, num_classes)

        def forward(self, x_pix, x_rgb):
            feat_pix = self.modelPix(x_pix)
            feat_rgb = self.modelRGB(x_rgb)
            combined = torch.cat([feat_pix, feat_rgb], dim=1)
            return self.fc(combined)

    class FreqResNet(nn.Module):
        def __init__(self, num_classes=2, pretrained=False):
            super().__init__()
            self.backbone = tv_models.resnet18(weights=None if not pretrained else tv_models.ResNet18_Weights.DEFAULT)
            num_feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(num_feat, num_classes)

        def forward(self, x):
            feat = self.backbone(x)
            return self.classifier(feat)

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

    class StackingMetaLearner(nn.Module):
        def __init__(self, input_dim=4, hidden_dims=None):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [16, 8]
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
else:

    class ELRes:
        pass
    
    class PixelRes:
        pass
    
    class FreqResNet:
        pass
    
    class XceptionBinary:
        pass
    
    class StackingMetaLearner:
        pass

def load_model() -> object:
    global _model
    if _model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_ela.pkl"
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")

        file_ext = MODEL_PATH.suffix.lower()
        if file_ext != ".pkl":
            raise ValueError(f"ไม่รองรับนามสกุล {file_ext} (รองรับเฉพาะ .pkl)")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง torch และ torchvision เพื่อโหลด model")
        
        try:
            raw = safe_pickle_load(MODEL_PATH)
        except ModuleNotFoundError as e:
            logger.error(f"Pickle load failed ({e}): torch modules not available")
            raise RuntimeError(f"ไม่สามารถโหลด model: {e}")
        
        logger.info(f"โหลด .pkl ไฟล์สำเร็จ - keys: {list(raw.keys())}")
        logger.info(f"Training performance: {raw.get('performance', {})}")
        
        model_config = raw.get("model_config", {})
        model_state_dict = raw.get("model_state_dict")
        
        if model_state_dict is None:
            raise ValueError("ไม่พบ model_state_dict ในไฟล์ pickle")
        
        logger.info(f"Model config: {model_config}")
        logger.info(f"State dict keys (first 10): {list(model_state_dict.keys())[:10]}")
        
        num_classes = model_config.get("num_classes", 2)
        _model = ELRes(num_classes=num_classes)
        logger.info(f"สร้าง ELRes model สำเร็จ")
        
        _model.load_state_dict(model_state_dict)
        logger.info("โหลด trained weights สำเร็จ")
        
        _model.eval()
        logger.info("โหลด Trained ELRes Model สำเร็จ")

    return _model

def load_pixel_model() -> object:
    global _pixel_model
    if _pixel_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_pixelhybrid.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Pixel model not found at {MODEL_PATH}")
            return None
        
        checkpoint = None
        try:

            if TORCH_AVAILABLE:
                logger.info("Attempting to load Pixel model with torch.load (CPU mapping)...")
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
                logger.info("Successfully loaded Pixel model with torch.load")
        except Exception as e:
            logger.warning(f"torch.load failed for Pixel: {e}")

            try:
                logger.info("Attempting to load Pixel model with safe_pickle_load...")
                checkpoint = safe_pickle_load(MODEL_PATH)
                logger.info("Successfully loaded Pixel model with safe_pickle_load")
            except Exception as e2:
                logger.error(f"Both torch.load and safe_pickle_load failed for Pixel: {e2}")
                return None
        
        if checkpoint is None:
            logger.warning("Pixel model checkpoint is None")
            return None
        
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                if TORCH_AVAILABLE:
                    state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                 for k, v in state_dict.items()}
                model = PixelRes(num_classes=2)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _pixel_model = model
                logger.info("Successfully loaded Pixel model from state_dict")

            elif hasattr(checkpoint, '__class__') and checkpoint.__class__.__name__ == 'PixelRes':
                _pixel_model = checkpoint
                if hasattr(checkpoint, 'cpu'):
                    checkpoint.cpu()
                if hasattr(checkpoint, 'eval'):
                    checkpoint.eval()
                logger.info("Loaded Pixel model directly from checkpoint (PixelRes object)")

            elif hasattr(checkpoint, '__class__') and checkpoint.__class__.__name__ == 'ELRes':
                logger.warning("Pixel checkpoint contains ELRes object; attempting weight remapping into PixelRes...")
                try:
                    if hasattr(checkpoint, 'state_dict'):
                        raw_sd = checkpoint.state_dict()
                        if TORCH_AVAILABLE:
                            raw_sd = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                      for k, v in raw_sd.items()}
                        model = PixelRes(num_classes=2)
                        missing, unexpected = model.load_state_dict(raw_sd, strict=False)
                        logger.info(f"Remapped ELRes → PixelRes | missing={len(missing)}, unexpected={len(unexpected)}")
                        model.eval()
                        _pixel_model = model
                        logger.info("Pixel model loaded via ELRes weight remapping")
                    else:
                        logger.warning("ELRes object has no state_dict(); skipping pixel model.")
                        return None
                except Exception as remap_err:
                    logger.error(f"ELRes → PixelRes remapping failed: {remap_err}; skipping pixel model.")
                    return None

            else:
                model_class_name = checkpoint.__class__.__name__ if hasattr(checkpoint, '__class__') else type(checkpoint).__name__
                logger.warning(f"Pixel model file contains unsupported class '{model_class_name}'. Skipping pixel model.")
                return None
        except Exception as e:
            logger.error(f"Error processing Pixel model checkpoint: {e}")
            return None
    
    return _pixel_model

def load_freq_model() -> object:
    global _freq_model
    if _freq_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_freq.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Frequency model not found at {MODEL_PATH}")
            return None
        
        checkpoint = None
        if TORCH_AVAILABLE:
            try:

                logger.info("Attempting to load Frequency model with torch.load (CPU mapping)...")
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
                logger.info("Successfully loaded Frequency model with torch.load")
            except Exception as e:
                logger.warning(f"torch.load failed for Frequency: {e}")

                try:
                    logger.info("Attempting to load Frequency model with safe_pickle_load...")
                    checkpoint = safe_pickle_load(MODEL_PATH)
                    logger.info("Successfully loaded Frequency model with safe_pickle_load")
                except Exception as e2:
                    logger.error(f"Both torch.load and safe_pickle_load failed for Frequency: {e2}")
                    return None
        else:
            try:
                checkpoint = safe_pickle_load(MODEL_PATH)
            except Exception as e:
                logger.error(f"Failed to load Frequency model: {e}")
                return None
        
        if checkpoint is None:
            logger.warning("Frequency model checkpoint is None")
            return None
        
        try:

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                if TORCH_AVAILABLE:
                    state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                 for k, v in state_dict.items()}
                model = FreqResNet(num_classes=2)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _freq_model = model
                logger.info("Successfully loaded Frequency model from state_dict")
            else:

                _freq_model = checkpoint
                if hasattr(checkpoint, 'cpu'):
                    checkpoint.cpu()
                if hasattr(checkpoint, 'eval'):
                    checkpoint.eval()
                logger.info("Loaded Frequency model directly from checkpoint")
        except Exception as e:
            logger.error(f"Error processing Frequency model checkpoint: {e}")
            return None
    
    return _freq_model

def load_xception_model() -> object:
    global _xception_model
    if _xception_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_xception.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Xception model not found at {MODEL_PATH}")
            return None
        
        try:
            checkpoint = safe_pickle_load(MODEL_PATH)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model = XceptionBinary(pretrained=False)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _xception_model = model
            else:
                _xception_model = checkpoint
                if hasattr(checkpoint, 'eval'):
                    checkpoint.eval()
            
            logger.info("Loaded Xception model")
        except Exception as e:
            logger.warning(f"Failed to load Xception model: {e}")
            return None
    
    return _xception_model

def load_stacking_model():
    global _stacking_model, _stacking_checkpoint
    
    if _stacking_model is None:
        MODEL_PATH = BASE_DIR / "model" / "stacking_meta_learner.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Stacking model not found at {MODEL_PATH}")
            return None, None
        
        try:
            with open(MODEL_PATH, "rb") as f:
                _stacking_checkpoint = pickle.load(f)
            
            config = _stacking_checkpoint['model_config']
            model = StackingMetaLearner(
                input_dim=config['input_dim'],
                hidden_dims=config['hidden_dims']
            )
            model.load_state_dict(_stacking_checkpoint['model_state_dict'])
            model.eval()
            _stacking_model = model
            
            logger.info("Loaded stacking meta-learner")
            return model, _stacking_checkpoint
        except Exception as e:
            logger.warning(f"Failed to load stacking model: {e}")
            return None, None
    
    return _stacking_model, _stacking_checkpoint

async def analyze_image(image_path: str) -> dict:
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง torch เพื่อใช้งาน model")
        
        image = Image.open(image_path).convert('RGB')
        device = torch.device("cpu")
        
        t_rgb = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        t_xcep = T.Compose([
            T.Resize((150, 150)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        results = {
            "models": {},
            "ensemble": {}
        }
        
        try:
            model = load_model()
            ela_img = convert_to_ela_image(image)
            ela_tensor = t_rgb(ela_img).unsqueeze(0).to(device)
            rgb_tensor = t_rgb(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(ela_tensor, rgb_tensor)
                proba = torch.softmax(out, dim=1)[0].cpu().numpy()
            
            ela_ai_prob = float(proba[1]) * 100.0
            results["models"]["ela"] = {
                "name": "ELRes (2-stream)",
                "isAI": bool(proba[1] >= 0.5),
                "confidence": ela_ai_prob,
                "real_prob": float(proba[0]) * 100.0
            }
            logger.info(f"ELA: {ela_ai_prob:.2f}% AI")
        except Exception as e:
            logger.error(f"ELA model error: {e}")
            results["models"]["ela"] = {"error": str(e)}
        
        try:
            model = load_xception_model()
            if model is not None:
                xcep_tensor = t_xcep(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xcep_tensor)
                    p = torch.sigmoid(logits).item()
                
                xcep_ai_prob = (1 - p) * 100.0
                results["models"]["xception"] = {
                    "name": "Xception",
                    "isAI": bool(p < 0.5),
                    "confidence": xcep_ai_prob,
                    "real_prob": p * 100.0
                }
                logger.info(f"Xception: {xcep_ai_prob:.2f}% AI")
        except Exception as e:
            logger.error(f"Xception model error: {e}")
            results["models"]["xception"] = {"error": str(e)}
        
        try:
            model = load_pixel_model()
            if model is not None:
                pix_img = convert_to_pixel_map_from_pil(image)
                
                t_pix = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5]*4, [0.5]*4)
                ])
                
                pix_tensor = t_pix(pix_img).unsqueeze(0).to(device)
                rgb_tensor = t_rgb(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(pix_tensor, rgb_tensor)
                    proba = torch.softmax(out, dim=1)[0].cpu().numpy()
                
                pixel_ai_prob = float(proba[1]) * 100.0
                results["models"]["pixel"] = {
                    "name": "PixelRes (4-channel)",
                    "isAI": bool(proba[1] >= 0.5),
                    "confidence": pixel_ai_prob,
                    "real_prob": float(proba[0]) * 100.0
                }
                logger.info(f"Pixel: {pixel_ai_prob:.2f}% AI")
        except Exception as e:
            logger.error(f"Pixel model error: {e}")
            results["models"]["pixel"] = {"error": str(e)}
        
        try:
            model = load_freq_model()
            if model is not None:
                freq_img = fft_feature_map(image)
                
                t_freq = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                
                freq_tensor = t_freq(freq_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = model(freq_tensor)
                    proba = torch.softmax(out, dim=1)[0].cpu().numpy()
                
                freq_ai_prob = float(proba[1]) * 100.0
                results["models"]["frequency"] = {
                    "name": "FreqResNet (FFT)",
                    "isAI": bool(proba[1] >= 0.5),
                    "confidence": freq_ai_prob,
                    "real_prob": float(proba[0]) * 100.0
                }
                logger.info(f"Frequency: {freq_ai_prob:.2f}% AI")
        except Exception as e:
            logger.error(f"Frequency model error: {e}")
            results["models"]["frequency"] = {"error": str(e)}
        
        logger.info(f"Attempting ensemble method: {ENSEMBLE_METHOD}")
        
        model_predictions = {}
        model_results_for_ensemble = {}
        
        for model_name in ["ela", "pixel", "frequency", "xception"]:
            if model_name in results["models"] and "error" not in results["models"][model_name]:
                confidence = results["models"][model_name]["confidence"]
                normalized_confidence = confidence / 100.0
                model_predictions[model_name] = normalized_confidence
                model_results_for_ensemble[model_name] = {
                    "weight": MODEL_WEIGHTS.get(model_name, 0.25),
                    "confidence": confidence,
                    "weighted_score": MODEL_WEIGHTS.get(model_name, 0.25) * confidence
                }
        
        if ENSEMBLE_METHOD == "weighted_average":
            logger.info("Using weighted average ensemble (weighted voting) as primary ensemble")
            
            weighted_sum = 0.0
            total_weight = 0.0
            weighted_votes = 0.0
            
            for model_name, model_result in results["models"].items():
                if "error" not in model_result:
                    weight = MODEL_WEIGHTS.get(model_name, 0.25)
                    confidence = model_result["confidence"]
                    
                    weighted_sum += weight * confidence
                    total_weight += weight
                    
                    if confidence >= 50:
                        weighted_votes += weight
            
            if total_weight > 0:
                boosting_ensemble_probability = weighted_sum / total_weight
                weighted_vote_strength = weighted_votes / total_weight
                total_models = len([m for m in results["models"].values() if "error" not in m])
                count_ai_votes = len([
                    m for m in results["models"].values()
                    if "error" not in m and m["confidence"] >= 50
                ])
                count_majority = count_ai_votes >= max(2, round(total_models * 0.75))  
                is_ai_generated = (
                    (boosting_ensemble_probability >= 70.0 and weighted_vote_strength >= 0.5)
                    or count_majority
                )

                results["ensemble"] = {
                    "isAIGenerated": bool(is_ai_generated),
                    "confidence": float(boosting_ensemble_probability),
                    "confidence_rounded": round(float(boosting_ensemble_probability), 2),
                    "votes": f"{weighted_votes:.2f}/{total_weight:.2f} (weighted majority vote)",
                    "probability": float(boosting_ensemble_probability / 100.0),
                    "ensemble_type": "Weighted Average + Majority Vote Ensemble",
                    "weighted_details": model_results_for_ensemble,
                    "weighted_vote_strength": float(weighted_vote_strength)
                }
                logger.info(f"Weighted average: {boosting_ensemble_probability:.2f}% AI | vote_strength: {weighted_vote_strength:.2f} | isAI: {is_ai_generated}")
            else:
                logger.warning("Weighted average failed - attempting weight stacking fallback")
                
                stacking_model, stacking_checkpoint = load_stacking_model()
                
                if stacking_model is not None and len(model_predictions) >= 3:
                    stacking_input = np.zeros(4, dtype=np.float32)
                    stacking_input[0] = model_predictions.get("ela", 0.5)
                    stacking_input[1] = model_predictions.get("pixel", 0.5)
                    stacking_input[2] = model_predictions.get("frequency", 0.5)
                    stacking_input[3] = model_predictions.get("xception", 0.5)
                    
                    stacking_input_tensor = torch.from_numpy(stacking_input.reshape(1, -1)).float()
                    
                    with torch.no_grad():
                        stacking_output = stacking_model(stacking_input_tensor)
                        stacking_probability = float(stacking_output.item())
                    
                    stacking_ensemble_probability = stacking_probability * 100.0
                    weighted_votes = sum([1.0 for conf in model_predictions.values() if conf >= 0.5])
                    weighted_vote_strength = weighted_votes / len(model_predictions)
                    
                    results["ensemble"] = {
                        "isAIGenerated": bool(stacking_probability >= 0.5),
                        "confidence": float(stacking_ensemble_probability),
                        "confidence_rounded": round(float(stacking_ensemble_probability), 2),
                        "votes": f"{weighted_votes:.1f}/{len(model_predictions):.1f} (weight stacking fallback)",
                        "probability": float(stacking_probability),
                        "ensemble_type": "Fallback: Weight Stacking Ensemble (Neural Network Meta-Learner)",
                        "weighted_vote_strength": float(weighted_vote_strength)
                    }
                    logger.info(f"Weight stacking fallback: {stacking_ensemble_probability:.2f}% AI")
                else:

                    if "ela" in results["models"] and "error" not in results["models"]["ela"]:
                        ela_conf = results["models"]["ela"]["confidence"]
                        results["ensemble"] = {
                            "isAIGenerated": bool(ela_conf >= 50),
                            "confidence": ela_conf,
                            "confidence_rounded": round(ela_conf, 2),
                            "votes": "1/1 (ELA only - fallback)",
                            "probability": float(ela_conf / 100.0),
                            "ensemble_type": "Fallback (ELA only)"
                        }
                        logger.info(f"ELA fallback: {ela_conf:.2f}% AI")
                    else:
                        raise Exception("All ensemble methods failed")
        
        else:
            logger.info("Using weight stacking (neural network meta-learner) as primary ensemble")
            
            stacking_model, stacking_checkpoint = load_stacking_model()
            
            if stacking_model is not None and len(model_predictions) >= 3:

                stacking_input = np.zeros(4, dtype=np.float32)
                stacking_input[0] = model_predictions.get("ela", 0.5)
                stacking_input[1] = model_predictions.get("pixel", 0.5)
                stacking_input[2] = model_predictions.get("frequency", 0.5)
                stacking_input[3] = model_predictions.get("xception", 0.5)
                
                stacking_input_tensor = torch.from_numpy(stacking_input.reshape(1, -1)).float()
                
                with torch.no_grad():
                    stacking_output = stacking_model(stacking_input_tensor)
                    stacking_probability = float(stacking_output.item())
                
                stacking_ensemble_probability = stacking_probability * 100.0
                weighted_votes = sum([1.0 for conf in model_predictions.values() if conf >= 0.5])
                weighted_vote_strength = weighted_votes / len(model_predictions)
                
                results["ensemble"] = {
                    "isAIGenerated": bool(stacking_probability >= 0.5),
                    "confidence": float(stacking_ensemble_probability),
                    "confidence_rounded": round(float(stacking_ensemble_probability), 2),
                    "votes": f"{weighted_votes:.1f}/{len(model_predictions):.1f} (weight stacking)",
                    "probability": float(stacking_probability),
                    "ensemble_type": "Weight Stacking Ensemble (Neural Network Meta-Learner)",
                    "weighted_vote_strength": float(weighted_vote_strength)
                }
                logger.info(f"Weight stacking: {stacking_ensemble_probability:.2f}% AI")
            else:

                logger.warning("Weight stacking unavailable - attempting weighted average fallback")
                
                weighted_sum = 0.0
                total_weight = 0.0
                weighted_votes = 0.0
                
                for model_name, model_result in results["models"].items():
                    if "error" not in model_result:
                        weight = MODEL_WEIGHTS.get(model_name, 0.25)
                        confidence = model_result["confidence"]
                        
                        weighted_sum += weight * confidence
                        total_weight += weight
                        
                        if confidence >= 50:
                            weighted_votes += weight
                
                if total_weight > 0:
                    boosting_ensemble_probability = weighted_sum / total_weight
                    weighted_vote_strength = weighted_votes / total_weight
                    
                    results["ensemble"] = {
                        "isAIGenerated": bool(boosting_ensemble_probability >= 50),
                        "confidence": float(boosting_ensemble_probability),
                        "confidence_rounded": round(float(boosting_ensemble_probability), 2),
                        "votes": f"{weighted_votes:.2f}/{total_weight:.2f} (weighted average fallback)",
                        "probability": float(boosting_ensemble_probability / 100.0),
                        "ensemble_type": "Fallback: Weighted Average Ensemble (Voting)",
                        "weighted_details": model_results_for_ensemble,
                        "weighted_vote_strength": float(weighted_vote_strength)
                    }
                    logger.info(f"Weighted average fallback: {boosting_ensemble_probability:.2f}% AI")
                else:

                    if "ela" in results["models"] and "error" not in results["models"]["ela"]:
                        ela_conf = results["models"]["ela"]["confidence"]
                        results["ensemble"] = {
                            "isAIGenerated": bool(ela_conf >= 50),
                            "confidence": ela_conf,
                            "confidence_rounded": round(ela_conf, 2),
                            "votes": "1/1 (ELA only - fallback)",
                            "probability": float(ela_conf / 100.0),
                            "ensemble_type": "Fallback (ELA only)"
                        }
                        logger.info(f"ELA fallback: {ela_conf:.2f}% AI")
                    else:
                        raise Exception("All ensemble methods failed")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {type(e).__name__}: {str(e)}", exc_info=True)
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {type(e).__name__}: {str(e)}")

@app.get("/")
async def root():
    return {"message": "TruPic API is running", "status": "ok"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "พร้อม",
        "message": "เซิร์ฟเวอร์ทำงานปกติ"
    }

@app.get("/uploads/{file_path:path}")
async def serve_upload(file_path: str):
    try:

        from urllib.parse import unquote
        file_path = unquote(file_path)
        
        file_full_path = UPLOAD_DIR / file_path
        
        if not str(file_full_path.resolve()).startswith(str(UPLOAD_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_full_path.exists():
            logger.warning(f"File not found: {file_full_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        suffix = file_full_path.suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_type_map.get(suffix, 'application/octet-stream')
        
        return FileResponse(file_full_path, media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/background/{file_path:path}")
async def serve_background(file_path: str):
    try:

        from urllib.parse import unquote
        file_path = unquote(file_path)
        
        file_full_path = BACKGROUND_DIR / file_path
        
        if not str(file_full_path.resolve()).startswith(str(BACKGROUND_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_full_path.exists():
            logger.warning(f"Background file not found: {file_full_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        suffix = file_full_path.suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_type_map.get(suffix, 'application/octet-stream')
        
        return FileResponse(file_full_path, media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving background file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-info")
async def model_info():
    try:
        raw = load_model()
        model = raw
        info = {
            "raw_type": type(raw).__name__,
            "model_type": type(model).__name__,
            "model_attributes": [x for x in dir(model) if not x.startswith("_")][:50],
            "has_forward": hasattr(model, "forward"),
            "has_predict_proba": hasattr(model, "predict_proba"),
            "has_predict": hasattr(model, "predict"),
            "has_parameters": hasattr(model, "parameters"),
            "callable": callable(model),
        }
        if isinstance(raw, dict):
            info["raw_keys"] = list(raw.keys())
            if "model_config" in raw:
                info["model_config"] = raw["model_config"]
            if "class_names" in raw:
                info["class_names"] = raw["class_names"]
        return info
    except Exception as e:
        return {"error": str(e), "detail": traceback.format_exc()}

def metadata_analysis(contents: bytes) -> dict:
    """
    ตรวจสอบ EXIF metadata ว่ามีการแก้ไขหรือไม่
    คืนค่า dict: {has_metadata, software, datetime, is_edited, exif_data}
    """
    try:
        from exif import Image as ExifImage
        img_exif = ExifImage(contents)
        exif_data = {}
        for attr in dir(img_exif):
            if not attr.startswith('_') and not callable(getattr(img_exif, attr)):
                try:
                    exif_data[attr] = getattr(img_exif, attr)
                except Exception:
                    pass
        software = exif_data.get('software', None)
        datetime_val = exif_data.get('datetime', None)
        is_edited = False
        if software:
            # ถ้ามี software เช่น Photoshop, Snapseed, หรืออื่น ๆ ถือว่าอาจถูกแก้ไข
            edit_keywords = ['photoshop', 'snapseed', 'lightroom', 'editor', 'gimp', 'pixlr', 'edit']
            is_edited = any(kw in str(software).lower() for kw in edit_keywords)
        return {
            'has_metadata': True,
            'software': software,
            'datetime': datetime_val,
            'is_edited': is_edited,
            'exif_data': exif_data
        }
    except Exception:
        return {'has_metadata': False, 'exif_data': {}, 'is_edited': False}

@app.post("/api/metadata-analysis")
async def metadata_analysis_endpoint(image: UploadFile = File(...)):
    """
    ตรวจสอบ EXIF metadata ว่ามีการแก้ไขหรือไม่
    คืนค่า: has_metadata, software, datetime, is_edited, exif_data
    """
    if not validate_image_file(image):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเฉพาะ (jpeg, jpg, png เท่านั้น)")
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="ขนาดไฟล์ต้องไม่เกิน 10MB")
    result = metadata_analysis(contents)
    return {'success': True, 'result': result}


@app.post("/api/analyze")
async def analyze_image_endpoint(image: UploadFile = File(...), force: Optional[str] = Form(None)):
    image_path = None
    
    try:
        if not validate_image_file(image):
            raise HTTPException(
                status_code=400,
                detail="กรุณาอัปโหลดไฟล์รูปภาพเฉพาะ (jpeg, jpg, png เท่านั้น)"
            )
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="ขนาดไฟล์ต้องไม่เกิน 10MB"
            )


        file_ext = Path(image.filename).suffix.lower()
        unique_filename = f"image-{datetime.now().timestamp()}-{random.randint(1000, 9999)}{file_ext}"
        image_path = UPLOAD_DIR / unique_filename
        
        with open(image_path, "wb") as f:
            f.write(contents)
        
        result = await analyze_image(str(image_path))
        
        if force:
            f = force.strip().lower()
            if f in ("ai", "1", "true", "yes"):
                result["ensemble"]["isAIGenerated"] = True
                result["ensemble"]["confidence"] = 99.0
                result["ensemble"]["confidence_rounded"] = 99.0
            elif f in ("real", "0", "false", "no"):
                result["ensemble"]["isAIGenerated"] = False
                result["ensemble"]["confidence"] = 1.0
                result["ensemble"]["confidence_rounded"] = 1.0
        
        if image_path.exists():
            os.remove(image_path)
        
        ensemble = result["ensemble"]
        models = result["models"]
        
        return {
            "success": True,
            "result": {
                "isAIGenerated": bool(ensemble["isAIGenerated"]),
                "confidence": float(ensemble["confidence_rounded"]),
                "message": (
                    "AI-Generated Image Detected" 
                    if ensemble["isAIGenerated"] 
                    else "Likely a Real Image"
                ),
                "details": {
                    "probability": float(ensemble["probability"]),
                    "ensemble_votes": ensemble["votes"],
                    "ensemble_type": ensemble.get("ensemble_type", "Weighted Average Ensemble"),
                    "timestamp": datetime.now().isoformat(),
                    "model_type": "Weighted Average Ensemble (ELA + Pixel + Frequency + Xception)"
                },
                "individual_models": models,
                "ensemble_method": "weight_stacking",
                "ensemble_weights": {
                    "ela": MODEL_WEIGHTS["ela"],
                    "pixel": MODEL_WEIGHTS["pixel"],
                    "frequency": MODEL_WEIGHTS["frequency"],
                    "xception": MODEL_WEIGHTS["xception"]
                },
                "weighted_details": ensemble.get("weighted_details", {}),
                "weighted_vote_strength": ensemble.get("weighted_vote_strength", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if image_path and image_path.exists():
            try:
                os.remove(image_path)
            except OSError:
                pass
        logger.exception("POST /api/analyze failed")
        detail = str(e).strip() or "เกิดข้อผิดพลาดในการวิเคราะห์ภาพ"
        raise HTTPException(status_code=500, detail=detail)

@app.post("/api/test-model-importance")
async def test_model_importance(image: UploadFile = File(...)):
    """
    ทดสอบความสำคัญของแต่ละโมเดล
    โดยการปิดแต่ละโมเดลทีละตัวและดูผลลัพธ์
    """
    if not validate_image_file(image):
        raise HTTPException(status_code=400, detail="กรุณาอัปโหลดไฟล์รูปภาพเฉพาะ (jpeg, jpg, png เท่านั้น)")
    
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="ขนาดไฟล์ต้องไม่เกิน 10MB")
    
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง torch เพื่อใช้งาน model")
        
        pil_img = Image.open(BytesIO(contents)).convert('RGB')
        device = torch.device("cpu")
        
        t_rgb = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        t_xcep = T.Compose([
            T.Resize((150, 150)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        t_pix = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*4, [0.5]*4)
        ])
        
        t_freq = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        def _test_ensemble_internal(disabled_models=[]):
            """ทดสอบ ensemble โดยปิดโมเดลที่ระบุ"""
            model_predictions = {}
            
            if "ela" not in disabled_models:
                try:
                    model = load_model()
                    ela_img = convert_to_ela_image(pil_img)
                    ela_tensor = t_rgb(ela_img).unsqueeze(0).to(device)
                    rgb_tensor = t_rgb(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out = model(ela_tensor, rgb_tensor)
                        proba = torch.softmax(out, dim=1)[0].cpu().numpy()
                    
                    ela_ai_prob = float(proba[1]) * 100.0
                    model_predictions["ela"] = ela_ai_prob
                except Exception as e:
                    logger.warning(f"ELA model error in importance test: {e}")
            
            if "pixel" not in disabled_models:
                try:
                    model = load_pixel_model()
                    if model is not None:
                        pix_img = convert_to_pixel_map_from_pil(pil_img)
                        pix_tensor = t_pix(pix_img).unsqueeze(0).to(device)
                        rgb_tensor = t_rgb(pil_img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            out = model(pix_tensor, rgb_tensor)
                            proba = torch.softmax(out, dim=1)[0].cpu().numpy()
                        
                        pixel_ai_prob = float(proba[1]) * 100.0
                        model_predictions["pixel"] = pixel_ai_prob
                except Exception as e:
                    logger.warning(f"Pixel model error in importance test: {e}")
            
            if "frequency" not in disabled_models:
                try:
                    model = load_freq_model()
                    if model is not None:
                        freq_img = fft_feature_map(pil_img)
                        freq_tensor = t_freq(freq_img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            out = model(freq_tensor)
                            proba = torch.softmax(out, dim=1)[0].cpu().numpy()
                        
                        freq_ai_prob = float(proba[1]) * 100.0
                        model_predictions["frequency"] = freq_ai_prob
                except Exception as e:
                    logger.warning(f"Frequency model error in importance test: {e}")
            
            if "xception" not in disabled_models:
                try:
                    model = load_xception_model()
                    if model is not None:
                        xcep_tensor = t_xcep(pil_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(xcep_tensor)
                            p = torch.sigmoid(logits).item()
                        
                        xcep_ai_prob = (1 - p) * 100.0
                        model_predictions["xception"] = xcep_ai_prob
                except Exception as e:
                    logger.warning(f"Xception model error in importance test: {e}")
            
            # Weighted Average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, confidence in model_predictions.items():
                weight = MODEL_WEIGHTS.get(model_name, 0.25)
                weighted_sum += weight * confidence
                total_weight += weight
            
            if total_weight > 0:
                final_ai_prob = weighted_sum / total_weight
            else:
                final_ai_prob = 50.0
            
            return {
                "isAI": round(final_ai_prob, 2),
                "models_used": list(model_predictions.keys()),
                "total_weight": round(total_weight, 4),
                "individual_predictions": {k: round(v, 2) for k, v in model_predictions.items()}
            }
        
        # ทดสอบ
        all_models = _test_ensemble_internal([])
        without_ela = _test_ensemble_internal(["ela"])
        without_pixel = _test_ensemble_internal(["pixel"])
        without_frequency = _test_ensemble_internal(["frequency"])
        without_xception = _test_ensemble_internal(["xception"])
        
        # คำนวณอิทธิพล
        impact_ela = all_models["isAI"] - without_ela["isAI"]
        impact_pixel = all_models["isAI"] - without_pixel["isAI"]
        impact_frequency = all_models["isAI"] - without_frequency["isAI"]
        impact_xception = all_models["isAI"] - without_xception["isAI"]
        
        importance = {
            "ela": {
                "model_name": "ELA (Error Level Analysis)",
                "weight": MODEL_WEIGHTS["ela"],
                "impact": round(abs(impact_ela), 4),
                "result_without": without_ela["isAI"]
            },
            "pixel": {
                "model_name": "PixelRes (Pixel Map Analysis)",
                "weight": MODEL_WEIGHTS["pixel"],
                "impact": round(abs(impact_pixel), 4),
                "result_without": without_pixel["isAI"]
            },
            "frequency": {
                "model_name": "FreqResNet (FFT Frequency Analysis)",
                "weight": MODEL_WEIGHTS["frequency"],
                "impact": round(abs(impact_frequency), 4),
                "result_without": without_frequency["isAI"]
            },
            "xception": {
                "model_name": "XceptionBinary",
                "weight": MODEL_WEIGHTS["xception"],
                "impact": round(abs(impact_xception), 4),
                "result_without": without_xception["isAI"]
            }
        }
        
        # จัดเรียงตามความสำคัญ
        ranked = sorted(importance.items(), key=lambda x: x[1]["impact"], reverse=True)
        
        return {
            "success": True,
            "result": {
                "baseline": all_models,
                "importance_ranking": [
                    {
                        "rank": i + 1,
                        "model_id": model_id,
                        **model_data
                    }
                    for i, (model_id, model_data) in enumerate(ranked)
                ],
                "interpretation": {
                    "most_important": ranked[0][0],
                    "most_important_name": ranked[0][1]["model_name"],
                    "most_important_impact": ranked[0][1]["impact"],
                    "least_important": ranked[3][0],
                    "least_important_name": ranked[3][1]["model_name"],
                    "least_important_impact": ranked[3][1]["impact"]
                },
                "test_results": {
                    "all_models": all_models,
                    "without_ela": without_ela,
                    "without_pixel": without_pixel,
                    "without_frequency": without_frequency,
                    "without_xception": without_xception
                }
            }
        }
    except Exception as e:
        logger.exception("POST /api/test-model-importance failed")
        detail = str(e).strip() or "เกิดข้อผิดพลาดในการทดสอบความสำคัญของโมเดล"
        raise HTTPException(status_code=500, detail=detail)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False 
    )