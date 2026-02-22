from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import random
import logging
import traceback
import math
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import uvicorn

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as tv_models
    import timm
    import cv2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    tv_models = None
    timm = None
    cv2 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TruPic API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent  
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".gif", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  

ENSEMBLE_METHOD = "weight_boosting"  


MODEL_WEIGHTS = {
    "ela": 0.20,           
    "pixel": 0.15,         
    "frequency": 0.50,     
    "xception": 0.15       
}

CONFIDENCE_THRESHOLDS = {
    "ela": 0.50,
    "pixel": 0.50,
    "frequency": 0.50,
    "xception": 0.50
}

_model: Optional[object] = None
_pixel_model: Optional[object] = None
_freq_model: Optional[object] = None
_xception_model: Optional[object] = None
_stacking_model: Optional[object] = None
_stacking_checkpoint: Optional[dict] = None


def validate_image_file(file: UploadFile) -> bool:
    """ตรวจสอบว่าไฟล์เป็นรูปภาพที่รองรับหรือไม่"""
    if not file.filename:
        return False
    
    ext = Path(file.filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS





def convert_to_ela_image(img, quality=90):
    """Convert image to ELA representation"""
    import tempfile
    temp_path = None
    try:
        img = img.convert('RGB')
        
        # Resize if too large
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
    """Extract 4-channel pixel features: Laplacian, Variance, SobelX, Gray"""
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

    # Variance
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
    """Extract frequency domain features: Magnitude, Phase, BandPass"""
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
    phase = np.angle(fshift)  # [-pi, pi]
    phase_norm = (phase + np.pi) / (2 * np.pi)
    phase_uint8 = (phase_norm * 255.0).astype(np.uint8)

    # 3. Band-Pass Filtered Magnitude
    rows, cols = arr.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # radius for high-pass
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


def _build_elres_model(config: dict):
    """
    สร้าง ELRes model ด้วย ResNeXt50_32X4D backbone (ตรงกับ checkpoint)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("ต้องติดตั้ง torch")
    
    num_classes = config.get("num_classes", 2)
    
    class ELRes(nn.Module):
        """2-stream ELRes model with ResNeXt50_32X4D backbones"""
        def __init__(self, num_classes=2):
            super().__init__()
            # Use ResNeXt50_32X4D (matches checkpoint)
            self.modelELA = tv_models.resnext50_32x4d(weights=None)
            self.modelRGB = tv_models.resnext50_32x4d(weights=None)
            
            # Get feature dimension before classification layer
            in_features = self.modelELA.fc.in_features  # 2048
            
            # Replace classification layers with identity
            self.modelELA.fc = nn.Identity()
            self.modelRGB.fc = nn.Identity()
            
            # Fusion: concatenate features from both streams, then classify
            self.fc = nn.Linear(in_features * 2, num_classes)
        
        def forward(self, x_ela, x_rgb):
            feat_ela = self.modelELA(x_ela)  # [B, 2048]
            feat_rgb = self.modelRGB(x_rgb)  # [B, 2048]
            feat_combined = torch.cat([feat_ela, feat_rgb], dim=1)  # [B, 4096]
            out = self.fc(feat_combined)  # [B, 2]
            return out
    
    model = ELRes(num_classes=num_classes)
    return model


if TORCH_AVAILABLE:
    class PixelRes(nn.Module):
        """4-channel Pixel feature extractor + RGB stream"""
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
        """Frequency domain ResNet18"""
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
        """Xception backbone for binary classification"""
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
        """
        Neural network meta-learner for weight stacking ensemble.
        
        Takes 4 model predictions (ELA, Pixel, Frequency, Xception) as inputs
        and learns the optimal way to combine them into a final prediction.
        """
        def __init__(self, input_dim=4, hidden_dims=None):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [16, 8]
            
            # Build hidden layers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim
            
            # Output layer (sigmoid for binary classification: 0-1 probability)
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            """
            Args:
                x: Tensor of shape (batch_size, 4) with model predictions [ELA, Pixel, Freq, Xcep]
                   Each value should be in range [0, 1] (normalized confidence)
            Returns:
                Tensor of shape (batch_size, 1) with combined prediction (0-1 probability)
            """
            return self.network(x)
else:
    # Dummy classes when torch is not available
    class PixelRes:
        pass
    
    class FreqResNet:
        pass
    
    class XceptionBinary:
        pass
    
    class StackingMetaLearner:
        pass





def load_model() -> object:
    """
    โหลด Trained ELRes Model จาก pickle file
    Model architectureคือ: 2-stream ResNeXt50_32X4D (ELA + RGB streams)
    """
    global _model
    if _model is None:
        MODEL_PATH = BASE_DIR / "model" / "elres_model.pkl"
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")

        file_ext = MODEL_PATH.suffix.lower()
        if file_ext != ".pkl":
            raise ValueError(f"ไม่รองรับนามสกุล {file_ext} (รองรับเฉพาะ .pkl)")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง torch และ torchvision เพื่อโหลด model")
        
        # โหลดข้อมูลจาก pickle file
        try:
            with open(MODEL_PATH, "rb") as f:
                raw = pickle.load(f)
        except ModuleNotFoundError as e:
            logger.error(f"Pickle load failed ({e}): torch modules not available")
            raise RuntimeError(f"ไม่สามารถโหลด model: {e}")
        
        logger.info(f"โหลด .pkl ไฟล์สำเร็จ - keys: {list(raw.keys())}")
        logger.info(f"Training performance: {raw.get('performance', {})}")
        
        # Extract model config and state dict
        model_config = raw.get("model_config", {})
        model_state_dict = raw.get("model_state_dict")
        
        if model_state_dict is None:
            raise ValueError("ไม่พบ model_state_dict ในไฟล์ pickle")
        
        logger.info(f"Model config: {model_config}")
        logger.info(f"State dict keys (first 10): {list(model_state_dict.keys())[:10]}")
        
        # Build ELRes model
        _model = _build_elres_model(model_config)
        logger.info(f"สร้าง ELRes model สำเร็จ")
        
        # Load trained weights
        _model.load_state_dict(model_state_dict)
        logger.info("โหลด trained weights สำเร็จ")
        
        # Set to evaluation mode
        _model.eval()
        logger.info("โหลด Trained ELRes Model สำเร็จ")

    return _model


def load_pixel_model() -> object:
    """Load Pixel model"""
    global _pixel_model
    if _pixel_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_pixelhybrid.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Pixel model not found at {MODEL_PATH}")
            return None
        
        try:
            with open(MODEL_PATH, "rb") as f:
                checkpoint = pickle.load(f)
            
            # Extract and load state dict into model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model = PixelRes(num_classes=2)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _pixel_model = model
            else:
                _pixel_model = checkpoint
                if hasattr(checkpoint, 'eval'):
                    checkpoint.eval()
            
            logger.info("Loaded Pixel model")
        except Exception as e:
            logger.warning(f"Failed to load Pixel model: {e}")
            return None
    
    return _pixel_model


def load_freq_model() -> object:
    """Load Frequency model"""
    global _freq_model
    if _freq_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_freq.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Frequency model not found at {MODEL_PATH}")
            return None
        
        try:
            with open(MODEL_PATH, "rb") as f:
                checkpoint = pickle.load(f)
            
            # Extract and load state dict into model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                model = FreqResNet(num_classes=2)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                _freq_model = model
            else:
                _freq_model = checkpoint
                if hasattr(checkpoint, 'eval'):
                    checkpoint.eval()
            
            logger.info("Loaded Frequency model")
        except Exception as e:
            logger.warning(f"Failed to load Frequency model: {e}")
            return None
    
    return _freq_model


def load_xception_model() -> object:
    """Load Xception model"""
    global _xception_model
    if _xception_model is None:
        MODEL_PATH = BASE_DIR / "model" / "full_model_xception.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Xception model not found at {MODEL_PATH}")
            return None
        
        try:
            with open(MODEL_PATH, "rb") as f:
                checkpoint = pickle.load(f)
            
            # Extract and load state dict into model
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
    """Load trained stacking meta-learner for weight stacking ensemble"""
    global _stacking_model, _stacking_checkpoint
    
    if _stacking_model is None:
        MODEL_PATH = BASE_DIR / "model" / "stacking_meta_learner.pkl"
        if not MODEL_PATH.exists():
            logger.warning(f"Stacking model not found at {MODEL_PATH}")
            return None, None
        
        try:
            with open(MODEL_PATH, "rb") as f:
                _stacking_checkpoint = pickle.load(f)
            
            # Reconstruct stacking meta-learner
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
    """
    Analyze image using 4-model ensemble (ELA, Pixel, Frequency, Xception)
    Returns individual model predictions and ensemble consensus
    """
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง torch เพื่อใช้งาน model")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        device = torch.device("cpu")
        
        # Define transforms
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
        
        # ===== ELA MODEL =====
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
        
        # ===== XCEPTION MODEL =====
        try:
            model = load_xception_model()
            if model is not None:
                xcep_tensor = t_xcep(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xcep_tensor)
                    p = torch.sigmoid(logits).item()
                
                xcep_ai_prob = (1 - p) * 100.0  # AI is class 1
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
        
        # ===== PIXEL MODEL =====
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
        
        # ===== FREQUENCY MODEL =====
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
        
        # ===== ENSEMBLE METHODS (with fallback chain) =====
        # Primary: Weight Boosting (weighted average)
        # Fallback 1: Weight Stacking (neural network meta-learner)
        # Fallback 2: ELA only
        
        logger.info(f"Attempting ensemble method: {ENSEMBLE_METHOD}")
        
        # Collect predictions for ensemble
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
        
        # ===== PRIMARY: WEIGHTED BOOSTING (Weighted Average) =====
        if ENSEMBLE_METHOD == "weight_boosting":
            logger.info("Using weight boosting (weighted average) as primary ensemble")
            
            weighted_sum = 0.0
            total_weight = 0.0
            weighted_votes = 0.0
            
            for model_name, model_result in results["models"].items():
                if "error" not in model_result:
                    weight = MODEL_WEIGHTS.get(model_name, 0.25)
                    confidence = model_result["confidence"]
                    
                    # Add to weighted sum
                    weighted_sum += weight * confidence
                    total_weight += weight
                    
                    # Count weighted votes (if confidence > 50%)
                    if confidence >= 50:
                        weighted_votes += weight
            
            if total_weight > 0:
                # Weight boosting result (weighted average)
                boosting_ensemble_probability = weighted_sum / total_weight
                weighted_vote_strength = weighted_votes / total_weight
                
                results["ensemble"] = {
                    "isAIGenerated": bool(boosting_ensemble_probability >= 50),
                    "confidence": float(boosting_ensemble_probability),
                    "confidence_rounded": round(float(boosting_ensemble_probability), 2),
                    "votes": f"{weighted_votes:.2f}/{total_weight:.2f} (weight boosting)",
                    "probability": float(boosting_ensemble_probability / 100.0),
                    "ensemble_type": "Weight Boosting Ensemble (Weighted Average)",
                    "weighted_details": model_results_for_ensemble,
                    "weighted_vote_strength": float(weighted_vote_strength)
                }
                logger.info(f"Weight boosting (primary): {boosting_ensemble_probability:.2f}% AI")
            else:
                logger.warning("Weight boosting failed - attempting weight stacking fallback")
                
                # Fall back to weight stacking
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
                    # Last resort: use ELA only
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
        
        # ===== FALLBACK: WEIGHT STACKING if primary method is unavailable =====
        else:
            logger.info("Using weight stacking (neural network) as primary ensemble")
            
            stacking_model, stacking_checkpoint = load_stacking_model()
            
            if stacking_model is not None and len(model_predictions) >= 3:
                # Use stacking ensemble
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
                # Fall back to weighted boosting
                logger.warning("Weight stacking unavailable - attempting weight boosting fallback")
                
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
                        "votes": f"{weighted_votes:.2f}/{total_weight:.2f} (weight boosting fallback)",
                        "probability": float(boosting_ensemble_probability / 100.0),
                        "ensemble_type": "Fallback: Weight Boosting Ensemble (Weighted Average)",
                        "weighted_details": model_results_for_ensemble,
                        "weighted_vote_strength": float(weighted_vote_strength)
                    }
                    logger.info(f"Weight boosting fallback: {boosting_ensemble_probability:.2f}% AI")
                else:
                    # Last resort: ELA only
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


@app.get("/api/health")
async def health_check():
    """ตรวจสอบสถานะ server"""
    return {
        "status": "พร้อม",
        "message": "เซิร์ฟเวอร์ทำงานปกติ"
    }


@app.get("/api/model-info")
async def model_info():
    """
    ดูโครงสร้างโมเดลที่โหลด (ใช้ debug เมื่อโมเดลใช้ inference ไม่ได้)
    """
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


@app.post("/api/analyze-debug")
async def analyze_image_debug(image: UploadFile = File(...)):
    """
    API endpoint สำหรับ debug - DISABLED (uses torch)
    """
    raise HTTPException(
        status_code=501,
        detail="Debug endpoint disabled - use /api/analyze instead"
    )


@app.post("/api/analyze")
async def analyze_image_endpoint(image: UploadFile = File(...), force: Optional[str] = Form(None)):
    """
    API endpoint สำหรับอัปโหลดและวิเคราะห์ภาพด้วย 4-model ensemble
    Returns individual model predictions and ensemble consensus
    """
    image_path = None
    
    try:
        if not validate_image_file(image):
            raise HTTPException(
                status_code=400,
                detail="กรุณาอัปโหลดไฟล์รูปภาพเฉพาะ (jpeg, jpg, png, gif, webp เท่านั้น)"
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
        
        # Override with force parameter if provided
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
        
        # Format response with proper type conversions
        ensemble = result["ensemble"]
        models = result["models"]
        
        return {
            "success": True,
            "result": {
                "isAIGenerated": bool(ensemble["isAIGenerated"]),
                "confidence": float(ensemble["confidence_rounded"]),
                "message": (
                    "ภาพนี้น่าจะถูกสร้างด้วย AI" 
                    if ensemble["isAIGenerated"] 
                    else "ภาพนี้น่าจะเป็นภาพจริง"
                ),
                "details": {
                    "probability": float(ensemble["probability"]),
                    "ensemble_votes": ensemble["votes"],
                    "ensemble_type": ensemble.get("ensemble_type", "Weighted Boosting Ensemble"),
                    "timestamp": datetime.now().isoformat(),
                    "model_type": "Weighted Boosting Ensemble (ELA + Pixel + Frequency + Xception)"
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False 
    )
