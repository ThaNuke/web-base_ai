#!/usr/bin/env python3
"""
Fix model weights: Convert all checkpoints to CPU-compatible format
Handles both .pkl and .pth files, extracts state_dict, and re-saves cleanly
"""

import torch
import torch.nn as nn
import pickle
import io
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import timm
import torchvision.models as models

device = torch.device('cpu')

# Define model classes so pickle can deserialize them
class ELRes(nn.Module):
    def __init__(self, num_classes=2):
        super(ELRes, self).__init__()
        self.modelELA = models.resnext50_32x4d(weights=None)
        self.modelRGB = models.resnext50_32x4d(weights=None)
        
        self.modelELA.fc = nn.Identity()
        self.modelRGB.fc = nn.Identity()
        
        self.fc = nn.Linear(2048*2, num_classes)
    
    def forward(self, x_ela, x_rgb):
        ela_feat = self.modelELA(x_ela)
        rgb_feat = self.modelRGB(x_rgb)
        combined = torch.cat((ela_feat, rgb_feat), dim=1)
        return self.fc(combined)


class PixelRes(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        self.modelPix = models.resnext50_32x4d(weights=weights)
        
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


class FreqResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
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

class CPUUnpickler(pickle.Unpickler):
    """Custom unpickler that forces CPU device mapping"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # Override default torch loading to use CPU
            def custom_load(b):
                return torch.load(
                    __import__('io').BytesIO(b), 
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
            return custom_load
        return super().find_class(module, name)


def fix_model_weights(model_path, model_name):
    """
    Load checkpoint, extract state_dict, and re-save without CUDA references
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ {model_name}: File not found at {model_path}")
        return False
    
    try:
        print(f"\n🔹 Processing {model_name}...")
        print(f"   File: {model_path}")
        
        # Load checkpoint
        if str(model_path).endswith('.pkl'):
            try:
                # Try standard pickle first
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                print(f"   Loaded .pkl file")
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    # CUDA error - use custom unpickler with CPU mapping
                    print(f"   Detected CUDA tensor - using custom CPU unpickler")
                    with open(model_path, 'rb') as f:
                        checkpoint = CPUUnpickler(f).load()
                    print(f"   Loaded .pkl file (CPU mapped)")
                else:
                    raise
        else:
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
            print(f"   Loaded .pth file")
        
        # Extract state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"   Found model_state_dict")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"   Found state_dict")
            else:
                state_dict = checkpoint
                print(f"   Using checkpoint as state_dict")
        else:
            # If it's a model object, get state_dict
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
                print(f"   Extracted state_dict from model object")
            else:
                state_dict = checkpoint
                print(f"   Using checkpoint directly")
        
        # Verify state_dict
        print(f"   State dict has {len(state_dict)} keys")
        if len(state_dict) > 0:
            first_key = list(state_dict.keys())[0]
            first_val = state_dict[first_key]
            print(f"   Sample key: {first_key}")
            print(f"   Sample shape: {first_val.shape if hasattr(first_val, 'shape') else type(first_val)}")
        
        # Prepare clean checkpoint (state_dict only, no CUDA refs)
        clean_checkpoint = {
            'model_state_dict': state_dict,
            'device': 'cpu',
            'pytorch_version': torch.__version__
        }
        
        # If original had config, keep it
        if isinstance(checkpoint, dict):
            if 'model_config' in checkpoint:
                clean_checkpoint['model_config'] = checkpoint['model_config']
            if 'performance' in checkpoint:
                clean_checkpoint['performance'] = checkpoint['performance']
        
        # Save as .pkl (same format)
        with open(model_path, 'wb') as f:
            pickle.dump(clean_checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   ✅ Re-saved {model_name} successfully")
        print(f"   File: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fix all 4 model weights"""
    
    # Resolve project paths
    script_dir = Path(__file__).parent
    model_dir = script_dir.parent / 'model'
    
    print("=" * 60)
    print("🔧 Fixing Model Weights (CPU-compatible)")
    print("=" * 60)
    
    models = {
        'ELA': model_dir / 'elres_model.pkl',
        'Pixel': model_dir / 'full_model_pixelhybrid.pkl',
        'Frequency': model_dir / 'full_model_freq.pkl',
        'Xception': model_dir / 'full_model_xception.pkl'
    }
    
    results = {}
    for name, path in models.items():
        results[name] = fix_model_weights(path, name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}: {'Fixed' if success else 'Failed'}")
    
    success_count = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nTotal: {success_count}/{total} models fixed")
    
    if success_count == total:
        print("\n✅ All models fixed successfully!")
        print("You can now run combined_inference_folder.py without CUDA errors")
        return 0
    else:
        print(f"\n⚠️ {total - success_count} model(s) failed to fix")
        return 1


if __name__ == "__main__":
    sys.exit(main())
