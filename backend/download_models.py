"""
Download models from Google Drive automatically on startup
"""
import os
import gdown
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Google Drive file IDs - can be overridden by environment variables
MODELS = {
    "full_model_ela.pkl": "1ib2WLtJPfDOQeaWJaJM_HQhM8qeDgeux",
    "full_model_freq.pkl": "1m15sxulaBU5_YPzCkDp-lHzmZy_6dp7n",
    "full_model_pixelhybrid.pkl": "1QUHlEKU3lXZ22ZzounKCBnwM3zf4Cc5W",
    "full_model_xception.pkl": "1cxzQYMMwmVzLnuXoTsRsGi12l9e43Dfg",
}

MODEL_DIR = Path(__file__).parent.parent / "model"


def get_gdrive_id(model_name: str) -> str:
    """Get Google Drive ID from environment variable or hardcoded value"""
    env_key = f"GDRIVE_{model_name.upper().replace('.', '_')}"
    return os.getenv(env_key, MODELS.get(model_name, ""))


def download_model(model_name: str, file_id: str) -> bool:
    """Download single model from Google Drive"""
    if not file_id:
        logger.warning(f"⚠️  Skipping {model_name} - Google Drive ID not set")
        logger.warning(f"   Set environment variable: GDRIVE_{model_name.upper().replace('.', '_')}")
        return False
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / model_name
    
    # Skip if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {model_name} already exists ({size_mb:.1f}MB)")
        return True
    
    try:
        logger.info(f"⏳ Downloading {model_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        gdown.download(url, str(model_path), quiet=False)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {model_name} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def ensure_models_exist():
    """Ensure all models are downloaded"""
    logger.info("🚀 Checking models...")
    
    downloaded = []
    failed = []
    
    for model_name in MODELS.keys():
        file_id = get_gdrive_id(model_name)
        if download_model(model_name, file_id):
            downloaded.append(model_name)
        else:
            failed.append(model_name)
    
    logger.info(f"📊 Downloaded: {len(downloaded)}/{len(MODELS)}")
    
    if failed:
        logger.warning(f"⚠️  {len(failed)} model(s) not available: {failed}")
        logger.warning(f"📝 Set up environment variables with Google Drive IDs")
        logger.warning(f"   See .env.example for configuration")
    
    return len(failed) == 0


if __name__ == "__main__":
    ensure_models_exist()

