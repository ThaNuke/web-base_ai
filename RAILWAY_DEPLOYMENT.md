# Railway Deployment Guide

## Problem: API Returns 404

The API crashes on startup because the ML model files are missing. They need to be downloaded from Google Drive during deployment.

## Solution: Set Environment Variables in Railway

### Step 1: Go to Railway Dashboard

1. Open [railway.app](https://railway.app)
2. Select your project
3. Click on the service (likely `web-production` or similar)
4. Go to the **Variables** tab

### Step 2: Add Environment Variables

Add these four variables with the Google Drive model IDs:

```
GDRIVE_FULL_MODEL_ELA_PKL=1ib2WLtJPfDOQeaWJaJM_HQhM8qeDgeux
GDRIVE_FULL_MODEL_FREQ_PKL=1m15sxulaBU5_YPzCkDp-lHzmZy_6dp7n
GDRIVE_FULL_MODEL_PIXELHYBRID_PKL=1QUHlEKU3lXZ22ZzounKCBnwM3zf4Cc5W
GDRIVE_FULL_MODEL_XCEPTION_PKL=1cxzQYMMwmVzLnuXoTsRsGi12l9e43Dfg
```

### Step 3: Redeploy

1. Go to the **Deployments** tab
2. Click on the latest deployment
3. Click **Redeploy**
4. Wait for the deployment to complete

The app will now:
- Download the 4 ML models from Google Drive (~500MB total)
- Load them into memory
- Start the API server on port 8000

### Step 4: Test the API

Once deployed, test your API:

```bash
curl https://your-railway-app.up.railway.app/api/health
```

You should get a response like:
```json
{
  "status": "พร้อม",
  "message": "เซิร์ฟเวอร์ทำงานปกติ"
}
```

## Troubleshooting

### Still seeing "Not Found"?

1. **Check the logs**: In Railway dashboard → Deployments → View Logs
2. **Look for**: `Downloading...` messages - models should be downloading
3. **Wait longer**: First deployment takes ~5-10 minutes to download 500MB

### Models not downloading?

1. Verify all 4 Google Drive IDs are set in Variables (copy from above)
2. Check that the Google Drive files are still accessible
3. Look for error messages mentioning "404" or "Permission denied"

### How to update model files?

1. Upload new models to Google Drive
2. Get the share link and extract the ID
3. Update the environment variable with the new ID
4. Redeploy

## What happens during deployment?

```
1. Railway builds Docker image
2. Installs Python dependencies (PyTorch, etc.)
3. On container startup:
   - download_models.py checks for models
   - Downloads from Google Drive if missing
   - Saves to /app/model/ directory
4. FastAPI server starts on port 8000
5. App serves requests at https://your-app.railway.app
```

## Important Notes

- ⚠️ **First deployment takes 5-10 minutes** (models are ~500MB)
- ✓ Subsequent deployments are faster (models already cached)
- 📊 Models are stored in Railway's ephemeral filesystem
- 🔄 Models are re-downloaded on each new deployment
- 💾 This is normal - use S3 storage for production optimization

## For Local Testing

Set environment variables locally and run:

```bash
cd backend
python main.py
```

Or test the download directly:
```bash
python download_models.py
```
