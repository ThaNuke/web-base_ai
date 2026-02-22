# TruPic Backend Server

Backend server สำหรับวิเคราะห์ภาพ AI-generated images (FastAPI + Python)

## การติดตั้ง

### 1. ติดตั้ง Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

หรือใช้ virtual environment (แนะนำ):

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## การรัน Server

```bash
# Development mode (auto-reload)
python main.py

# หรือใช้ uvicorn โดยตรง
uvicorn main:app --reload --port 3001

# Production mode
uvicorn main:app --host 0.0.0.0 --port 3001
```

Server จะรันที่ `http://localhost:3001`

## API Documentation

เมื่อรัน server แล้ว สามารถดู API documentation ได้ที่:
- Swagger UI: `http://localhost:3001/docs`
- ReDoc: `http://localhost:3001/redoc`

## API Endpoints

### GET /api/health
ตรวจสอบสถานะ server

**Response:**
```json
{
  "status": "ok",
  "message": "Backend server is running"
}
```

### POST /api/analyze
วิเคราะห์ภาพว่าถูกสร้างด้วย AI หรือไม่

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "success": true,
  "result": {
    "isAIGenerated": true,
    "confidence": 85.5,
    "message": "ภาพนี้น่าจะถูกสร้างด้วย AI",
    "details": {
      "probability": 0.855,
      "model": "Mock Detection Model",
      "timestamp": "2024-01-01T00:00:00.000Z"
    }
  }
}
```

## การเชื่อมต่อกับ AI Model

ตอนนี้ใช้ mock data สำหรับทดสอบ คุณสามารถแก้ไขฟังก์ชัน `analyze_image()` ใน `main.py` เพื่อเชื่อมต่อกับ AI model จริง เช่น:

- Hugging Face API / Transformers
- PyTorch / TensorFlow models
- Custom ML model
- Cloud AI services (AWS Rekognition, Google Vision API, etc.)

### ตัวอย่างการเชื่อมต่อกับ Hugging Face Transformers

```python
from transformers import pipeline
from PIL import Image

async def analyze_image(image_path: str) -> dict:
    # โหลด model
    classifier = pipeline(
        "image-classification", 
        model="your-model-name"
    )
    
    # วิเคราะห์ภาพ
    image = Image.open(image_path)
    result = classifier(image)
    
    # ประมวลผลผลลัพธ์
    return {
        "isAIGenerated": result[0]["label"] == "ai-generated",
        "confidence": result[0]["score"] * 100,
        "details": {
            "probability": result[0]["score"],
            "model": "your-model-name",
            "timestamp": datetime.now().isoformat()
        }
    }
```

### ตัวอย่างการเชื่อมต่อกับ Hugging Face API

```python
import requests
import os

async def analyze_image(image_path: str) -> dict:
    api_url = "https://api-inference.huggingface.co/models/YOUR_MODEL"
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
    }
    
    with open(image_path, "rb") as f:
        data = f.read()
    
    response = requests.post(api_url, headers=headers, data=data)
    result = response.json()
    
    return {
        "isAIGenerated": result[0]["label"] == "ai-generated",
        "confidence": result[0]["score"] * 100,
        "details": result
    }
```

## Environment Variables

สร้างไฟล์ `.env` ในโฟลเดอร์ `backend`:

```
PORT=3001
HUGGINGFACE_API_KEY=your_api_key_here
```
