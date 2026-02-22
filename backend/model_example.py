"""
ตัวอย่างการเชื่อมต่อกับ AI Model สำหรับตรวจจับ AI-generated images

คุณสามารถ copy code จากไฟล์นี้ไปใช้ใน main.py ได้
"""

from PIL import Image
from datetime import datetime
import os

# ============================================
# ตัวอย่างที่ 1: ใช้ Hugging Face Transformers
# ============================================

async def analyze_image_with_transformers(image_path: str) -> dict:
    """
    ใช้ Hugging Face Transformers library
    
    ติดตั้ง: pip install transformers torch torchvision pillow
    """
    try:
        from transformers import pipeline
        
        # โหลด model (ตัวอย่าง: ใช้ image classification model)
        # คุณสามารถเปลี่ยนเป็น model ที่เหมาะสมสำหรับตรวจจับ AI images
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"  # เปลี่ยนเป็น model ที่เหมาะสม
        )
        
        # เปิดและวิเคราะห์ภาพ
        image = Image.open(image_path)
        result = classifier(image)
        
        # ประมวลผลผลลัพธ์
        # ตัวอย่าง: ถ้า label เป็น "ai-generated" หรือ confidence สูง
        top_result = result[0]
        is_ai_generated = "ai" in top_result["label"].lower() or top_result["score"] > 0.7
        
        return {
            "isAIGenerated": is_ai_generated,
            "confidence": top_result["score"] * 100,
            "details": {
                "probability": top_result["score"],
                "label": top_result["label"],
                "model": "transformers",
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")


# ============================================
# ตัวอย่างที่ 2: ใช้ Hugging Face API
# ============================================

async def analyze_image_with_hf_api(image_path: str) -> dict:
    """
    ใช้ Hugging Face Inference API
    
    ติดตั้ง: pip install requests
    ต้องมี HUGGINGFACE_API_KEY ใน environment variable
    """
    try:
        import requests
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise Exception("กรุณาตั้งค่า HUGGINGFACE_API_KEY ใน environment variable")
        
        # เปลี่ยนเป็น model ที่เหมาะสมสำหรับตรวจจับ AI images
        model_name = "your-model-name"  # เช่น "microsoft/git-base"
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # อ่านไฟล์ภาพ
        with open(image_path, "rb") as f:
            data = f.read()
        
        # ส่ง request ไปยัง API
        response = requests.post(api_url, headers=headers, data=data)
        response.raise_for_status()
        
        result = response.json()
        
        # ประมวลผลผลลัพธ์
        if isinstance(result, list) and len(result) > 0:
            top_result = result[0]
            is_ai_generated = "ai" in top_result.get("label", "").lower()
            
            return {
                "isAIGenerated": is_ai_generated,
                "confidence": top_result.get("score", 0) * 100,
                "details": {
                    "probability": top_result.get("score", 0),
                    "label": top_result.get("label", "unknown"),
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            raise Exception("ไม่ได้รับผลลัพธ์จาก API")
            
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")


# ============================================
# ตัวอย่างที่ 3: ใช้ PyTorch/TensorFlow Custom Model
# ============================================

async def analyze_image_with_custom_model(image_path: str) -> dict:
    """
    ใช้ Custom PyTorch/TensorFlow model
    
    ติดตั้ง: pip install torch torchvision pillow
    หรือ: pip install tensorflow pillow
    """
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # โหลด model (ตัวอย่าง)
        # model = torch.load("path/to/your/model.pth")
        # model.eval()
        
        # Preprocess ภาพ
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path)
        input_tensor = transform(image).unsqueeze(0)
        
        # ทำ prediction
        # with torch.no_grad():
        #     output = model(input_tensor)
        #     probabilities = torch.nn.functional.softmax(output[0], dim=0)
        #     confidence, predicted = torch.max(probabilities, 0)
        
        # Mock result สำหรับตัวอย่าง
        confidence = 0.85
        predicted = 1  # 1 = AI-generated, 0 = Real
        
        return {
            "isAIGenerated": bool(predicted),
            "confidence": float(confidence) * 100,
            "details": {
                "probability": float(confidence),
                "model": "custom-pytorch-model",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")


# ============================================
# ตัวอย่างที่ 4: ใช้ Cloud AI Services
# ============================================

async def analyze_image_with_aws_rekognition(image_path: str) -> dict:
    """
    ใช้ AWS Rekognition
    
    ติดตั้ง: pip install boto3
    ต้องตั้งค่า AWS credentials
    """
    try:
        import boto3
        
        rekognition = boto3.client('rekognition')
        
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        
        # ใช้ AWS Rekognition Content Moderation หรือ Custom Labels
        response = rekognition.detect_moderation_labels(
            Image={'Bytes': image_bytes}
        )
        
        # ประมวลผลผลลัพธ์
        # (ปรับแต่งตาม API response ของ AWS)
        
        return {
            "isAIGenerated": False,  # ปรับแต่งตามผลลัพธ์
            "confidence": 0.0,
            "details": {
                "model": "aws-rekognition",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {str(e)}")
