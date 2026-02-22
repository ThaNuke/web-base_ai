# TruPic - AI Image Detection

เว็บแอปพลิเคชันสำหรับตรวจจับภาพที่ถูกสร้างด้วย AI

## โครงสร้างโปรเจกต์

```
web-baseai/
├── backend/          # Backend server (FastAPI + Python)
├── src/             # Frontend (React + Vite)
└── public/          # Static files
```

## การติดตั้ง

### 1. ติดตั้ง Frontend Dependencies

```bash
npm install
```

### 2. ติดตั้ง Backend Dependencies (Python)

```bash
cd backend

# สร้าง virtual environment (แนะนำ)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt
```

## การรันโปรเจกต์

### รัน Frontend (Port 5173)
```bash
npm run dev
```

### รัน Backend (Port 3001)

**วิธีที่ 1: ใช้ Python โดยตรง**
```bash
cd backend
python main.py
```

**วิธีที่ 2: ใช้ uvicorn**
```bash
cd backend
uvicorn main:app --reload --port 3001
```

**วิธีที่ 3: ใช้ script (Windows)**
```bash
cd backend
run.bat
```

**วิธีที่ 4: ใช้ script (Linux/Mac)**
```bash
cd backend
chmod +x run.sh
./run.sh
```

หรือรันทั้งสองพร้อมกันใน terminal แยกกัน

## การใช้งาน

1. เปิดเบราว์เซอร์ไปที่ `http://localhost:5173`
2. คลิกปุ่ม "Get Started"
3. อัปโหลดภาพที่ต้องการวิเคราะห์
4. รอผลการวิเคราะห์

## Backend API

ดูรายละเอียด API ใน [backend/README.md](./backend/README.md)

## การเชื่อมต่อกับ AI Model

ตอนนี้ backend ใช้ mock data สำหรับทดสอบ คุณสามารถแก้ไขฟังก์ชัน `analyze_image()` ใน `backend/main.py` เพื่อเชื่อมต่อกับ AI model จริง

### ตัวเลือกที่แนะนำ:
- Hugging Face Transformers / API
- PyTorch / TensorFlow models
- Cloud AI Services (AWS Rekognition, Google Vision API)

## Environment Variables

สร้างไฟล์ `.env` ในโฟลเดอร์ `backend`:

```
PORT=3001
HUGGINGFACE_API_KEY=your_api_key_here
```

## เทคโนโลยีที่ใช้

- **Frontend**: React, Vite
- **Backend**: Python, FastAPI, Uvicorn
- **File Upload**: FastAPI File Upload

## API Documentation

เมื่อรัน backend server แล้ว สามารถดู API documentation ได้ที่:
- Swagger UI: `http://localhost:3001/docs`
- ReDoc: `http://localhost:3001/redoc`
