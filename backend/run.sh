#!/bin/bash
# Script สำหรับรัน FastAPI server

# ตรวจสอบว่ามี virtual environment หรือไม่
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน server
uvicorn main:app --reload --port 3001
