@echo off
REM Script สำหรับรัน FastAPI server บน Windows

REM ตรวจสอบว่ามี virtual environment หรือไม่
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM ติดตั้ง dependencies
pip install -r requirements.txt

REM รัน server
uvicorn main:app --reload --port 3001
