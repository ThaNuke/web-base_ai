FROM python:3.11-slim

WORKDIR /app

# Copy backend directory
COPY backend/ .

# Install system dependencies first (required for opencv, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision \
    && pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pillow>=10.4.0 \
    numpy \
    scikit-learn \
    opencv-python-headless \
    timm \
    exif \
    gdown>=4.6.0 \
    requests

# Create model directory
RUN mkdir -p /app/model

# Expose port (Railway overrides with $PORT)
EXPOSE 8000

# Download models (if environment variables are provided)
# Models will be downloaded on container startup if not present
ENV PYTHONUNBUFFERED=1

# Run uvicorn - Railway sets $PORT dynamically, must use shell form for variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
