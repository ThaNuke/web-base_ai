FROM python:3.11

WORKDIR /app

# Copy backend directory
COPY backend/ .

# Install Python dependencies
RUN pip install --no-cache-dir -q \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pillow>=10.4.0 \
    numpy \
    scikit-learn \
    opencv-python \
    exif \
    gdown>=4.6.0 \
    requests

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
