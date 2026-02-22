#!/usr/bin/env python3
"""Test the 4-model ensemble API with image upload"""

import urllib.request
import json
from pathlib import Path

image_path = 'backend/uploads/AI.png'
print(f'Testing ensemble API: {image_path}\n')

# Read image
with open(image_path, 'rb') as f:
    image_data = f.read()

# Create multipart form data
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = []
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="image"; filename="AI.png"')
body.append(b'Content-Type: image/png')
body.append(b'')
body.append(image_data)
body.append(f'--{boundary}--'.encode())
body.append(b'')

body_bytes = b'\r\n'.join(body)

# Create request
req = urllib.request.Request(
    'http://localhost:8001/api/analyze',
    data=body_bytes,
    method='POST'
)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

# Send request
try:
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
