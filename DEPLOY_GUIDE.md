# 🚀 Deploy Guide - Render.com

## Step 1: ขั้นแรก Commit & Push บน GitHub

```bash
# ที่โฟลเดอร์ root ของโปรเจกต์
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

---

## Step 2: อัพโหลดไฟล์โมเดลไปยัง Google Drive

1. **สร้างโฟลเดอร์ใหม่บน Google Drive ชื่อ "TruPic-Models"**

2. **ดาวน์โหลด 4 ไฟล์โมเดลจากเครื่องของคุณ:**
   - `model/full_model_ela.pkl` (176MB)
   - `model/full_model_freq.pkl` (42MB)
   - `model/full_model_pixelhybrid.pkl` (176MB)
   - `model/full_model_xception.pkl` (79MB)

3. **อัพโหลดทั้ง 4 ไฟล์ไปยังโฟลเดอร์ "TruPic-Models"**

4. **สำหรับแต่ละไฟล์ - ดึง Google Drive ID:**
   - คลิก Right-click บนไฟล์ → **Share**
   - คลิก **Copy link** 
   - ลิงก์จะเป็นแบบนี้:
     ```
     https://drive.google.com/file/d/1AbCdEf_123456_XyZ/view?usp=sharing
                                      ↑
                                      ← Google Drive ID
     ```
   - **Copy ID ส่วนนั้น** (ตัวอักษรตัวเลขยาว ๆ)

5. **สร้างไฟล์ `.env`** ที่ root ของโปรเจกต์:

```bash
# .env
GDRIVE_FULL_MODEL_ELA_PKL=ใส่_ID_ของ_full_model_ela.pkl
GDRIVE_FULL_MODEL_FREQ_PKL=ใส่_ID_ของ_full_model_freq.pkl
GDRIVE_FULL_MODEL_PIXELHYBRID_PKL=ใส่_ID_ของ_full_model_pixelhybrid.pkl
GDRIVE_FULL_MODEL_XCEPTION_PKL=ใส่_ID_ของ_full_model_xception.pkl
```

**ตัวอย่าง:**
```bash
GDRIVE_FULL_MODEL_ELA_PKL=1A-2b_3c4D5eF6-7gHiJ8kLmNoPqRsTuVwX
GDRIVE_FULL_MODEL_FREQ_PKL=2B_3c-4d5E6fG7-8hIjK9lMnOpQrStUvWxY
...
```

6. **Commit & Push:**
```bash
git add .env
git commit -m "Add environment variables"
git push origin main
```

---

## Step 3: Deploy บน Render

### วิธี A: ใช้ render.yaml (อัตโนมัติ - แนะนำ)

1. **เข้า https://render.com**
   - Log in / Sign up (GitHub account ได้)

2. **สร้าง New Service:**
   - คลิก **"New +"** → **"Web Service"**
   - เลือก **"Connect a repository"** 
   - เลือก `web-baseai` repository

3. **Render จะอ่าน `render.yaml` อัตโนมัติ**
   - ตั้งค่า Backend
   - ตั้งค่า Frontend
   - ทั้งหมด automatic ✓

4. **ไป Tab "Environment"** → Add variables:
   ```
   GDRIVE_FULL_MODEL_ELA_PKL=your_id_here
   GDRIVE_FULL_MODEL_FREQ_PKL=your_id_here
   GDRIVE_FULL_MODEL_PIXELHYBRID_PKL=your_id_here
   GDRIVE_FULL_MODEL_XCEPTION_PKL=your_id_here
   ```

5. **Deploy!** 🚀

### วิธี B: ตั้งค่าผ่าน Dashboard (ถ้า render.yaml ไม่ทำงาน)

1. **สร้าง Backend Service:**
   - **Web Service** → GitHub `web-baseai`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Add Environment:**
     ```
     GDRIVE_FULL_MODEL_ELA_PKL=...
     GDRIVE_FULL_MODEL_FREQ_PKL=...
     GDRIVE_FULL_MODEL_PIXELHYBRID_PKL=...
     GDRIVE_FULL_MODEL_XCEPTION_PKL=...
     ```
   - Deploy

2. **ได้ Backend URL** (ตัวอย่าง):
   ```
   https://trupic-backend.onrender.com
   ```

3. **สร้าง Frontend Service:**
   - **Static Site** → GitHub `web-baseai`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `dist`
   - Deploy

---

## Step 4: ตั้งค่า Frontend

### ถ้าใช้ render.yaml (automatic):
- Render จะ auto deploy frontend ที่ `https://trupic-frontend.onrender.com`
- Environment variables เตรียมไว้แล้วใน render.yaml ✓

### ถ้าตั้งค่าเอง:

1. **ได้ Backend URL จาก Render Dashboard**
   - ตัวอย่าง: `https://trupic-backend.onrender.com`

2. **Code เตรียมไว้แล้ว** `src/App.jsx`:
   ```javascript
   const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
   ```
   - Render จะ set `VITE_API_URL` อัตโนมัติจาก render.yaml ✓

3. **Deploy Frontend:**
   ```bash
   npm run build
   ```
   - Render จะ host ที่ Static Site ของมัน

---

## Step 5: ทดสอบ

### Backend:
```bash
# ตรวจสอบ Health
curl https://trupic-backend.onrender.com/api/health

# Response:
# {"status": "ok", "message": "..."}
```

### Frontend:
- ไปที่ `https://trupic-frontend.onrender.com`
- หรือ localhost:5173 (development)
- อัพโหลดรูปภาพและทดสอบ

### Logs:
- **Backend Log:** Render Dashboard → Service → Logs
  ```
  🚀 Starting backend server...
  📥 Ensuring models are downloaded...
  ✓ Models are ready!
  ```

---

## ⚠️ Troubleshooting

### Models ไม่ Download:
1. ตรวจสอบ Google Drive IDs ถูก?
   ```bash
   # Render Dashboard → Environment
   # ดู values ของ GDRIVE_FULL_MODEL_* ถูกไหม
   ```

2. ลองเพิ่มเติม manually ใน Dashboard:
   - Settings → Environment Variables
   - Update ทั้ง 4 IDs

### 502 Bad Gateway:
- รอสักครู่ Render อาจยังรัน startup script
- ดู Logs ว่า models download สำเร็จหรือไม่

### Frontend ไม่ connect Backend:
1. ตรวจสอบ Backend URL:
   ```
   Render Dashboard → Backend Service → URL
   ```
2. Check `render.yaml` หรือ env var `VITE_API_URL` ถูก?

### Upload Timeout:
- Render free tier มี timeout 30 นาที
- ถ้าอัพโหลดไฟล์ใหญ่ → compress ให้เล็ก

---

## 💀 หลังจาก Deploy สำหรับ GitHub

เมื่อทำงานกับ GitHub แล้ว อย่าลืมเพิ่ม `.gitignore`:

```bash
# .gitignore มี:
model/*.pkl
.env
backend/uploads/
```

---

## 🎉 สำเร็จ!

โปรเจกต์ของคุณ live บน Render แล้ว!

**URLs:**
- Backend: `https://trupic-backend.onrender.com`
- Frontend: `https://trupic-frontend.onrender.com`

ส่ง Frontend URL ให้คนอื่นใช้ได้เลย 🚀
