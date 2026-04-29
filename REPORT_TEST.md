# รายงานผลการทดสอบ — ระบบตรวจจับภาพ AI

**โครงงาน**: AI Image Detection Web (TruPic API)  
**วันที่ทำรายงาน**: 2026-04-26  
**ผู้ทดสอบ**: _(กรอกชื่อ)_  
**Backend URL**: `http://localhost:8001`  
**ชุดข้อมูล (สำหรับทดสอบแบบชุด/Batch)**: `backend/Dataset Senior Project/`  

---

## 1) วัตถุประสงค์ (Objective)

ตรวจสอบความถูกต้องของความสามารถหลักของระบบ:

- รองรับการอัปโหลดรูปภาพเท่านั้น (`.jpg`, `.jpeg`, `.png`) และขนาดไฟล์ต้องไม่เกิน **10MB**
- วิเคราะห์รูปภาพด้วยโมเดล ML และส่งผลลัพธ์กลับมาเป็น:
  - ป้ายกำกับ **Real / AI-generated**
  - **ความมั่นใจ (Confidence %)**
- รูปแบบ JSON response จาก API ต้องสม่ำเสมอและใช้งานต่อได้
- จัดการ input ผิดพลาดอย่างปลอดภัย (ชนิดไฟล์ผิด / ไฟล์ใหญ่เกิน)
- ลบไฟล์ชั่วคราวหลังประมวลผลเสร็จ (ฝั่งเซิร์ฟเวอร์)

---

## 2) ขอบเขตการทดสอบ (Scope)

**อยู่ในขอบเขต (In scope)**

- `GET /api/health`
- `POST /api/analyze`
- การตรวจสอบความถูกต้องของไฟล์: นามสกุล/ชนิดไฟล์, ขนาดไฟล์สูงสุด
- การตรวจสอบโครงสร้าง JSON response (schema)

**นอกขอบเขต (Out of scope)**

- การทดสอบ UI แบบ end-to-end ผ่านเบราว์เซอร์ทั้งหมด
- การทดสอบโหลด/สเตรสระดับ production concurrency
- การประเมินความแม่นยำเชิงวิชาการของโมเดล (เช่น accuracy/F1 บนชุดข้อมูลที่มี label จริงครบถ้วน)

---

## 3) สภาพแวดล้อมการทดสอบ (Test Environment)

- **OS**: Windows 10 (win32 10.0.26200)
- **Python**: venv (local)
- **เครื่องมือทดสอบ**:
  - `test_dataset.py` (Python + `requests`) สำหรับทดสอบแบบชุด (Batch)
  - การเรียก API ตรง (Python) สำหรับทดสอบเคสเดี่ยวเพื่อเก็บหลักฐาน

---

## 4) สรุปผลการทดสอบ (Test Summary)

ผลรวม: **PASS** สำหรับ requirement หลัก (การตรวจสอบไฟล์, การวิเคราะห์, โครงสร้าง JSON response)

หมายเหตุ:

- ข้อความ error ภาษาไทยบางส่วนอาจแสดงเพี้ยนในบาง terminal เนื่องจาก **encoding** แต่ HTTP status code และโครงสร้าง JSON ถูกต้อง

---

## 5) รายการ Test Case (ID | Objective | Steps | Expected | Actual | Status)

| ID | Objective | Steps | Expected | Actual (Observed) | Status |
|---|---|---|---|---|---|
| **TC_UI_001** | อัปโหลดรูปภาพที่ถูกต้องได้สำเร็จ | 1) เลือกไฟล์ `.jpg/.png` ≤ 10MB 2) ส่งเพื่อวิเคราะห์ | HTTP 200 และได้ผลวิเคราะห์กลับมา | `POST /api/analyze` ด้วย `0082_T.jpg` → **HTTP 200**, `success:true`, `isAIGenerated:false`, `confidence:0.25` | **PASS** |
| **TC_UI_002** | ปฏิเสธไฟล์ผิดประเภท | 1) เลือกไฟล์ `.pdf` 2) ส่งเพื่อวิเคราะห์ | HTTP 400 + ข้อความแจ้งเตือน | `POST /api/analyze` ด้วย `temp_invalid.pdf` → **HTTP 400**, JSON `detail` ระบุชนิดไฟล์ที่อนุญาต (jpeg/jpg/png) | **PASS** |
| **TC_UI_003** | ปฏิเสธไฟล์ขนาดเกิน (>10MB) | 1) เลือกไฟล์ > 10MB 2) ส่งเพื่อวิเคราะห์ | HTTP 400 + แจ้งเตือนขนาดไฟล์ | `POST /api/analyze` ด้วยไฟล์จำลอง 11MB `temp_large.jpg` → **HTTP 400**, JSON `detail` ระบุห้ามเกิน 10MB | **PASS** |
| **TC_AI_001** | ระบบประมวลผลได้โดยไม่ล่ม | 1) ส่งรูปที่ถูกต้อง 2) รอผลวิเคราะห์ | ประมวลผลเสถียร ไม่ crash และตอบกลับภายในเวลา | เคสตัวอย่างตอบกลับสำเร็จใน ~**6.47s** (เวลารวมทั้ง request) | **PASS** |
| **TC_RESULT_001** | การแสดงผลลัพธ์ถูกต้อง | 1) ส่งรูปที่ถูกต้อง 2) อ่านผลลัพธ์ | มี label และเปอร์เซ็นต์ในช่วง 0–100 | Response มี `message:"Likely a Real Image"` และ `confidence` เป็นตัวเลขในช่วงถูกต้อง | **PASS** |
| **TC_SEC_001** | ลบไฟล์ชั่วคราวหลังประมวลผล | 1) ส่งรูป 2) รอ response | เซิร์ฟเวอร์ลบไฟล์หลังจบ request | โค้ด backend ลบไฟล์ชั่วคราวหลังประมวลผล (`os.remove(image_path)`) *(สามารถตรวจ `uploads/` หลังยิง request เพื่อยืนยันเพิ่มได้)* | **PASS (ตามการออกแบบ)** |
| **TC_UI_004** | อัปโหลดหลายครั้งระบบไม่พัง/ไม่ค้าง state | 1) ส่งรูป A 2) ส่งรูป B | ผลลัพธ์ของรูปหลังสุดถูกต้อง ไม่มี error | การทดสอบแบบชุดแสดงว่าระบบตอบกลับได้ต่อเนื่อง และไม่ขึ้นกับ state ของ UI | **PASS** |
| **TC_API_001** | โครงสร้าง JSON response ตรงตามที่กำหนด | 1) ส่งรูปที่ถูกต้อง 2) ตรวจ JSON | JSON มี `success`, `result`, `details.probability`, `individual_models` | พบว่ามี `success`, `result.details.probability`, `result.individual_models`, `ensemble_weights` | **PASS** |

---

## 6) หลักฐาน (ตัวอย่าง Response)

ตัวอย่าง (รูปที่ถูกต้อง):

```json
{
  "success": true,
  "result": {
    "isAIGenerated": false,
    "confidence": 0.25,
    "message": "Likely a Real Image",
    "details": {
      "probability": 0.002537102869246155,
      "ensemble_votes": "0.00/1.00 (weighted majority vote)",
      "ensemble_type": "Weighted Average + Majority Vote Ensemble",
      "timestamp": "2026-04-26T17:27:12.977447",
      "model_type": "Weighted Average Ensemble (ELA + Pixel + Frequency + Xception)"
    },
    "individual_models": {
      "ela": { "confidence": 0.010768831998575479 },
      "xception": { "confidence": 1.0003209114074707 },
      "frequency": { "confidence": 0.07517478079535067 },
      "pixel": { "confidence": 0.015554687706753612 }
    }
  }
}
```

---

## 7) ประเด็นที่พบ / ความเสี่ยง (Known Issues / Risks)

- **Encoding ของ Terminal**: ข้อความภาษาไทยบางครั้งอาจแสดงไม่ถูกต้องในบาง terminal
  - **ผลกระทบ**: ต่ำ (สัญญา API/HTTP ถูกต้อง กระทบเฉพาะการแสดงผล)
  - **ข้อเสนอแนะ**: ใช้ terminal แบบ UTF-8 หรือปรับ PowerShell output encoding

- **เวลาในการประมวลผลแปรผัน**: ขึ้นกับ CPU/RAM และวิธีโหลดโมเดล
  - **ข้อเสนอแนะ**: กำหนดเป้าหมายเวลา (เช่น < 10 วินาทีต่อภาพบนเครื่อง local) และแสดงสถานะ loading ใน UI

---

## 8) บทสรุป (Conclusion)

ระบบ backend ผ่าน requirement หลักดังนี้:

- อัปโหลดและวิเคราะห์รูปภาพได้
- ตรวจสอบ input ได้ถูกต้อง (ชนิด/ขนาดไฟล์)
- ส่ง JSON response ได้สม่ำเสมอและใช้งานต่อได้

ข้อเสนอแนะขั้นถัดไป:

- ทำชุด regression test ขนาดเล็ก (10–20 ภาพ) และรันทุกครั้งก่อน deploy
- (ถ้าจำเป็นตาม rubric วิชา) เพิ่ม UI E2E tests

