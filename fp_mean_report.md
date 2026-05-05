## Mean AI Score (ภาพจริง) และความเสี่ยง False Positive

- **Dataset**: `backend\Dataset Senior Project`
- **Backend**: `http://localhost:8001`
- **Threshold (FP on Real)**: **50.00%**
- **API success**: 298 | **fail**: 2

| โมเดล | Mean AI Score (ภาพจริง) | False Positive (ภาพจริง) | ความเสี่ยง False Positive |
|---|---:|---:|---|
| Frequency | 42.80% | 42.00% | สูง |
| ELA | 49.31% | 51.33% | สูง |
| CNN | 55.91% | 56.67% | สูง |
| Pixel-level | 63.62% | 64.67% | สูง |

หมายเหตุ:
- Mean AI Score (ภาพจริง) = ค่าเฉลี่ยของ `confidence` (AI probability %) ของโมเดลนั้น เฉพาะรูปที่เป็น Real
- False Positive (ภาพจริง) = สัดส่วนรูป Real ที่โมเดลให้ `confidence >= threshold`
- ระดับความเสี่ยง: ต่ำ (<5%), ปานกลาง (5–<15%), สูง (>=15%)
