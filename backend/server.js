import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// สร้างโฟลเดอร์สำหรับเก็บไฟล์ชั่วคราว
const uploadsDir = join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// ตั้งค่า multer สำหรับอัปโหลดไฟล์
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + '.' + file.originalname.split('.').pop());
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(file.originalname.toLowerCase().split('.').pop());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('กรุณาอัปโหลดไฟล์รูปภาพเท่านั้น (jpeg, jpg, png, gif, webp)'));
    }
  }
});

// ฟังก์ชันสำหรับวิเคราะห์ภาพ (จะเชื่อมต่อกับ AI model ต่อไป)
async function analyzeImage(imagePath) {
  try {
    // TODO: เชื่อมต่อกับ AI model ที่นี่
    // ตอนนี้จะ return mock data ก่อน
    
    // ตัวอย่าง: ใช้ Hugging Face API หรือ model อื่นๆ
    // const response = await fetch('https://api-inference.huggingface.co/models/...', {
    //   method: 'POST',
    //   headers: { 'Authorization': 'Bearer YOUR_TOKEN' },
    //   body: fs.readFileSync(imagePath)
    // });
    
    // Mock response สำหรับทดสอบ
    const mockResult = {
      isAIGenerated: Math.random() > 0.5,
      confidence: Math.random() * 100,
      details: {
        probability: Math.random(),
        model: 'Mock Detection Model',
        timestamp: new Date().toISOString()
      }
    };
    
    return mockResult;
  } catch (error) {
    throw new Error(`เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: ${error.message}`);
  }
}

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Backend server is running' });
});

app.post('/api/analyze', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'กรุณาอัปโหลดไฟล์รูปภาพ' });
    }

    const imagePath = req.file.path;
    
    // วิเคราะห์ภาพ
    const result = await analyzeImage(imagePath);
    
    // ลบไฟล์ชั่วคราวหลังจากวิเคราะห์เสร็จ
    fs.unlinkSync(imagePath);
    
    res.json({
      success: true,
      result: {
        isAIGenerated: result.isAIGenerated,
        confidence: Math.round(result.confidence * 100) / 100,
        message: result.isAIGenerated 
          ? 'ภาพนี้น่าจะถูกสร้างด้วย AI' 
          : 'ภาพนี้น่าจะเป็นภาพจริง',
        details: result.details
      }
    });
  } catch (error) {
    // ลบไฟล์ถ้ามีข้อผิดพลาด
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({
      success: false,
      error: error.message || 'เกิดข้อผิดพลาดในการประมวลผล'
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📁 Upload directory: ${uploadsDir}`);
});
