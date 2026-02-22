import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

function App() {
  const [showUpload, setShowUpload] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setResult(null)
      
      // สร้าง preview
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('choose an image')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'error')
      }

      setResult(data.result)
    } catch (err) {
      setError(err.message || 'error server')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo-container">
          <div className="logo-icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M16 2L4 8V16C4 22.6274 9.37258 28 16 28C22.6274 28 28 22.6274 28 16V8L16 2Z" fill="#7C3AED" stroke="#7C3AED" strokeWidth="1"/>
              <path d="M16 10L12 12V18C12 20.2091 13.7909 22 16 22C18.2091 22 20 20.2091 20 18V12L16 10Z" fill="white"/>
            </svg>
          </div>
          <span className="logo-text">TruPic</span>
        </div>
        <nav className="nav">
          <a href="#features" className="nav-link">Features</a>
          <a href="#how-it-works" className="nav-link">How it work</a>
          <a href="#about" className="nav-link">About</a>
        </nav>
      </header>
      
      <main className="main-content">
        {!showUpload ? (
          <>
            <h1 className="headline">
              Unmask <span className="highlight">AI-Generated</span> Images
            </h1>
            <button 
              className="cta-button"
              onClick={() => setShowUpload(true)}
            >
              Get Started
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="10" cy="10" r="9" fill="white" fillOpacity="0.25"/>
                <path d="M10 6V12M10 12L7 9M10 12L13 9" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </>
        ) : (
          <div className="upload-section">
            <h2 className="upload-title">Upload</h2>
            
            {!preview ? (
              <div className="upload-area">
                <input
                  type="file"
                  id="file-input"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
                <label htmlFor="file-input" className="upload-label">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  <span>คลิกเพื่อเลือกไฟล์หรือลากไฟล์มาวางที่นี่</span>
                  <small>รองรับไฟล์: JPEG, PNG, GIF, WebP (สูงสุด 10MB)</small>
                </label>
              </div>
            ) : (
              <div className="preview-section">
                <div className="image-preview">
                  <img src={preview} alt="Preview" />
                  <button className="remove-image" onClick={handleReset}>×</button>
                </div>
                
                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}
                
                {result && (
                  <div className={`result-card ${result.isAIGenerated ? 'ai-detected' : 'real-image'}`}>
                    <div className="result-header">
                      <div className="result-icon">
                        {result.isAIGenerated ? '🤖' : '✅'}
                      </div>
                      <div className="result-text">
                        <h3>{result.message}</h3>
                        <p className="confidence-value">ความมั่นใจ: {result.confidence}%</p>
                      </div>
                    </div>

                    <div className="probability-section">
                      <div className="probability-item">
                        <div className="probability-label">
                          <span>AI Probability</span>
                          <span className="probability-value">{result.confidence}%</span>
                        </div>
                        <div className="probability-bar ai-bar">
                          <div 
                            className="probability-fill" 
                            style={{ width: `${result.confidence}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="probability-item">
                        <div className="probability-label">
                          <span>Real Probability</span>
                          <span className="probability-value">{100 - result.confidence}%</span>
                        </div>
                        <div className="probability-bar real-bar">
                          <div 
                            className="probability-fill" 
                            style={{ width: `${100 - result.confidence}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="action-buttons">
                  <button 
                    className="cta-button"
                    onClick={handleUpload}
                    disabled={loading}
                  >
                    {loading ? 'กำลังวิเคราะห์...' : 'วิเคราะห์ภาพ'}
                  </button>
                  <button 
                    className="secondary-button"
                    onClick={handleReset}
                    disabled={loading}
                  >
                    เลือกภาพใหม่
                  </button>
                </div>
              </div>
            )}
            
            <button 
              className="back-button"
              onClick={() => {
                setShowUpload(false)
                handleReset()
              }}
            >
              ← กลับหน้าแรก
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
