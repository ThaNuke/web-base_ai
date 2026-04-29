import { useState, useEffect, useRef } from 'react'
import './App.css'
import { FiShield, FiArrowDown, FiLock, FiTarget, FiUpload, FiX, FiBarChart2, FiChevronDown, FiMenu } from 'react-icons/fi'
import { BiChevronRight } from 'react-icons/bi'

const API_URL = import.meta.env.VITE_API_URL || 'https://web-baseai-production.up.railway.app'

const heroImages = [
  '/background/5501.jpg',
  '/background/5502.jpg',
  '/background/5503.jpg',
  '/background/5504.jpg',
  '/background/5505.jpg',
  '/background/5506.jpg',
  '/background/5507.jpg',
  '/background/5508.jpg',
  '/background/5509.jpg',
  '/background/5510.jpg',
  '/background/5511.jpg',
]

const howToUseSteps = [
  {
    step: '1',
    title: 'Upload an Image',
    description:
      'Upload an image from your device (JPG/PNG, max 10MB). You can also drag and drop to begin.',
    image: '/How_to_use/1.png'
  },
  {
    step: '2',
    title: 'Run Image Analysis',
    description:
      'Our AI detector analyzes visual patterns using multiple models to identify AI-generated content.',
    image: '/How_to_use/2.png'
  },
  {
    step: '3',
    title: 'Review the Detection Result',
    description:
      'Get a clear verdict with confidence and detailed insights from each model.',
    image: '/How_to_use/3.png'
  }
]

const useCaseCards = [
  {
    title: 'Inspect Product and Marketplace Images',
    description:
      'Use the AI image detector to verify product photos and promotional visuals across online marketplaces.',
    image: '/use_cases/products.png'
  },
  {
    title: 'Verify News and Media Images',
    description:
      'Use this AI image checker to review images used in news articles, blogs, and media content, and check whether AI was involved.',
    image: '/use_cases/news.png'
  },
  {
    title: 'Analyze AI-Generated Artwork',
    description:
      'This AI art detector helps identify illustrations, designs, and creative images produced by AI image generators.',
    image: '/use_cases/artwork.png'
  }
]

function App() {
  const uploadSectionRef = useRef(null)
  const [showUpload, setShowUpload] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(null) 
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [selectedMethod, setSelectedMethod] = useState(null)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [selectedAudience, setSelectedAudience] = useState('auditors')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const scrollToSection = (id) => {
    setMobileMenuOpen(false)
    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentImageIndex((prev) => (prev + 1) % heroImages.length)
    }, 4000)

    return () => clearInterval(interval)
  }, [])

  const methodsData = {
    '01': {
      number: '01',
      title: 'Convolutional Neural Network',
      subtitle: 'Neural network-based image analysis',
      description: 'Xception, a CNN, is designed to extract more detailed features than conventional models. The input images are scaled to 150x150 pixels and adjusted to a range of [0-1] before the actual training process. The model utilizes Depthwise Separable Convolution, which separates processing by color channels (R, G, B) to extract image characteristics, such as sharpness, detail, or shading. Pointwise Convolution then combines information from all color channels, allowing the model to better understand the image holistically.'
    },
    '02': {
      number: '02',
      title: 'Frequency-domain analysis',
      subtitle: 'Frequency spectrum (FFT) analysis',
      description: 'The Frequency-Domain method begins by converting the image into a frequency spectrum using FFT (Fast Fourier Transform) to create a three-channel map of magnitude, phase, and bandpass. This helps amplify the unnatural frequency characteristics often found in AI-generated images. After generating the FFT feature map, the system loads all the data and feeds it into the FreqResNet model, which is specifically designed to handle frequency data. During training, the model learns balanced frequency profiles of real images versus those that appear distorted in fake images. Cross-Entropy Loss is used for training and evaluation at every epoch, as with other techniques. The best version is recorded, allowing the model to more clearly understand the differences in frequency spectrum between real and fake images, particularly those generated or amplified by AI.'
    },
    '03': {
      number: '03',
      title: 'Pixel-level analysis',
      subtitle: 'Per-pixel artifact inspection',
      description: 'The pixel-level technique begins by creating a pixel map containing detailed information about each pixel, such as edges (laplacian), texture variance (local variance), pixel flow direction (SobelX), and gray scale values. This information is combined into a 4-channel image, which provides more detail than normal RGB. The system loads both RGB and pixel map data and sends them to the PixelRes model in two streams, similar to the ELA technique. The pixel stream captures micro-pixel irregularities, while the RGB stream captures the overall context. Training and evaluation follow the same pattern as other techniques: Loss and Accuracy are measured at each epoch, with the best model recorded. The result is a model that can detect finer-grained differences in image structure.'
    },
    '04': {
      number: '04',
      title: 'Error Level Analysis',
      subtitle: 'Compression error artifact analysis (ELA)',
      description: 'Using the ELA technique, the process starts by generating an ELA image by recompressing the original image with reduced JPEG quality. Pixel differences are then calculated to reveal "anomalous traces" such as bright spots or uneven textures, which are often found in AI-generated or manipulated images. Once the ELA image is generated, both the RGB and ELA images are fed into a dual-stream ELRes model, which processes the two data sets simultaneously, allowing the model to see both the structure of the real image and any artificial anomalies. Training is performed using Cross-Entropy Loss and evaluation is performed at every Epoch; the best model is saved if the test results improve.'
    }
  }

  const benefitsData = {
    auditors: {
      category: 'Online Platform Auditors',
      description: 'Reviewing promotional or advertising images ensures that visual materials are accurate and reliable before public dissemination.'
    },
    media: {
      category: 'News Media Practitioners',
      description: 'Verify the authenticity of photographs and visual content to maintain journalistic integrity and combat misinformation.'
    },
    ecommerce: {
      category: 'E-Commerce Platform Operators',
      description: 'Protect customers from fraudulent product listings by detecting AI-generated or manipulated product images.'
    },
    appraisers: {
      category: 'Art Appraisers',
      description: 'Authenticate artwork and detect digitally manipulated or AI-created artwork to provide accurate valuations.'
    },
    marketers: {
      category: 'Marketing & Advertising Professionals',
      description: 'Ensure brand authenticity and compliance with advertising regulations by detecting fabricated visual content.'
    }
  }

  const handleViewMore = (methodNumber) => {
    setSelectedMethod(methodNumber)
  }

  const handleBackFromMethod = () => {
    setSelectedMethod(null)
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setError(null)
      setResult(null)

      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please choose an image.')
      return
    }

    setLoading(true)
    setLoadingStep('ai')
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedFile)

      const aiRes = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData
      })
      const aiData = await aiRes.json()

      if (!aiRes.ok) {
        throw new Error(aiData.detail || aiData.error || 'AI analysis failed')
      }

      setResult(aiData.result)
    } catch (err) {
      setError(err.message || 'Server error.')
    } finally {
      setLoading(false)
      setLoadingStep(null)
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
            <FiShield size={32} color="#FFFFFF" />
          </div>
          <span className="logo-text">TruPic</span>
        </div>
        {/* Desktop nav */}
        <nav className="nav desktop-nav">
          <a href="#how-to-use" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('how-to-use') }}>How to Use</a>
          <a href="#upload" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('upload') }}>Try It</a>
          <a href="#features" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('features') }}>Features</a>
          <a href="#how-it-works" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('how-it-works') }}>How It Works</a>
          <a href="#use-cases" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('use-cases') }}>Use Cases</a>
          <a href="#benefits" className="nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('benefits') }}>Benefits</a>
        </nav>

        {/* Mobile hamburger button */}
        <button className="hamburger-btn" onClick={() => setMobileMenuOpen(!mobileMenuOpen)} aria-label="Toggle menu">
          {mobileMenuOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>

        {/* Mobile drawer overlay */}
        <div className={`mobile-menu-overlay ${mobileMenuOpen ? 'open' : ''}`} onClick={() => setMobileMenuOpen(false)} />
        <nav className={`mobile-nav ${mobileMenuOpen ? 'open' : ''}`}>
          <a href="#how-to-use" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('how-to-use') }}>How to Use</a>
          <a href="#upload" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('upload') }}>Try It</a>
          <a href="#features" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('features') }}>Features</a>
          <a href="#how-it-works" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('how-it-works') }}>How It Works</a>
          <a href="#use-cases" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('use-cases') }}>Use Cases</a>
          <a href="#benefits" className="mobile-nav-link" onClick={(e) => { e.preventDefault(); scrollToSection('benefits') }}>Benefits</a>
        </nav>
      </header>

      <main className="main-content">
        <div className="landing-page">
          <div className="landing-hero">
            <div className="carousel-background">
              <div className="carousel-track">
                {[...heroImages, ...heroImages].map((image, index) => (
                  <div key={index} className="carousel-item-bg">
                    <img src={image} alt="carousel" />
                  </div>
                ))}
              </div>
            </div>

            <h1 className="headline">
              Unmask <span className="highlight">AI-Generated</span> Images
            </h1>
            <button
              className="cta-button"
              onClick={() => {
                uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
              }}
            >
              Get Started
              <FiArrowDown size={20} />
            </button>
          </div>

          <section className="how-to-use-section" id="how-to-use">
            <div className="how-to-use-container">
              <h2 className="how-to-use-title">How to Use Our AI Image Detector?</h2>

              <div className="how-to-use-steps-grid">
                {howToUseSteps.map((s) => (
                  <div key={s.step} className="how-to-use-step-card">
                    <div className="how-to-use-step-media">
                      <img src={s.image} alt={`Step ${s.step}`} />
                    </div>

                    <div className="how-to-use-step-meta">
                      <div className="how-to-use-step-number">Step {s.step}</div>
                      <h3 className="how-to-use-step-title">{s.title}</h3>
                      <p className="how-to-use-step-description">{s.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <div className="upload-section" id="upload" ref={uploadSectionRef}>
            <h2 className="upload-title">Upload Your Image</h2>

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
                  <FiUpload size={48} />
                  <span>Click to choose a file, or drag and drop it here</span>
                  <small>Supported: JPEG, PNG (max 10MB)</small>
                </label>
              </div>
            ) : (
              <div className="preview-section">
                {!result ? (
                  <div className="image-preview">
                    <img src={preview} alt="Preview" />
                    <button className="remove-image" onClick={handleReset}>
                      <FiX size={24} />
                    </button>

                    {loading && (
                      <div className="loading-container">
                        <div className="loading-spinner">
                          <div className="spinner-circle"></div>
                          <div className="spinner-circle"></div>
                          <div className="spinner-circle"></div>
                        </div>
                        <h3 className="loading-text"> Analyzing with AI Models...</h3>
                        <div className="loading-steps">
                          <div className={`loading-step ${loadingStep === 'ai' ? 'active' : ''}`}>
                            <span className="step-dot">1</span>
                            <span>AI Model Analysis</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}

                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}

                {!loading && !result && (
                  <div className="action-buttons">
                    <button
                      className="cta-button"
                      onClick={handleUpload}
                      disabled={loading}
                    >
                      Analyze Image
                    </button>
                    <button
                      className="secondary-button"
                      onClick={handleReset}
                      disabled={loading}
                    >
                      Choose Another Image
                    </button>
                  </div>
                )}

                {result && (() => {
                  // Backend `confidence` is an AI-likelihood percentage (0-100).
                  // For UI consistency, derive the headline/badge/classification from this same number.
                  const aiProbRaw = Number(result.confidence)
                  const aiProb = Number.isFinite(aiProbRaw) ? Math.min(100, Math.max(0, aiProbRaw)) : 0
                  const realProb = 100 - aiProb
                  // Use backend's final verdict when available (more consistent than local thresholds).
                  const isAI = typeof result.isAIGenerated === 'boolean' ? result.isAIGenerated : aiProb >= 50

                  const displayPercent = isAI ? aiProb.toFixed(0) : realProb.toFixed(0)
                  const displayLabel = isAI ? 'AI' : 'REAL'

                  // Determine confidence level based on the displayed side
                  let confidenceLevel = 'Low'
                  const dominantProb = isAI ? aiProb : realProb
                  if (dominantProb >= 90) confidenceLevel = 'High'
                  else if (dominantProb >= 75) confidenceLevel = 'Medium'

                  const ringValue = isAI ? aiProb : realProb
                  const ringPercent = ringValue.toFixed(0)
                  const ringLabel = isAI ? 'AI Likelihood' : 'Real Likelihood'
                  const ringColorClass = isAI ? 'ring-ai' : 'ring-real'
                  const themeClass = isAI ? 'theme-ai' : 'theme-real'

                  return (
                    <div className={`result-card-v2 ${themeClass} success-animation`}>
                      <div className="result-v2-accent" aria-hidden="true" />
                      <div className="result-v2-image">
                        <img src={preview} alt="Analyzed" />
                        <button className="result-v2-close" onClick={handleReset}>
                          <FiX size={16} />
                        </button>
                      </div>

                      <div className="result-v2-info">
                        <div className="result-v2-summary">
                          <span className="result-v2-summary-text">
                            {isAI ? 'This image is likely AI-generated' : 'This image is likely real'}
                          </span>
                        </div>

                        <div className="result-v2-metrics">
                          <div className={`result-v2-ring ${ringColorClass}`} aria-label={`${ringLabel}: ${ringPercent}%`}>
                            <div
                              className="ring"
                              style={{ ['--p']: ringPercent }}
                            >
                              <div className="ring-center">
                                <div className="ring-percent">{ringPercent}%</div>
                                <div className="ring-caption">{ringLabel}</div>
                              </div>
                            </div>
                          </div>

                          <div className="result-v2-details">
                            <div className="result-v2-row">
                              <span className="result-v2-label">AI Likelihood</span>
                              <span className={`result-v2-value ${aiProb >= 50 ? 'text-danger' : 'text-muted'}`}>
                                {aiProb.toFixed(0)}%
                              </span>
                            </div>
                            <div className="result-v2-row">
                              <span className="result-v2-label">Confidence</span>
                              <span className="result-v2-value text-bold">{confidenceLevel}</span>
                            </div>
                            <div className="result-v2-row">
                              <span className="result-v2-label">Classification</span>
                              <span className={`result-v2-value ${isAI ? 'text-danger' : 'text-success'}`}>
                                {isAI ? 'AI Generated' : 'Real'}
                              </span>
                            </div>
                          </div>
                        </div>

                        <button className="result-v2-retry" onClick={handleReset}>
                          ⟳ Try Another Image
                        </button>
                      </div>
                    </div>
                  )
                })()}
              </div>
            )}
          </div>

          <section className="features-section" id="features">
            <div className="features-container">
              <h2 className="features-title">Why Choose TruPic?</h2>
              <p className="features-subtitle">Our cutting-edge technology provides reliable AI image detection for professionals and enthusiasts alike</p>

              <div className="features-grid">
                <div className="feature-card">
                  <div className="feature-icon privacy-icon">
                    <FiLock size={48} />
                  </div>
                  <h3 className="feature-title">Privacy First</h3>
                  <p className="feature-description">Your images are processed securely and never stored on our servers</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon analysis-icon">
                    <FiBarChart2 size={48} />
                  </div>
                  <h3 className="feature-title">AI Analysis</h3>
                  <p className="feature-description">Receive comprehensive reports with confidence scores and detailed insights</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon detection-icon">
                    <FiTarget size={48} />
                  </div>
                  <h3 className="feature-title">Advanced Detection</h3>
                  <p className="feature-description">State-of-the-art AI algorithms to accurately identify AI-generated images</p>
                </div>
              </div>
            </div>
          </section>

          <section className="how-it-works-section" id="how-it-works">
            <div className="how-it-works-container">
              <h2 className="how-it-works-title">How It Works</h2>
              <p className="how-it-works-subtitle">Advanced detection powered by state-of-the-art AI models</p>

              {!selectedMethod ? (
                <div className="methods-grid">
                  <div className="method-card">
                    <div className="method-number">01</div>
                    <h3 className="method-title">Convolutional Neural Network</h3>
                    <p className="method-description">Neural network-based image analysis</p>
                    <button className="view-more-btn" onClick={() => handleViewMore('01')}>View more <BiChevronRight size={16} /></button>
                  </div>

                  <div className="method-card">
                    <div className="method-number">02</div>
                    <h3 className="method-title">Frequency-domain analysis</h3>
                    <p className="method-description">Frequency spectrum (FFT) analysis</p>
                    <button className="view-more-btn" onClick={() => handleViewMore('02')}>View more <BiChevronRight size={16} /></button>
                  </div>

                  <div className="method-card">
                    <div className="method-number">03</div>
                    <h3 className="method-title">Pixel-level analysis</h3>
                    <p className="method-description">Per-pixel artifact inspection</p>
                    <button className="view-more-btn" onClick={() => handleViewMore('03')}>View more <BiChevronRight size={16} /></button>
                  </div>

                  <div className="method-card">
                    <div className="method-number">04</div>
                    <h3 className="method-title">Error Level Analysis</h3>
                    <p className="method-description">Compression error artifact analysis (ELA)</p>
                    <button className="view-more-btn" onClick={() => handleViewMore('04')}>View more <BiChevronRight size={16} /></button>
                  </div>
                </div>
              ) : (
                <div className="method-detail">
                  <div className="method-detail-left">
                    <div className="detail-number">{methodsData[selectedMethod].number}</div>
                    <h3 className="detail-title">{methodsData[selectedMethod].title}</h3>
                    <p className="detail-thai">{methodsData[selectedMethod].subtitle}</p>
                  </div>
                  <div className="method-detail-right">
                    <div className="detail-description-box">
                      {methodsData[selectedMethod].description}
                    </div>
                  </div>
                  <button className="back-from-method-btn" onClick={handleBackFromMethod}>
                    ← Back
                  </button>
                </div>
              )}
            </div>
          </section>

          <section className="use-cases-section" id="use-cases">
            <div className="use-cases-container">
              <h2 className="use-cases-title">What You Can Check With TruPic</h2>
              <p className="use-cases-subtitle">
                Use TruPic's AI image detector to verify different types of images and understand whether AI was used to generate, edit, or manipulate visual content.
              </p>

              <div className="use-cases-grid">
                {useCaseCards.map((card, idx) => (
                  <div key={idx} className="use-case-card">
                    <div className="use-case-image">
                      <img src={card.image} alt={card.title} />
                    </div>
                    <h3 className="use-case-card-title">{card.title}</h3>
                    <p className="use-case-card-description">{card.description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="benefits-section" id="benefits">
            <div className="benefits-container">
              <h2 className="benefits-title">Who Can Benefit from AI Image Detection?</h2>

              <div className="category-tabs">
                {Object.entries(benefitsData).map(([key, data]) => (
                  <button
                    key={key}
                    className={`category-tab ${selectedAudience === key ? 'active' : ''}`}
                    onClick={() => setSelectedAudience(key)}
                  >
                    {data.category}
                  </button>
                ))}
              </div>

              <div className="benefits-content">
                <div className="benefits-description">
                  {benefitsData[selectedAudience].description}
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
