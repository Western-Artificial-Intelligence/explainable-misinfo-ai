# Image/Text Processing Production Pipeline

## Overview

This document describes the end-to-end image and text processing pipeline for misinformation detection on video content (TikTok, YouTube Shorts, etc.) when voice content is absent or minimal.

## Architecture

```
Chrome Extension (TikTok/YouTube)
    │
    ├─ Analyze Audio Levels (RMS-based)
    │   ├─ If Voice Present → Use Audio Pipeline (existing Whisper)
    │   └─ If No Voice → Use Image/Text Pipeline (this)
    │
    ├─ Frame Capture (Smart Sampling)
    │   ├─ 1 frame per 1.5 seconds
    │   ├─ OR on significant visual change
    │   └─ Max 15 frames per capture session
    │
    ├─ DOM Text Extraction
    │   ├─ Extract captions, overlays
    │   └─ Filter out UI controls
    │
    └─→ Backend /api/image/analyze-claim
        │
        ├─ OCR on Frames (Tesseract.js client-side)
        │   └─ Extract text from images
        │
        ├─ Text Cleaning (Production-Grade)
        │   ├─ Remove emojis, URLs, mentions
        │   ├─ Normalize whitespace
        │   ├─ Remove repeated characters
        │   ├─ Confidence filtering
        │   └─ Unicode normalization
        │
        ├─ Merge Text Sources
        │   ├─ OCR text from frames
        │   ├─ DOM text from page  
        │   └─ Deduplicate and merge
        │
        ├─ Analyze for Misinformation
        │   ├─ Split into segments
        │   ├─ Run classification pipeline
        │   └─ Generate predictions
        │
        └─→ Response to Extension UI
```

## Key Components

### 1. Audio Analysis (Chrome Extension)

**File**: `chrome-extension/image-capture.js`

**Function**: `analyzeAudioLevelsAdvanced()`

**Algorithm** (Option A: RMS-Based - **Recommended**):

```javascript
// Frequency domain RMS calculation
- Sample frequency data every 100ms for ~1 second
- Calculate RMS from FFT bins
- Thresholds:
  * SILENCE_THRESHOLD = 0.02 RMS (~-34 dB)
  * VOICE_THRESHOLD = 0.05 RMS (~-26 dB)
  * MIN_VOICE_RATIO = 10% (voice must be 10%+ of audio)
  * MIN_VOICE_DURATION = 0.5 seconds

// If audio fails these checks → Trigger image/text pipeline
if (voiceRatio < 10% || avgEnergy < VOICE_THRESHOLD) {
  useImageCapture()
}
```

**Advantages**:
- Fast, local processing (no server round-trip)
- Privacy-preserving
- Works offline
- Reduces unnecessary OCR (expensive operation)

### 2. Frame Capture (Chrome Extension)

**File**: `chrome-extension/image-capture.js`

**Function**: `startFrameCaptureLoop()`

**Smart Sampling Strategy**:

```javascript
Capture decision logic:
1. Check time since last capture: if (now - lastCaptureTime > 1500ms)
2. Check frame change: if (calculateFrameHash() differs)
3. Limits:
   - Max 15 frames per session
   - Check every 300ms, but only save every 1.5s+
   - JPEG quality: 0.65 (balance size vs quality)

Frame Hash Calculation:
- Sample every 1000th pixel (fast, not expensive)
- Compare hashes to detect visual changes
- Avoid capturing duplicate similar frames
```

**Why Smart Sampling?**:
- TikTok videos are typically 15-60 seconds
- 15 frames = 1 frame per 4 seconds (very sparse)
- Reduces payload and OCR processing
- Still captures key text moments
- Avoids OCR spamming on static content

### 3. Text Extraction

**Text Sources**:

1. **OCR from Frames** (Tesseract.js, client-side)
   - Pros: No privacy loss, fast, works offline
   - Applied to each captured frame
   - Tesseract.js v5.0.2 from CDN

2. **DOM Text Extraction** (Real-time)
   - Grab captions, overlays, and on-screen text
   - Filter out UI controls (like, share, follow, etc.)
   - Lower latency than OCR

### 4. Text Cleaning (Backend)

**File**: `api/utils/text_processor.py`

**TextCleaner Class**:

```python
Clean operations (in order):
1. Unicode normalization (NFC)
2. Remove emojis (regex pattern)
3. Remove URLs (http[s]://...)
4. Remove @mentions
5. Remove #hashtags (or convert to words)
6. Remove repeated characters (hellooooo → hello)
7. Normalize whitespace (collapse spaces, normalize newlines)
8. Validate (min length, alphanumeric ratio)

Confidence filtering:
- Discard text if OCR confidence < min threshold
- Default: 0.5 (50% confidence minimum)

Validation criteria:
- Min 3 characters
- At least 30% alphanumeric
- Not excessively repeated characters
```

**Example**:

```python
raw_text = "AMAZING 😱😱 OFFER!!! Visit https://fake.com @everyone #BUY_NOW"
cleaned = cleaner.clean(raw_text)
# Result: "AMAZING OFFER Visit @everyone BUY NOW"
```

### 5. Misinformation Analysis (Backend)

**File**: `api/routes/imageprocessing.py`

**Function**: `analyze_text_for_misinfo()`

```python
# Split cleaned text into segments
segments = merged_text.split('.')
segments = [s.strip() for s in segments if len(s.strip()) > 5]

# Analyze each segment
for segment in segments:
    result = predict_text(segment)  # Existing RoBERTa model
    predictions.append({
        "text": segment,
        "prediction": "REAL" or "FAKE",
        "confidence": 0.0-1.0,
        "explanation": "..."
    })
    
# Return all predictions to popup
```

## API Endpoints

### POST `/api/image/analyze-claim`

**Request**:
```json
{
  "request_id": "req_img_1234567890_abc123",
  "claim_id": "claim_1234567890",
  "frame_count": 5,
  "audio_analysis": "{...json...}",
  "captured_text": "[\"text1\", \"text2\", ...]",
  "frame_0": <binary image>,
  "frame_0_text": "OCR extracted text",
  ...
}
```

**Response**:
```json
{
  "request_id": "req_img_...",
  "claim_id": "claim_...",
  "status": "success",
  "analysis_type": "image_and_text_extraction",
  "extracted_text": "Cleaned merged text...",
  "text_statistics": {
    "char_count": 234,
    "word_count": 45,
    "sentence_count": 3,
    "avg_word_length": 5.2,
    "avg_sentence_length": 15
  },
  "frame_count": 5,
  "audio_analysis": {...},
  "misinfo_predictions": [
    {
      "text": "Claim segment",
      "prediction": "FAKE",
      "confidence": 0.87,
      "explanation": "Similar to known false claims..."
    }
  ],
  "meta": {
    "timestamp": "2026-02-24T...",
    "extraction_sources": 7,
    "final_text_length": 234,
    "text_valid": true
  }
}
```

### POST `/api/image/detect-silence`

**Request**:
```json
{
  "request_id": "req_...",
  "audio_energy": 0.045,
  "duration_seconds": 15.5,
  "sample_rate": 16000
}
```

**Response**:
```json
{
  "request_id": "req_...",
  "is_silent": false,
  "has_voice": true,
  "confidence": 0.85,
  "recommendation": "Use audio transcription (voice detected)"
}
```

## Dependencies

**Backend**:
- `opencv-python >= 4.8.0` - Image processing, preprocessing
- `pytesseract >= 0.3.10` - OCR wrapper
- `Pillow >= 10.0.0` - Image manipulation
- `librosa >= 10.0` - Audio processing (optional, for advanced analysis)
- `scipy >= 1.11.0` - Scientific computing (optional)

**Frontend** (Chrome Extension):
- `Tesseract.js v5.0.2` - Client-side OCR (loaded from CDN)
- Web Audio API - Audio analysis (built-in)

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install Tesseract-OCR system package
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Mac:
brew install tesseract

# Windows:
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Configuration

**Backend** (api/main.py):
- Imageprocessing router auto-registered if dependencies available
- Falls back gracefully if optional deps missing

**Extension** (popup.html + popup.js):
- Auto-detect mode: Analyzes audio, then chooses capture method
- Force image capture: Skip audio analysis, capture immediately
- Backend URL: Configurable via chrome.storage.sync

## Thresholds & Tuning

### Audio Analysis Thresholds

```python
# api/utils/audio_utils.py
SILENCE_THRESHOLD_RMS = 0.02  # Very quiet background noise (<-34 dB)
VOICE_THRESHOLD_RMS = 0.05    # Conversational speech (~-26 dB)
VOICE_PRESENCE_RATIO = 0.1    # At least 10% of audio should be voice
MIN_VOICE_DURATION_SEC = 0.5  # At least 0.5s of voice content
```

### Text Cleaning Thresholds

```python
# api/utils/text_processor.py
MIN_TEXT_LENGTH = 3
MAX_REPEATED_CHAR_RATIO = 0.3  # Max 30% of same character
MIN_CONFIDENCE_THRESHOLD = 0.5  # 50% confidence minimum
```

### Frame Capture Thresholds

```javascript
// chrome-extension/image-capture.js
MIN_CAPTURE_INTERVAL_MS = 1500   // 1.5 seconds minimum
MAX_FRAMES = 15                   // Limit to 15 frames
JPEG_QUALITY = 0.65              // 65% JPEG quality
```

## Debugging & Logging

**Console Logs** (Browser):
```javascript
[TruthLens] Audio Analysis - Avg RMS: 0.045, Voice Ratio: 85.0%, Has Voice: true
[TruthLens] Starting image capture: req_img_1234567890_abc123
[TruthLens] Frame 5: 234 chars from OCR
[TruthLens] Extracted 7 DOM text elements
```

**Server Logs** (Python):
```
INFO: Text cleaning: 234 -> 198 chars, valid=True
INFO: Extracted 234 characters from image
INFO: Image analysis complete: req_img_..., predictions=3
```

## Error Handling

**Extension**:
- All async operations have try/catch
- Graceful fallbacks (e.g., if Tesseract fails to load)
- User-friendly error messages in popup

**Backend**:
- HTTPException with detail messages
- Comprehensive logging with exc_info=True
- Optional dependency checks with fallbacks

## Performance Characteristics

**Frame Capture**:
- ~15 frames max per session
- ~5-10 seconds total capture time
- Network payload: ~2-5 MB (depends on frame count & quality)

**OCR Processing**:
- Tesseract.js: ~200-500ms per frame (client-side)
- Linear with frame count
- Can be parallelized in future versions

**Backend Processing**:
- Text cleaning: ~10-50ms
- Misinformation analysis: ~100-500ms (depends on # segments)
- Total API response: <1 second typical

## Testing

```bash
# Test silence detection
curl -X POST http://localhost:8000/api/image/detect-silence \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_1",
    "audio_energy": 0.025,
    "duration_seconds": 10.0
  }'

# Test image analysis endpoint
# (Use multipart/form-data with image files)
```

## Security Considerations

1. **Privacy**: OCR runs client-side (no images sent to server)
2. **Input Validation**: All form inputs validated
3. **Payload Size**: Frames limited to 15 (prevents gigantic uploads)
4. **CORS**: Enabled for frontend, but can be restricted

## Future Improvements

1. **GPU Acceleration**: CUDA-enabled Tesseract for batch OCR
2. **Caching**: Cache OCR results for duplicate frames
3. **Model Enhancements**: Fine-tune TextCleaner for social media
4. **Analytics**: Track success rates and optimize thresholds
5. **Multi-Language**: Support non-English videos
6. **Real-time Streaming**: Process frames as video plays

## References

- Tesseract.js: https://github.com/naptha/tesseract.js
- PyTesseract: https://github.com/madmaze/pytesseract
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- FastAPI: https://fastapi.tiangolo.com/
