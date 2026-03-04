# Stage 0: Image/Text Processing (Parallel Audio Extraction)

## Overview

This stage handles **video content with minimal or no voice/audio**, providing an alternative processing path to the traditional audio extraction pipeline.

When the audio analysis detects insufficient voice content (RMS energy below threshold or voice ratio < 10%), this pipeline automatically triggers instead of the Audio Extraction stage.

## Architecture

```
Decision Point: Analyze Audio Levels
    |
    |- Voice Detected (>10% voice ratio, energy > 0.05 RMS)
    |  +-> Go to Stage 1: Audio Extraction / Whisper
    |
    +- Silent/Minimal Voice
       +-> Use Image/Text Processing (This Stage)
           |
           |- Frame Capture (Smart Sampling)
           |- DOM Text Extraction
           |- Client-Side OCR (Tesseract.js)
           +-> Backend Text Processing
               |
               |- Text Cleaning
               |- Misinformation Analysis
               +-> Output Claims
```

## Input

**Source**: Chrome Extension (`chrome-extension/image-capture.js`)

```json
{
  "request_id": "req_img_1234567890_abc123",
  "claim_id": "claim_1234567890",
  "frame_count": 5,
  "audio_analysis": {
    "is_silent": true,
    "has_voice": false,
    "voice_ratio": 0.05,
    "avg_energy": 0.032
  },
  "captured_text": ["text from frame 1", "text from frame 2"],
  "dom_text": ["captions", "on-screen text"],
  "frames": [
    {
      "index": 0,
      "blob": "<binary jpeg>",
      "ocr_text": "Extracted text from frame"
    }
  ]
}
```

See [input_schema.json](input_schema.json) for full schema.

## Processing Steps

### Step 1: Frame Extraction (Chrome Extension)
- Analyze audio to decide if image capture needed
- Capture frames at 1.5s intervals or on visual change
- Limit to max 15 frames per session
- Send frames to backend via multipart form

**File**: `chrome-extension/image-capture.js`
**Key Function**: `startFrameCaptureLoop()`

### Step 2: OCR Processing (Client-Side + Backend)
- Client: Tesseract.js (privacy-preserving)
- Backend: OpenCV preprocessing + pytesseract

**File**: `api/routes/imageprocessing.py`
**Key Function**: `extract_text_from_image()`

### Step 3: Text Cleaning (Backend)
- Remove emojis, URLs, mentions, hashtags
- Normalize whitespace
- Remove repeated characters
- Validate text quality
- Confidence filtering

**File**: `api/utils/text_processor.py`
**Class**: `TextCleaner`

### Step 4: Misinformation Analysis (Backend)
- Split cleaned text into segments
- Run RoBERTa classification on each segment
- Generate predictions with confidence scores
- Return results to popup

**File**: `api/routes/imageprocessing.py`
**Key Function**: `analyze_text_for_misinfo()`

## Output

**Destination**: Chrome Popup UI

```json
{
  "request_id": "req_img_...",
  "claim_id": "claim_...",
  "status": "success",
  "extracted_text": "Cleaned merged text from all sources",
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
    "timestamp": "2026-02-25T...",
    "processing_time_ms": 2340,
    "text_valid": true
  }
}
```

See [output_schema.json](output_schema.json) for full schema.

## API Endpoints

### POST `/api/image/analyze-claim`
Main pipeline endpoint. Receives frames, extracts text, analyzes for misinformation.

**Response Code**: 200 OK (success), 400 Bad Request (invalid input), 500 Server Error

### POST `/api/image/detect-silence`
Audio decision endpoint. Determines if voice is present.

**Input**:
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

### GET `/api/image/health`
Health check endpoint.

## Configuration

### Audio Decision Thresholds

**File**: `api/utils/audio_utils.py`

```python
SILENCE_THRESHOLD_RMS = 0.02          # Very quiet background noise
VOICE_THRESHOLD_RMS = 0.05            # Conversational speech
VOICE_PRESENCE_RATIO = 0.1            # At least 10% of audio must be voice
MIN_VOICE_DURATION_SEC = 0.5          # At least 0.5s of voice content
```

### Frame Capture Thresholds

**File**: `chrome-extension/image-capture.js`

```javascript
MIN_CAPTURE_INTERVAL_MS = 1500   // 1.5 seconds between frames
MAX_FRAMES = 15                   // Maximum frames per session
JPEG_QUALITY = 0.65              // JPEG quality (0-1)
```

### Text Cleaning Thresholds

**File**: `api/utils/text_processor.py`

```python
MIN_TEXT_LENGTH = 3
MAX_REPEATED_CHAR_RATIO = 0.3    # Max 30% same character
MIN_CONFIDENCE_THRESHOLD = 0.5   # 50% OCR confidence minimum
```

## Performance

- Audio Analysis: 1-2 seconds
- Frame Capture: 3-5 seconds (15 frames max)
- OCR Processing: 200-500ms per frame
- Text Cleaning: 10-50ms
- Misinformation Analysis: 300-800ms
- **Total E2E**: 8-15 seconds typical
- **Payload**: 2-4 MB typical

## Testing

### Unit Tests
```bash
pytest tests/test_audio_utils.py::test_silence_detection
pytest tests/test_text_processor.py::test_emoji_removal
```

### Integration Tests
```bash
pytest tests/test_integration.py::test_image_claim_analysis_end_to_end
```

### Manual Testing
1. Open TikTok/YouTube Shorts with silent video
2. Click "Auto-Detect & Start" in TruthLens popup
3. Wait 10-15 seconds for frame capture
4. Click "Stop & Analyze"
5. Review extracted text and predictions in popup

## Dependencies

**Backend**:
- opencv-python >= 4.8.0
- pytesseract >= 0.3.10
- Pillow >= 10.0.0
- librosa >= 10.0 (optional)
- scipy >= 1.11.0 (optional)

**Frontend**:
- Tesseract.js v5.0.2 (CDN)
- Web Audio API (built-in)

Install with:
```bash
pip install -r requirements.txt
```

Optional system package (Tesseract-OCR):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Troubleshooting

### Tesseract.js loads slowly
- Increase timeout in `chrome-extension/image-capture.js`
- Use browser cache (CDN caching)

### Backend returns 500 error
- Install system `tesseract-ocr` package
- Check Python dependencies: `pip install -r requirements.txt`
- Review server logs for detailed error

### Audio detection says "voice found" but it's silent
- Increase thresholds in `api/utils/audio_utils.py`:
  ```python
  VOICE_THRESHOLD_RMS = 0.07  # from 0.05
  VOICE_PRESENCE_RATIO = 0.15 # from 0.10
  ```

### Capturing too many frames
- Increase minimum interval in `chrome-extension/image-capture.js`:
  ```javascript
  const MIN_CAPTURE_INTERVAL_MS = 2000;  // from 1500
  ```

## References

- Complete Documentation: [IMAGE_TEXT_PROCESSING.md](../IMAGE_TEXT_PROCESSING.md)
- Chrome Extension Code: [image-capture.js](../../../chrome-extension/image-capture.js)
- Backend Route: [imageprocessing.py](../../../api/routes/imageprocessing.py)
- Utilities: [text_processor.py](../../../api/utils/text_processor.py), [audio_utils.py](../../../api/utils/audio_utils.py)
- Frontend UI: [popup.js](../../../chrome-extension/popup.js)

## Status

**Production Ready** - Full implementation with comprehensive error handling, logging, and documentation.

**Branch**: `mrida-image/text_processing`
