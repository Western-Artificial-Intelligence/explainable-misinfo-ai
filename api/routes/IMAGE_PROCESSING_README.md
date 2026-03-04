# Image/Text Processing - Backend Routes

## Overview

This directory contains **backend API endpoints** for the Image/Text Processing pipeline (Stage 0).

**Main File**: `imageprocessing.py` (600+ lines)

## Endpoints

### POST `/api/image/analyze-claim`

**Purpose**: Main pipeline endpoint for image/text analysis

**Input**: Multipart form data with:
- `request_id` - Request identifier
- `claim_id` - Associated claim ID
- `frame_count` - Number of frames
- `audio_analysis` - Audio decision JSON
- `captured_text` - DOM text extraction
- `frame_N` - Frame image blobs (N=0,1,2,...)
- `frame_N_text` - OCR text for each frame

**Processing**:
1. Extract text from image frames (OpenCV + pytesseract)
2. Merge all text sources (OCR + DOM + captured)
3. Clean text (remove emojis, URLs, mentions, etc.)
4. Segment text and analyze each segment
5. Return predictions with explanations

**Response**: JSON with extracted text, predictions, statistics

**Status Codes**:
- 200 OK - Success
- 400 Bad Request - Invalid form data
- 500 Internal Server Error - Processing failed

### POST `/api/image/detect-silence`

**Purpose**: Audio analysis decision endpoint

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
  "recommendation": "Use audio transcription"
}
```

### GET `/api/image/health`

**Purpose**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-25T...",
  "version": "1.0.0"
}
```

## Key Functions

### Text Extraction
```python
extract_text_from_image(image_data: bytes) -> str
  Purpose: OCR on image frame
  Uses: OpenCV preprocessing + pytesseract
  Returns: Extracted text string
```

### Text Cleaning
```python
clean_and_validate_text(text: str) -> TextValidationResult
  Purpose: Clean text and validate quality
  Uses: TextCleaner utility class
  Returns: Cleaned text + validation result
```

### Misinformation Analysis
```python
analyze_text_for_misinfo(text: str) -> List[MisinfoPrediction]
  Purpose: Classify text segments for misinformation
  Uses: Existing RoBERTa model from predictor service
  Returns: List of predictions with confidence
```

### Audio Analysis
```python
analyze_audio_for_silence(audio_data, sr=16000) -> bool
  Purpose: Determine if audio has significant voice
  Uses: AudioAnalyzer utility class
  Returns: is_silent boolean
```

## Response Models (Pydantic)

### ImageClaimAnalysisResponse
```python
request_id: str
claim_id: Optional[str]
status: str
extracted_text: str
text_statistics: TextStatistics
frame_count: int
misinfo_predictions: List[MisinfoPrediction]
meta: AnalysisMeta
```

### SilenceDetectionResponse
```python
request_id: str
is_silent: bool
has_voice: bool
confidence: float
recommendation: str
```

### MisinfoPrediction
```python
text: str
prediction: str  # "REAL", "FAKE", "MIXED"
confidence: float
explanation: str
```

## Configuration

### Dependencies
- opencv-python >= 4.8.0
- pytesseract >= 0.3.10
- Pillow >= 10.0.0
- python-multipart >= 0.0.6
- aiofiles >= 23.2.0

### System Requirements
- Tesseract-OCR system package (optional, falls back gracefully)

Install:
```bash
pip install -r requirements.txt

# Optional system package
# Ubuntu: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### Thresholds
- Text min length: 3 characters
- OCR confidence threshold: 50% (0.5)
- Max repeated character ratio: 30%

## Error Handling

All endpoints return appropriate HTTP status codes:
- **400 Bad Request** - Invalid form data, missing fields
- **422 Unprocessable** - Validation errors
- **500 Internal Server Error** - Processing failures

Error responses include:
```json
{
  "detail": "User-friendly error message",
  "error_code": "code_identifier",
  "timestamp": "ISO 8601 timestamp"
}
```

## Performance

**Expected Timings**:
- Text extraction (OCR): 200-500ms per frame
- Text cleaning: 10-50ms
- Misinformation analysis: 300-800ms (5 segments)
- Total request: 1-2 seconds typical

## Integration

This route is registered in `api/main.py`:
```python
from routes import imageprocessing
app.include_router(imageprocessing.router, prefix="/api/image")
```

Falls back gracefully if optional dependencies missing.

## Testing

### Unit Tests
```bash
pytest api/tests/test_imageprocessing.py -v
pytest api/tests/test_text_processor.py -v
pytest api/tests/test_audio_utils.py -v
```

### Integration Tests
```bash
pytest api/tests/test_integration.py::test_image_claim_analysis_end_to_end -v
```

### Manual Testing
```bash
# Test silence detection
curl -X POST http://localhost:8000/api/image/detect-silence \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_1",
    "audio_energy": 0.025,
    "duration_seconds": 10.0
  }'

# Test health check
curl http://localhost:8000/api/image/health
```

## Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Processing frame: {frame_index}")
```

Check server logs:
```bash
# Terminal where uvicorn is running
# Shows: INFO: Image analysis complete: req_img_..., predictions=3
```

## Production Pipeline Reference

**Full Documentation**: [../../production_pipeline_instruction/0_Image_Text_Processing/README.md](../../production_pipeline_instruction/0_Image_Text_Processing/README.md)

**Input Schema**: [../../production_pipeline_instruction/0_Image_Text_Processing/input_schema.json](../../production_pipeline_instruction/0_Image_Text_Processing/input_schema.json)

**Output Schema**: [../../production_pipeline_instruction/0_Image_Text_Processing/output_schema.json](../../production_pipeline_instruction/0_Image_Text_Processing/output_schema.json)

**Extension Code**: [../../chrome-extension/image-capture.js](../../chrome-extension/image-capture.js)

**Utilities**: [../utils/](../utils/)

## Status

Production Ready - Branch: `mrida-image/text_processing`
