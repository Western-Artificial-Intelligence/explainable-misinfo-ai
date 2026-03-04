# Backend Integration: Audio Pipeline & Pipeline Runner

## Overview

The backend now has a unified runner that chains together the entire misinformation detection pipeline:

```
Media Input (Audio/Video/URL)
    ↓
[0] Audio Extraction & Transcription (Whisper)
    ↓
[1] Ingest & Normalize Claim
    ↓
[2] RoBERTa Inference (Classification)
    ↓
[3] Routing Policy (Decide next steps)
    ↓
[4-10] Conditional pipeline steps
    ↓
Final Result: REAL/FAKE + Evidence
```

---

## New Modules

### 1. `api/production_pipeline/pipeline_runner.py`

**Purpose:** Runs entire pipeline flow

**Main Class:** `PipelineRunner`

**Key Methods:**
- `run()` - Main entry point
- `_stage_0_audio_extraction()` - Audio → Text
- `_stage_1_ingest_claim()` - Text normalization
- `_stage_2_roberta_inference()` - Classification
- `_stage_3_routing_policy()` - Routing logic

**Features:**
- Handles both audio and text inputs
- Automatic request/claim ID generation
- Execution logging
- Error handling at each stage
- Returns complete pipeline trace

**Example Usage:**
```python
runner = PipelineRunner()

# Audio input (TikTok video)
result = runner.run(
    media_source="/path/to/video.mp4",
    media_type="video",
    request_id="req_123"
)

# Text input (existing flow)
result = runner.run(
    text_input="The moon is made of cheese",
    request_id="req_124"
)
```

---

### 2. `api/routes/pipeline.py`

**Purpose:** REST API endpoints for pipeline processing

**Endpoints:**

#### `POST /api/pipeline/process-media`
Process video/audio file
```bash
curl -X POST http://localhost:8000/api/pipeline/process-media \
  -H "Content-Type: application/json" \
  -d '{
    "media_source": "/path/to/video.mp4",
    "media_type": "video",
    "language": "en",
    "request_id": "req_123"
  }'
```

#### `POST /api/pipeline/process-text`
Process text directly (skip audio extraction)
```bash
curl -X POST http://localhost:8000/api/pipeline/process-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The vaccine causes autism",
    "request_id": "req_124"
  }'
```

#### `POST /api/pipeline/process-direct` ⭐ FOR CHROME EXTENSION
Process audio file uploaded by extension
```bash
curl -X POST http://localhost:8000/api/pipeline/process-direct \
  -F "request_id=req_125" \
  -F "claim_id=claim_125" \
  -F "audio_file=@/path/to/audio.wav" \
  -F "language=en"
```

#### `GET /api/pipeline/status/{request_id}`
Check processing status
```bash
curl http://localhost:8000/api/pipeline/status/req_123
```

#### `GET /api/pipeline/health`
Health check
```bash
curl http://localhost:8000/api/pipeline/health
```

---

## Integration Architecture

### Flow 1: Chrome Extension → Backend

```
User scrolls TikTok
    ↓
[Chrome Extension] detects video
    ↓
[Audio Capture] records audio
    ↓
[POST /api/pipeline/process-direct] sends audio blob
    ↓
[Backend Orchestrator] runs full pipeline
    ↓
[Response] Returns: transcription + verdict
    ↓
[Chrome Extension] displays results on video
```

**Extension calls:** `POST /api/pipeline/process-direct`

### Flow 2: Direct API Usage

```
Text/Media Input
    ↓
[API Client] calls /process-text or /process-media
    ↓
[Orchestrator] coordinates all steps
    ↓
[Response] Complete analysis result
```

### Flow 3: Batch Processing

```
Multiple Claims
    ↓
[Loop] Call /process-text for each
    ↓
[Store] Results in database
    ↓
[Report] Generate statistics
```

---

## Data Flow

### Input
```json
{
  "media_source": "/path/or/url",
  "media_type": "video|audio|url",
  "language": "en",
  "request_id": "req_123"
}
```

### Pipeline Stages Output

**Stage 0: Audio Extraction**
```json
{
  "request_id": "req_123",
  "claim_id": "claim_xyz",
  "transcription": {
    "full_text": "The earth is flat",
    "segments": [...],
    "language_detected": "en",
    "duration_seconds": 45.2
  }
}
```

**Stage 1: Ingest Claim**
```json
{
  "request_id": "req_123",
  "claim_id": "claim_xyz",
  "user_claim": "The earth is flat",
  "normalized_claim": "the earth is flat",
  "meta": {...}
}
```

**Stage 2: RoBERTa Inference**
```json
{
  "request_id": "req_123",
  "claim_id": "claim_xyz",
  "classification": {
    "prediction": "FAKE",
    "confidence": 0.95,
    "logits": {
      "real": 0.02,
      "fake": 0.95,
      "requires_verification": 0.03
    }
  }
}
```

**Stage 3: Routing Policy**
```json
{
  "request_id": "req_123",
  "claim_id": "claim_xyz",
  "routing_decision": "FULL_ANALYSIS",
  "prediction": "FAKE",
  "confidence": 0.95
}
```

### Final Output
```json
{
  "request_id": "req_123",
  "claim_id": "claim_xyz",
  "final_prediction": "FAKE",
  "confidence": 0.95,
  "stages": {
    "0_audio_extraction": {...},
    "1_ingest_claim": {...},
    "2_roberta_inference": {...},
    "3_routing_policy": {...}
  },
  "meta": {
    "total_stages_executed": 4,
    "pipeline_completed_at": "2026-02-22T15:30:45Z",
    "execution_log": [...]
  }
}
```

---

## Key Features

### ✅ Auto-Detection & Auto-Capture (Backend Ready)

The orchestrator automatically:
- Detects media type (video/audio/URL)
- Extracts audio using ffmpeg
- Transcribes using Whisper
- All without user interaction

### ✅ Request Tracking

Every request has:
- `request_id`: Unique identifier for tracking
- `claim_id`: Identifier for the claim being analyzed
- Execution log: What ran and when
- Stage outputs: Results from each step

### ✅ Error Handling

- Validates input at each stage
- Returns helpful error messages
- Logs failures for debugging
- Gracefully handles missing dependencies

### ✅ Extensible Design

Supports adding new stages:
```python
def _stage_4_query_building(self, ...):
    # New stage implementation
    result = query_builder.build(...)
    self._log_stage("4_query_building", result)
    return result
```

---

## Configuration

### Environment Variables

```bash
# Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL_SIZE=base

# Backend URL (for extension)
BACKEND_URL=http://localhost:8000

# Max file size for audio (MB)
MAX_FILE_SIZE_MB=500
```

### Audio Preprocessing Settings

In `audio_extraction.py`:
```python
WHISPER_MODEL_SIZE = "base"
AUDIO_SAMPLE_RATE = 16000  # 16kHz
CHUNK_DURATION_MS = 30000   # 30 second chunks
```

Modify as needed for:
- Better accuracy: use `medium` or `large` model
- Faster processing: use `tiny` or `base` model
- Different audio quality: adjust sample rate

---

## Testing the Pipeline

### 1. Test Audio Extraction Only
```bash
curl -X POST http://localhost:8000/api/audio/transcribe-file \
  -F "request_id=req_test" \
  -F "claim_id=claim_test" \
  -F "file=@test_video.mp4"
```

### 2. Test Text Analysis Only
```bash
curl -X POST http://localhost:8000/api/pipeline/process-text \
  -H "Content-Type: application/json" \
  -d '{"text": "The vaccines are safe"}'
```

### 3. Test Full Pipeline (Audio → Analysis)
```bash
curl -X POST http://localhost:8000/api/pipeline/process-direct \
  -F "request_id=req_full" \
  -F "claim_id=claim_full" \
  -F "audio_file=@test_video.mp4"
```

### 4. Check Health
```bash
curl http://localhost:8000/api/pipeline/health
```

---

## Performance Considerations

| Operation | Typical Time |
|-----------|--------------|
| Video → Audio extraction | 2-5 seconds |
| Audio → Transcription (Whisper base) | 10-30 seconds |
| Text → RoBERTa inference | 0.5-2 seconds |
| Total (end-to-end) | 15-40 seconds |

**Optimization Tips:**
1. Use smaller Whisper model (`tiny`) for speed
2. Cache model after first load (already done)
3. Batch process multiple claims
4. Use GPU if available (modify audio_extraction.py: `fp16=True`)

---

## Next Steps

### For Backend
- [ ] Connect RoBERTa model (currently placeholder)
- [ ] Implement stages 4-10
- [ ] Add database for storing results
- [ ] Implement async job queue (Celery/Redis)
- [ ] Add rate limiting
- [ ] Add authentication for production

### For Frontend Team
- [ ] Auto-detect video play events
- [ ] Call `/api/pipeline/process-direct` endpoint
- [ ] Display results in floating widget
- [ ] Handle errors gracefully

### For DevOps
- [ ] Container configuration (Docker)
- [ ] Environment setup
- [ ] Database setup (if needed)
- [ ] GPU support (if available)
- [ ] Deployment to production

---

## API Documentation

Full OpenAPI docs available at:
```
http://localhost:8000/docs  (Swagger UI)
http://localhost:8000/redoc (ReDoc)
```

Endpoints appear under:
- `/api/audio` - Audio transcription
- `/api/pipeline` - Full pipeline processing
- `/api/predict` - Text prediction (existing)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│         Chrome Extension                             │
│  (Detects video, captures audio, uploads file)     │
└────────────────────┬────────────────────────────────┘
                     │ POST /api/pipeline/process-direct
                     ↓
┌─────────────────────────────────────────────────────┐
│         FastAPI Backend (main.py)                   │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         ↓                        ↓
    ┌─────────────┐        ┌──────────────┐
    │ /api/audio  │        │ /api/pipeline│
    │  routes     │        │  routes      │
    └─────────────┘        └──────────────┘
         │                        │
         ↓                        ↓
    ┌─────────────┐        ┌──────────────┐
    │ Audio       │        │ Orchestrator │
    │ Extraction  │        │              │
    │ (Whisper)   │        └──────────────┘
    └─────────────┘                │
                          ┌─────────┴──────────┬──────────┐
                          ↓                    ↓          ↓
                      ┌────────┐        ┌─────────┐   ┌──────┐
                      │ Stage1 │        │ Stage2  │   │Stage3│
                      │ Ingest │        │RoBERTa  │   │Route │
                      └────────┘        └─────────┘   └──────┘
                          
                          Response → Chrome Extension
                                → Display on Video
```

---

## Questions?

See the production pipeline instruction folders for detailed stage documentation:
```
production_pipeline_instruction/
├── 1_Ingest_claim/
├── 2_RoBERTa_inference/
├── 3_Routing_policy/
└── ... (stages 4-10)
```
