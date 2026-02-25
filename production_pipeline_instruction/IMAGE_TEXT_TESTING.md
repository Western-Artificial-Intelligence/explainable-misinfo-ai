# Image/Text Processing - Integration & Testing Guide

## End-to-End Flow

### 1. User Interaction (Chrome Extension)

User plays TikTok/YouTube Shorts video → Clicks "Auto-Detect & Start" button in TruthLens popup

### 2. Audio Analysis Decision (Browser)

```
Extension analyzes audio using RMS
│
├─ Voice detected (energy > VOICE_THRESHOLD & ratio > 10%)
│  └─→ "Use audio transcription (Whisper)"
│
└─ No voice (energy below threshold or insufficient voice ratio)
   └─→ Immediately start image/frame capture
```

### 3. Frame Capture Loop (Browser - 3-5 seconds)

```
Every 300ms:
  ├─ Draw current video frame to canvas
  ├─ Calculate frame hash
  ├─ Check if time > 1.5s since last capture OR frame changed
  │  └─ If yes: Convert to JPEG blob, queue for OCR
  │
  └─ Run Tesseract.js on frame (async)
     └─ Extract text, add to TLX_IMAGE_STATE.extractedText

Also running in parallel:
  └─ Extract DOM text elements (captions, overlays, etc.)
     └─ Filter out UI controls, deduplicate
```

### 4. User Stops Capture (Browser)

Click "Stop & Analyze" button →

```
Extension triggers stopImageCapture():
  ├─ Clear frame capture interval
  ├─ Set isCapturing = false
  ├─ Package all data:
  │  ├─ All frame blobs (image/jpeg)
  │  ├─ OCR text from each frame
  │  ├─ Extracted DOM text
  │  └─ Audio analysis results
  └─→ Send multi-part form to backend

```

### 5. Backend Processing (Server)

**Endpoint**: `POST /api/image/analyze-claim`

**Processing**:

```python
1. Receive form data (request_id, claim_id, frames, text)

2. Extract text from images:
   for each frame in request:
      image = read_blob(frame)
      preprocessed = denoise + threshold (OpenCV)
      ocr_text = pytesseract.image_to_string()

3. Merge all text sources:
   all_texts = [ocr_from_frames] + [dom_text_from_extension]
   merged = merge_text_blocks(all_texts, deduplicate=True)

4. Clean text:
   cleaner = TextCleaner()
   result = cleaner.clean(merged_text)
   cleaned_text = result.cleaned

5. Analyze for misinformation:
   segments = split_into_sentences(cleaned_text)
   predictions = []
   for segment in segments[:5]:  # Max 5 segments
      pred = predict_text(segment)  # Use existing RoBERTa model
      predictions.append(pred)

6. Generate response:
   return {
     extracted_text: cleaned_text,
     text_statistics: word_count, sentence_count, etc,
     misinfo_predictions: [
       {text, prediction, confidence, explanation},
       ...
     ]
   }
```

**Response Status**:
- `200 OK` - Analysis complete
- `400 Bad Request` - Invalid form data
- `500 Internal Server Error` - Processing failed

### 6. Results Display (Popup UI)

Backend response → Extension processes → Popup displays:

```
✅ Image & Text Analysis Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Captured Content:
   • 5 frames analyzed
   • 7 text elements extracted
   • 234 total characters

🔍 Extracted Text:
   ┌─────────────────────────────────┐
   │ "Amazing deal! Click here now!  │
   │  Limited time offer..."          │
   └─────────────────────────────────┘

⚠️  Misinformation Detection:
   1. "Amazing deal" → FAKE (78% confidence)
      Similar to known deceptive marketing claims
      
   2. "Limited time offer" → MIXED (62% confidence)
      Could be legitimate or misleading
```

## Testing Checklist

### Unit Tests

**Audio Analysis**:
```python
# test_audio_utils.py
def test_silence_detection():
    # Silent audio (white noise)
    silent_audio = generate_silence(duration=2.0, sample_rate=16000)
    assert AudioAnalyzer.is_silent(silent_audio) == True
    
    # Voice audio
    voice_audio = generate_voice_sample(speech="Hello world", duration=2.0)
    assert AudioAnalyzer.is_silent(voice_audio) == False

def test_rms_calculation():
    # Generate known RMS value
    audio = np.sin(np.linspace(0, 10*np.pi, 16000)) * 0.1
    rms = AudioAnalyzer.calculate_rms(audio)
    assert 0.05 < rms < 0.15
```

**Text Cleaning**:
```python
# test_text_processor.py
def test_emoji_removal():
    text = "AMAZING 😱😱 OFFER!!!"
    result = TextCleaner.clean(text)
    assert "😱" not in result.cleaned
    assert "AMAZING" in result.cleaned
    assert result.is_valid == True

def test_url_removal():
    text = "Visit https://fake-scam.com for free money"
    result = TextCleaner.clean(text)
    assert "fake-scam.com" not in result.cleaned
    assert "free money" in result.cleaned

def test_repeated_char_removal():
    text = "Hellooooo woooorld!!!"
    result = TextCleaner.clean(text)
    assert result.cleaned == "Hello world!"
```

### Integration Tests

**Full Pipeline**:
```python
# test_integration.py
async def test_image_claim_analysis_end_to_end():
    # 1. Create test image with known text
    test_image = create_test_image("FAKE NEWS HERE")
    
    # 2. Create form data
    form_data = {
        'request_id': 'test_req_123',
        'claim_id': 'test_claim_456',
        'frame_count': 1,
        'captured_text': '["Additional context text"]',
        'audio_analysis': '{"has_voice": false, "is_silent": true}',
    }
    
    # 3. Call endpoint
    response = client.post(
        "/api/image/analyze-claim",
        data=form_data,
        files={'frame_0': test_image}
    )
    
    # 4. Verify response
    assert response.status_code == 200
    data = response.json()
    assert "FAKE NEWS" in data['extracted_text'] or "fake" in data['extracted_text'].lower()
    assert len(data['misinfo_predictions']) > 0

async def test_silence_detection_endpoint():
    request = {
        "request_id": "test_1",
        "audio_energy": 0.015,  # Below VOICE_THRESHOLD
        "duration_seconds": 5.0
    }
    
    response = client.post("/api/image/detect-silence", json=request)
    
    assert response.status_code == 200
    data = response.json()
    assert data['is_silent'] == True
    assert data['has_voice'] == False
```

### Manual Browser Testing

1. **Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Start backend
   cd api
   python -m uvicorn main:app --reload --port 8000
   ```

2. **Test Audio Detection**:
   - Open TikTok/YouTube Shorts with silent video
   - Click "Auto-Detect & Start" in TruthLens
   - Should detect "No voice" and start frame capture
   - Check console: `[TruthLens] Audio Analysis - Voice Ratio: 5.0%, Has Voice: false`

3. **Test Frame Capture**:
   - Should capture frames at 1.5s intervals
   - Check console for frame counts
   - Should show OCR progress

4. **Test Backend Processing**:
   - Stop capture after 5-10 seconds
   - Check server logs for analysis messages
   - Popup should show extracted text and predictions

5. **Test Text Cleaning**:
   - Capture video with emojis, hashtags, mentions
   - Verify they're removed in final output
   - Check predictions are based on cleaned text

## Common Issues & Troubleshooting

### Issue: "Tesseract.js takes too long to load"

**Cause**: CDN slow or blocked
**Solution**: 
```javascript
// Increase timeout in image-capture.js
setTimeout(() => {
  clearInterval(sampleInterval);
  // ... fallback
}, 5000);  // Increase from 2500ms to 5000ms
```

### Issue: "Backend returns 500 - pytesseract module error"

**Cause**: pytesseract or Tesseract-OCR system package not installed
**Solution**:
```bash
# Install pytesseract
pip install pytesseract

# Install Tesseract system package
# Ubuntu: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
# Windows: Download installer from GitHub
```

### Issue: "Audio analysis says 'voice detected' but it's silent video"

**Cause**: Thresholds too sensitive (background noise > VOICE_THRESHOLD)
**Solution**: Increase thresholds in `api/utils/audio_utils.py`
```python
VOICE_THRESHOLD_RMS = 0.07  # Increase from 0.05
VOICE_PRESENCE_RATIO = 0.15  # Increase from 0.10
```

### Issue: "Frame capturing too many frames (>15)"

**Cause**: Frame capture interval too frequent or visual changes detected
**Solution**: 
```javascript
// In image-capture.js, increase minimum interval
const MIN_CAPTURE_INTERVAL_MS = 2000;  // 2 seconds instead of 1.5
```

### Issue: "Text cleaning removes important text"

**Cause**: Cleaning regex too aggressive
**Solution**: Review `isCommonUIText()` and `isLikelyUIControl()` in extension
```javascript
// Add domain-specific filters
const userDefinedFilters = ['specific_false_positive'];
```

## Performance Metrics

**Expected Timings**:
- Audio analysis: 1-2 seconds
- Frame capture: 3-5 seconds (15 frames max)
- OCR per frame: 200-500ms (Tesseract.js)
- Text cleaning: <100ms
- Misinformation analysis: 300-800ms (5 segments)
- **Total E2E**: ~8-15 seconds

**Expected Payload**:
- 15 frames at 640x360 JPEG (65% quality): 1.5-3 MB
- Text data: <100 KB
- **Total request**: 2-4 MB

## Monitoring & Analytics

**Metrics to Track**:
1. Frame capture success rate
2. OCR confidence distribution
3. Text cleaning effectiveness
4. Backend API response times
5. Error rate (failed analyses)
6. User adoption (extension usage)

**Example Logging**:
```python
# In imageprocessing.py
logger.info(f"Image analysis: {request_id}")
logger.info(f"  Frames: {frame_count}")
logger.info(f"  Text sources: {len(all_text_sources)}")
logger.info(f"  Final text length: {len(final_text)}")
logger.info(f"  Predictions: {len(predictions)}")
logger.info(f"  Processing time: {elapsed_seconds:.2f}s")
```

## Security Checklist

- [ ] Input validation on all form fields
- [ ] File size limits enforced
- [ ] Rate limiting on image analysis endpoint
- [ ] CORS properly configured for deployment
- [ ] No sensitive data in logs
- [ ] Error messages don't expose system details
- [ ] Dependencies regularly updated
- [ ] OCR runs client-side (privacy)
