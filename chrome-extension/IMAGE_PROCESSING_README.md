# Image/Text Processing - Chrome Extension

## Overview

This directory contains the **client-side components** for the Image/Text Processing pipeline (Stage 0, alternative to audio extraction for silent videos).

**Main File**: `image-capture.js` (850+ lines)

## Pipeline Integration

When a user plays a TikTok/YouTube Shorts video:

1. **Audio Analysis Decision**
   - `image-capture.js` analyzes video audio using Web Audio API
   - Calculates RMS energy (frequency domain)
   - Measures voice ratio (voice vs silence)

2. **Decision Logic**
   ```
   Voice Ratio > 10% AND Energy > 0.05 RMS
   ├─ YES: Continue with existing Whisper audio extraction
   └─ NO:  Start image/text processing (this extension)
   ```

3. **Frame Capture**
   - Frames captured at 1.5s intervals OR on visual change
   - Maximum 15 frames per session
   - Frames converted to JPEG (65% quality)

4. **Client-Side OCR**
   - Tesseract.js v5.0.2 (CDN)
   - Extracts text from each frame (privacy-preserving)

5. **Text Extraction**
   - DOM text extraction (captions, overlays)
   - UI filtering (removes "Like", "Share", "Follow", etc.)

6. **Backend Submission**
   - Multipart form submission to `/api/image/analyze-claim`
   - Includes frames, OCR text, DOM text, audio analysis

## Key Files

- `image-capture.js` - Main image capture and analysis module
- `popup.js` - Updated with image capture UI handlers
- `popup.html` - Updated with image capture buttons
- `manifest.json` - Updated to register image-capture.js

## Key Functions

### Audio Analysis
```javascript
analyzeAudioLevelsAdvanced()
  Purpose: RMS-based silence detection
  Returns: { voiceRatio, avgEnergy, hasVoice }
  
shouldUseImageCapture()
  Purpose: Decision logic for image vs audio pipeline
  Returns: boolean
```

### Frame Capture
```javascript
startImageCapture()
  Purpose: Initialize capture session
  
startFrameCaptureLoop()
  Purpose: Smart frame sampling with hash-based change detection
  Captures: Max 15 frames, 1.5s minimum interval
  
calculateFrameHash()
  Purpose: Detect visual changes (sample every 1000th pixel)
  Returns: Hash string for comparison
```

### Text Extraction
```javascript
extractOnScreenText()
  Purpose: Extract text from page DOM
  Filters: UI controls, duplicates
  Returns: Array of text strings

performOCROnFrame()
  Purpose: Run Tesseract.js on frame
  Returns: Extracted text with confidence
```

### Backend Communication
```javascript
sendImageAndTextForAnalysis()
  Purpose: Submit data to backend
  Endpoint: POST /api/image/analyze-claim
  Format: multipart/form-data with frames + metadata
```

## Configuration

### Thresholds

**Audio Decision** (in `image-capture.js`):
```javascript
const SILENCE_THRESHOLD = 0.02;      // ~-34 dB
const VOICE_THRESHOLD = 0.05;        // ~-26 dB  
const MIN_VOICE_RATIO = 0.10;        // 10%
const MIN_VOICE_DURATION = 0.5;      // 0.5 seconds
```

**Frame Capture**:
```javascript
const MIN_CAPTURE_INTERVAL_MS = 1500; // 1.5 seconds
const MAX_FRAMES = 15;                // Frame limit
const JPEG_QUALITY = 0.65;            // Quality (0-1)
```

## Testing

### Manual Testing
1. Open TikTok/YouTube Shorts with silent video
2. Open Developer Tools (F12)
3. Look for `[TruthLens]` log messages:
   ```
   [TruthLens] Audio Analysis - Avg RMS: 0.032, Voice Ratio: 5.0%, Has Voice: false
   [TruthLens] Starting image capture: req_img_...
   [TruthLens] Frame 1: 234 chars from OCR
   ```

### Console Logs
```javascript
[TruthLens] Audio Analysis - Avg RMS: X.XXX, Voice Ratio: Y.Y%, Has Voice: true/false
[TruthLens] Starting image capture: [request_id]
[TruthLens] Frame N: XXX chars from OCR
[TruthLens] Extracted Y DOM text elements
[TruthLens] Sending analysis request to backend...
[TruthLens] Analysis complete! Status: success/error
```

## Troubleshooting

### Issue: "Tesseract.js takes too long to load"
**Solution**: Check browser cache, verify CDN connectivity
```javascript
// In image-capture.js
setTimeout(() => { /* fallback */ }, 5000);  // Increase timeout
```

### Issue: "Audio analysis never triggers image capture"
**Solution**: Check thresholds, verify Web Audio API access
```javascript
// Temporarily lower thresholds for testing
VOICE_THRESHOLD = 0.08;      // from 0.05
MIN_VOICE_RATIO = 0.15;      // from 0.10
```

### Issue: "Frame capture not working"
**Solution**: Check console logs, verify canvas access
```javascript
// Enable debug logging
console.log('[TruthLens] Frame hash:', calculateFrameHash());
```

## Production Pipeline Reference

**Full Documentation**: [../0_Image_Text_Processing/README.md](../../production_pipeline_instruction/0_Image_Text_Processing/README.md)

**Input Schema**: [../0_Image_Text_Processing/input_schema.json](../../production_pipeline_instruction/0_Image_Text_Processing/input_schema.json)

**Backend Route**: [../api/routes/imageprocessing.py](../api/routes/imageprocessing.py)

**Utilities**: [../api/utils/](../api/utils/)

## Status

Production Ready - Branch: `mrida-image/text_processing`
