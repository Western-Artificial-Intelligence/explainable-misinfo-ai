# Image/Text Processing - Utility Modules

## Overview

These utility modules support the Image/Text Processing pipeline (Stage 0).

## Modules

### 1. text_processor.py (500+ lines)

**Purpose**: Comprehensive text cleaning for OCR output and extracted text

**Main Class**: `TextCleaner`

**Cleaning Operations** (in order):
1. Unicode normalization (NFC)
2. Emoji removal (regex pattern)
3. URL removal (http/https)
4. @mention removal
5. #hashtag removal/conversion
6. Repeated character removal
7. Whitespace normalization
8. Text validation

**Key Methods**:
```python
TextCleaner.clean(text: str) -> CleaningResult
  Purpose: Clean text and return result object
  Returns: CleaningResult with cleaned_text, is_valid, statistics
  
merge_text_blocks(texts: List[str]) -> str
  Purpose: Deduplicate and merge multiple text sources
  Returns: Single cleaned text string
  
extract_sentences(text: str) -> List[str]
  Purpose: Segment text by sentences
  Returns: List of sentence strings
  
get_text_statistics(text: str) -> TextStatistics
  Purpose: Calculate text metrics
  Returns: Statistics object with counts
  
is_valid_text(text: str) -> bool
  Purpose: Validate text quality
  Checks: Min length, alphanumeric ratio, repeated chars
  Returns: boolean
```

**Configuration**:
```python
MIN_TEXT_LENGTH = 3
MAX_REPEATED_CHAR_RATIO = 0.3    # 30%
MIN_CONFIDENCE_THRESHOLD = 0.5   # 50%
```

**Example**:
```python
cleaner = TextCleaner()
raw_text = "AMAZING [emoji] OFFER!!! Visit https://fake.com @user #BUY_NOW"
result = cleaner.clean(raw_text)
# result.cleaned = "AMAZING OFFER Visit fake.com user BUY NOW"
# result.is_valid = True
```

### 2. audio_utils.py (400+ lines)

**Purpose**: Audio analysis for voice/silence detection

**Main Class**: `AudioAnalyzer`

**Key Methods**:
```python
AudioAnalyzer.calculate_rms(audio_data: np.ndarray) -> float
  Purpose: Calculate RMS energy from audio samples
  Input: Audio samples as numpy array
  Returns: RMS value (float)
  
AudioAnalyzer.detect_voice_frames(audio_data, sr=16000)
  Purpose: Detect which frames contain voice
  Returns: Array of frame-level voice indicators
  
AudioAnalyzer.is_silent(audio_data, sr=16000) -> bool
  Purpose: Determine if audio is silent/no voice
  Returns: boolean
  
AudioAnalyzer.analyze_audio(audio_data, sr=16000) -> AudioAnalysis
  Purpose: Comprehensive audio analysis
  Returns: AudioAnalysis object with all metrics
  
AudioAnalyzer.audio_bytes_to_numpy(audio_bytes) -> np.ndarray
  Purpose: Convert audio bytes to numpy format
  Returns: numpy array
  
AudioAnalyzer.get_rms_in_db(audio_data) -> float
  Purpose: Calculate RMS in decibels
  Returns: dB value (float)
```

**Configuration**:
```python
SILENCE_THRESHOLD_RMS = 0.02      # ~-34 dB (very quiet)
VOICE_THRESHOLD_RMS = 0.05        # ~-26 dB (conversational)
VOICE_PRESENCE_RATIO = 0.1        # 10% voice minimum
MIN_VOICE_DURATION_SEC = 0.5      # 0.5 second minimum
```

**Example**:
```python
analyzer = AudioAnalyzer()
analysis = analyzer.analyze_audio(audio_data, sr=16000)
# analysis.is_silent -> True/False
# analysis.voice_ratio -> 0.05 (5%)
# analysis.avg_energy -> 0.032 (RMS)
```

## Integration Points

### Text Processing
**Used By**: `imageprocessing.py` route handler
```python
from utils.text_processor import TextCleaner

cleaner = TextCleaner()
result = cleaner.clean(extracted_text)
if result.is_valid:
    proceed_with_analysis(result.cleaned)
```

### Audio Analysis
**Used By**: `imageprocessing.py` route handler
```python
from utils.audio_utils import AudioAnalyzer

analyzer = AudioAnalyzer()
analysis = analyzer.analyze_audio(audio_bytes)
if analysis.is_silent:
    use_image_processing()
else:
    use_audio_transcription()
```

## Dependencies

**text_processor.py**:
- re (built-in)
- unicodedata (built-in)
- string (built-in)
- logging (built-in)

**audio_utils.py**:
- numpy >= 1.20.0
- scipy >= 1.7.0
- librosa >= 10.0 (optional)
- logging (built-in)

## Performance

**Text Cleaning**:
- Speed: 10-50ms per 1000 characters
- Memory: Minimal (in-place processing)

**Audio Analysis**:
- Speed: 100-300ms for 10 second audio
- Memory: ~10MB for 16kHz audio

## Testing

### Unit Tests

**Text Processor**:
```bash
pytest api/tests/test_text_processor.py::test_emoji_removal -v
pytest api/tests/test_text_processor.py::test_url_removal -v
pytest api/tests/test_text_processor.py::test_repeated_char_removal -v
```

**Audio Utils**:
```bash
pytest api/tests/test_audio_utils.py::test_silence_detection -v
pytest api/tests/test_audio_utils.py::test_rms_calculation -v
```

### Example Tests

**Text Cleaning**:
```python
def test_emoji_removal():
    text = "Amazing [emoji] offer!!!"
    result = TextCleaner().clean(text)
    assert "[emoji]" not in result.cleaned
    assert result.is_valid == True
```

**Audio Analysis**:
```python
def test_silence_detection():
    # Silent audio (white noise)
    silent = np.random.normal(0, 0.01, 16000)
    assert AudioAnalyzer.is_silent(silent) == True
    
    # Voice audio (sine wave)
    voice = np.sin(2*np.pi*440*np.linspace(0,1,16000)) * 0.1
    assert AudioAnalyzer.is_silent(voice) == False
```

## Troubleshooting

### Issue: "TextCleaner removes too much text"
**Solution**: Adjust validation thresholds
```python
cleaner = TextCleaner()
cleaner.min_text_length = 2  # from 3
cleaner.max_repeated_char_ratio = 0.4  # from 0.3
```

### Issue: "AudioAnalyzer detects voice in silent video"
**Solution**: Increase thresholds
```python
analyzer = AudioAnalyzer()
analyzer.voice_threshold_rms = 0.07  # from 0.05
analyzer.voice_presence_ratio = 0.15  # from 0.10
```

### Issue: "OCR confidence filtering removes valid text"
**Solution**: Lower confidence threshold
```python
cleaner = TextCleaner()
cleaner.min_confidence_threshold = 0.4  # from 0.5
```

## Production Pipeline Reference

**Full Documentation**: [../../production_pipeline_instruction/0_Image_Text_Processing/README.md](../../production_pipeline_instruction/0_Image_Text_Processing/README.md)

**Route Handler**: [../routes/imageprocessing.py](../routes/imageprocessing.py)

**Extension Code**: [../../chrome-extension/image-capture.js](../../chrome-extension/image-capture.js)

## Status

Production Ready - Branch: `mrida-image/text_processing`
