# 0_Audio_Extraction_VoiceToText Pipeline

## Overview
This pipeline step handles extracting audio from various media sources (videos, MP3 files, etc.) and converting speech to text using voice recognition models.

### Supported Input Formats
- **Video Files**: MP4, MOV, AVI, WebM, etc. (via ffmpeg)
- **Audio Files**: MP3, WAV, FLAC, OGG, M4A
- **Video URLs**: TikTok, YouTube, etc. (download & extract audio)

### Output
Extracted text transcriptions with metadata (timestamps, confidence scores, speaker info)

## Use Cases
- Process TikTok videos for misinformation detection
- Extract audio from various video sources
- Batch process multiple media files
- Real-time audio stream processing (future)

## Dependencies
- `whisper` - OpenAI's speech-to-text model
- `ffmpeg` - Audio/video processing
- `pydub` - Audio manipulation
- `yt-dlp` - Download videos from URLs

## Pipeline Steps
1. **Input Validation** - Check file type and size
2. **Audio Extraction** - Extract audio from video/file
3. **Audio Preprocessing** - Normalize, chunk, resample
4. **Transcription** - Convert speech to text using Whisper
5. **Post-processing** - Clean, structure output with metadata
6. **Output** - Return transcribed text to downstream pipeline
