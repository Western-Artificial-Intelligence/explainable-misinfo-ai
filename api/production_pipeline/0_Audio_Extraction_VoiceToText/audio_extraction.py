"""
Audio Extraction and Voice-to-Text Pipeline

This module handles:
1. Audio extraction from video files (MP4, MOV, AVI, etc.)
2. Audio file handling (MP3, WAV, FLAC, etc.)
3. Video download from URLs (YouTube, TikTok, etc.)
4. Speech-to-text conversion using Whisper
5. Output structuring with metadata
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

import whisper
from pydub import AudioSegment
import numpy as np

logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL_SIZE = "base"  # Can be: tiny, base, small, medium, large
MAX_FILE_SIZE_MB = 500  # Max file size in MB
AUDIO_SAMPLE_RATE = 16000  # Whisper requires 16kHz
CHUNK_DURATION_MS = 30000  # 30 second chunks for processing
PIPELINE_VERSION = "0.1.0"

# Supported formats
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


class AudioExtractionError(Exception):
    """Base exception for audio extraction pipeline"""
    pass


class InvalidMediaSource(AudioExtractionError):
    """Raised when media source is invalid"""
    pass


class FileToolargeError(AudioExtractionError):
    """Raised when file exceeds size limit"""
    pass


class AudioExtractionFailed(AudioExtractionError):
    """Raised when audio extraction fails"""
    pass


class TranscriptionFailed(AudioExtractionError):
    """Raised when transcription fails"""
    pass


class UnsupportedFormat(AudioExtractionError):
    """Raised when media format is not supported"""
    pass


class AudioExtractionPipeline:
    """Pipeline for extracting audio and converting to text"""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        """
        Initialize the pipeline
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model (cached after first load)"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise TranscriptionFailed(f"Failed to load model: {e}")

    def process(
        self,
        request_id: str,
        claim_id: str,
        media_source: str,
        media_type: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process media and extract transcription
        
        Args:
            request_id: Unique request identifier
            claim_id: Unique claim identifier
            media_source: File path, URL, or media reference
            media_type: 'video', 'audio', or 'url'
            language: Optional language code (e.g., 'en')
            
        Returns:
            Dictionary with transcription and metadata
            
        Raises:
            InvalidMediaSource: If media source is invalid
            FileToolargeError: If file exceeds size limit
            AudioExtractionFailed: If audio extraction fails
            TranscriptionFailed: If transcription fails
        """
        try:
            # 1. Validate input
            self._validate_input(media_source, media_type)
            
            # 2. Prepare/download media
            audio_path = self._prepare_media(media_source, media_type)
            
            # 3. Preprocess audio
            audio_segments = self._preprocess_audio(audio_path)
            
            # 4. Transcribe
            transcription_result = self._transcribe_audio(
                audio_segments, language
            )
            
            # 5. Post-process
            output = self._build_output(
                request_id,
                claim_id,
                media_source,
                transcription_result,
            )
            
            # Cleanup
            self._cleanup(audio_path)
            
            return output
            
        except AudioExtractionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            raise TranscriptionFailed(f"Pipeline error: {e}")

    def _validate_input(self, media_source: str, media_type: str) -> None:
        """Validate input parameters"""
        if not media_source:
            raise InvalidMediaSource("Media source cannot be empty")
        
        if media_type not in ["video", "audio", "url"]:
            raise InvalidMediaSource(f"Invalid media_type: {media_type}")
        
        # For file paths, check if file exists and size
        if media_type in ["video", "audio"]:
            if not os.path.exists(media_source):
                raise InvalidMediaSource(f"File not found: {media_source}")
            
            file_size_mb = os.path.getsize(media_source) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise FileToolargeError(
                    f"File size ({file_size_mb}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)"
                )
            
            # Check file extension
            ext = Path(media_source).suffix.lower()
            if media_type == "video" and ext not in SUPPORTED_VIDEO_FORMATS:
                raise UnsupportedFormat(f"Unsupported video format: {ext}")
            elif media_type == "audio" and ext not in SUPPORTED_AUDIO_FORMATS:
                raise UnsupportedFormat(f"Unsupported audio format: {ext}")

    def _prepare_media(self, media_source: str, media_type: str) -> str:
        """
        Prepare media for processing
        
        Returns:
            Path to audio file
        """
        try:
            if media_type == "audio":
                # Audio file, use as-is
                return media_source
            
            elif media_type == "video":
                # Extract audio from video
                return self._extract_audio_from_video(media_source)
            
            elif media_type == "url":
                # Download and extract audio from URL
                return self._download_and_extract_audio(media_source)
                
        except Exception as e:
            logger.error(f"Failed to prepare media: {e}")
            raise AudioExtractionFailed(f"Media preparation failed: {e}")

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file using ffmpeg"""
        try:
            import subprocess
            
            # Create temporary file for audio
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"extracted_audio_{datetime.now().timestamp()}.wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:a", "9",  # Quality
                "-n",  # Don't overwrite
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise AudioExtractionFailed(f"ffmpeg error: {result.stderr}")
            
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except Exception as e:
            raise AudioExtractionFailed(f"Audio extraction failed: {e}")

    def _download_and_extract_audio(self, url: str) -> str:
        """Download video from URL and extract audio"""
        try:
            import subprocess
            
            temp_dir = tempfile.gettempdir()
            temp_video = os.path.join(temp_dir, f"downloaded_{datetime.now().timestamp()}.mp4")
            audio_path = os.path.join(temp_dir, f"extracted_audio_{datetime.now().timestamp()}.wav")
            
            # Download video using yt-dlp
            logger.info(f"Downloading video from: {url}")
            dl_cmd = [
                "yt-dlp",
                "-f", "best",
                "-o", temp_video,
                url
            ]
            
            result = subprocess.run(dl_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise AudioExtractionFailed(f"Download failed: {result.stderr}")
            
            # Extract audio from downloaded video
            ext_cmd = [
                "ffmpeg",
                "-i", temp_video,
                "-q:a", "9",
                "-n",
                audio_path
            ]
            
            result = subprocess.run(ext_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise AudioExtractionFailed(f"Audio extraction failed: {result.stderr}")
            
            # Cleanup downloaded video
            if os.path.exists(temp_video):
                os.remove(temp_video)
            
            return audio_path
            
        except Exception as e:
            raise AudioExtractionFailed(f"Download and extraction failed: {e}")

    def _preprocess_audio(self, audio_path: str) -> List[np.ndarray]:
        """
        Preprocess audio: normalize sample rate, chunk, normalize levels
        
        Returns:
            List of audio segments as numpy arrays
        """
        try:
            logger.info(f"Preprocessing audio: {audio_path}")
            
            # Load audio using pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Normalize to mono and resample to 16kHz
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            if audio.frame_rate != AUDIO_SAMPLE_RATE:
                audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE)
            
            # Normalize audio levels
            max_amplitude = audio.max
            if max_amplitude > 0:
                normalized = audio.apply_gain(-max_amplitude + (-3.0103))  # -3dB headroom
            else:
                normalized = audio
            
            # Convert to numpy array
            samples = np.array(normalized.get_array_of_samples())
            
            # Chunk into 30-second segments
            samples_per_chunk = AUDIO_SAMPLE_RATE * (CHUNK_DURATION_MS // 1000)
            segments = [
                samples[i:i + samples_per_chunk]
                for i in range(0, len(samples), int(samples_per_chunk))
            ]
            
            logger.info(f"Audio split into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise AudioExtractionFailed(f"Audio preprocessing failed: {e}")

    def _transcribe_audio(
        self,
        audio_segments: List[np.ndarray],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio segments using Whisper
        
        Returns:
            Dictionary with transcription segments and metadata
        """
        try:
            logger.info(f"Starting transcription of {len(audio_segments)} segments")
            
            segments = []
            full_text_parts = []
            language_detected = language or "en"
            total_duration = 0
            
            for idx, segment in enumerate(audio_segments):
                logger.info(f"Transcribing segment {idx + 1}/{len(audio_segments)}")
                
                # Normalize segment to [-1, 1] range (Whisper requirement)
                segment_float = segment.astype(np.float32) / 32768.0
                
                # Transcribe
                result = self.model.transcribe(
                    segment_float,
                    language=language,
                    fp16=False  # Set to True if GPU available
                )
                
                # Detect language from first segment
                if idx == 0:
                    language_detected = result.get("language", "en")
                
                # Extract segments and timing
                for whisper_segment in result["segments"]:
                    text = whisper_segment.get("text", "").strip()
                    if text:
                        full_text_parts.append(text)
                        
                        segments.append({
                            "text": text,
                            "start_time": whisper_segment["start"] + (idx * CHUNK_DURATION_MS / 1000),
                            "end_time": whisper_segment["end"] + (idx * CHUNK_DURATION_MS / 1000),
                            "confidence": 0.95  # Whisper doesn't provide confidence; default to high
                        })
                
                total_duration += len(segment) / AUDIO_SAMPLE_RATE
            
            return {
                "full_text": " ".join(full_text_parts),
                "segments": segments,
                "language_detected": language_detected,
                "duration_seconds": total_duration
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise TranscriptionFailed(f"Transcription failed: {e}")

    def _build_output(
        self,
        request_id: str,
        claim_id: str,
        media_source: str,
        transcription_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final output object"""
        return {
            "request_id": request_id,
            "claim_id": claim_id,
            "media_source": media_source,
            "transcription": transcription_result,
            "meta": {
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "model_used": f"whisper-{self.model_size}",
                "version": PIPELINE_VERSION
            }
        }

    def _cleanup(self, audio_path: str) -> None:
        """Clean up temporary files"""
        try:
            if audio_path and os.path.exists(audio_path):
                # Check if it's in temp directory (to avoid deleting user files)
                if audio_path.startswith(tempfile.gettempdir()):
                    os.remove(audio_path)
                    logger.info(f"Cleaned up: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {audio_path}: {e}")
