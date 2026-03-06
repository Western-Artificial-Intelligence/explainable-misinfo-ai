"""
API route handler for audio extraction and voice-to-text processing

Endpoints:
    POST /api/audio/transcribe - Transcribe audio or video file
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os
import tempfile
import json
from datetime import datetime
import uuid

import importlib

_audio_mod = importlib.import_module("api.production_pipeline.0_Audio_Extraction_VoiceToText")
AudioExtractionPipeline = _audio_mod.AudioExtractionPipeline
AudioExtractionError = _audio_mod.AudioExtractionError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/audio", tags=["audio"])

# Initialize pipeline (could be cached globally)
pipeline = None
whisper_model = None


def get_pipeline():
    """Get or initialize the audio extraction pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = AudioExtractionPipeline(model_size="base")
    return pipeline


def get_whisper_model():
    """Lazy-load a direct Whisper model for short audio chunks.

    This bypasses the heavier production_pipeline for small uploads like
    live transcript chunks and standalone MP3 tests.
    """
    global whisper_model
    if whisper_model is None:
        import whisper

        logger.info("Loading direct Whisper model for /transcribe-file")
        whisper_model = whisper.load_model("base")
    return whisper_model


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    media_source: str = Field(..., description="File path, URL, or media reference")
    media_type: str = Field(..., description="'video', 'audio', or 'url'")
    language: Optional[str] = Field(None, description="ISO 639-1 language code (e.g., 'en')")
    model_size: Optional[str] = Field(
        "base",
        description="Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'"
    )


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    request_id: str
    claim_id: str
    media_source: str
    transcription: dict
    meta: dict


class SegmentInfo(BaseModel):
    """Single transcription segment"""
    text: str
    start_time: float
    end_time: float
    confidence: float


class TranscriptionDetail(BaseModel):
    """Detailed transcription info"""
    full_text: str
    segments: List[SegmentInfo]
    language_detected: str
    duration_seconds: float


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_media(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    media_source: str = Form(...),
    media_type: str = Form(...),
    language: Optional[str] = Form(None),
    model_size: Optional[str] = Form("base"),
):
    """
    Transcribe audio or video file
    
    Args:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        media_source: File path, URL, or media reference
        media_type: 'video', 'audio', or 'url'
        language: Optional language code
        model_size: Optional model size
        
    Returns:
        Transcription result with metadata
        
    Raises:
        HTTPException: If transcription fails
    """
    try:
        pipeline = get_pipeline()
        
        logger.info(f"Processing transcription request: {request_id}")
        
        result = pipeline.process(
            request_id=request_id,
            claim_id=claim_id,
            media_source=media_source,
            media_type=media_type,
            language=language,
        )
        
        logger.info(f"Transcription completed: {request_id}")
        return result
        
    except AudioExtractionError as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in transcription: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.post("/transcribe-file")
async def transcribe_file(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe an uploaded audio/video file.

    For audio files (mp3, wav, etc.) this uses a lightweight direct Whisper
    model to improve robustness for short clips and live chunks, without
    touching the production_pipeline. Video files still go through the
    production pipeline.
    """
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        original_name = file.filename or "upload"
        ext = os.path.splitext(original_name)[1].lower()
        safe_ext = ext if ext else ".bin"
        # Use a unique temp path per request to avoid cross-request overwrites
        temp_name = f"tlx_{request_id}_{uuid.uuid4().hex}{safe_ext}"
        temp_path = os.path.join(temp_dir, temp_name)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Determine media type from file extension
        if ext in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}:
            media_type = "video"
        elif ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
            media_type = "audio"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {ext}"
            )
        
        # Fast path: small audio files -> direct Whisper (more tolerant of short clips)
        if media_type == "audio":
            model = get_whisper_model()
            # Let Whisper handle decoding/resampling from the file path directly
            result = model.transcribe(temp_path, language=language, fp16=False)

            full_text = (result.get("text") or "").strip()
            segments_payload = []
            for seg in result.get("segments", []) or []:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                segments_payload.append(
                    {
                        "text": text,
                        "start_time": float(seg.get("start", 0.0)),
                        "end_time": float(seg.get("end", 0.0)),
                        "confidence": 0.95,
                    }
                )

            duration_sec = float(result.get("duration", 0.0) or 0.0)
            language_detected = str(result.get("language", language or "en"))

            transcription = {
                "full_text": full_text,
                "segments": segments_payload,
                "language_detected": language_detected,
                "duration_seconds": duration_sec,
            }

            logger.info("Direct Whisper transcription completed: %s (len=%d)", request_id, len(full_text))
            return {
                "request_id": request_id,
                "claim_id": claim_id,
                "media_source": temp_path,
                "transcription": transcription,
                "meta": {
                    "processed_at": datetime.utcnow().isoformat() + "Z",
                    "model_used": "whisper-base",
                    "version": "0.1.0",
                },
            }

        # Video or other supported media: fall back to production pipeline
        pipeline = get_pipeline()
        result = pipeline.process(
            request_id=request_id,
            claim_id=claim_id,
            media_source=temp_path,
            media_type=media_type,
            language=language,
        )
        
        logger.info(f"File transcription completed: {request_id}")
        return result
        
    except AudioExtractionError as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@router.get("/status/{request_id}")
async def get_transcription_status(request_id: str):
    """
    Get status of a transcription request
    
    Note: This is a placeholder. Implement with your job queue system.
    """
    return {
        "request_id": request_id,
        "status": "completed",
        "message": "Implement with your job queue system (e.g., Celery, RQ)"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_pipeline()
        return {
            "status": "healthy",
            "service": "audio-extraction",
            "model_loaded": pipeline.model is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "audio-extraction",
            "error": str(e)
        }


