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

from api.production_pipeline.0_Audio_Extraction_VoiceToText import (
    AudioExtractionPipeline,
    AudioExtractionError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/audio", tags=["audio"])

# Initialize pipeline (could be cached globally)
pipeline = None


def get_pipeline():
    """Get or initialize the audio extraction pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = AudioExtractionPipeline(model_size="base")
    return pipeline


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
    """
    Transcribe an uploaded audio/video file
    
    Args:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        file: Uploaded audio or video file
        language: Optional language code
        
    Returns:
        Transcription result with metadata
    """
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Determine media type from file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}:
            media_type = "video"
        elif ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
            media_type = "audio"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {ext}"
            )
        
        # Process
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
