"""
Unified Production Pipeline API Routes

Endpoints:
    POST /api/pipeline/process-media     - Process audio/video → full pipeline
    POST /api/pipeline/process-text      - Process text → full pipeline
    POST /api/pipeline/process-direct    - Process with direct audio blob
    GET  /api/pipeline/status/{request_id} - Check request status
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import tempfile
import os

from api.production_pipeline.pipeline_runner import PipelineRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Store orchestrator instances (could use Redis/database in production)
active_requests = {}


class ProcessMediaRequest(BaseModel):
    """Request model for processing media"""
    media_source: str = Field(..., description="File path or URL")
    media_type: str = Field(..., description="'video', 'audio', or 'url'")
    language: Optional[str] = Field("en", description="Language code")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")


class ProcessTextRequest(BaseModel):
    """Request model for processing text"""
    text: str = Field(..., description="Text to analyze")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")


class PipelineResponse(BaseModel):
    """Response model for pipeline"""
    request_id: str
    claim_id: str
    final_prediction: str
    confidence: float
    stages: Dict[str, Any]
    meta: Dict[str, Any]


@router.post("/process-media")
async def process_media(request: ProcessMediaRequest) -> PipelineResponse:
    """
    Process audio/video file through full pipeline
    
    Flow:
    1. Extract audio from media (if video)
    2. Transcribe to text (Whisper)
    3. Ingest & normalize claim
    4. RoBERTa inference
    5. Routing policy
    6. Continue through remaining pipeline
    
    Args:
        media_source: File path or URL
        media_type: 'video', 'audio', or 'url'
        language: Language code (default: 'en')
        request_id: Optional custom request ID
        
    Returns:
        Complete pipeline result
    """
    try:
        logger.info(f"Processing media: {request.media_source}")
        
        runner = PipelineRunner()
        result = runner.run(
            media_source=request.media_source,
            media_type=request.media_type,
            text_input=None,
            request_id=request.request_id,
            language=request.language
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Media processing completed: {result['request_id']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-text")
async def process_text(request: ProcessTextRequest) -> PipelineResponse:
    """
    Process text directly through pipeline
    
    Skips audio extraction, goes straight to text normalization
    
    Args:
        text: Text to analyze
        request_id: Optional custom request ID
        
    Returns:
        Complete pipeline result
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Processing text claim")
        
        runner = PipelineRunner()
        result = runner.run(
            media_source=None,
            media_type=None,
            text_input=request.text,
            request_id=request.request_id
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Text processing completed: {result['request_id']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-direct")
async def process_direct(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("en")
) -> PipelineResponse:
    """
    Process audio file directly (uploaded from extension)
    
    This is the primary endpoint for the Chrome extension
    
    Args:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        audio_file: Uploaded audio/video file
        language: Language code
        
    Returns:
        Complete pipeline result with transcription + analysis
    """
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"media_{request_id}")
        
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Determine media type from extension
        filename = audio_file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}:
            media_type = "video"
        elif ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
            media_type = "audio"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
        
        logger.info(f"Processing direct media: {media_type} from extension")
        
        # Run full pipeline
        runner = PipelineRunner()
        result = runner.run(
            media_source=temp_path,
            media_type=media_type,
            text_input=None,
            request_id=request_id,
            language=language
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Direct media processing completed: {request_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """
    Get processing status and results
    
    Args:
        request_id: Request identifier
        
    Returns:
        Current status and results if available
    """
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    return active_requests[request_id]


@router.get("/health")
async def health_check():
    """Health check for pipeline"""
    try:
        runner = PipelineRunner()
        return {
            "status": "healthy",
            "service": "production-pipeline",
            "audio_extraction_available": runner.audio_extractor is not None,
            "active_requests": len(active_requests)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.delete("/clear-requests")
async def clear_requests():
    """Clear completed requests (for cleanup)"""
    count = len(active_requests)
    active_requests.clear()
    return {"message": f"Cleared {count} completed requests"}
