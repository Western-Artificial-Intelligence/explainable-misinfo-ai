"""
API route handler for image/text extraction and processing

Endpoints:
    POST /api/image/analyze-claim - Analyze images and text for misinfo detection
    POST /api/image/detect-silence - Analyze audio to determine if voice present
"""

import logging
import io
import base64
import tempfile
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from api.utils.text_processor import TextCleaner, merge_text_blocks, get_text_statistics
from api.services.predictor import predict_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/image", tags=["image"])

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available - image processing will be limited")
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("Pytesseract not available - OCR will be limited")
    TESSERACT_AVAILABLE = False

try:
    from api.utils.audio_utils import AudioAnalyzer, audio_bytes_to_numpy
    AUDIO_UTILS_AVAILABLE = True
except ImportError:
    logger.warning("Audio utils not available")
    AUDIO_UTILS_AVAILABLE = False


# =====================
# Request/Response Models
# =====================

class TextExtractionResult(BaseModel):
    """Result of text extraction from images"""
    frame_index: int
    raw_text: str
    cleaned_text: str
    confidence: float
    is_valid: bool
    word_count: int


class AudioAnalysisResult(BaseModel):
    """Result of audio analysis"""
    average_energy: float
    silence_ratio: float
    is_silent: bool
    has_voice: bool
    confidence: float
    details: Dict[str, Any]


class ImageClaimAnalysisRequest(BaseModel):
    """Request for image-based claim analysis"""
    request_id: str = Field(..., description="Unique request ID")
    claim_id: str = Field(..., description="Unique claim ID")
    frame_count: int = Field(default=0, description="Number of frames captured")
    audio_analysis: Optional[str] = Field(None, description="JSON audio analysis data")
    captured_text: Optional[str] = Field(None, description="JSON array of captured text")


class ImageClaimAnalysisResponse(BaseModel):
    """Response for image-based claim analysis"""
    request_id: str
    claim_id: str
    status: str
    analysis_type: str
    extracted_text: str
    text_statistics: Dict[str, Any]
    frame_count: int
    audio_analysis: Optional[Dict[str, Any]] = None
    misinfo_predictions: List[Dict[str, Any]]
    meta: Dict[str, Any]


class SilenceDetectionRequest(BaseModel):
    """Request for silence detection"""
    request_id: str = Field(..., description="Request ID")
    audio_energy: float = Field(..., description="Average audio energy (RMS)")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(default=16000, description="Audio sample rate")


class SilenceDetectionResponse(BaseModel):
    """Response for silence detection"""
    request_id: str
    is_silent: bool
    has_voice: bool
    confidence: float
    recommendation: str


# =====================
# Processing Functions
# =====================

def extract_text_from_image(
    image_bytes: bytes,
    confidence_threshold: float = 0.5,
) -> str:
    """
    Extract text from image bytes using OCR.
    
    Args:
        image_bytes: Raw image data
        confidence_threshold: Minimum confidence for extracted text
        
    Returns:
        Extracted text
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract not available - cannot extract text from image")
        return ""
    
    try:
        # Convert bytes to image
        image = io.BytesIO(image_bytes)
        image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available for image processing")
            return ""
        
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return ""
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Thresholding for better contrast
        _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)
        
        # Upscale for small text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(processed)
        
        logger.info(f"Extracted {len(text)} characters from image")
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""


def clean_and_validate_text(
    raw_text: str,
    confidence: float = 1.0,
    min_confidence: float = 0.5,
) -> TextExtractionResult:
    """
    Clean and validate extracted text.
    
    Args:
        raw_text: Raw OCR output
        confidence: OCR confidence score
        min_confidence: Minimum confidence threshold
        
    Returns:
        TextExtractionResult with cleaned text
    """
    cleaner = TextCleaner()
    cleaned_result = cleaner.clean(
        raw_text,
        confidence=confidence,
        min_confidence=min_confidence,
    )
    
    return TextExtractionResult(
        frame_index=0,
        raw_text=raw_text,
        cleaned_text=cleaned_result.cleaned,
        confidence=cleaned_result.confidence,
        is_valid=cleaned_result.is_valid,
        word_count=len(cleaned_result.cleaned.split()),
    )


def analyze_text_for_misinfo(
    text: str,
    max_segments: int = 5,
) -> List[Dict[str, Any]]:
    """
    Analyze text segments for misinformation.
    
    Args:
        text: Text to analyze
        max_segments: Maximum number of segments to analyze
        
    Returns:
        List of prediction results
    """
    if not text or len(text.strip()) < 3:
        return []
    
    # Split into sentences/segments
    segments = text.split('.')
    segments = [s.strip() for s in segments if len(s.strip()) > 5]
    segments = segments[:max_segments]
    
    predictions = []
    
    for segment in segments:
        try:
            result = predict_text(segment)
            
            # Convert to prediction output format
            label = str(result.get("label", "mixed")).lower().strip()
            real_labels = {"real", "true", "factual", "reliable"}
            prediction = "REAL" if label in real_labels else "FAKE"
            
            predictions.append({
                "text": segment,
                "prediction": prediction,
                "confidence": float(result.get("confidence", 0)),
                "explanation": str(result.get("explanation", "")),
            })
        except Exception as e:
            logger.error(f"Error analyzing segment: {e}")
            predictions.append({
                "text": segment,
                "prediction": "ERROR",
                "confidence": 0.0,
                "explanation": f"Analysis error: {str(e)}",
            })
    
    return predictions


def analyze_audio_for_silence(
    audio_energy: float,
    duration_seconds: float,
) -> Dict[str, Any]:
    """
    Analyze audio metrics to determine if voice is present.
    
    Args:
        audio_energy: Average RMS energy
        duration_seconds: Audio duration
        
    Returns:
        Analysis result dictionary
    """
    if not AUDIO_UTILS_AVAILABLE:
        logger.warning("Audio utils not available")
        return {
            "is_silent": False,
            "has_voice": True,
            "confidence": 0.5,
        }
    
    analyzer = AudioAnalyzer()
    
    # If audio is very short, likely silence
    if duration_seconds < 0.5:
        return {
            "is_silent": True,
            "has_voice": False,
            "confidence": 0.9,
            "reason": "Audio duration too short",
        }
    
    # Check RMS thresholds
    if audio_energy < analyzer.SILENCE_THRESHOLD_RMS:
        return {
            "is_silent": True,
            "has_voice": False,
            "confidence": 0.95,
            "reason": "Audio energy below silence threshold",
        }
    elif audio_energy < analyzer.VOICE_THRESHOLD_RMS:
        return {
            "is_silent": True,
            "has_voice": False,
            "confidence": 0.85,
            "reason": "Audio energy minimal (likely background noise)",
        }
    else:
        return {
            "is_silent": False,
            "has_voice": True,
            "confidence": 0.8,
            "reason": "Sufficient voice energy detected",
        }


# =====================
# API Endpoints
# =====================

@router.post("/analyze-claim", response_model=ImageClaimAnalysisResponse)
async def analyze_image_claim(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    frame_count: int = Form(default=0),
    audio_analysis: Optional[str] = Form(None),
    captured_text: Optional[str] = Form(None),
    **form_data
):
    """
    Analyze images and text extracted from video for misinformation.
    
    Args:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        frame_count: Number of frames captured
        audio_analysis: JSON string of audio analysis data
        captured_text: JSON string array of captured text
        **form_data: Image files as frame_N
        
    Returns:
        Analysis result with predictions
    """
    try:
        # Parse audio analysis if provided
        audio_analysis_parsed = None
        if audio_analysis:
            import json
            try:
                audio_analysis_parsed = json.loads(audio_analysis)
            except json.JSONDecodeError:
                logger.warning("Failed to parse audio analysis JSON")
        
        # Parse captured text if provided
        captured_text_list = []
        if captured_text:
            import json
            try:
                captured_text_list = json.loads(captured_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse captured text JSON")
        
        # Extract text from uploaded images
        extracted_texts = []
        
        for i in range(frame_count):
            frame_key = f"frame_{i}"
            frame_text_key = f"frame_{i}_text"
            
            # Check if frame is in form data
            if frame_key in form_data:
                file = form_data[frame_key]
                try:
                    image_bytes = await file.read()
                    text = extract_text_from_image(image_bytes)
                    
                    if text.strip():
                        extracted_texts.append(text)
                        logger.info(f"Extracted {len(text)} chars from frame {i}")
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
            
            # Check if pre-extracted text is provided
            if frame_text_key in form_data:
                text = form_data[frame_text_key]
                if text:
                    extracted_texts.append(str(text))
        
        # Merge all extracted text
        all_text_sources = extracted_texts + captured_text_list
        merged_text = merge_text_blocks(all_text_sources, remove_duplicates=True)
        
        # Clean and validate merged text
        cleaner = TextCleaner()
        cleaned_result = cleaner.clean(merged_text)
        final_text = cleaned_result.cleaned
        
        # Get text statistics
        text_stats = get_text_statistics(final_text)
        
        # Analyze for misinformation
        predictions = analyze_text_for_misinfo(final_text)
        
        # Prepare response
        response_data = {
            "request_id": request_id,
            "claim_id": claim_id,
            "status": "success" if final_text else "no_text_extracted",
            "analysis_type": "image_and_text_extraction",
            "extracted_text": final_text,
            "text_statistics": text_stats,
            "frame_count": frame_count,
            "audio_analysis": audio_analysis_parsed,
            "misinfo_predictions": predictions,
            "meta": {
                "timestamp": datetime.utcnow().isoformat(),
                "extraction_sources": len(all_text_sources),
                "final_text_length": len(final_text),
                "text_valid": cleaned_result.is_valid,
            },
        }
        
        logger.info(f"Image analysis complete: {request_id}, predictions={len(predictions)}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error analyzing image claim: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-silence", response_model=SilenceDetectionResponse)
async def detect_silence(request: SilenceDetectionRequest):
    """
    Determine if audio is silent (no meaningful voice).
    
    This endpoint helps decide whether to use audio transcription
    or fall back to image/text extraction.
    
    Args:
        request: SilenceDetectionRequest
        
    Returns:
        Silence detection result and recommendation
    """
    try:
        audio_analysis = analyze_audio_for_silence(
            request.audio_energy,
            request.duration_seconds,
        )
        
        is_silent = audio_analysis.get("is_silent", False)
        has_voice = audio_analysis.get("has_voice", True)
        confidence = audio_analysis.get("confidence", 0.5)
        
        # Recommendation
        if is_silent:
            recommendation = "Use image/text capture instead of audio transcription"
        else:
            recommendation = "Use audio transcription (voice detected)"
        
        logger.info(
            f"Silence detection: {request.request_id}, "
            f"is_silent={is_silent}, confidence={confidence}"
        )
        
        return SilenceDetectionResponse(
            request_id=request.request_id,
            is_silent=is_silent,
            has_voice=has_voice,
            confidence=confidence,
            recommendation=recommendation,
        )
        
    except Exception as e:
        logger.error(f"Error in silence detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "tesseract_available": TESSERACT_AVAILABLE,
        "opencv_available": OPENCV_AVAILABLE,
        "audio_utils_available": AUDIO_UTILS_AVAILABLE,
    }
