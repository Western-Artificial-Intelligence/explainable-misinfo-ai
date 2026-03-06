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

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel, Field

from api.utils.text_processor import TextCleaner, merge_text_blocks, get_text_statistics
from api.services.predictor import predict_text
from api.utils.analysis_store import add_analysis_record

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


def _persist_image_record(
    *,
    session_id: str,
    page_url: str,
    source_context: str,
    input_type: str,
    input_text: str,
    response_data: dict[str, Any],
) -> None:
    predictions = response_data.get("misinfo_predictions")
    first_prediction = predictions[0] if isinstance(predictions, list) and predictions else {}
    verdict = str(first_prediction.get("prediction") or "MIXED")
    confidence = _safe_float(first_prediction.get("confidence"), default=0.0)
    reasoning = str(first_prediction.get("explanation") or "Image/text analysis completed.")
    add_analysis_record(
        {
            "session_id": session_id or response_data.get("request_id") or "image-session",
            "input_type": input_type or "image_text_capture",
            "input_text": input_text,
            "transcript": "",
            "page_url": page_url or "",
            "analysis_result": response_data,
            "confidence": confidence,
            "reasoning": reasoning,
            "verdict": verdict,
            "source_context": source_context or "chrome_extension",
        }
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =====================
# API Endpoints
# =====================

@router.post("/analyze-claim", response_model=ImageClaimAnalysisResponse)
async def analyze_image_claim(request: Request):
    """
    Analyze images and text extracted from video for misinformation.
    
    Form fields:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        frame_count: Number of frames captured
        audio_analysis: JSON string of audio analysis data
        captured_text: JSON string array of captured text
        frame_N: Image files (optional)
        frame_N_text: Pre-extracted OCR text (optional)
        
    Returns:
        Analysis result with predictions
    """
    import json as json_module

    try:
        form = await request.form()

        request_id = form.get("request_id", "")
        claim_id = form.get("claim_id", "")
        frame_count = int(form.get("frame_count", 0) or 0)
        audio_analysis = form.get("audio_analysis")
        captured_text = form.get("captured_text")
        session_id = str(form.get("session_id", "") or "")
        page_url = str(form.get("page_url", "") or "")
        source_context = str(form.get("source_context", "chrome_extension") or "chrome_extension")
        input_type = str(form.get("input_type", "image_text_capture") or "image_text_capture")

        # Parse audio analysis if provided
        audio_analysis_parsed = None
        if audio_analysis:
            try:
                audio_analysis_parsed = json_module.loads(audio_analysis)
            except json_module.JSONDecodeError:
                logger.warning("Failed to parse audio analysis JSON")

        # Parse captured text if provided
        captured_text_list = []
        if captured_text:
            try:
                captured_text_list = json_module.loads(captured_text)
            except json_module.JSONDecodeError:
                logger.warning("Failed to parse captured text JSON")

        # Extract text from uploaded images and pre-extracted frame text
        extracted_texts = []

        for i in range(frame_count):
            frame_key = f"frame_{i}"
            frame_text_key = f"frame_{i}_text"

            if frame_key in form:
                item = form[frame_key]
                if hasattr(item, "read"):
                    try:
                        image_bytes = await item.read()
                        text = extract_text_from_image(image_bytes)
                        if text.strip():
                            extracted_texts.append(text)
                            logger.info(f"Extracted {len(text)} chars from frame {i}")
                    except Exception as e:
                        logger.error(f"Error processing frame {i}: {e}")

            if frame_text_key in form:
                text = form.get(frame_text_key)
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

        # System prompt pipeline: extract claim via LLM when available
        predictions = []
        try:
            from api.services.claim_extractor import extract_claim_from_ocr
            extracted_claim = await extract_claim_from_ocr(final_text)
            if extracted_claim and len(extracted_claim.strip()) >= 5:
                result = predict_text(extracted_claim)
                label = str(result.get("label", "mixed")).lower().strip()
                real_labels = {"real", "true", "factual", "reliable"}
                pred = "REAL" if label in real_labels else "FAKE"
                predictions.append({
                    "text": extracted_claim[:200] + ("…" if len(extracted_claim) > 200 else ""),
                    "prediction": pred,
                    "confidence": float(result.get("confidence", 0)),
                    "explanation": str(result.get("explanation", "")),
                })
        except Exception as e:
            logger.debug("Claim extraction skipped: %s", e)

        # Segment-level analysis
        segment_predictions = analyze_text_for_misinfo(final_text)
        predictions.extend(segment_predictions)
        
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
        try:
            _persist_image_record(
                session_id=session_id,
                page_url=page_url,
                source_context=source_context,
                input_type=input_type,
                input_text=final_text,
                response_data=response_data,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist image analysis record: %s", exc)

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
