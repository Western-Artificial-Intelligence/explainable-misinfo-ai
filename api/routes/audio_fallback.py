"""Fallback audio routes when Whisper is not installed.

Provides /api/audio/transcribe-file that returns a structured response
so the Chrome extension can show a proper message instead of 404.
"""

from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/transcribe-file")
async def transcribe_file_fallback(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    """Fallback when Whisper is not installed. Returns structured error for extension."""
    return {
        "request_id": request_id,
        "claim_id": claim_id,
        "error": "Audio transcription requires OpenAI Whisper. Install with: pip install openai-whisper",
        "transcription": None,
    }
