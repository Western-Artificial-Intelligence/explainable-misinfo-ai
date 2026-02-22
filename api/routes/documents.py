from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.document_qa import answer_question, ingest_document_text, ingest_document_url, list_documents

router = APIRouter()


class DocumentUploadRequest(BaseModel):
    filename: str
    content: str


class DocumentURLRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None


@router.post("/documents/upload")
def upload_document(payload: DocumentUploadRequest):
    try:
        return ingest_document_text(filename=payload.filename, content=payload.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/documents/url")
def upload_document_from_url(payload: DocumentURLRequest):
    try:
        return ingest_document_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")


@router.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@router.post("/documents/ask")
def ask_document_question(payload: AskRequest):
    try:
        return answer_question(question=payload.question, document_id=payload.document_id)
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if "No document found" in message else 400
        raise HTTPException(status_code=status_code, detail=message)

