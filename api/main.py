# Import FastAPI framework
import os  # noqa: F401
from pathlib import Path
import logging

from dotenv import load_dotenv

# Load .env before importing routes so pipeline steps (e.g. Step 2) see ROBERTA_USE_LLM etc.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.is_file():
    load_dotenv(_env_path, override=False)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import classify, health, production

# Create FastAPI app instance
app = FastAPI(title="TruthLens API", version="0.1.0")

# Enable CORS:
# CORS (Cross-Origin Resource Sharing) allows frontend hosted on a different domain
# to make API requests to this backend without being blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # "*" = allow requests from any origin. Can restrict later for security.
    allow_methods=["*"],  # Allow all HTTP methods: GET, POST, PUT, DELETE...
    allow_headers=["*"],  # Allow all HTTP headers (like content-type, authorization)
)
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(documents.router)

# Audio extraction and voice-to-text route
try:
    from .routes import audio
    app.include_router(audio.router)
except (ModuleNotFoundError, ImportError) as exc:
    logger.warning(
        "Skipping /api/audio routes because dependency is missing: %s. "
        "Install optional audio deps to enable it.",
        getattr(exc, "name", exc),
    )

# Image processing and OCR route
try:
    from .routes import imageprocessing
    app.include_router(imageprocessing.router)
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api/image routes because dependency is missing: %s. "
        "Install optional image processing deps to enable it.",
        exc.name,
    )

# Unified production pipeline route
try:
    from .routes import pipeline
    app.include_router(pipeline.router)
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api/pipeline routes because dependency is missing: %s. "
        "Install optional deps to enable it.",
        exc.name,
    )

# Production route is optional because it relies on extra dependencies.
try:
    from .routes import production
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api production routes because dependency is missing: %s. "
        "Install optional deps to enable it.",
        exc.name,
    )
else:
    app.include_router(production.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "TruthLens backend running!"}
