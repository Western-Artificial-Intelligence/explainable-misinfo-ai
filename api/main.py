import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import classify, documents, health

load_dotenv()
logger = logging.getLogger(__name__)

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
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api/audio routes because dependency is missing: %s. "
        "Install optional audio deps to enable it.",
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
