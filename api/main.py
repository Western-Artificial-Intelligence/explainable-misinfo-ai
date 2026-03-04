# Import FastAPI framework
import os  # noqa: F401
from pathlib import Path

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
# Include routers
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(production.router, prefix="/api")
# app.include_router(ollama_blackboxes.router, prefix="/api/ollama")


# simple endpoint to verify the server is running
@app.get("/")
def root():
    return {"message": "TruthLens backend running!"}
