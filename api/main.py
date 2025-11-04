# Import FastAPI framework
import os  # noqa: F401

# allows your Python app to read environment variables from a
from dotenv import load_dotenv
from fastapi import FastAPI
# Import CORS middleware to allow frontend requests from other origins
from fastapi.middleware.cors import CORSMiddleware

# Import endpoint routers from other files (modular structure)
from .routes import classify, health

load_dotenv()

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


# simple endpoint to verify the server is running
@app.get("/")
def root():
    return {"message": "TruthLens backend running!"}
