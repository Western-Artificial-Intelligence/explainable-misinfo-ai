from fastapi import APIRouter

# Create a router instance for this module
router = APIRouter()


# Define a GET endpoint at /healthz
@router.get("/healthz")
def health_check():
    # Returns a simple JSON indicating the server is healthy
    return {"status": "ok"}
