<#
Start the FastAPI backend in a terminal window. Assumes .venv exists.
#>

Write-Host "Activating Python environment and launching backend..."
. ".\.venv\Scripts\Activate.ps1"
cd api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
