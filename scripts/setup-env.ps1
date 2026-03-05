<#
PowerShell helper to initialize the project for development.
Runs from workspace root.
#>

# create python virtual environment if missing
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

# activate and install Python packages
Write-Host "Activating venv and installing Python requirements (this may take a while)..."
. ".\.venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install -r requirements.txt

# install frontend dependencies
if (Test-Path "frontend/package.json") {
    Write-Host "Installing frontend npm dependencies..."
    Push-Location frontend
    npm install
    Pop-Location
}

Write-Host "Setup complete. Use 'scripts\start-backend.ps1' and 'scripts\start-frontend.ps1' or run VS Code tasks."