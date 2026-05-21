# Install Python deps. Run once.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install_deps.ps1

$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host "deps installed"
