# Download a small Wikipedia subset for pretraining.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\download_wiki.ps1

$ErrorActionPreference = "Stop"
python -m src.data.download_wiki --out data\raw\wiki_edu.txt --max-mb 50
