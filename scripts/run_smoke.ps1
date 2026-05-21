# Run the end-to-end smoke test on all 3 attention variants.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1

$ErrorActionPreference = "Stop"
python smoke_test.py
