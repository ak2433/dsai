# runme.ps1 — Run the agent FastAPI app locally (Windows).
# From repo root:  powershell -File 10_data_management/agentpy/runme.ps1
# Or: cd 10_data_management/agentpy; .\runme.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
