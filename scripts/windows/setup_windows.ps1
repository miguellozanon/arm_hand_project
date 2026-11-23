<#
.SYNOPSIS
  One-time setup on Windows. Does NOT modify C:\isaacsim or C:\isaaclab source files.

  1. Checks Isaac Sim / Isaac Lab paths.
  2. Installs the arm_hand_lab task package (editable) into Isaac Sim's bundled Python.
     Isaac Lab 3.0 discovers it through the "isaaclab.tasks" entry point.
  3. Lists the new tasks to confirm registration.
  4. Pulls the local LLM used by the agents (if Ollama is installed).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File C:\arm_hand_robot\scripts\windows\setup_windows.ps1
  powershell -ExecutionPolicy Bypass -File ...\setup_windows.ps1 -SkipOllama
#>
param(
    [switch]$SkipOllama,
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"
if ($Model -eq "") { $Model = $AHR_LLM }

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

Step "Checking paths"
foreach ($p in @($ISAACLAB_BAT, $ISAACSIM_PY, $AHR_PACKAGE)) {
    if (-not (Test-Path $p)) { throw "Missing: $p (edit scripts\windows\config.ps1)" }
    Write-Host "  ok  $p"
}
if ($env:VIRTUAL_ENV -or $env:CONDA_PREFIX) {
    throw "Deactivate your virtual/conda environment first: isaaclab.bat refuses to mix it with the bundled Isaac Sim Python."
}

Step "Installing arm_hand_lab (editable, no dependency changes) into Isaac Sim's Python"
& $ISAACSIM_PY -m pip install --no-deps -e $AHR_PACKAGE
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Step "Checking task registration (expects ArmHand-Lift-Cube-SO101-v0)"
Push-Location $AHR_ROOT
& $ISAACLAB_BAT list_envs --keyword ArmHand
Pop-Location

New-Item -ItemType Directory -Force -Path (Join-Path $AHR_ROOT "logs") | Out-Null

if (-not $SkipOllama) {
    Step "Local LLM for the agents ($Model)"
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -eq $ollama) {
        Write-Warning "Ollama not found on PATH. Install it from https://ollama.com, then run: ollama pull $Model"
    } else {
        & ollama pull $Model
    }
}

Step "Done. Next: scripts\windows\train.ps1 (see README.md)"
