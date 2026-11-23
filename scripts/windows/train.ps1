<#
.SYNOPSIS
  Train a task with RSL-RL. Logs go to C:\arm_hand_robot\logs\rsl_rl\<experiment>\<run>.

.EXAMPLE
  .\train.ps1                                   # SO-101 lift, 512 envs, headless
  .\train.ps1 -NumEnvs 256 -MaxIterations 300   # smaller / shorter
  .\train.ps1 -Viz                              # open the Kit viewer (slower, uses more VRAM)
  .\train.ps1 -Task <Allegro task id> -NumEnvs 1024
  .\train.ps1 -Extra "env.rewards.lifting.weight=20"   # Hydra-style override
#>
param(
    [string]$Task = "ArmHand-Lift-Cube-SO101-v0",
    [int]$NumEnvs = 512,
    [int]$MaxIterations = 0,
    [int]$Seed = 42,
    [switch]$Viz,
    [string]$RunName = "",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

$argsList = @("train", "--rl_library", "rsl_rl", "--task", $Task, "--num_envs", $NumEnvs, "--seed", $Seed)
if ($MaxIterations -gt 0) { $argsList += @("--max_iterations", $MaxIterations) }
if ($Viz) { $argsList += @("--viz", "kit") }
if ($RunName -ne "") { $argsList += @("--run_name", $RunName) }
if ($Extra -ne "") { $argsList += $Extra.Split(" ") }

Push-Location $AHR_ROOT   # logs/ is created relative to the current directory
try {
    Write-Host "isaaclab.bat $($argsList -join ' ')" -ForegroundColor Cyan
    & $ISAACLAB_BAT @argsList
} finally { Pop-Location }
