<#
.SYNOPSIS
  Watch the latest trained policy in the Kit viewer (16 envs).

.EXAMPLE
  .\play.ps1
  .\play.ps1 -Task <Allegro task id>
#>
param(
    [string]$Task = "ArmHand-Lift-Cube-SO101-v0",
    [int]$NumEnvs = 16,
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

$argsList = @("play", "--rl_library", "rsl_rl", "--task", $Task, "--num_envs", $NumEnvs, "--viz", "kit")
if ($Extra -ne "") { $argsList += $Extra.Split(" ") }

Push-Location $AHR_ROOT   # play looks for checkpoints under .\logs
try { & $ISAACLAB_BAT @argsList } finally { Pop-Location }
