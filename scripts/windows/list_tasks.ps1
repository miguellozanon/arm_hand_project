<#
.SYNOPSIS
  List registered tasks, filtered by keyword. Useful to find the exact hand task IDs.

.EXAMPLE
  .\list_tasks.ps1                 # our tasks
  .\list_tasks.ps1 -Keyword Allegro
  .\list_tasks.ps1 -Keyword Shadow
  .\list_tasks.ps1 -Keyword SO101
#>
param([string]$Keyword = "ArmHand")
. "$PSScriptRoot\config.ps1"
Push-Location $AHR_ROOT
try { & $ISAACLAB_BAT list_envs --keyword $Keyword } finally { Pop-Location }
