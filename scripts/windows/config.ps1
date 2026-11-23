# Shared paths for the Windows scripts. Edit here if your install locations change.
$Global:AHR_ROOT     = "C:\arm_hand_robot"
$Global:ISAACLAB_DIR = "C:\isaaclab"
$Global:ISAACLAB_BAT = Join-Path $ISAACLAB_DIR "isaaclab.bat"
# Isaac Lab on Windows runs on Isaac Sim's bundled Python (through the _isaac_sim link).
$Global:ISAACSIM_PY  = Join-Path $ISAACLAB_DIR "_isaac_sim\python.bat"
$Global:AHR_PACKAGE  = Join-Path $AHR_ROOT "sim\arm_hand_lab"
# Default local model for the agents. 7-8B models in 4-bit fit in 8 GB VRAM when Isaac Sim is closed.
$Global:AHR_LLM      = "qwen2.5-coder:7b"
