#!/usr/bin/env bash
# Activate one of the project environments in the CURRENT shell, without ROS paths leaking in.
#   source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh lerobot
#   source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh mujoco
# Use a separate terminal for ROS 2 Jazzy work (or `deactivate` and re-source ROS).

_ahr_env="${1:-lerobot}"
_ahr_path="${HOME}/.venvs/ahr-${_ahr_env}"
if [[ ! -d "${_ahr_path}" ]]; then
    echo "No env at ${_ahr_path}. Run scripts/wsl/setup_wsl.sh ${_ahr_env} first."
    return 1 2>/dev/null || exit 1
fi
if [[ -n "${ROS_DISTRO:-}" ]]; then
    echo "[activate] ROS ${ROS_DISTRO} is sourced in this shell; clearing its Python paths for this session."
    unset PYTHONPATH
fi
# shellcheck disable=SC1091
source "${_ahr_path}/bin/activate"
export AHR_ROOT="/mnt/c/arm_hand_robot"
# Big datasets and checkpoints are faster on the Linux filesystem than on /mnt/c.
export HF_HOME="${HOME}/.cache/huggingface"
export HF_LEROBOT_HOME="${HOME}/ahr_data/lerobot"
mkdir -p "${HF_LEROBOT_HOME}"
echo "[activate] ahr-${_ahr_env} active | AHR_ROOT=${AHR_ROOT} | datasets: ${HF_LEROBOT_HOME}"
unset _ahr_env _ahr_path
