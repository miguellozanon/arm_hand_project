#!/usr/bin/env bash
# One-time setup inside WSL (Ubuntu 24.04). Safe for an existing ROS 2 Jazzy install:
#   - never touches /opt/ros, the system Python or system pip packages
#   - everything goes into two uv virtual environments under ~/.venvs/
#       ahr-lerobot : LeRobot + SmolVLA + PyTorch (imitation learning / VLAs / compression)
#       ahr-mujoco  : MuJoCo + MuJoCo Playground + JAX (fast GPU RL, LEAP hand)
#
# Usage (from WSL):
#   bash /mnt/c/arm_hand_robot/scripts/wsl/setup_wsl.sh            # both envs
#   bash /mnt/c/arm_hand_robot/scripts/wsl/setup_wsl.sh lerobot    # only one
#   WITH_APT=1 bash .../setup_wsl.sh      # also apt-install ffmpeg + build tools (asks for sudo)
set -euo pipefail

VENVS="${HOME}/.venvs"
TARGETS="${1:-all}"
PY=3.12

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# Run in a clean env: a sourced ROS setup adds its own Python paths, which would leak into pip.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION || true

say "GPU visible from WSL?"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi not found in WSL. Update the Windows NVIDIA driver (WSL uses it directly)."; exit 1
fi

if [[ "${WITH_APT:-0}" == "1" ]]; then
    say "apt packages (ffmpeg for LeRobot video datasets, build tools)"
    sudo apt-get update
    sudo apt-get install -y ffmpeg build-essential git git-lfs
fi

say "uv (Python package manager, user-local)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version
mkdir -p "${VENVS}"

setup_lerobot() {
    say "ahr-lerobot: LeRobot + SmolVLA (Python ${PY})"
    uv venv --python "${PY}" "${VENVS}/ahr-lerobot"
    # shellcheck disable=SC1091
    source "${VENVS}/ahr-lerobot/bin/activate"
    # --torch-backend=auto picks the CUDA build matching the driver (RTX 50xx needs CUDA >= 12.8).
    uv pip install --torch-backend=auto torch torchvision
    uv pip install --torch-backend=auto "lerobot[smolvla]"
    uv pip install tensorboard nvidia-ml-py
    python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
import lerobot
print("lerobot", getattr(lerobot, "__version__", "installed"))
EOF
    deactivate
}

setup_mujoco() {
    say "ahr-mujoco: MuJoCo + Playground + JAX (Python ${PY})"
    uv venv --python "${PY}" "${VENVS}/ahr-mujoco"
    # shellcheck disable=SC1091
    source "${VENVS}/ahr-mujoco/bin/activate"
    uv pip install "jax[cuda12]" mujoco mujoco-mjx playground mediapy
    python - <<'EOF'
import jax, mujoco
print("jax", jax.__version__, "devices:", jax.devices())
print("mujoco", mujoco.__version__)
EOF
    deactivate
}

case "${TARGETS}" in
    all) setup_lerobot; setup_mujoco ;;
    lerobot) setup_lerobot ;;
    mujoco) setup_mujoco ;;
    *) echo "unknown target '${TARGETS}' (use: all | lerobot | mujoco)"; exit 2 ;;
esac

say "Done. Activate with:  source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh lerobot|mujoco"
