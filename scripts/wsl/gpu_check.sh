#!/usr/bin/env bash
# GPU check inside WSL: driver view plus CUDA from each project environment that exists.
unset PYTHONPATH
nvidia-smi || echo "nvidia-smi not available in WSL"
for e in lerobot mujoco; do
    py="${HOME}/.venvs/ahr-${e}/bin/python"
    if [[ ! -x "${py}" ]]; then echo "ahr-${e}: not installed"; continue; fi
    echo "--- ahr-${e}"
    "${py}" - <<'PY' 2>&1 | tail -4
import importlib.util as u
if u.find_spec("torch"):
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if u.find_spec("jax"):
    import jax
    print("jax", jax.__version__, jax.devices())
PY
done
