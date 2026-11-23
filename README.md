# arm_hand_robot

Train a robot arm (SO-101) and hands in simulation using **only local AI** (local RL, local
VLAs, local LLM agents). This is groundwork for the PhD project *Reliable On-Board Intelligence
for Autonomous Robots* (closed-loop-aware compression of VLA policies).

Hardware budget: none. Everything runs on the laptop (RTX 5060 Laptop, 8 GB VRAM, 32 GB RAM,
Windows 11 + WSL Ubuntu 24.04).

## Launcher (recommended)

A local web page at **http://127.0.0.1:8790** that runs everything below with buttons, same scheme
as the Go2 (8765), B2 (8770), Performance Lab (8090) and Airbox (8780) launchers: Isaac Sim's own
Python, standard library only, listening on 127.0.0.1, fixed actions only, one job at a time.

| To | Do |
| --- | --- |
| Install (once) | `Install.bat`: task package, Ollama model, the **Arm Hand Robot** shortcut and, if you want, start at login |
| Open | **Arm Hand Robot** shortcut or `Open_launcher.bat` (starts it hidden if needed, then opens the browser) |
| Close the browser | Nothing happens: the job keeps running |
| Stop everything | **Shut down** on the page, or `Stop_launcher.bat` |

Tabs: **Train** (smoke test, train, play, find hand tasks), **Local AI** (reward agent, model
download, sessions and "train with these" weights), **WSL & VLA** (WSL environments, Playground,
GPU check), **Results** (runs and checkpoints), **Log & report** (live log and a plain-text report
for Claude, also at http://127.0.0.1:8790/api/report).

GPU jobs refuse to start while another Isaac Sim (e.g. Go2/B2) is open, unless "Run anyway" is
ticked; Stop only kills what this launcher started. Launcher logs: `logs\launcher\`.

## Layout

```
C:\arm_hand_robot
├─ sim\arm_hand_lab\          Isaac Lab 3.0 task package (installed editable, auto-discovered)
│   └─ arm_hand_lab\tasks\so101_lift\   SO-101 lift-cube RL task + PPO config
├─ scripts\windows\           setup, train, play, list tasks (PowerShell)
├─ scripts\wsl\               WSL envs: LeRobot/SmolVLA and MuJoCo Playground (uv venvs)
├─ agents\reward_agent.py     local LLM (Ollama) reward-design loop
├─ docs\ROADMAP.md            phases from simulation to the PhD work packages
└─ logs\                      training runs (created on first run)
```

## What stays untouched

- `C:\isaacsim`, `C:\isaaclab`: no source edits. The task package is installed into Isaac Sim's
  bundled Python with `pip install -e`, and Isaac Lab finds it through its `isaaclab.tasks`
  entry point. To remove it: `C:\isaaclab\_isaac_sim\python.bat -m pip uninstall arm-hand-lab`.
- WSL ROS 2 Jazzy and the system Python: WSL tools go into `~/.venvs/ahr-*` only.

## 1. Setup (Windows, once)

Close Isaac Sim first (it holds GPU memory). Deactivate any conda/venv in the terminal.

```powershell
powershell -ExecutionPolicy Bypass -File C:\arm_hand_robot\scripts\windows\setup_windows.ps1
```

You should see `ArmHand-Lift-Cube-SO101-v0` and `ArmHand-Lift-Cube-SO101-Quick-v0` listed.

## 2. Train the SO-101 arm (Isaac Lab, RL)

```powershell
cd C:\arm_hand_robot\scripts\windows
train.bat -NumEnvs 64 -MaxIterations 2       # smoke test (~1-2 min)
train.bat                                      # full run: 512 envs, 1500 iterations
play.bat                                       # watch the latest checkpoint (16 envs)
```

Each `.ps1` script has a `.bat` launcher next to it (`train.bat`, `play.bat`, `list_tasks.bat`,
`setup_windows.bat`), so they run even when Windows blocks PowerShell scripts. Arguments are
the same.

The warning `_isaac_sim\setup_conda_env.bat is missing` is expected with the zip install of
Isaac Sim: `isaaclab.bat` then runs Isaac Sim's own `python.bat`, which sets the needed paths.

TensorBoard: `C:\isaaclab\_isaac_sim\python.bat -m tensorboard.main --logdir C:\arm_hand_robot\logs`

Key metric: `Episode_Reward/success` (time fraction the cube is held at 12 cm).

**8 GB VRAM guide (state-based RL):** 512 envs is the default; 1024 usually fits headless;
drop to 256 if you get CUDA out-of-memory errors or when using `-Viz`.

**First check:** in `play.ps1` the gripper frames are drawn. The reaching rewards use the
midpoint of the `gripper` and `moving_jaw` frames as the grasp point. If that midpoint is far
from the space between the jaws, set an offset in the stack scene's `make_ee_frame_cfg` call
by overriding `self.scene.ee_frame` in `env_cfg.py`.

## 3. Train a hand

Built-in Isaac Lab hand tasks (Allegro, Shadow, Kuka+Allegro):

```powershell
list_tasks.bat -Keyword Allegro
train.bat -Task <id from the list> -NumEnvs 1024
```

MuJoCo Playground LEAP hand (WSL, JAX on GPU):

```bash
bash /mnt/c/arm_hand_robot/scripts/wsl/setup_wsl.sh mujoco
source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh mujoco
python /mnt/c/arm_hand_robot/scripts/wsl/train_playground.py --list
python /mnt/c/arm_hand_robot/scripts/wsl/train_playground.py --env LeapCubeRotateZAxis --num-envs 2048
```

## 4. Local AI agent: reward design

Needs Ollama (`ollama pull qwen2.5-coder:7b`, or any model that answers in JSON).
The agent proposes reward weights, runs a 300-iteration training per trial, reads the results,
and repeats. The LLM is unloaded before each training so the GPU is never shared.

```powershell
C:\isaaclab\_isaac_sim\python.bat C:\arm_hand_robot\agents\reward_agent.py --trials 6
```

Results: `agents\runs\<timestamp>\` (`history.jsonl`, `summary.md`, `best.json`, per-trial
logs). Without Ollama it still runs, using random perturbations instead of LLM proposals.
Rough time: about 5-15 min per trial at 512 envs on this laptop.

To train the best weights fully:

```powershell
$env:ARM_HAND_OVERRIDES = "C:\arm_hand_robot\agents\runs\<timestamp>\trial_XX\overrides.json"
train.bat
Remove-Item Env:ARM_HAND_OVERRIDES   # otherwise later runs keep using it
```

## 5. WSL: LeRobot / SmolVLA (next phase)

```bash
bash /mnt/c/arm_hand_robot/scripts/wsl/setup_wsl.sh lerobot     # add WITH_APT=1 for ffmpeg
source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh lerobot
```

See `docs/ROADMAP.md` for how this connects to the PhD work packages.
# arm_hand_project
