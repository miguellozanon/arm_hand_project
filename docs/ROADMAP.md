# Roadmap

Constraint: no hardware budget. Everything in simulation on the laptop
(RTX 5060 Laptop 8 GB, 32 GB RAM). Isaac Sim 6.1 / Isaac Lab 3.0 on Windows,
LeRobot / MuJoCo / JAX in WSL Ubuntu 24.04, ROS 2 Jazzy left as is.

## Phase 1: arm and hand RL in simulation (now)

- [ ] Run `setup_windows.ps1`, smoke-test `ArmHand-Lift-Cube-SO101-v0`.
- [ ] Full SO-101 lift training; check the grasp-point frames in `play.ps1`.
- [ ] Run the local reward agent (6-10 trials), compare against the hand-set baseline.
- [ ] Train one Isaac Lab hand task (Allegro reorient) and the Playground LEAP hand.
- Output: a working SO-101 lift policy ("expert") plus a first local-AI result
  (LLM-designed rewards vs. baseline).

## Phase 2: from RL expert to VLA (still simulation)

- Add cameras (top + wrist) to a copy of the lift task; use small resolutions
  (e.g. 128x128) and 16-32 envs to fit 8 GB.
- Roll out the RL expert and record demonstrations with images and a language instruction.
- Convert to LeRobot dataset format (WSL), train ACT, then fine-tune SmolVLA
  (bf16, small batch, gradient accumulation on 8 GB).
- Evaluate closed loop: policy server in WSL, Isaac Lab on Windows as the client.

## Phase 3: PhD WP1 pilot (closed-loop compression benchmark)

- Quantize SmolVLA (fp16 / int8 / int4, per-module mixes) and measure in closed loop:
  success rate, p50/p99 latency, control rate, GPU energy per episode (NVML),
  peak memory, and trajectory deviation from the full-precision policy.
- The laptop GPU at a fixed power cap acts as a stand-in for an embedded device
  until a Jetson is available.

## Phase 4: local agents on the robot side (WP3/WP4 prototypes)

- Small local VLM (Ollama) as a slow planner/supervisor above the fast policy.
- Reliability monitor: action-sample disagreement and progress checks triggering a fallback.
- ROS 2 Jazzy integration through Isaac Sim's ROS 2 bridge.
