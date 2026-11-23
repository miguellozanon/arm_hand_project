"""Local reward-design agent (Eureka-style, fully offline).

Loop:
  1. A local LLM (Ollama) reads the task description and the results of earlier trials, and
     proposes new reward weights as JSON.
  2. The weights are clamped to safe bounds and written to trial_XX/overrides.json.
  3. Isaac Lab trains the SO-101 lift task for a short budget with ARM_HAND_OVERRIDES set.
  4. The agent parses the training log and scores the trial by Episode_Reward/success.
Repeat. If Ollama is unreachable or returns bad JSON, the agent falls back to random
perturbation of the best weights so far, so the loop never stalls.

Only the Python standard library is used. Run it with Isaac Sim's bundled Python:

    C:\\isaaclab\\_isaac_sim\\python.bat C:\\arm_hand_robot\\agents\\reward_agent.py --trials 6

The LLM is unloaded from VRAM (keep_alive=0) before each training run, so the LLM and Isaac Sim
never hold the 8 GB GPU at the same time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

AHR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAACLAB_BAT = Path(r"C:\isaaclab\isaaclab.bat")
DEFAULT_TASK = "ArmHand-Lift-Cube-SO101-Quick-v0"
SCORE_KEY = "Episode_Reward/success"

# Tunable reward weights: default and [min, max]. Mirrors LiftRewardsCfg in env_cfg.py.
SEARCH_SPACE: dict[str, dict] = {
    "reach_coarse": {"default": 1.0, "bounds": [0.0, 5.0], "meaning": "dense reward for moving the gripper towards the cube (std 0.1 m)"},
    "reach_fine": {"default": 1.0, "bounds": [0.0, 5.0], "meaning": "sharper reward for the gripper being very close to the cube (std 0.02 m)"},
    "lifting": {"default": 15.0, "bounds": [0.0, 40.0], "meaning": "constant reward while the cube is > 2.5 cm above the table AND within 5 cm of the gripper (held)"},
    "height_coarse": {"default": 16.0, "bounds": [0.0, 40.0], "meaning": "dense reward for cube height approaching the 12 cm target (only when lifted and held)"},
    "height_fine": {"default": 5.0, "bounds": [0.0, 20.0], "meaning": "sharper version of height tracking (std 2 cm)"},
    "dropped": {"default": -10.0, "bounds": [-50.0, 0.0], "meaning": "one-off penalty when the cube falls off the table (ends the episode)"},
    "action_rate": {"default": -1e-3, "bounds": [-0.05, 0.0], "meaning": "penalty on action changes (smoothness)"},
    "joint_vel": {"default": -1e-4, "bounds": [-0.01, 0.0], "meaning": "penalty on joint velocities"},
}

TASK_DESCRIPTION = """\
Task: a 5-DOF SO-101 robot arm with a single-jaw gripper must grasp a 4 cm cube lying on a table
and lift it to 12 cm (cube centre height) and hold it there. Policy: PPO (RSL-RL), 20 Hz control,
5 s episodes, joint-position arm actions plus a binary open/close gripper action.
The score is Episode_Reward/success: the time-averaged indicator that the cube is within 2 cm of
the target height (higher is better; 0 means the cube is never lifted to target).
Lift, height and success rewards only count while the cube is within 5 cm of the gripper's tool
centre, so throwing the cube earns nothing; a cube that falls off the table ends the episode with the
"dropped" penalty. Typical failure modes: the arm never learns to close the gripper on the cube
(reach rewards too weak or too strong relative to lifting), the cube is pushed or flicked instead of
grasped (Episode_Termination/cube_dropped high), jerky motion (penalties too weak) or no motion at
all (penalties too strong)."""

LINE_RE = re.compile(
    r"^\s*(Episode_Reward/\w+|Episode_Termination/\w+|Mean reward|Mean episode length)\s*:\s*"
    r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")


##
# LLM (Ollama HTTP API, stdlib only)
##


def ollama_chat(host: str, model: str, messages: list[dict], timeout: int = 600) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "keep_alive": 0,  # free the GPU right after answering
            "options": {"temperature": 0.7},
        }
    ).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def build_prompt(history: list[dict]) -> list[dict]:
    space = {k: {"current_default": v["default"], "allowed_range": v["bounds"], "meaning": v["meaning"]} for k, v in SEARCH_SPACE.items()}
    trials = [
        {"trial": h["trial"], "weights": h["weights"], "score": h["score"], "final_metrics": h["metrics"]}
        for h in history
    ]
    system = (
        "You are a reinforcement-learning reward engineer. You tune reward weights for a robot task. "
        "Reply with JSON only, shaped as {\"analysis\": str, \"weights\": {term: number}}. "
        "Include every tunable term. Stay inside allowed_range. Change at most 3 weights per trial "
        "so the effect of each change can be attributed."
    )
    user = (
        f"{TASK_DESCRIPTION}\n\nTunable reward terms:\n{json.dumps(space, indent=1)}\n\n"
        f"Previous trials (score = {SCORE_KEY}):\n{json.dumps(trials, indent=1)}\n\n"
        "Look at the per-term Episode_Reward values to diagnose what the policy is (not) doing, "
        "then propose the next weights."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clamp_weights(raw: dict) -> dict[str, float]:
    out = {}
    for name, spec in SEARCH_SPACE.items():
        lo, hi = spec["bounds"]
        try:
            value = float(raw.get(name, spec["default"]))
        except (TypeError, ValueError):
            value = spec["default"]
        if not math.isfinite(value):
            value = spec["default"]
        out[name] = min(max(value, lo), hi)
    return out


def perturb(weights: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Fallback proposal: log-normal jitter on up to 3 terms."""
    new = dict(weights)
    for name in rng.sample(list(SEARCH_SPACE), k=3):
        w = new[name] if new[name] != 0 else SEARCH_SPACE[name]["default"] or 1e-3
        new[name] = w * math.exp(rng.gauss(0.0, 0.5))
    return clamp_weights(new)


##
# Training run + log parsing
##


def run_training(args, trial_dir: Path, overrides_path: Path) -> tuple[int, Path]:
    log_path = trial_dir / "train.log"
    cmd = [
        "cmd", "/c", str(args.isaaclab_bat), "train",
        "--rl_library", "rsl_rl",
        "--task", args.task,
        "--num_envs", str(args.num_envs),
        "--seed", str(args.seed),
    ]
    if args.max_iterations:
        cmd += ["--max_iterations", str(args.max_iterations)]
    # UTF-8 output: Isaac Lab's banner uses characters that cp1252 (Windows default) cannot write to a file
    env = dict(os.environ, ARM_HAND_OVERRIDES=str(overrides_path), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    print(f"    $ {' '.join(cmd[2:])}")
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=trial_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        last_print = 0.0
        for line in proc.stdout:
            log.write(line)
            m = ITER_RE.search(line)
            if m and time.time() - last_print > 30:
                print(f"    iteration {m.group(1)}/{m.group(2)}  ({time.time() - start:.0f}s)", flush=True)
                last_print = time.time()
            if time.time() - start > args.trial_timeout_min * 60:
                proc.kill()
                log.write("\n[reward_agent] killed: trial timeout\n")
                break
        proc.wait()
    return proc.returncode, log_path


def parse_metrics(log_path: Path, tail_fraction: float = 0.2) -> dict[str, float]:
    """Average each logged metric over the last ``tail_fraction`` of its occurrences."""
    series: dict[str, list[float]] = {}
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line)
            if m:
                series.setdefault(m.group(1), []).append(float(m.group(2)))
    out = {}
    for key, values in series.items():
        n = max(1, int(len(values) * tail_fraction))
        out[key] = round(sum(values[-n:]) / n, 5)
    return out


##
# Main loop
##


def main() -> int:
    p = argparse.ArgumentParser(description="Local LLM reward-design loop for the SO-101 lift task.")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--model", default=os.environ.get("AHR_LLM", "qwen2.5-coder:7b"))
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--max_iterations", type=int, default=0, help="0 = use the task's quick budget (300)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trial_timeout_min", type=float, default=60)
    p.add_argument("--isaaclab_bat", type=Path, default=DEFAULT_ISAACLAB_BAT)
    p.add_argument("--resume", type=Path, default=None, help="existing session dir to continue")
    args = p.parse_args()

    session = args.resume or AHR_ROOT / "agents" / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    history_file = session / "history.jsonl"
    history: list[dict] = []
    if history_file.exists():
        history = [json.loads(line) for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed + len(history))
    print(f"Session: {session}\nModel: {args.model} @ {args.ollama}\nTask: {args.task}")

    for trial in range(len(history), len(history) + args.trials):
        print(f"\n=== Trial {trial} ===")
        analysis = ""
        if trial == 0:
            weights = clamp_weights({k: v["default"] for k, v in SEARCH_SPACE.items()})
            analysis = "baseline (default weights)"
        else:
            try:
                reply = ollama_chat(args.ollama, args.model, build_prompt(history))
                proposal = json.loads(reply)
                weights = clamp_weights(proposal.get("weights", {}))
                analysis = str(proposal.get("analysis", ""))[:1500]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                best = max(history, key=lambda h: h["score"])
                weights = perturb(best["weights"], rng)
                analysis = f"fallback random perturbation of trial {best['trial']} ({type(exc).__name__}: {exc})"
        print(f"  analysis: {analysis[:300]}")
        print(f"  weights:  {weights}")

        trial_dir = session / f"trial_{trial:02d}"
        trial_dir.mkdir(exist_ok=True)
        overrides_path = trial_dir / "overrides.json"
        overrides_path.write_text(json.dumps({"rewards": {k: {"weight": v} for k, v in weights.items()}}, indent=2))

        rc, log_path = run_training(args, trial_dir, overrides_path)
        metrics = parse_metrics(log_path)
        score = metrics.get(SCORE_KEY, 0.0)
        if rc != 0 and not metrics:
            print(f"  training failed (exit {rc}); see {log_path}")
        record = {"trial": trial, "weights": weights, "score": score, "metrics": metrics, "analysis": analysis, "exit_code": rc}
        history.append(record)
        with history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  score ({SCORE_KEY}) = {score}")

    best = max(history, key=lambda h: h["score"])
    (session / "best.json").write_text(json.dumps(best, indent=2))
    lines = ["| trial | score | weights |", "|---|---|---|"]
    lines += [f"| {h['trial']} | {h['score']} | {json.dumps(h['weights'])} |" for h in history]
    (session / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBest: trial {best['trial']} score {best['score']}\nSummary: {session / 'summary.md'}")
    best_overrides = session / "trial_{:02d}".format(best["trial"]) / "overrides.json"
    print(f"Train it fully with (PowerShell):\n  $env:ARM_HAND_OVERRIDES = '{best_overrides}'")
    print("  powershell -File C:\\arm_hand_robot\\scripts\\windows\\train.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
