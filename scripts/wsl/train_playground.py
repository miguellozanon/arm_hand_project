"""Train a MuJoCo Playground task (GPU, JAX) inside WSL. Default: LEAP hand cube rotation.

Run (WSL):
    source /mnt/c/arm_hand_robot/scripts/wsl/activate.sh mujoco
    python /mnt/c/arm_hand_robot/scripts/wsl/train_playground.py --list
    python /mnt/c/arm_hand_robot/scripts/wsl/train_playground.py --env LeapCubeRotateZAxis --num-envs 2048

Uses the published Playground PPO settings for the task. On an 8 GB GPU, lower
--num-envs (e.g. 1024-2048) if you see out-of-memory errors. Results go to
~/ahr_data/playground/<env>/<timestamp>/ (params + progress.csv).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

# Let JAX allocate on demand instead of grabbing 75% of VRAM up front (8 GB laptop GPU).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="LeapCubeRotateZAxis", help="Playground env name")
    parser.add_argument("--list", action="store_true", help="list manipulation envs and exit")
    parser.add_argument("--num-envs", type=int, default=None, help="override parallel envs")
    parser.add_argument("--timesteps", type=int, default=None, help="override total env steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mujoco_playground import registry, wrapper

    if args.list:
        try:
            names = registry.manipulation.ALL_ENVS
        except AttributeError:
            names = registry.ALL_ENVS
        print("Envs:", *sorted(names), sep="\n  ")
        return

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import manipulation_params

    env_cfg = registry.get_default_config(args.env)
    env = registry.load(args.env, config=env_cfg)
    ppo_params = manipulation_params.brax_ppo_config(args.env)
    if args.num_envs:
        ppo_params.num_envs = args.num_envs
        ppo_params.batch_size = min(ppo_params.batch_size, args.num_envs)
    if args.timesteps:
        ppo_params.num_timesteps = args.timesteps

    out_dir = Path.home() / "ahr_data" / "playground" / args.env / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.csv"
    t0 = time.time()

    def progress(step: int, metrics: dict) -> None:
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        print(f"[{time.time() - t0:7.1f}s] step {step:>12,d}  eval reward {reward:9.3f}", flush=True)
        new = not progress_file.exists()
        with progress_file.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["seconds", "step", "eval_episode_reward"])
            w.writerow([round(time.time() - t0, 1), step, reward])

    params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in params:
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **params.pop("network_factory"))

    print(f"Training {args.env}: num_envs={params.get('num_envs')} timesteps={params.get('num_timesteps'):,}")
    make_inference_fn, policy_params, _ = ppo.train(
        environment=env,
        eval_env=registry.load(args.env, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        progress_fn=progress,
        seed=args.seed,
        **params,
    )
    with (out_dir / "policy_params.pkl").open("wb") as f:
        pickle.dump(policy_params, f)
    print(f"Done in {time.time() - t0:.0f}s. Saved to {out_dir}")


if __name__ == "__main__":
    main()
