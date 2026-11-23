"""Apply reward/env overrides from a JSON file named by the ``ARM_HAND_OVERRIDES`` env var.

The local reward agent writes such a file and launches training with the variable set. This
avoids patching code between trials and keeps each trial reproducible (the JSON is saved with
the trial results).

JSON format::

    {
      "rewards": {
        "reach_coarse": {"weight": 1.0, "params": {"std": 0.1}},
        "lifting": {"weight": 15.0}
      },
      "env": {"episode_length_s": 5.0}
    }

Unknown terms and keys are reported and skipped, so a bad proposal cannot crash a run.
"""

from __future__ import annotations

import json
import os
from typing import Any

ENV_VAR = "ARM_HAND_OVERRIDES"
# Reward terms the agent must never change (they define the score).
PROTECTED_REWARDS = {"success"}
# Top-level env cfg attributes that may be overridden.
ALLOWED_ENV_KEYS = {"episode_length_s"}


def load_overrides() -> dict[str, Any] | None:
    path = os.environ.get(ENV_VAR)
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"[arm_hand_lab] {ENV_VAR}={path} does not exist; ignoring overrides.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_overrides(env_cfg: Any, overrides: dict[str, Any] | None = None) -> list[str]:
    """Apply overrides to ``env_cfg`` in place. Returns a list of human-readable changes."""
    overrides = overrides if overrides is not None else load_overrides()
    if not overrides:
        return []
    applied: list[str] = []

    for name, spec in (overrides.get("rewards") or {}).items():
        if name in PROTECTED_REWARDS:
            print(f"[arm_hand_lab] reward '{name}' is protected; skipping.")
            continue
        term = getattr(env_cfg.rewards, name, None)
        if term is None:
            print(f"[arm_hand_lab] unknown reward term '{name}'; skipping.")
            continue
        if "weight" in spec:
            term.weight = float(spec["weight"])
            applied.append(f"rewards.{name}.weight={term.weight}")
        for key, value in (spec.get("params") or {}).items():
            if term.params is None or key not in term.params:
                print(f"[arm_hand_lab] reward '{name}' has no param '{key}'; skipping.")
                continue
            term.params[key] = value
            applied.append(f"rewards.{name}.params.{key}={value}")

    for key, value in (overrides.get("env") or {}).items():
        if key not in ALLOWED_ENV_KEYS:
            print(f"[arm_hand_lab] env key '{key}' is not overridable; skipping.")
            continue
        setattr(env_cfg, key, value)
        applied.append(f"env.{key}={value}")

    if applied:
        print("[arm_hand_lab] overrides applied: " + ", ".join(applied))
    return applied
