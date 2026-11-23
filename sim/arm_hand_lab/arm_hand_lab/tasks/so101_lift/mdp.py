"""Observation and reward terms for the SO-101 lift task.

All heights are measured relative to each environment's origin. In the SO-101 stack scene the
table top is at z = 0 in the environment frame and a resting cube's centre is at about 0.0203 m.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


##
# Helpers
##


_TCP_OFFSET: dict[str, torch.Tensor | None] = {}


def _tcp_offset(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Position of the USD ``gripper/gripperframe`` marker in the ``gripper`` body frame [m] (cached).

    The SO-101 USD has a tool-centre marker between the jaw tips, but it is not a rigid body, so the
    FrameTransformer cannot track it directly. Its fixed offset from the gripper body is read once
    from the stage (env 0) and applied to the tracked gripper pose. Returns None if not found.
    """
    if "offset" in _TCP_OFFSET:
        return _TCP_OFFSET["offset"]
    offset = None
    try:
        from pxr import UsdGeom

        stage = getattr(env.sim, "stage", None)
        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        root = env.scene["robot"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("[^/]+", "0").replace(".*", "0")
        body = stage.GetPrimAtPath(f"{root}/gripper")
        marker = stage.GetPrimAtPath(f"{root}/gripper/gripperframe")
        if body.IsValid() and marker.IsValid():
            cache = UsdGeom.XformCache()
            rel = cache.GetLocalToWorldTransform(marker) * cache.GetLocalToWorldTransform(body).GetInverse()
            t = rel.ExtractTranslation()
            offset = torch.tensor([t[0], t[1], t[2]], dtype=torch.float32, device=env.device)
            print(f"[arm_hand_lab] grasp point = gripper body + {[round(float(v), 4) for v in t]} m (from gripper/gripperframe)")
        else:
            print(f"[arm_hand_lab] {root}/gripper/gripperframe not found; grasp point = gripper/jaw midpoint")
    except Exception as exc:  # never break training over the helper frame
        print(f"[arm_hand_lab] could not read gripperframe offset ({exc}); grasp point = gripper/jaw midpoint")
    _TCP_OFFSET["offset"] = offset
    return offset


def _grasp_point_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp point (tool centre between the jaw tips) in the world frame.

    ee_frame target 0 is the ``gripper`` body; the tool-centre offset comes from the USD marker
    ``gripper/gripperframe`` (see :func:`_tcp_offset`). Without it, fall back to the midpoint of the
    gripper link and moving-jaw origins, which sit a few cm behind the tips (first training video).
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    targets = ee_frame.data.target_pos_w.torch  # (num_envs, num_targets, 3)
    offset = _tcp_offset(env)
    if offset is not None:
        quat = ee_frame.data.target_quat_w.torch[:, 0, :]
        return targets[:, 0, :] + quat_apply(quat, offset.expand(targets.shape[0], 3))
    return targets[:, :2, :].mean(dim=1)


def _held(env: ManagerBasedRLEnv, max_distance: float, object_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 where the object centre is within ``max_distance`` [m] of the grasp point, else 0.

    Gates the lift rewards so a cube thrown into the air (not held) earns nothing.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w.torch - _grasp_point_w(env, ee_frame_cfg), dim=1)
    return (distance < max_distance).float()


def _object_height(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


##
# Observations
##


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Object position in the robot base frame [m], shape (num_envs, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch
    )
    return pos_b


def object_quat_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Object orientation (world frame quaternion). The base is fixed, so this carries the yaw."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w.torch


def grasp_point_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Grasp point position in the robot base frame [m], shape (num_envs, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, _grasp_point_w(env, ee_frame_cfg)
    )
    return pos_b


def object_to_grasp_point(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from grasp point to object (world axes) [m], shape (num_envs, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch - _grasp_point_w(env, ee_frame_cfg)


##
# Rewards
##


def reach_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1 - tanh(distance / std) between grasp point and object. In [0, 1]."""
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w.torch - _grasp_point_w(env, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    held_distance: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1 if the object centre is above ``minimal_height`` [m] AND held at the grasp point, else 0."""
    lifted = (_object_height(env, object_cfg) > minimal_height).float()
    return lifted * _held(env, held_distance, object_cfg, ee_frame_cfg)


def lift_height_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    minimal_height: float,
    held_distance: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward for holding the object near ``target_height``, paid only while lifted and held."""
    height = _object_height(env, object_cfg)
    lifted = (height > minimal_height).float() * _held(env, held_distance, object_cfg, ee_frame_cfg)
    return lifted * (1.0 - torch.tanh(torch.abs(height - target_height) / std))


def held_at_target(
    env: ManagerBasedRLEnv,
    target_height: float,
    tolerance: float,
    held_distance: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Success indicator: 1 while the object is held and within ``tolerance`` of ``target_height``.

    Its weight stays at 1.0 so its logged ``Episode_Reward/success`` is a clean success metric.
    The local reward agent uses it as the score and never changes it.
    """
    height = _object_height(env, object_cfg)
    at_target = (torch.abs(height - target_height) < tolerance).float()
    return at_target * _held(env, held_distance, object_cfg, ee_frame_cfg)
