"""SO-101 lift-cube task (reinforcement learning).

Reuses the SO-101 scene from Isaac Lab's contrib stack task (seated arm, table, ee frames,
PhysX/Newton presets, joint-position arm action, binary gripper) and turns it into a
reward-driven RL task with a single cube to lift to a target height.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.stack.config.so101.stack_joint_pos_env_cfg import SO101CubeStackEnvCfg
from isaaclab_tasks.contrib.stack.mdp import stack_events

from ..overrides import apply_overrides
from . import mdp

##
# Task constants (env frame; table top at z = 0)
##

CUBE = "cube_2"  # red cube from the stack scene
CUBE_REST_Z = 0.0203  # resting cube centre height [m]
CUBE_INIT_POS = (0.22, 0.0, CUBE_REST_Z)  # inside the SO-101 ~0.3 m reach
LIFTED_Z = 0.045  # counts as "lifted" above this centre height [m] (~2.5 cm off the table)
TARGET_Z = 0.12  # target centre height [m] (~10 cm lift)


@configclass
class LiftObservationsCfg:
    """State observations (no cameras), concatenated into one vector."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        object_pos = ObsTerm(func=mdp.object_pos_b, params={"object_cfg": SceneEntityCfg(CUBE)})
        object_quat = ObsTerm(func=mdp.object_quat_w, params={"object_cfg": SceneEntityCfg(CUBE)})
        grasp_point_pos = ObsTerm(func=mdp.grasp_point_pos_b)
        object_to_grasp = ObsTerm(func=mdp.object_to_grasp_point, params={"object_cfg": SceneEntityCfg(CUBE)})
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftEventCfg:
    """Reset events: small joint noise and a random cube pose on the table."""

    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )

    reset_cube = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # offsets around CUBE_INIT_POS, in the env frame
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.08, 0.08), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(CUBE),
        },
    )


@configclass
class LiftRewardsCfg:
    """Reward terms. Names are the keys the local reward agent may tune (except ``success``)."""

    reach_coarse = RewTerm(func=mdp.reach_object, weight=1.0, params={"std": 0.1, "object_cfg": SceneEntityCfg(CUBE)})
    reach_fine = RewTerm(func=mdp.reach_object, weight=1.0, params={"std": 0.02, "object_cfg": SceneEntityCfg(CUBE)})
    lifting = RewTerm(
        func=mdp.object_lifted, weight=15.0, params={"minimal_height": LIFTED_Z, "object_cfg": SceneEntityCfg(CUBE)}
    )
    height_coarse = RewTerm(
        func=mdp.lift_height_tracking,
        weight=16.0,
        params={"std": 0.1, "target_height": TARGET_Z, "minimal_height": LIFTED_Z, "object_cfg": SceneEntityCfg(CUBE)},
    )
    height_fine = RewTerm(
        func=mdp.lift_height_tracking,
        weight=5.0,
        params={"std": 0.02, "target_height": TARGET_Z, "minimal_height": LIFTED_Z, "object_cfg": SceneEntityCfg(CUBE)},
    )
    # dropping the cube off the table ends the episode; without a penalty, flicking it was rewarded
    dropped = RewTerm(func=base_mdp.is_terminated_term, weight=-10.0, params={"term_keys": ["cube_dropped"]})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
    # score term: keep weight 1.0 (protected from overrides)
    success = RewTerm(
        func=mdp.held_at_target,
        weight=1.0,
        params={"target_height": TARGET_Z, "tolerance": 0.02, "object_cfg": SceneEntityCfg(CUBE)},
    )


@configclass
class LiftTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(CUBE)},
    )


@configclass
class SO101LiftEnvCfg(SO101CubeStackEnvCfg):
    """SO-101 lifts a cube to ``TARGET_Z``. Sized for an 8 GB laptop GPU."""

    def __post_init__(self):
        super().__post_init__()

        # scene: one cube only, many parallel envs
        self.scene.num_envs = 512
        self.scene.env_spacing = 2.5
        self.scene.cube_1 = None
        self.scene.cube_3 = None
        self.scene.cube_2.init_state = RigidObjectCfg.InitialStateCfg(pos=CUBE_INIT_POS, rot=[0, 0, 0, 1])

        # MDP: replace the imitation-learning managers of the stack task with RL ones
        self.observations = LiftObservationsCfg()
        self.events = LiftEventCfg()
        self.rewards = LiftRewardsCfg()
        self.terminations = LiftTerminationsCfg()
        self.commands = None
        self.curriculum = None

        # timing: sim 100 Hz (parent), policy 20 Hz, 5 s episodes
        self.decimation = 5
        self.episode_length_s = 5.0

        # viewer / video camera: close-up of the arm and the cube workspace (env 0 at the origin)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.75, 0.55, 0.45), lookat=(0.18, 0.0, 0.05))

        # optional overrides written by the local reward agent (no-op if ARM_HAND_OVERRIDES unset)
        apply_overrides(self)

    def play_mode(self):
        parent = getattr(super(), "play_mode", None)
        if callable(parent):
            parent()
        self.scene.num_envs = 16
        self.scene.ee_frame.debug_vis = True  # shows the grasp frames, useful to check the offset
