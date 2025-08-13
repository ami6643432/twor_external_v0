# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for TwoR Variable Impedance Control RL Environment."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
from twor_external_v0.robots.twor import TWOR_CONFIG

##
# Scene definition
##

@configclass
class TworExternalV0SceneCfg(InteractiveSceneCfg):
    """Configuration for TwoR variable impedance control scene."""

    # Ground plane - exactly as in working example
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights - exactly as in working example  
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot - keep scene entity name "robot" but set prim_path to "Twor" (matches direct workflow USD structure)
    # This aligns with direct env where sensor is attached to Link2 under the Twor prim.
    robot: ArticulationCfg = TWOR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Twor")

    # Cube object - exactly as in working add_new_robot.py
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0, 0.25)),
    )

    # Contact sensor - align with direct workflow (attached to Link2 end-effector)
    # In direct env: prim_path="/World/envs/env_.*/Twor/Link2" stored as contact_L2. We keep the scene key
    # "contact_sensor" for consistency with manager-based observation/reward configuration.
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Twor/Link2",     # End-effector link for force sensing
        update_period=0.0,                          # every physics step
        history_length=1,                           # only latest contact
        debug_vis=False,                            # disable visualization (set True to debug)
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],  # only collisions with Cube
    )

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",  # This must match the scene entity name
        joint_names=["Servo1", "Servo2", "Clamp"],  # From URDF
        scale=100.0
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Contact force magnitude (functional helper returns [B,1])
        contact_force_mag = ObsTerm(
            func=mdp.contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    contact_reward = RewTerm(
        func=mdp.contact_force_magnitude,
        weight=0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##

@configclass
class TworExternalV0EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for TwoR Variable Impedance Control RL Environment."""

    scene: TworExternalV0SceneCfg = TworExternalV0SceneCfg(num_envs=4096, env_spacing=2.0)  # match add_new_robot.py
    observations: ObservationsCfg = ObservationsCfg()
    # actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5
        # Camera similar to add_new_robot.py (eye only)
        self.viewer.eye = (0.0, 3.5, 1)  # now from +Y instead of +X
        self.viewer.lookat = (0.0, 0.0, 0.5)  # aim toward scene center

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation