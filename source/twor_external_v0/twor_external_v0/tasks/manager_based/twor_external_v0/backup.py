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

# Import TWOR configuration - CORRECTED PATH
from twor_external_v0.robots.twor import TWOR_CONFIG  # isort:skip


##
# Scene definition
##


@configclass
class TworExternalV0SceneCfg(InteractiveSceneCfg):
    """Configuration for TwoR variable impedance control scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot - Use consistent naming with test script
    robot: ArticulationCfg = TWOR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Twor")

    # Cube object - Match the working example exactly
    Cube = RigidObjectCfg(
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

    # Contact sensor - EXACTLY as in working add_new_robot.py
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Twor/Sensor",   # Use Sensor, not Link2
        update_period=0.0,                       # every physics step
        history_length=1,                        # only latest contact
        debug_vis=False,                         # visualize contact forces
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],  # only collisions with Cube
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Use joint effort action to match test script behavior
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",  # Match the scene entity name
        joint_names=["Servo1", "Servo2", "Clamp"],
        scale=100.0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Joint positions and velocities
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Contact forces from sensor
        contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset robot position
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Contact reward
    contact_reward = RewTerm(
        func=mdp.contact_forces,
        weight=0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class TworExternalV0EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for TwoR Variable Impedance Control RL Environment."""

    # Scene settings
    scene: TworExternalV0SceneCfg = TworExternalV0SceneCfg(num_envs=4096, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation