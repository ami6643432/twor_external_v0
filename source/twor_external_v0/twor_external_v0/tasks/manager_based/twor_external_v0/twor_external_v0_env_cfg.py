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
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
# from twor_external_v0.robots.twor import TWOR_CONFIG
from twor_external_v0.robots.twor_min import TWOR_MIN_CONFIG

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

    # robot_min - using the minimal robot configuration
    robot_min: ArticulationCfg = TWOR_MIN_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Twor_min")

    # Cube object - exactly as in working add_new_robot.py
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, -0.5, 0.25)),
    )

    # Contact sensor - aligned with robot_min configuration (attached to Link2 end-effector)
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Twor_min/Link2",     # End-effector link for force sensing
        update_period=0.0,                            # every physics step
        history_length=1,                             # only latest contact
        debug_vis=False,                              # disable visualization (set True to debug)
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],  # only collisions with Cube
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    joint_position_command = mdp.JointPositionCommandCfg(
        asset_name="robot_min",
        joint_names=["Servo1", "Servo2"],
        command_type="sinusoidal",  # or "step"
        amplitude=[0.3, 0.4],  # [rad] for each joint
        frequency=[0.1, 0.15], # [Hz] for each joint
        offset=[0.0, 1.5708],  # [rad] starting positions
        recompute_time=0.0,    # Update every step
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    variable_impedance = mdp.VariableImpedanceActionCfg(
        asset_name="robot_min",
        joint_names=["Servo1", "Servo2"],
        command_term_name="joint_position_command",
        stiffness_range=(10.0, 2000.0),
        damping_range=(0.1, 200.0),
        default_stiffness=100.0,
        default_damping=30.0,
        contact_sensor_name="contact_sensor",
        debug_contact_forces=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_min")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_min")})
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

    scene: TworExternalV0SceneCfg = TworExternalV0SceneCfg(num_envs=1, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()  # Add this line
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