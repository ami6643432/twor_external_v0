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
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

# Import TwoR robot configuration
from twor_external_v0.robots.twor import TWOR_CONFIG  # isort:skip


##
# Scene definition
##


@configclass
class TworExternalV0SceneCfg(InteractiveSceneCfg):
    """Configuration for TwoR variable impedance control scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # TwoR robot with variable impedance control
    robot: ArticulationCfg = TWOR_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=TWOR_CONFIG.spawn.replace(activate_contact_sensors=True)
    )

    # Interaction object (cube for force interaction tasks)
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.5)  # Position cube in front of robot end-effector
        ),
    )

    # Contact sensor on robot end-effector
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link2",  # End-effector link
        update_period=0.0,  # Update every step
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=[],  # Empty means contact with any object
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Variable impedance action - controls stiffness and damping parameters
    impedance_action = mdp.VariableImpedanceActionTermCfg(
        asset_name="robot",
        joint_names=["Servo1", "Servo2"],
        stiffness_range=(10.0, 5000.0),
        damping_range=(0.1, 100.0),
        default_stiffness=1000.0,
        default_damping=20.0,
        position_command_source="fixed_target",  # or "external_planner"
        fixed_target_positions=[0.0, 1.5708],  # Default target positions [rad]
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Current impedance state (stiffness, damping, tracking errors, joint states)
        impedance_state = ObsTerm(
            func=mdp.ImpedanceStateObsTerm,
            params={
                "asset_name": "robot",
                "action_term_name": "impedance_action",
                "include_current_impedance": True,
                "include_tracking_errors": True,
                "include_joint_states": True,
                "stiffness_range": (10.0, 5000.0),
                "damping_range": (0.1, 100.0),
                "max_position_error": 0.5,
                "max_velocity_error": 5.0,
            },
        )

        # Contact forces from end-effector sensor
        contact_forces = ObsTerm(
            func=mdp.ContactForceObsTerm,
            params={
                "asset_name": "robot",
                "contact_sensor_names": ["contact_sensor"],
                "max_expected_force": 100.0,
                "include_force_magnitude": True,
                "include_force_components": False,
            },
        )

        # Joint efforts/torques
        joint_efforts = ObsTerm(
            func=mdp.JointEffortObsTerm,
            params={
                "asset_name": "robot",
                "joint_names": ["Servo1", "Servo2"],
                "max_expected_effort": 100.0,
            },
        )

        # Distance to target positions
        target_distance = ObsTerm(
            func=mdp.TargetDistanceObsTerm,
            params={
                "asset_name": "robot",
                "action_term_name": "impedance_action",
                "joint_names": ["Servo1", "Servo2"],
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot joint positions to random values around targets
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Servo1", "Servo2"]),
            "position_range": (-0.2, 0.2),  # Small random offset from target
            "velocity_range": (-0.1, 0.1),
        },
    )

    # Reset cube position for interaction tasks
    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (0.7, 0.9),
                "z": (0.45, 0.55),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary reward: tracking performance
    tracking_performance = RewTerm(
        func=mdp.TrackingPerformanceReward,
        weight=10.0,
        params={
            "asset_name": "robot",
            "action_term_name": "impedance_action",
            "position_tolerance": 0.1,
            "velocity_tolerance": 1.0,
            "use_exponential_reward": True,
            "exponential_scale": 10.0,
        },
    )

    # Force stability reward
    force_stability = RewTerm(
        func=mdp.ForceStabilityReward,
        weight=5.0,
        params={
            "contact_sensor_names": ["contact_sensor"],
            "max_force_variation": 20.0,
            "target_force_range": (5.0, 50.0),
            "stability_window": 10,
        },
    )

    # Contact quality reward
    contact_quality = RewTerm(
        func=mdp.ContactQualityReward,
        weight=3.0,
        params={
            "contact_sensor_names": ["contact_sensor"],
            "desired_force_magnitude": 20.0,
            "force_tolerance": 10.0,
        },
    )

    # Parameter regularization (penalty for extreme values)
    impedance_regularization = RewTerm(
        func=mdp.ImpedanceParameterRegularization,
        weight=-0.01,  # Negative weight for penalty
        params={
            "action_term_name": "impedance_action",
            "stiffness_penalty_scale": 1e-6,
            "damping_penalty_scale": 1e-4,
            "parameter_change_penalty": 1e-3,
        },
    )

    # Effort efficiency reward
    effort_efficiency = RewTerm(
        func=mdp.EffortEfficiencyReward,
        weight=1.0,
        params={
            "asset_name": "robot",
            "joint_names": ["Servo1", "Servo2"],
            "max_acceptable_effort": 50.0,
        },
    )

    # Task completion bonus
    task_completion = RewTerm(
        func=mdp.TaskCompletionReward,
        weight=50.0,
        params={
            "action_term_name": "impedance_action",
            "asset_name": "robot",
            "position_threshold": 0.05,
            "velocity_threshold": 0.1,
            "completion_time_requirement": 50,
        },
    )
        func=mdp.ImpedanceParameterRegularization,
        weight=-0.01,  # Negative weight for penalty
        action_term_name="impedance_action",
        stiffness_penalty_scale=1e-6,
        damping_penalty_scale=1e-4,
        parameter_change_penalty=1e-3,
    )

    # Effort efficiency reward
    effort_efficiency = RewTerm(
        func=mdp.EffortEfficiencyReward,
        weight=1.0,
        asset_name="robot",
        joint_names=["Servo1", "Servo2"],
        max_acceptable_effort=50.0,
    )

    # Task completion bonus
    task_completion = RewTerm(
        func=mdp.TaskCompletionReward,
        weight=50.0,
        action_term_name="impedance_action",
        asset_name="robot",
        position_threshold=0.05,
        velocity_threshold=0.1,
        completion_time_requirement=50,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Joint limits violation
    joint_limits = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Servo1", "Servo2"]),
            "bounds": (-3.14, 3.14),  # Joint limits in radians
        },
    )


##
# Environment configuration
##


@configclass
class TworExternalV0EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for TwoR Variable Impedance Control RL Environment."""

    # Scene settings
    scene: TworExternalV0SceneCfg = TworExternalV0SceneCfg(num_envs=1024, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # General settings
        self.decimation = 2  # Control frequency = sim_freq / decimation
        self.episode_length_s = 10.0  # Episode duration in seconds
        
        # Viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)  # Camera position
        self.viewer.lookat = (0.0, 0.0, 0.5)  # Camera target
        
        # Simulation settings
        self.sim.dt = 1 / 120  # Simulation timestep (120 Hz)
        self.sim.render_interval = self.decimation  # Render frequency
        
        # Enable GPU pipeline for better performance
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625