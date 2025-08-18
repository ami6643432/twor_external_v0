# File: twor_external_v0/twor_external_v0_env_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers...
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from gymnasium.spaces import Box

from twor_external_v0.robots.twor import TWOR_CONFIG

# =============================================================================
# TworExternalV0 Environment Configuration
# =============================================================================

Kmax = 5000.0    # Max joint stiffness
Dmax = 100.0     # Max joint damping

@configclass
class TworExternalV0EnvCfg(DirectRLEnvCfg):
    """
    Configuration for the TworExternalV0 RL environment.
    """

    # -------------------------------------------------------------------------
    # Run settings
    # -------------------------------------------------------------------------
    decimation: int         = 1
    episode_length_s: float = 5.0  # Slightly longer than trajectory to allow completion

    # -------------------------------------------------------------------------
    # Action / Observation spaces
    # -------------------------------------------------------------------------
    # Use Box spaces for Isaac Lab 4.5
    action_space: Box = Box(low=0.0, high=1.0, shape=(4,))  # 4D action space: [k1, d1, k2, d2]
    observation_space: Box = Box(low=-float('inf'), high=float('inf'), shape=(15,))  # 15D observation space
    state_space: int = 0  # no extra state vector

    # -------------------------------------------------------------------------
    # Simulation / Robot / Scene
    # -------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    robot_cfg: ArticulationCfg = TWOR_CONFIG.replace(
        prim_path="/World/envs/env_.*/Twor",
        spawn=TWOR_CONFIG.spawn.replace(activate_contact_sensors=True)
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=True #Increase the num_envs later
    )

    # -------------------------------------------------------------------------
    # Joint & Sensor Names
    # -------------------------------------------------------------------------
    servo1_dof_name: str   = "Servo1"
    servo2_dof_name: str   = "Servo2"
    sensor_body_name: str  = "Link2"   # body on which ContactSensor is mounted

    # -------------------------------------------------------------------------
    # Rewards / Termination
    # -------------------------------------------------------------------------
    force_scale: float       = 1.0
    max_joint_pos: float     = 1.5708
    max_allowed_force: float = 200.0

    target_pos_x: float = -1.0
    w_pos:            float = 10.0
    w_tracking:       float = 100.0
    w_stiffness:      float = 1e-6
    w_damping:        float = 0.01
    w_prog:           float = 10.0
    w_terminal:       float = 10.0

    imp_M: tuple[float, float] = (1.0, 1.0)
    
    # Action scaling - convert from [0,1] to actual ranges
    action_scale: list[float] = [Kmax, Dmax, Kmax, Dmax]
    action_offset: list[float] = [1e-3, 1e-3, 1e-3, 1e-3]

    # -------------------------------------------------------------------------
    # Impedance Filter Parameters
    # -------------------------------------------------------------------------
    impedance_method: str = "euler"  # "euler" or "iir"
    virtual_mass: tuple[float, float] = (1.0, 1.0)     # [kg·m²]
    virtual_damping: tuple[float, float] = (10.0, 10.0) # [N·m·s/rad]  
    virtual_stiffness: tuple[float, float] = (100.0, 100.0) # [N·m/rad]
    
    # Variable impedance limits (if RL controls parameters)
    mass_limits: tuple[float, float] = (0.1, 10.0)
    damping_limits: tuple[float, float] = (1.0, 100.0)
    stiffness_limits: tuple[float, float] = (10.0, 1000.0)

    # -------------------------------------------------------------------------
    # Manual Trajectory Waypoints
    # -------------------------------------------------------------------------
    # Define waypoints manually as a list of (x, y, time) tuples
    # Each waypoint: (x_pos, y_pos, time_to_reach)
    # Based on add_new_robot.py working motion, use conservative waypoints
    manual_waypoints: list[tuple[float, float, float]] = [
        (0.3, 0.0, 0.1),    # Waypoint 1: reach forward to +x direction
        (0.0, 0.3, 1.0),    # Waypoint 2: move to +y direction  
        (0.3, 0.0, 1.0),   # Waypoint 3: move to -x direction
        # (0.0, -0.3, 4.0),   # Waypoint 4: move to -y direction (simple circle)
    ]
    """Manual waypoints for trajectory generation [(x, y, time), ...] - simple circular motion"""
    
    use_manual_waypoints: bool = True
    """Use manual waypoints instead of automatic box pushing trajectory"""
    
    # -------------------------------------------------------------------------
    # Box Pushing Trajectory Parameters (fallback if manual waypoints disabled)
    # -------------------------------------------------------------------------
    box_start_pos: tuple[float, float, float] = (-0.3, -0.5, 0.1250)    # Initial box position [x, y, z] - CONSISTENT WITH SCENE
    box_target_pos: tuple[float, float, float] = (-0.4, -0.5, 0.1250)   # Target box position [x, y, z]
    trajectory_total_time: float = 4.0                                 # Total trajectory execution time [s]
    trajectory_approach_offset: float = 0.02                           # Smaller approach offset for contact [m]
    trajectory_max_count: int = int(4.0 / (1/120))                     # Steps for one trajectory (4 seconds at 120Hz)
