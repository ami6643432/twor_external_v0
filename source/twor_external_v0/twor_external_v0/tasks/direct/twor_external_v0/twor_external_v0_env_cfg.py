# File: twor_external_v0/twor_external_v0_env_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers...
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from gymnasium.spaces import Box
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg

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
    episode_length_s: float = 10.0

    # -------------------------------------------------------------------------
    # Action / Observation spaces
    # -------------------------------------------------------------------------
    action_space: Box = Box(
        low  = np.array([1e-3,1e-3,1e-3,1e-3], dtype=np.float32),
        high = np.array([Kmax,Dmax,Kmax,Dmax], dtype=np.float32),
        dtype=np.float32
    )
    observation_space: Box = Box(
        low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # Simulation / Robot / Scene
    # -------------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    robot_cfg: ArticulationCfg = TWOR_CONFIG.replace(
        prim_path="/World/envs/env_.*/Twor",
        spawn=TWOR_CONFIG.spawn.replace(activate_contact_sensors=True)
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=2.0, replicate_physics=True
    )

    # -------------------------------------------------------------------------
    # Joint & Sensor Names
    # -------------------------------------------------------------------------
    servo1_dof_name: str   = "Servo1"
    servo2_dof_name: str   = "Servo2"
    sensor_body_name: str  = "Link2"   # body on which ContactSensor is mounted :contentReference[oaicite:3]{index=3}

    # -------------------------------------------------------------------------
    # Rewards / Termination
    # -------------------------------------------------------------------------
    force_scale: float       = 1.0
    max_joint_pos: float     = 1.5708
    max_allowed_force: float = 200.0

    state_space: int = 0  # no extra state vector (avoids Hydra space error) :contentReference[oaicite:10]{index=10}

    target_pos_x: float = -1.0
    w_pos:            float = 10.0
    w_tracking:       float = 100.0
    w_stiffness:      float = 1e-6
    w_damping:        float = 0.01
    w_prog:           float = 10.0
    w_terminal:       float = 10.0

    imp_M: tuple[float, float] = (1.0, 1.0)
