# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for variable impedance control RL training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTerm, ObservationTermCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class ImpedanceStateObsTermCfg(ObservationTermCfg):
    """Configuration for impedance state observation term."""
    
    class_type: type[ObservationTerm] = "ImpedanceStateObsTerm"
    
    # Asset and action term names
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    action_term_name: str = "impedance_action"
    """Name of the variable impedance action term."""
    
    # What to include in observations
    include_current_impedance: bool = True
    """Include normalized current stiffness and damping values."""
    
    include_tracking_errors: bool = True
    """Include position and velocity tracking errors."""
    
    include_joint_states: bool = True
    """Include current joint positions and velocities."""
    
    # Normalization ranges for impedance parameters
    stiffness_range: tuple[float, float] = (10.0, 5000.0)
    """Range for normalizing stiffness values."""
    
    damping_range: tuple[float, float] = (0.1, 100.0)
    """Range for normalizing damping values."""
    
    # Normalization for tracking errors
    max_position_error: float = 0.5
    """Maximum expected position error for normalization [rad]."""
    
    max_velocity_error: float = 5.0
    """Maximum expected velocity error for normalization [rad/s]."""


class ImpedanceStateObsTerm(ObservationTerm):
    """Observation term for current impedance parameters and control state.
    
    Provides the RL agent with:
    - Current stiffness and damping parameters (normalized)
    - Position and velocity tracking errors
    - Current joint positions and velocities
    """
    
    cfg: ImpedanceStateObsTermCfg
    
    def __init__(self, cfg: ImpedanceStateObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get variable impedance action term
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices (should be 2 for Servo1, Servo2)
        self._joint_ids = self._action_term._joint_ids
        self._num_joints = len(self._joint_ids)
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns impedance control state observations.
        """
        obs_list = []
        
        # Current impedance parameters (normalized to [-1, 1])
        if self.cfg.include_current_impedance:
            current_stiffness = self._action_term.current_stiffness
            current_damping = self._action_term.current_damping
            
            # Normalize stiffness to [-1, 1]
            stiffness_normalized = (
                2.0 * (current_stiffness - self.cfg.stiffness_range[0]) / 
                (self.cfg.stiffness_range[1] - self.cfg.stiffness_range[0]) - 1.0
            )
            
            # Normalize damping to [-1, 1]
            damping_normalized = (
                2.0 * (current_damping - self.cfg.damping_range[0]) / 
                (self.cfg.damping_range[1] - self.cfg.damping_range[0]) - 1.0
            )
            
            obs_list.extend([stiffness_normalized, damping_normalized])
        
        # Current joint states
        if self.cfg.include_joint_states:
            joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
            joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
            
            # Normalize joint positions (assuming typical range [-π, π])
            joint_pos_normalized = joint_pos / 3.14159
            
            # Normalize joint velocities
            joint_vel_normalized = joint_vel / self.cfg.max_velocity_error
            
            obs_list.extend([joint_pos_normalized, joint_vel_normalized])
        
        # Tracking errors
        if self.cfg.include_tracking_errors:
            desired_pos = self._action_term.desired_joint_positions
            current_pos = self._asset.data.joint_pos[:, self._joint_ids]
            current_vel = self._asset.data.joint_vel[:, self._joint_ids]
            
            # Position tracking error
            pos_error = desired_pos - current_pos
            pos_error_normalized = pos_error / self.cfg.max_position_error
            
            # Velocity tracking error (assuming desired velocity is zero)
            vel_error = -current_vel  # Since desired vel is typically 0
            vel_error_normalized = vel_error / self.cfg.max_velocity_error
            
            obs_list.extend([pos_error_normalized, vel_error_normalized])
        
        # Concatenate all observations
        return torch.cat(obs_list, dim=-1)


@configclass
class ContactForceObsTermCfg(ObservationTermCfg):
    """Configuration for contact force observation term."""
    
    class_type: type[ObservationTerm] = "ContactForceObsTerm"
    
    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    contact_sensor_names: list[str] = ["contact_sensor"]
    """Names of contact sensors to read forces from."""
    
    # Force normalization
    max_expected_force: float = 100.0
    """Maximum expected contact force for normalization [N]."""
    
    include_force_magnitude: bool = True
    """Include magnitude of contact force."""
    
    include_force_components: bool = False
    """Include x, y, z components of contact force."""


class ContactForceObsTerm(ObservationTerm):
    """Observation term for contact forces from sensors.
    
    Provides the RL agent with contact force information from sensors
    mounted on the robot (typically end-effector or specific links).
    """
    
    cfg: ContactForceObsTermCfg
    
    def __init__(self, cfg: ContactForceObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get contact sensors
        self._contact_sensors = []
        for sensor_name in cfg.contact_sensor_names:
            if sensor_name in env.scene:
                self._contact_sensors.append(env.scene[sensor_name])
            else:
                print(f"Warning: Contact sensor '{sensor_name}' not found in scene")
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns contact force observations.
        """
        obs_list = []
        
        for sensor in self._contact_sensors:
            # Get contact forces from sensor
            contact_forces = sensor.data.net_forces_w  # Shape: [num_envs, 3]
            
            if self.cfg.include_force_magnitude:
                # Compute force magnitude
                force_magnitude = torch.norm(contact_forces, dim=-1, keepdim=True)
                force_magnitude_normalized = force_magnitude / self.cfg.max_expected_force
                obs_list.append(force_magnitude_normalized)
            
            if self.cfg.include_force_components:
                # Include x, y, z components
                force_components_normalized = contact_forces / self.cfg.max_expected_force
                obs_list.append(force_components_normalized)
        
        if obs_list:
            return torch.cat(obs_list, dim=-1)
        else:
            # Return zero observation if no sensors
            return torch.zeros((self._env.num_envs, 1), device=self._env.device)


@configclass
class JointEffortObsTermCfg(ObservationTermCfg):
    """Configuration for joint effort observation term."""
    
    class_type: type[ObservationTerm] = "JointEffortObsTerm"
    
    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to observe efforts for."""
    
    # Effort normalization
    max_expected_effort: float = 100.0
    """Maximum expected joint effort for normalization [N⋅m]."""


class JointEffortObsTerm(ObservationTerm):
    """Observation term for joint efforts/torques.
    
    Provides the RL agent with current joint efforts, which can indicate
    interaction forces and impedance control performance.
    """
    
    cfg: JointEffortObsTermCfg
    
    def __init__(self, cfg: JointEffortObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get joint indices
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns joint effort observations.
        """
        # Get joint efforts for controlled joints
        joint_efforts = self._asset.data.applied_torque[:, self._joint_ids]
        
        # Normalize efforts
        efforts_normalized = joint_efforts / self.cfg.max_expected_effort
        
        return efforts_normalized


@configclass
class TargetDistanceObsTermCfg(ObservationTermCfg):
    """Configuration for target distance observation term."""
    
    class_type: type[ObservationTerm] = "TargetDistanceObsTerm"
    
    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    action_term_name: str = "impedance_action"
    """Name of the variable impedance action term."""
    
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to compute distance for."""


class TargetDistanceObsTerm(ObservationTerm):
    """Observation term for distance to target positions.
    
    Provides the RL agent with information about how close the robot
    is to its target positions.
    """
    
    cfg: TargetDistanceObsTermCfg
    
    def __init__(self, cfg: TargetDistanceObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get action term
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns target distance observations.
        """
        # Get current and desired positions
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        desired_pos = self._action_term.desired_joint_positions
        
        # Compute distance to target
        position_error = desired_pos - current_pos
        distance_to_target = torch.norm(position_error, dim=-1, keepdim=True)
        
        return distance_to_target