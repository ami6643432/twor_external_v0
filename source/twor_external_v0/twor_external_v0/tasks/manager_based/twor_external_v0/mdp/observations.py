# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for learned impedance control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTerm, ObservationTermCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ImpedanceStateObsTerm(ObservationTerm):
    """Observation term for current impedance parameters and tracking errors."""
    
    def __init__(self, cfg: "ImpedanceStateObsTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get action term to access impedance parameters
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices from action term
        self._joint_ids = self._action_term._joint_ids
        self._num_joints = self._action_term._num_joints
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns impedance-related observations:
        - Current stiffness values (normalized)
        - Current damping values (normalized)  
        - Position tracking errors
        - Velocity tracking errors
        """
        obs_list = []
        
        # Get current joint states
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Get target positions from action term
        desired_pos = self._action_term.desired_positions
        desired_vel = self._action_term._desired_vel
        
        # Compute tracking errors
        pos_error = desired_pos - joint_pos
        vel_error = desired_vel - joint_vel
        
        # Add tracking errors
        if self.cfg.include_position_error:
            obs_list.append(pos_error)
        
        if self.cfg.include_velocity_error:
            obs_list.append(vel_error)
        
        # Add current impedance parameters (normalized to [-1, 1])
        if self.cfg.include_stiffness:
            current_stiffness = self._action_term.current_stiffness
            stiffness_normalized = (
                2.0 * (current_stiffness - self.cfg.stiffness_min) / 
                (self.cfg.stiffness_max - self.cfg.stiffness_min) - 1.0
            )
            obs_list.append(stiffness_normalized)
        
        if self.cfg.include_damping:
            current_damping = self._action_term.current_damping
            # Normalize damping based on configured range
            damping_normalized = (
                2.0 * (current_damping - self.cfg.damping_min) / 
                (self.cfg.damping_max - self.cfg.damping_min) - 1.0
            )
            obs_list.append(damping_normalized)
        
        # Add joint torques if requested
        if self.cfg.include_joint_torques:
            joint_torques = self._asset.data.joint_effort_target[:, self._joint_ids]
            # Normalize torques
            if self.cfg.torque_normalization is not None:
                joint_torques = joint_torques / self.cfg.torque_normalization
            obs_list.append(joint_torques)
        
        # Concatenate all observations
        return torch.cat(obs_list, dim=-1)


class ExternalForceObsTerm(ObservationTerm):
    """Observation term for external force/torque estimates using momentum observer."""
    
    def __init__(self, cfg: "ExternalForceObsTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get joint indices
        if cfg.joint_names is not None:
            self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        else:
            self._joint_ids = slice(None)
        
        # Momentum observer parameters
        self._observer_gain = cfg.observer_gain
        self._cutoff_freq = cfg.cutoff_freq
        
        # Internal states for momentum observer
        num_joints = len(self._joint_ids) if isinstance(self._joint_ids, list) else self._asset.num_joints
        self._momentum_estimate = torch.zeros(env.num_envs, num_joints, device=env.device)
        self._external_torque_estimate = torch.zeros(env.num_envs, num_joints, device=env.device)
        
        # Previous velocity for numerical differentiation
        self._prev_joint_vel = torch.zeros(env.num_envs, num_joints, device=env.device)
        
    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset observer states for specified environments."""
        self._momentum_estimate[env_ids] = 0.0
        self._external_torque_estimate[env_ids] = 0.0
        current_vel = self._asset.data.joint_vel[env_ids, self._joint_ids]
        self._prev_joint_vel[env_ids] = current_vel
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns external force estimates using momentum observer:
        - Estimated external joint torques
        - Filtered external torques (optional)
        """
        # Get current joint states
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        joint_torques = self._asset.data.joint_effort_target[:, self._joint_ids]
        
        # Estimate joint accelerations (simple finite difference)
        dt = self._env.physics_dt
        joint_acc = (joint_vel - self._prev_joint_vel) / dt
        self._prev_joint_vel = joint_vel.clone()
        
        # Momentum observer: p = I*q_dot
        # dp/dt = I*q_ddot = tau + tau_ext
        # tau_ext_estimate = dp/dt - tau_applied
        
        # For simplicity, assume unit inertia (can be improved with actual inertia)
        estimated_inertia = self.cfg.estimated_inertia
        momentum_current = estimated_inertia * joint_vel
        
        # Update momentum estimate with observer
        momentum_error = momentum_current - self._momentum_estimate
        self._momentum_estimate += self._observer_gain * momentum_error * dt
        
        # Estimate external torques
        momentum_derivative = estimated_inertia * joint_acc
        self._external_torque_estimate = momentum_derivative - joint_torques
        
        # Apply low-pass filter to reduce noise
        alpha = dt * self._cutoff_freq / (1 + dt * self._cutoff_freq)
        self._external_torque_estimate = (
            alpha * self._external_torque_estimate + 
            (1 - alpha) * self._external_torque_estimate
        )
        
        obs_list = []
        
        # Add raw external torque estimates
        if self.cfg.include_raw_estimates:
            obs_list.append(self._external_torque_estimate)
        
        # Add filtered/processed estimates
        if self.cfg.include_filtered_estimates:
            # Simple magnitude and direction
            torque_magnitude = torch.norm(self._external_torque_estimate, dim=-1, keepdim=True)
            obs_list.append(torque_magnitude)
        
        return torch.cat(obs_list, dim=-1) if obs_list else self._external_torque_estimate


class JointStateObsTerm(ObservationTerm):
    """Basic joint state observations for impedance control."""
    
    def __init__(self, cfg: "JointStateObsTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get joint indices
        if cfg.joint_names is not None:
            self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        else:
            self._joint_ids = slice(None)
    
    @property
    def data(self) -> torch.Tensor:
        """Returns normalized joint positions and velocities."""
        obs_list = []
        
        # Get joint states
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Normalize joint positions to [-1, 1] based on limits
        if self.cfg.normalize_positions:
            joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            joint_lower = joint_limits[..., 0]
            joint_upper = joint_limits[..., 1]
            # Avoid division by zero
            joint_range = joint_upper - joint_lower
            joint_range = torch.where(joint_range > 1e-6, joint_range, torch.ones_like(joint_range))
            joint_pos_normalized = 2.0 * (joint_pos - joint_lower) / joint_range - 1.0
            obs_list.append(joint_pos_normalized)
        else:
            obs_list.append(joint_pos)
        
        # Add joint velocities (optionally normalized)
        if self.cfg.velocity_normalization is not None:
            joint_vel_normalized = torch.clamp(
                joint_vel / self.cfg.velocity_normalization, -1.0, 1.0
            )
            obs_list.append(joint_vel_normalized)
        else:
            obs_list.append(joint_vel)
        
        return torch.cat(obs_list, dim=-1)


@configclass
class ImpedanceStateObsTermCfg(ObservationTermCfg):
    """Configuration for impedance state observations."""
    
    func = ImpedanceStateObsTerm
    
    # Asset and action term references
    asset_name: str = "robot"
    action_term_name: str = "variable_impedance"
    
    # What to include in observations
    include_position_error: bool = True
    include_velocity_error: bool = True
    include_stiffness: bool = True
    include_damping: bool = True
    include_joint_torques: bool = False
    
    # Normalization parameters 
    stiffness_min: float = 10.0
    stiffness_max: float = 5000.0
    damping_min: float = 0.1
    damping_max: float = 100.0
    
    # Torque normalization (if including torques)
    torque_normalization: float | None = 50.0


@configclass
class ExternalForceObsTermCfg(ObservationTermCfg):
    """Configuration for external force estimation observations."""
    
    func = ExternalForceObsTerm
    
    # Asset settings
    asset_name: str = "robot"
    joint_names: list[str] | None = None
    
    # Observer parameters
    observer_gain: float = 10.0
    cutoff_freq: float = 20.0  # Hz
    estimated_inertia: float = 0.1  # kg⋅m²
    
    # What to include
    include_raw_estimates: bool = True
    include_filtered_estimates: bool = True


@configclass
class JointStateObsTermCfg(ObservationTermCfg):
    """Configuration for joint state observations."""
    
    func = JointStateObsTerm
    
    # Asset settings
    asset_name: str = "robot"
    joint_names: list[str] | None = None
    
    # Normalization settings
    normalize_positions: bool = True
    velocity_normalization: float | None = 5.0  # rad/s