# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Variable impedance action terms for the MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VariableImpedanceActionTerm(ActionTerm):
    """Variable impedance action term that combines position tracking with learned impedance parameters.
    
    This action term:
    - Receives joint position commands from an external planner
    - Uses RL agent outputs for stiffness and damping parameters
    - Applies impedance control law: τ = K*(q_des - q) + D*(q̇_des - q̇)
    - Ensures stability through positive definiteness constraints
    """

    cfg: VariableImpedanceActionTermCfg
    """Configuration for the action term."""

    def __init__(self, cfg: VariableImpedanceActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the variable impedance action term.

        Args:
            cfg: Configuration for the action term.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Store the articulation asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Find the controlled joint indices (only Servo1 and Servo2)
        if cfg.joint_names is None:
            self._joint_ids = slice(None)
            self._num_joints = self._asset.num_joints
        else:
            self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
            self._num_joints = len(self._joint_ids)
        
        # Expected action dimension: [K1, D1, K2, D2] for 2 joints
        expected_action_dim = 2 * self._num_joints  # stiffness + damping for each joint
        if self.action_dim != expected_action_dim:
            raise ValueError(
                f"Expected action dimension {expected_action_dim} for {self._num_joints} joints, "
                f"got {self.action_dim}"
            )

        # Initialize impedance parameter bounds
        self._stiffness_min = torch.tensor(cfg.stiffness_min, device=self._env.device)
        self._stiffness_max = torch.tensor(cfg.stiffness_max, device=self._env.device)
        self._damping_min = torch.tensor(cfg.damping_min, device=self._env.device)
        self._damping_max = torch.tensor(cfg.damping_max, device=self._env.device)
        
        # Ensure bounds are broadcastable to joint dimensions
        if self._stiffness_min.numel() == 1:
            self._stiffness_min = self._stiffness_min.repeat(self._num_joints)
        if self._stiffness_max.numel() == 1:
            self._stiffness_max = self._stiffness_max.repeat(self._num_joints)
        if self._damping_min.numel() == 1:
            self._damping_min = self._damping_min.repeat(self._num_joints)
        if self._damping_max.numel() == 1:
            self._damping_max = self._damping_max.repeat(self._num_joints)

        # State variables for tracking
        self._desired_pos = torch.zeros(self._env.num_envs, self._num_joints, device=self._env.device)
        self._desired_vel = torch.zeros(self._env.num_envs, self._num_joints, device=self._env.device)
        self._processed_stiffness = torch.zeros(self._env.num_envs, self._num_joints, device=self._env.device)
        self._processed_damping = torch.zeros(self._env.num_envs, self._num_joints, device=self._env.device)
        
        # Initialize with default values
        self._processed_stiffness[:] = cfg.default_stiffness
        self._processed_damping[:] = cfg.default_damping

        # External planner interface
        self._planner_interface = None  # Will be set by environment if needed
        
        # Stability monitoring
        self._stability_margin = cfg.stability_margin

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the raw actions from the RL agent.

        Args:
            actions: Raw actions from agent [batch_size, action_dim]
                    Format: [K1, D1, K2, D2, ...] for each joint
        """
        # Reshape actions: [batch_size, num_joints, 2] where last dim is [K, D]
        actions_reshaped = actions.view(self._env.num_envs, self._num_joints, 2)
        
        # Extract stiffness and damping
        raw_stiffness = actions_reshaped[..., 0]  # [batch_size, num_joints]
        raw_damping = actions_reshaped[..., 1]    # [batch_size, num_joints]
        
        # Apply bounds and ensure positive definiteness
        self._processed_stiffness = torch.clamp(
            raw_stiffness, 
            min=self._stiffness_min, 
            max=self._stiffness_max
        )
        
        self._processed_damping = torch.clamp(
            raw_damping,
            min=self._damping_min,
            max=self._damping_max
        )
        
        # Ensure stability: D² < 4*M*K (for second-order systems)
        # Using estimated inertia or conservative bound
        if self.cfg.enforce_stability:
            max_stable_damping = 2.0 * torch.sqrt(self.cfg.estimated_inertia * self._processed_stiffness)
            self._processed_damping = torch.min(
                self._processed_damping,
                max_stable_damping * self._stability_margin
            )

    def apply_actions(self) -> None:
        """Apply the impedance control law to compute joint torques."""
        # Get current joint states
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Get desired positions from planner (if available) or use current as default
        if self._planner_interface is not None:
            self._desired_pos = self._planner_interface.get_desired_positions()
            self._desired_vel = self._planner_interface.get_desired_velocities()
        else:
            # For now, use current position as desired (can be overridden by environment)
            if torch.allclose(self._desired_pos, torch.zeros_like(self._desired_pos)):
                self._desired_pos = current_pos.clone()
            # Desired velocity is zero for position holding
            self._desired_vel.zero_()
        
        # Compute position and velocity errors
        pos_error = self._desired_pos - current_pos
        vel_error = self._desired_vel - current_vel
        
        # Apply impedance control law: τ = K*(q_des - q) + D*(q̇_des - q̇)
        stiffness_torque = self._processed_stiffness * pos_error
        damping_torque = self._processed_damping * vel_error
        total_torque = stiffness_torque + damping_torque
        
        # Apply torque limits for safety
        if self.cfg.torque_limit is not None:
            torque_limit = torch.tensor(self.cfg.torque_limit, device=self._env.device)
            if torque_limit.numel() == 1:
                torque_limit = torque_limit.repeat(self._num_joints)
            total_torque = torch.clamp(total_torque, -torque_limit, torque_limit)
        
        # Set the computed torques to the asset
        self._asset.set_joint_effort_target(total_torque, joint_ids=self._joint_ids)

    def set_desired_position(self, desired_pos: torch.Tensor, desired_vel: torch.Tensor | None = None) -> None:
        """Set desired position and velocity from external planner.
        
        Args:
            desired_pos: Desired joint positions [batch_size, num_joints]
            desired_vel: Desired joint velocities [batch_size, num_joints], defaults to zero
        """
        self._desired_pos = desired_pos.clone()
        if desired_vel is not None:
            self._desired_vel = desired_vel.clone()
        else:
            self._desired_vel.zero_()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the action term for specified environment IDs.
        
        Args:
            env_ids: Environment IDs to reset.
        """
        # Reset to default impedance parameters
        self._processed_stiffness[env_ids] = self.cfg.default_stiffness
        self._processed_damping[env_ids] = self.cfg.default_damping
        
        # Reset desired positions to current positions
        current_pos = self._asset.data.joint_pos[env_ids, self._joint_ids]
        self._desired_pos[env_ids] = current_pos
        self._desired_vel[env_ids] = 0.0

    @property
    def current_stiffness(self) -> torch.Tensor:
        """Get current stiffness parameters."""
        return self._processed_stiffness.clone()

    @property
    def current_damping(self) -> torch.Tensor:
        """Get current damping parameters."""
        return self._processed_damping.clone()

    @property
    def desired_positions(self) -> torch.Tensor:
        """Get current desired positions."""
        return self._desired_pos.clone()

    @property
    def control_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current position and velocity errors."""
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        pos_error = self._desired_pos - current_pos
        vel_error = self._desired_vel - current_vel
        return pos_error, vel_error


@configclass
class VariableImpedanceActionTermCfg(ActionTermCfg):
    """Configuration for variable impedance action term."""

    class_type: type[ActionTerm] = VariableImpedanceActionTerm
    """Class type for the action term."""

    # Joint specification
    joint_names: list[str] | None = None
    """Names of joints to control. If None, all joints are controlled."""

    # Impedance parameter bounds
    stiffness_min: float | list[float] = 10.0
    """Minimum stiffness value(s) [N⋅m/rad]."""
    
    stiffness_max: float | list[float] = 5000.0
    """Maximum stiffness value(s) [N⋅m/rad]."""
    
    damping_min: float | list[float] = 0.1
    """Minimum damping value(s) [N⋅m⋅s/rad]."""
    
    damping_max: float | list[float] = 100.0
    """Maximum damping value(s) [N⋅m⋅s/rad]."""

    # Default values
    default_stiffness: float | list[float] = 1000.0
    """Default stiffness value(s) [N⋅m/rad]."""
    
    default_damping: float | list[float] = 20.0
    """Default damping value(s) [N⋅m⋅s/rad]."""

    # Safety and stability
    torque_limit: float | list[float] | None = None
    """Maximum torque output [N⋅m]. If None, no limit is applied."""
    
    enforce_stability: bool = True
    """Whether to enforce stability constraints on damping."""
    
    estimated_inertia: float | list[float] = 0.1
    """Estimated joint inertia for stability calculations [kg⋅m²]."""
    
    stability_margin: float = 0.8
    """Safety margin for stability constraint (< 1.0)."""