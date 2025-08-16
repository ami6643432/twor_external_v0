# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Variable impedance action terms for RL-based impedance autotuning."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class VariableImpedanceActionCfg(ActionTermCfg):
    """Configuration for variable impedance action term for RL autotuning."""
    
    class_type: type[ActionTerm] = "VariableImpedanceAction"
    
    asset_name: str = "robot_min"
    """Name of the robot asset."""
    
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to control impedance for."""
    
    command_term_name: str = "joint_position_command"
    """Name of the command term that provides position references."""
    
    stiffness_range: tuple[float, float] = (10.0, 5000.0)
    """Range for joint stiffness values [N⋅m/rad]."""
    
    damping_range: tuple[float, float] = (0.1, 100.0)
    """Range for joint damping values [N⋅m⋅s/rad]."""
    
    default_stiffness: float = 100.0
    """Default stiffness value [N⋅m/rad]."""
    
    default_damping: float = 30.0
    """Default damping value [N⋅m⋅s/rad]."""
    
    debug_contact_forces: bool = True
    """Whether to print contact forces during control."""
    
    contact_sensor_name: str = "contact_sensor"
    """Name of contact sensor for force feedback."""


class VariableImpedanceAction(ActionTerm):
    """Variable impedance action term for RL-based impedance parameter autotuning."""

    cfg: VariableImpedanceActionCfg

    def __init__(self, cfg: VariableImpedanceActionCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the variable impedance action term."""
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device
        
        # Get joint indices
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
        
        self._num_joints = len(self._joint_ids)
        
        # Initialize impedance parameters
        self._current_stiffness = torch.full(
            (self._num_envs, self._num_joints), 
            cfg.default_stiffness,
            device=self._device,
            dtype=torch.float32
        )
        
        self._current_damping = torch.full(
            (self._num_envs, self._num_joints),
            cfg.default_damping,
            device=self._device,
            dtype=torch.float32
        )
        
        # Get contact sensor for debugging
        self._contact_sensor = None
        if cfg.contact_sensor_name in env.scene:
            self._contact_sensor = env.scene[cfg.contact_sensor_name]
        
        # Step counter for periodic printing
        self._step_count = 0

    @property 
    def action_dim(self) -> int:
        """Dimension of the action space: 4 (2 stiffness + 2 damping)."""
        return 4
        
    @property
    def current_stiffness(self) -> torch.Tensor:
        """Current joint stiffness parameters."""
        return self._current_stiffness
        
    @property
    def current_damping(self) -> torch.Tensor:
        """Current joint damping parameters."""
        return self._current_damping
        
    @property  
    def desired_joint_positions(self) -> torch.Tensor:
        """Current desired joint positions from command manager."""
        command_term = self._env.command_manager._terms[self.cfg.command_term_name]
        return command_term.command

    @property
    def contact_forces(self) -> torch.Tensor:
        """Get current contact forces from sensor."""
        if self._contact_sensor is not None:
            forces = self._contact_sensor.data.net_forces_w
            while forces.ndim > 2:
                forces = forces.sum(dim=1)
            return forces
        return torch.zeros((self._num_envs, 3), device=self._device)

    @property
    def contact_force_magnitude(self) -> torch.Tensor:
        """Get contact force magnitude."""
        forces = self.contact_forces
        return torch.norm(forces, dim=-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process RL actions to update impedance parameters."""
        # Split actions into stiffness and damping components
        stiffness_actions = actions[:, [0, 2]]  # K1, K2
        damping_actions = actions[:, [1, 3]]    # D1, D2
        
        # Map actions from [-1, 1] to parameter ranges
        self._current_stiffness = self._map_to_stiffness_range(stiffness_actions)
        self._current_damping = self._map_to_damping_range(damping_actions)
        
        # Get desired positions from command manager
        desired_positions = self.desired_joint_positions
        
        # Apply impedance control
        self._asset.set_joint_position_target(desired_positions, joint_ids=self._joint_ids)
        
        # Debug printing
        if self.cfg.debug_contact_forces and self._step_count % 10 == 0:
            current_pos = self._asset.data.joint_pos[:, self._joint_ids]
            current_vel = self._asset.data.joint_vel[:, self._joint_ids]
            contact_forces = self.contact_forces
            force_magnitude = self.contact_force_magnitude
            
            print(f"\n--- Variable Impedance Control (Step {self._step_count}) ---")
            print(f"RL Actions (normalized): {actions[0]}")
            print(f"Current Stiffness [Nm/rad]: {self._current_stiffness[0]}")
            print(f"Current Damping [Nms/rad]: {self._current_damping[0]}")
            print(f"Desired positions [rad]: {desired_positions[0]}")
            print(f"Current positions [rad]: {current_pos[0]}")
            print(f"Position errors [rad]: {(desired_positions - current_pos)[0]}")
            print(f"Current velocities [rad/s]: {current_vel[0]}")
            print(f"Contact forces [N]: {contact_forces[0]}")
            print(f"Force magnitude [N]: {force_magnitude[0]}")
            print("-" * 60)
        
        self._step_count += 1

    def _map_to_stiffness_range(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] to stiffness range."""
        actions = torch.clamp(actions, -1.0, 1.0)
        stiffness_range = self.cfg.stiffness_range[1] - self.cfg.stiffness_range[0]
        stiffness = self.cfg.stiffness_range[0] + (actions + 1.0) * 0.5 * stiffness_range
        return stiffness

    def _map_to_damping_range(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] to damping range."""
        actions = torch.clamp(actions, -1.0, 1.0)
        damping_range = self.cfg.damping_range[1] - self.cfg.damping_range[0]
        damping = self.cfg.damping_range[0] + (actions + 1.0) * 0.5 * damping_range
        return damping

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
            
        self._current_stiffness[env_ids] = self.cfg.default_stiffness
        self._current_damping[env_ids] = self.cfg.default_damping
        
        if len(env_ids) == self._num_envs:
            self._step_count = 0