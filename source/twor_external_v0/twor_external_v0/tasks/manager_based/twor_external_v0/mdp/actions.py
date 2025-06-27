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
class VariableImpedanceActionTermCfg(ActionTermCfg):
    """Configuration for variable impedance action term for RL autotuning."""
    
    class_type: type[ActionTerm] = "VariableImpedanceActionTerm"
    
    # Robot asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    # Joint specification - only control Servo1 and Servo2
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to control impedance for."""
    
    # RL action space configuration (stiffness and damping for 2 joints)
    stiffness_range: tuple[float, float] = (10.0, 5000.0)
    """Range for joint stiffness values [N⋅m/rad]."""
    
    damping_range: tuple[float, float] = (0.1, 100.0)
    """Range for joint damping values [N⋅m⋅s/rad]."""
    
    # Default impedance parameters
    default_stiffness: float = 1000.0
    """Default stiffness value [N⋅m/rad]."""
    
    default_damping: float = 20.0
    """Default damping value [N⋅m⋅s/rad]."""
    
    # Position command source
    position_command_source: str = "external_planner"
    """Source of position commands: 'external_planner' or 'fixed_target'."""
    
    fixed_target_positions: list[float] = [0.0, 1.5708]  # Default positions for Servo1, Servo2
    """Fixed target positions if using fixed target [rad]."""


class VariableImpedanceActionTerm(ActionTerm):
    """Variable impedance action term for RL-based impedance parameter autotuning.
    
    This action term implements Isaac Lab's variable impedance control mode where:
    - RL agent outputs stiffness and damping parameters for 2 joints (Servo1, Servo2)
    - Position commands come from external planner or fixed targets
    - Uses Isaac Lab's built-in impedance control with variable gains
    - Action space: [K1, D1, K2, D2] - 4 parameters total
    """

    cfg: VariableImpedanceActionTermCfg
    """Configuration for the action term."""

    def __init__(self, cfg: VariableImpedanceActionTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the variable impedance action term.

        Args:
            cfg: Configuration for the action term.
            env: The environment object.
        """
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._num_envs = env.num_envs
        self._device = env.device
        
        # Get joint indices for controlled joints only
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
        
        self._num_joints = len(self._joint_ids)
        assert self._num_joints == 2, f"Expected 2 joints (Servo1, Servo2), got {self._num_joints}"
        
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
        
        # Initialize command tracking
        self._desired_joint_pos = torch.zeros(
            (self._num_envs, self._num_joints), 
            device=self._device, 
            dtype=torch.float32
        )
        
        # Set initial positions
        if cfg.position_command_source == "fixed_target":
            fixed_pos = torch.tensor(cfg.fixed_target_positions, device=self._device, dtype=torch.float32)
            self._desired_joint_pos[:] = fixed_pos.unsqueeze(0).expand(self._num_envs, -1)
        else:
            # Initialize with current positions
            current_pos = self._asset.data.joint_pos[:, self._joint_ids]
            self._desired_joint_pos[:] = current_pos

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
        """Current desired joint positions."""
        return self._desired_joint_pos

    def set_external_commands(self, joint_pos: torch.Tensor) -> None:
        """Set external joint position commands from planner.
        
        Args:
            joint_pos: Desired joint positions [num_envs, 2] for [Servo1, Servo2]
        """
        if self.cfg.position_command_source == "external_planner":
            self._desired_joint_pos[:] = joint_pos

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process RL actions to update impedance parameters.

        Args:
            actions: RL actions [num_envs, 4] containing impedance parameters.
                    Format: [K1, D1, K2, D2] for [Servo1, Servo2]
                    Values are assumed to be in range [-1, 1]
        """
        # Split actions into stiffness and damping components
        stiffness_actions = actions[:, [0, 2]]  # K1, K2
        damping_actions = actions[:, [1, 3]]    # D1, D2
        
        # Map actions from [-1, 1] to parameter ranges
        self._current_stiffness = self._map_to_stiffness_range(stiffness_actions)
        self._current_damping = self._map_to_damping_range(damping_actions)

    def _map_to_stiffness_range(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] to stiffness range."""
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Linear mapping from [-1, 1] to [min, max]
        stiffness_range = self.cfg.stiffness_range[1] - self.cfg.stiffness_range[0]
        stiffness = self.cfg.stiffness_range[0] + (actions + 1.0) * 0.5 * stiffness_range
        
        return stiffness

    def _map_to_damping_range(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized actions [-1, 1] to damping range."""
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Linear mapping from [-1, 1] to [min, max]
        damping_range = self.cfg.damping_range[1] - self.cfg.damping_range[0]
        damping = self.cfg.damping_range[0] + (actions + 1.0) * 0.5 * damping_range
        
        return damping

    def apply_actions(self) -> None:
        """Apply the variable impedance control.
        
        This creates the command vector in the format expected by Isaac Lab's
        variable impedance mode: [positions, stiffness, damping]
        """
        # Create full command vector for all joints (including non-controlled ones)
        num_total_joints = self._asset.num_joints
        
        # Position commands (all joints, but only controlled joints will use impedance)
        position_commands = self._asset.data.joint_pos.clone()
        position_commands[:, self._joint_ids] = self._desired_joint_pos
        
        # Stiffness commands (only for controlled joints)
        stiffness_commands = torch.zeros((self._num_envs, num_total_joints), device=self._device)
        stiffness_commands[:, self._joint_ids] = self._current_stiffness
        
        # Damping commands (only for controlled joints)
        damping_commands = torch.zeros((self._num_envs, num_total_joints), device=self._device)
        damping_commands[:, self._joint_ids] = self._current_damping
        
        # Combine into full command vector as expected by Isaac Lab's variable impedance mode
        # Format: [positions, stiffness, damping]
        full_commands = torch.cat([
            position_commands,
            stiffness_commands, 
            damping_commands
        ], dim=-1)
        
        # Set the joint position targets with variable impedance
        # Note: This assumes the actuator is configured for variable impedance mode
        self._asset.set_joint_position_target(
            position_commands, 
            stiffness=stiffness_commands,
            damping=damping_commands,
            joint_ids=self._joint_ids
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term.

        Args:
            env_ids: Environment indices to reset. If None, reset all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
            
        # Reset impedance parameters to default values
        self._current_stiffness[env_ids] = self.cfg.default_stiffness
        self._current_damping[env_ids] = self.cfg.default_damping
        
        # Reset desired positions
        if self.cfg.position_command_source == "fixed_target":
            fixed_pos = torch.tensor(self.cfg.fixed_target_positions, device=self._device, dtype=torch.float32)
            self._desired_joint_pos[env_ids] = fixed_pos.unsqueeze(0).expand(len(env_ids), -1)
        else:
            # Reset to current positions
            current_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
            self._desired_joint_pos[env_ids] = current_pos