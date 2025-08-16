# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for TwoR trajectory following."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class JointPositionCommandCfg(CommandTermCfg):
    """Configuration for joint position command term."""
    
    class_type: type[CommandTerm] = "JointPositionCommand"
    
    # Robot configuration
    asset_name: str = "robot_min"
    """Name of the robot asset."""
    
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to command."""
    
    # Command generation
    command_type: str = "sinusoidal"  # "sinusoidal", "step", or "trajectory"
    """Type of reference trajectory to generate."""
    
    # Sinusoidal trajectory parameters
    amplitude: list[float] = [0.5, 0.5]  # [rad] for each joint
    frequency: list[float] = [0.2, 0.3]  # [Hz] for each joint  
    offset: list[float] = [0.0, 1.5708]  # [rad] offset for each joint
    
    # Step trajectory parameters
    step_positions: list[float] = [0.5, 1.0]  # [rad] target positions
    step_duration: float = 2.0  # [s] time to hold each position
    
    # Command limits
    position_range: tuple[float, float] = (-1.57, 1.57)  # [rad] joint limits
    
    # Update frequency
    recompute_time: float = 0.0  # Recompute every step (set > 0 for periodic updates)


class JointPositionCommand(CommandTerm):
    """Command term that generates reference joint position trajectories.
    
    This acts as the high-level planner that provides position references
    for the impedance controller to track.
    """
    
    cfg: JointPositionCommandCfg
    
    def __init__(self, cfg: JointPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get joint indices
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
        
        self._num_joints = len(self._joint_ids)
        
        # Initialize command buffer
        self.pos_command_w = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self.vel_command_w = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        
        # Time tracking
        self._time = torch.zeros(env.num_envs, device=env.device)
        
        # Step command state
        self._step_phase = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._step_timer = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        """String representation."""
        msg = f"JointPositionCommand:\n"
        msg += f"\tCommand type: {self.cfg.command_type}\n"
        msg += f"\tJoints: {self.cfg.joint_names}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Current position command. Shape: [num_envs, num_joints]"""
        return self.pos_command_w
        
    @property
    def command_velocity(self) -> torch.Tensor:
        """Current velocity command. Shape: [num_envs, num_joints]"""
        return self.vel_command_w

    def _update_metrics(self):
        """Update command metrics (if needed for logging)."""
        pass

    def _resample_command(self, env_ids: torch.Tensor):
        """Resample command for given environment indices."""
        # Update time
        dt = self._env.step_dt
        self._time[env_ids] += dt
        
        if self.cfg.command_type == "sinusoidal":
            self._generate_sinusoidal_command(env_ids)
        elif self.cfg.command_type == "step":
            self._generate_step_command(env_ids)
        else:
            # Default: hold current position
            current_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
            self.pos_command_w[env_ids] = current_pos
            self.vel_command_w[env_ids] = 0.0

    def _generate_sinusoidal_command(self, env_ids: torch.Tensor):
        """Generate sinusoidal reference trajectory."""
        t = self._time[env_ids].unsqueeze(-1)  # [num_envs, 1]
        
        # Convert lists to tensors
        amplitude = torch.tensor(self.cfg.amplitude, device=self._env.device)
        frequency = torch.tensor(self.cfg.frequency, device=self._env.device)
        offset = torch.tensor(self.cfg.offset, device=self._env.device)
        
        # Generate sinusoidal positions
        omega = 2 * math.pi * frequency.unsqueeze(0)  # [1, num_joints]
        self.pos_command_w[env_ids] = offset + amplitude * torch.sin(omega * t)
        
        # Generate corresponding velocities
        self.vel_command_w[env_ids] = amplitude * omega * torch.cos(omega * t)
        
        # Apply position limits
        self.pos_command_w[env_ids] = torch.clamp(
            self.pos_command_w[env_ids], 
            self.cfg.position_range[0], 
            self.cfg.position_range[1]
        )

    def _generate_step_command(self, env_ids: torch.Tensor):
        """Generate step reference trajectory."""
        # Update step timer
        dt = self._env.step_dt
        self._step_timer[env_ids] += dt
        
        # Check if we need to switch to next position
        switch_mask = self._step_timer[env_ids] >= self.cfg.step_duration
        if switch_mask.any():
            switch_env_ids = env_ids[switch_mask]
            self._step_phase[switch_env_ids] = (self._step_phase[switch_env_ids] + 1) % 2
            self._step_timer[switch_env_ids] = 0.0
        
        # Set position commands based on phase
        step_positions = torch.tensor(self.cfg.step_positions, device=self._env.device)
        phase_0_mask = self._step_phase[env_ids] == 0
        phase_1_mask = self._step_phase[env_ids] == 1
        
        # Phase 0: go to step_positions
        self.pos_command_w[env_ids[phase_0_mask]] = step_positions.unsqueeze(0)
        
        # Phase 1: return to offset positions  
        offset_positions = torch.tensor(self.cfg.offset, device=self._env.device)
        self.pos_command_w[env_ids[phase_1_mask]] = offset_positions.unsqueeze(0)
        
        # Zero velocity for step commands
        self.vel_command_w[env_ids] = 0.0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset command for given environment indices."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        # Reset time tracking
        self._time[env_ids] = 0.0
        self._step_phase[env_ids] = 0
        self._step_timer[env_ids] = 0.0
        
        # Initialize with current robot position
        current_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self.pos_command_w[env_ids] = current_pos
        self.vel_command_w[env_ids] = 0.0
        
        # Generate initial command
        self._resample_command(env_ids)

    def compute(self, dt: float):
        """Compute the command for all environments."""
        # Update for all environments
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self._resample_command(env_ids)