"""Observation terms for learned impedance control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTerm, ObservationTermCfg
from omni.isaac.lab.assets import Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class ImpedanceStateObsTerm(ObservationTerm):
    """Observation term for current impedance parameters and tracking errors."""
    
    def __init__(self, cfg: ImpedanceStateObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get action term to access impedance parameters
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices
        self._joint_ids = self._action_term._joint_ids
        self._num_joints = len(self._joint_ids)
    
    def update(self, dt: float) -> None:
        """Update any internal buffers (optional)."""
        # Nothing to update for simple observations
        pass
    
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
        
        # Get target positions from action term's planner
        pos_targets = self._action_term._position_targets
        vel_targets = self._action_term._velocity_targets
        
        # Compute tracking errors
        pos_error = pos_targets - joint_pos
        vel_error = vel_targets - joint_vel
        
        # Add tracking errors
        if self.cfg.include_position_error:
            obs_list.append(pos_error)
        
        if self.cfg.include_velocity_error:
            obs_list.append(vel_error)
        
        # Add current impedance parameters (normalized to [-1, 1])
        if self.cfg.include_stiffness:
            stiffness_normalized = (
                2.0 * (self._action_term._stiffness - self.cfg.stiffness_min) / 
                (self.cfg.stiffness_max - self.cfg.stiffness_min) - 1.0
            )
            obs_list.append(stiffness_normalized)
        
        if self.cfg.include_damping:
            # Normalize damping based on expected range
            damping_max = 2.0 * torch.sqrt(torch.tensor(self.cfg.stiffness_max))
            damping_normalized = self._action_term._damping / damping_max
            obs_list.append(damping_normalized)
        
        # Add joint torques if requested
        if self.cfg.include_joint_torques:
            joint_torques = self._asset.data.joint_torque_target[:, self._joint_ids]
            # Normalize torques
            if self.cfg.torque_normalization is not None:
                joint_torques = joint_torques / self.cfg.torque_normalization
            obs_list.append(joint_torques)
        
        # Concatenate all observations
        return torch.cat(obs_list, dim=-1)


class TrajectoryInfoObsTerm(ObservationTerm):
    """Observation term for trajectory planner information."""
    
    def __init__(self, cfg: TrajectoryInfoObsTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get action term to access planner
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        self._planner = self._action_term._trajectory_planner
    
    @property
    def data(self) -> torch.Tensor:
        """
        Returns trajectory-related observations:
        - Current target positions
        - Current target velocities
        - Time progress in trajectory (if applicable)
        """
        obs_list = []
        
        # Add target positions
        if self.cfg.include_target_position:
            obs_list.append(self._action_term._position_targets)
        
        # Add target velocities
        if self.cfg.include_target_velocity:
            obs_list.append(self._action_term._velocity_targets)
        
        # Add trajectory phase/progress
        if self.cfg.include_trajectory_phase:
            # Normalized episode progress [0, 1]
            progress = self._env.episode_length_buf.float() / self._env.max_episode_length
            # Expand to match joint dimensions
            progress_expanded = progress.unsqueeze(-1).repeat(1, self._action_term._num_joints)
            obs_list.append(progress_expanded)
        
        return torch.cat(obs_list, dim=-1)


class JointStateObsTerm(ObservationTerm):
    """Basic joint state observations for impedance control."""
    
    def __init__(self, cfg: JointStateObsTermCfg, env: ManagerBasedEnv):
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
            joint_pos_normalized = 2.0 * (joint_pos - joint_lower) / (joint_upper - joint_lower) - 1.0
            obs_list.append(joint_pos_normalized)
        else:
            obs_list.append(joint_pos)
        
        # Add joint velocities (optionally normalized)
        if self.cfg.velocity_normalization is not None:
            joint_vel_normalized = joint_vel / self.cfg.velocity_normalization
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
    action_term_name: str = "learned_impedance"
    
    # What to include in observations
    include_position_error: bool = True
    include_velocity_error: bool = True
    include_stiffness: bool = True
    include_damping: bool = True
    include_joint_torques: bool = False
    
    # Normalization parameters (for stiffness normalization)
    stiffness_min: float = 10.0
    stiffness_max: float = 300.0
    
    # Torque normalization (if including torques)
    torque_normalization: float | None = 10.0


@configclass
class TrajectoryInfoObsTermCfg(ObservationTermCfg):
    """Configuration for trajectory information observations."""
    
    func = TrajectoryInfoObsTerm
    
    # Action term reference
    action_term_name: str = "learned_impedance"
    
    # What to include
    include_target_position: bool = True
    include_target_velocity: bool = False
    include_trajectory_phase: bool = True


@configclass
class JointStateObsTermCfg(ObservationTermCfg):
    """Configuration for joint state observations."""
    
    func = JointStateObsTerm
    
    # Asset settings
    asset_name: str = "robot"
    joint_names: list[str] | None = ["Servo1", "Servo2"]
    
    # Normalization settings
    normalize_positions: bool = True
    velocity_normalization: float | None = 5.0  # rad/s