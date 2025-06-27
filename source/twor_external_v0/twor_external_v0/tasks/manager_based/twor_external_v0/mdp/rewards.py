# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for variable impedance control RL training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.managers import RewardTerm, RewardTermCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class TrackingPerformanceRewardCfg(RewardTermCfg):
    """Configuration for tracking performance reward."""
    
    class_type: type[RewardTerm] = "TrackingPerformanceReward"
    
    # Asset and action term names
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    action_term_name: str = "impedance_action"
    """Name of the variable impedance action term."""
    
    # Reward weighting
    weight: float = 1.0
    """Weight for this reward term."""
    
    # Tracking tolerances
    position_tolerance: float = 0.1
    """Position tracking tolerance [rad]."""
    
    velocity_tolerance: float = 1.0
    """Velocity tracking tolerance [rad/s]."""
    
    # Reward shaping parameters
    use_exponential_reward: bool = True
    """Use exponential reward shaping vs linear."""
    
    exponential_scale: float = 10.0
    """Scale factor for exponential reward."""


class TrackingPerformanceReward(RewardTerm):
    """Reward term for position and velocity tracking performance.
    
    Encourages the RL agent to learn impedance parameters that provide
    good tracking of desired joint positions and velocities.
    """
    
    cfg: TrackingPerformanceRewardCfg
    
    def __init__(self, cfg: TrackingPerformanceRewardCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get action term
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices
        self._joint_ids = self._action_term._joint_ids
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute tracking performance reward."""
        
        # Get current and desired positions/velocities
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        desired_pos = self._action_term.desired_joint_positions
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Compute tracking errors
        pos_error = torch.abs(desired_pos - current_pos)
        vel_error = torch.abs(current_vel)  # Assuming desired velocity is zero
        
        # Compute normalized errors
        pos_error_norm = pos_error / self.cfg.position_tolerance
        vel_error_norm = vel_error / self.cfg.velocity_tolerance
        
        # Combined tracking error
        total_error = torch.mean(pos_error_norm + vel_error_norm, dim=-1)
        
        if self.cfg.use_exponential_reward:
            # Exponential reward: high reward for low error, quickly decreasing
            reward = torch.exp(-self.cfg.exponential_scale * total_error)
        else:
            # Linear reward: 1.0 - normalized_error
            reward = torch.clamp(1.0 - total_error, min=0.0)
        
        return reward


@configclass
class ForceStabilityRewardCfg(RewardTermCfg):
    """Configuration for force stability reward."""
    
    class_type: type[RewardTerm] = "ForceStabilityReward"
    
    # Sensor configuration
    contact_sensor_names: list[str] = ["contact_sensor"]
    """Names of contact sensors to monitor."""
    
    # Stability criteria
    max_force_variation: float = 20.0
    """Maximum acceptable force variation for stability [N]."""
    
    target_force_range: tuple[float, float] = (5.0, 50.0)
    """Acceptable range of contact forces [N]."""
    
    # Reward parameters
    weight: float = 1.0
    """Weight for this reward term."""
    
    stability_window: int = 10
    """Number of timesteps to consider for force stability."""


class ForceStabilityReward(RewardTerm):
    """Reward term for stable force interactions.
    
    Encourages impedance parameters that result in stable contact forces
    without excessive oscillations or force spikes.
    """
    
    cfg: ForceStabilityRewardCfg
    
    def __init__(self, cfg: ForceStabilityRewardCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get contact sensors
        self._contact_sensors = []
        for sensor_name in cfg.contact_sensor_names:
            if sensor_name in env.scene:
                self._contact_sensors.append(env.scene[sensor_name])
        
        # Force history for stability assessment
        self._force_history = torch.zeros(
            (env.num_envs, cfg.stability_window),
            device=env.device
        )
        self._history_idx = 0
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute force stability reward."""
        
        if not self._contact_sensors:
            return torch.zeros(env.num_envs, device=env.device)
        
        # Get current contact force magnitude
        total_force = torch.zeros(env.num_envs, device=env.device)
        for sensor in self._contact_sensors:
            contact_forces = sensor.data.net_forces_w
            force_magnitude = torch.norm(contact_forces, dim=-1)
            total_force += force_magnitude
        
        # Update force history
        self._force_history[:, self._history_idx] = total_force
        self._history_idx = (self._history_idx + 1) % self.cfg.stability_window
        
        # Compute force stability (low variance = high stability)
        force_variance = torch.var(self._force_history, dim=-1)
        stability_score = torch.exp(-force_variance / self.cfg.max_force_variation)
        
        # Compute force range penalty (reward forces within acceptable range)
        in_range_bonus = torch.zeros_like(total_force)
        in_range = (total_force >= self.cfg.target_force_range[0]) & \
                   (total_force <= self.cfg.target_force_range[1])
        in_range_bonus[in_range] = 1.0
        
        # Combined reward
        reward = stability_score * (0.5 + 0.5 * in_range_bonus)
        
        return reward


@configclass
class ImpedanceParameterRegularizationCfg(RewardTermCfg):
    """Configuration for impedance parameter regularization."""
    
    class_type: type[RewardTerm] = "ImpedanceParameterRegularization"
    
    # Action term
    action_term_name: str = "impedance_action"
    """Name of the variable impedance action term."""
    
    # Regularization parameters
    weight: float = 0.01
    """Weight for this penalty term (typically small)."""
    
    stiffness_penalty_scale: float = 1e-6
    """Scale for stiffness magnitude penalty."""
    
    damping_penalty_scale: float = 1e-4
    """Scale for damping magnitude penalty."""
    
    parameter_change_penalty: float = 1e-3
    """Penalty for rapid parameter changes."""


class ImpedanceParameterRegularization(RewardTerm):
    """Regularization term for impedance parameters.
    
    Encourages reasonable impedance parameter values and smooth changes
    to avoid aggressive or unstable parameter selections.
    """
    
    cfg: ImpedanceParameterRegularizationCfg
    
    def __init__(self, cfg: ImpedanceParameterRegularizationCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get action term
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Previous parameters for change detection
        self._prev_stiffness = None
        self._prev_damping = None
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute parameter regularization penalty."""
        
        current_stiffness = self._action_term.current_stiffness
        current_damping = self._action_term.current_damping
        
        # Parameter magnitude penalties (encourage reasonable values)
        stiffness_penalty = self.cfg.stiffness_penalty_scale * torch.mean(current_stiffness**2, dim=-1)
        damping_penalty = self.cfg.damping_penalty_scale * torch.mean(current_damping**2, dim=-1)
        
        # Parameter change penalty (encourage smooth changes)
        change_penalty = torch.zeros(env.num_envs, device=env.device)
        if self._prev_stiffness is not None:
            stiffness_change = torch.mean((current_stiffness - self._prev_stiffness)**2, dim=-1)
            damping_change = torch.mean((current_damping - self._prev_damping)**2, dim=-1)
            change_penalty = self.cfg.parameter_change_penalty * (stiffness_change + damping_change)
        
        # Store current parameters for next iteration
        self._prev_stiffness = current_stiffness.clone()
        self._prev_damping = current_damping.clone()
        
        # Return negative penalty as reward
        total_penalty = stiffness_penalty + damping_penalty + change_penalty
        return -total_penalty


@configclass
class ContactQualityRewardCfg(RewardTermCfg):
    """Configuration for contact quality reward."""
    
    class_type: type[RewardTerm] = "ContactQualityReward"
    
    # Sensor configuration
    contact_sensor_names: list[str] = ["contact_sensor"]
    """Names of contact sensors to monitor."""
    
    # Quality criteria
    desired_force_magnitude: float = 20.0
    """Desired contact force magnitude [N]."""
    
    force_tolerance: float = 10.0
    """Tolerance around desired force [N]."""
    
    # Reward parameters
    weight: float = 1.0
    """Weight for this reward term."""


class ContactQualityReward(RewardTerm):
    """Reward term for maintaining desired contact quality.
    
    Rewards the agent for maintaining contact forces close to a desired
    magnitude, which indicates good impedance tuning for the task.
    """
    
    cfg: ContactQualityRewardCfg
    
    def __init__(self, cfg: ContactQualityRewardCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get contact sensors
        self._contact_sensors = []
        for sensor_name in cfg.contact_sensor_names:
            if sensor_name in env.scene:
                self._contact_sensors.append(env.scene[sensor_name])
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute contact quality reward."""
        
        if not self._contact_sensors:
            return torch.zeros(env.num_envs, device=env.device)
        
        # Get current contact force magnitude
        total_force = torch.zeros(env.num_envs, device=env.device)
        for sensor in self._contact_sensors:
            contact_forces = sensor.data.net_forces_w
            force_magnitude = torch.norm(contact_forces, dim=-1)
            total_force += force_magnitude
        
        # Compute force error from desired
        force_error = torch.abs(total_force - self.cfg.desired_force_magnitude)
        
        # Reward decreases with distance from desired force
        normalized_error = force_error / self.cfg.force_tolerance
        reward = torch.exp(-normalized_error)
        
        return reward


@configclass
class EffortEfficiencyRewardCfg(RewardTermCfg):
    """Configuration for effort efficiency reward."""
    
    class_type: type[RewardTerm] = "EffortEfficiencyReward"
    
    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    joint_names: list[str] = ["Servo1", "Servo2"]
    """Names of joints to monitor efforts for."""
    
    # Efficiency parameters
    weight: float = 0.1
    """Weight for this reward term."""
    
    max_acceptable_effort: float = 50.0
    """Maximum acceptable joint effort [N⋅m]."""


class EffortEfficiencyReward(RewardTerm):
    """Reward term for efficient control effort.
    
    Encourages impedance parameters that achieve good performance
    without excessive joint efforts/torques.
    """
    
    cfg: EffortEfficiencyRewardCfg
    
    def __init__(self, cfg: EffortEfficiencyRewardCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get joint indices
        self._joint_ids = []
        for joint_name in cfg.joint_names:
            joint_idx, _ = self._asset.find_joints(joint_name)
            self._joint_ids.extend(joint_idx)
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute effort efficiency reward."""
        
        # Get joint efforts
        joint_efforts = self._asset.data.applied_torque[:, self._joint_ids]
        
        # Compute effort magnitude
        effort_magnitude = torch.norm(joint_efforts, dim=-1)
        
        # Normalize by maximum acceptable effort
        normalized_effort = effort_magnitude / self.cfg.max_acceptable_effort
        
        # Reward low effort (efficiency)
        reward = torch.exp(-normalized_effort)
        
        return reward


@configclass
class TaskCompletionRewardCfg(RewardTermCfg):
    """Configuration for task completion reward."""
    
    class_type: type[RewardTerm] = "TaskCompletionReward"
    
    # Action term
    action_term_name: str = "impedance_action"
    """Name of the variable impedance action term."""
    
    # Asset configuration
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    # Completion criteria
    position_threshold: float = 0.05
    """Position threshold for task completion [rad]."""
    
    velocity_threshold: float = 0.1
    """Velocity threshold for task completion [rad/s]."""
    
    completion_time_requirement: int = 50
    """Number of timesteps to maintain completion state."""
    
    # Reward parameters
    weight: float = 10.0
    """Weight for task completion bonus."""


class TaskCompletionReward(RewardTerm):
    """Reward term for successful task completion.
    
    Provides a large bonus when the robot successfully reaches and
    maintains the target position with low velocity.
    """
    
    cfg: TaskCompletionRewardCfg
    
    def __init__(self, cfg: TaskCompletionRewardCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get robot asset and action term
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Get joint indices
        self._joint_ids = self._action_term._joint_ids
        
        # Completion state tracking
        self._completion_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute task completion reward."""
        
        # Get current state
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        desired_pos = self._action_term.desired_joint_positions
        
        # Check completion criteria
        pos_error = torch.norm(desired_pos - current_pos, dim=-1)
        vel_magnitude = torch.norm(current_vel, dim=-1)
        
        is_complete = (pos_error < self.cfg.position_threshold) & \
                     (vel_magnitude < self.cfg.velocity_threshold)
        
        # Update completion counter
        self._completion_counter[is_complete] += 1
        self._completion_counter[~is_complete] = 0
        
        # Reward if completion criteria met for required duration
        completion_achieved = self._completion_counter >= self.cfg.completion_time_requirement
        
        reward = completion_achieved.float()
        
        return reward