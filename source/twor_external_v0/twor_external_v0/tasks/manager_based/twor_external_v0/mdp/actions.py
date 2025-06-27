"""Action terms for learned impedance control with planned trajectories."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.assets import Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class LearnedImpedanceActionTerm(ActionTerm):
    """
    Action term that applies impedance control with learned stiffness and damping.
    Position targets come from a trajectory planner, not from RL actions.
    """
    
    cfg: LearnedImpedanceActionTermCfg
    _asset: Articulation
    
    def __init__(self, cfg: LearnedImpedanceActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get the robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Get joint indices for impedance control
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Initialize trajectory planner
        self._trajectory_planner = self._create_trajectory_planner(cfg.planner_cfg)
        
        # Storage for current targets and parameters
        self._position_targets = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._velocity_targets = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._stiffness = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        self._damping = torch.zeros(env.num_envs, self._num_joints, device=env.device)
        
        # Initialize with default values
        self.reset()
    
    def _create_trajectory_planner(self, planner_cfg):
        """Create the trajectory planner based on configuration."""
        if planner_cfg.type == "sinusoidal":
            return SinusoidalPlanner(
                num_envs=self._env.num_envs,
                num_joints=self._num_joints,
                device=self._env.device,
                frequency=planner_cfg.frequency,
                amplitude=planner_cfg.amplitude,
                phase_offset=planner_cfg.phase_offset,
            )
        elif planner_cfg.type == "waypoint":
            return WaypointPlanner(
                num_envs=self._env.num_envs,
                num_joints=self._num_joints,
                device=self._env.device,
                waypoints=planner_cfg.waypoints,
                interpolation_time=planner_cfg.interpolation_time,
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_cfg.type}")
    
    @property
    def action_dim(self) -> int:
        """Dimension of the action term: stiffness + damping for each controlled joint."""
        return self._num_joints * 2  # Only stiffness and damping
    
    def process_actions(self, actions: torch.Tensor):
        """Process raw actions into stiffness and damping parameters."""
        # Split actions into stiffness and damping
        stiffness_raw = actions[:, :self._num_joints]
        damping_raw = actions[:, self._num_joints:2*self._num_joints]
        
        # Process stiffness (ensure positive and within bounds)
        self._stiffness = (
            torch.sigmoid(stiffness_raw) * 
            (self.cfg.stiffness_max - self.cfg.stiffness_min) + 
            self.cfg.stiffness_min
        )
        
        # Process damping (ensure stability)
        if self.cfg.auto_compute_damping:
            # Critical damping: d = 2 * sqrt(k * m), assuming m=1 for simplicity
            self._damping = 2 * torch.sqrt(self._stiffness) * self.cfg.damping_ratio
        else:
            # Ensure damping is within stable bounds
            damping_min = self.cfg.damping_ratio_min * 2 * torch.sqrt(self._stiffness)
            damping_max = self.cfg.damping_ratio_max * 2 * torch.sqrt(self._stiffness)
            self._damping = (
                torch.sigmoid(damping_raw) * (damping_max - damping_min) + damping_min
            )
    
    def apply_actions(self):
        """Apply impedance control with learned parameters and planned trajectories."""
        # Get current joint states
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Update trajectory planner and get targets
        current_time = self._env.episode_length_buf * self._env.step_dt
        self._position_targets, self._velocity_targets = self._trajectory_planner.compute_targets(
            current_time, joint_pos
        )
        
        # Compute position and velocity errors
        pos_error = self._position_targets - joint_pos
        vel_error = self._velocity_targets - joint_vel
        
        # Compute impedance control torques: tau = kp * e_pos + kd * e_vel
        torques = self._stiffness * pos_error + self._damping * vel_error
        
        # Optionally add gravity compensation
        if self.cfg.gravity_compensation:
            gravity = self._asset.data.generalized_gravity[:, self._joint_ids]
            torques += gravity
        
        # Apply torque limits
        if self.cfg.torque_limit is not None:
            torques = torch.clamp(torques, -self.cfg.torque_limit, self.cfg.torque_limit)
        
        # Set the computed torques
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        
        # Reset to default impedance values
        self._stiffness[env_ids] = self.cfg.stiffness_init
        self._damping[env_ids] = 2 * torch.sqrt(self.cfg.stiffness_init) * self.cfg.damping_ratio
        
        # Reset trajectory planner
        self._trajectory_planner.reset(env_ids)


class SinusoidalPlanner:
    """Simple sinusoidal trajectory planner for testing."""
    
    def __init__(self, num_envs, num_joints, device, frequency, amplitude, phase_offset):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device
        self.frequency = torch.tensor(frequency, device=device)
        self.amplitude = torch.tensor(amplitude, device=device)
        self.phase_offset = torch.tensor(phase_offset, device=device)
        self.initial_pos = torch.zeros(num_envs, num_joints, device=device)
    
    def compute_targets(self, time, current_pos):
        """Compute position and velocity targets."""
        # Simple sinusoidal motion around initial position
        pos_targets = self.initial_pos + self.amplitude * torch.sin(
            2 * torch.pi * self.frequency * time.unsqueeze(-1) + self.phase_offset
        )
        vel_targets = self.amplitude * 2 * torch.pi * self.frequency * torch.cos(
            2 * torch.pi * self.frequency * time.unsqueeze(-1) + self.phase_offset
        )
        return pos_targets, vel_targets
    
    def reset(self, env_ids):
        """Reset planner for specified environments."""
        # Store initial position as reference
        pass  # Will be set when first called


@configclass
class LearnedImpedanceActionTermCfg(ActionTermCfg):
    """Configuration for learned impedance action term."""
    
    class_type: type[ActionTerm] = LearnedImpedanceActionTerm
    
    # Asset settings
    asset_name: str = "robot"
    joint_names: list[str] = ["Servo1", "Servo2"]  # Only control these joints
    
    # Impedance bounds
    stiffness_min: float = 10.0
    stiffness_max: float = 300.0
    stiffness_init: float = 100.0
    
    # Damping settings
    auto_compute_damping: bool = True  # If True, compute damping from stiffness
    damping_ratio: float = 1.0  # For auto-computed damping (1.0 = critical)
    damping_ratio_min: float = 0.3  # For learned damping
    damping_ratio_max: float = 2.0  # For learned damping
    
    # Control settings
    gravity_compensation: bool = True
    torque_limit: float | None = None  # Set to limit maximum torques
    
    # Trajectory planner configuration
    @configclass
    class PlannerCfg:
        """Configuration for trajectory planner."""
        type: str = "sinusoidal"  # "sinusoidal", "waypoint", etc.
        # Sinusoidal planner params
        frequency: list[float] = [0.5, 0.3]  # Hz for each joint
        amplitude: list[float] = [0.3, 0.2]  # Radians
        phase_offset: list[float] = [0.0, 1.57]  # Phase offset
        # Waypoint planner params (if used)
        waypoints: list[list[float]] | None = None
        interpolation_time: float = 2.0
    
    planner_cfg: PlannerCfg = PlannerCfg()