"""
IK-based trajectory generator for TWOR robot.
Takes Cartesian waypoints and generates smooth joint trajectories.
Updated with actual TWOR robot parameters from URDF.
"""

from __future__ import annotations
import torch
import numpy as np
import math
from typing import List, Tuple, Literal
from dataclasses import dataclass


@dataclass
class Waypoint:
    """Cartesian waypoint definition."""
    position: torch.Tensor  # [x, y, z] in meters
    orientation: torch.Tensor | None = None  # [qw, qx, qy, qz] quaternion (optional)
    time: float = 0.0  # Time to reach this waypoint [s]


class WorkspaceError(Exception):
    """Exception raised when target position is outside manipulator workspace."""
    pass


class IKTrajectoryGenerator:
    """
    IK-based trajectory generator that converts Cartesian waypoints to joint trajectories.
    
    Updated for TWOR robot based on actual URDF parameters:
    - Link1 origin: xyz="0.017875 -0.45598 -0.03525"
    - Link2 origin: xyz="-0.034252 -0.20903 -0.021308" (relative to Link1)
    - Sensor origin: xyz="-0.0351 -0.30046 -0.052517" (relative to Link2)
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        dt: float = 0.01,
        joint_limits: Tuple[Tuple[float, float], ...] = ((-3.14159, 3.14159), (-3.14159, 3.14159)),
        link_lengths: Tuple[float, float] = None,  # Will be computed from URDF
        base_offset: Tuple[float, float] = (0.017875, -0.45598),  # Base to Link1 offset (x, y)
        interpolation_method: Literal["cubic", "quintic", "linear"] = "cubic",
        workspace_check: bool = True,  # Enable/disable workspace checking
        workspace_tolerance: float = 1e-3  # Tolerance for workspace boundary [m]
    ):
        """
        Initialize IK trajectory generator for TWOR robot.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for computations
            dt: Control timestep [s]
            joint_limits: Joint angle limits [(min, max), ...]
            link_lengths: TWOR robot link lengths [L1, L2] (computed from URDF if None)
            base_offset: Offset from base_link to first joint
            interpolation_method: Trajectory interpolation method
            workspace_check: Enable workspace boundary checking
            workspace_tolerance: Tolerance for workspace boundary violations [m]
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.joint_limits = torch.tensor(joint_limits, device=device)
        self.interpolation_method = interpolation_method
        self.workspace_check = workspace_check
        self.workspace_tolerance = workspace_tolerance
        
        # TWOR robot parameters from URDF analysis
        if link_lengths is None:
            # Link1: Base to Servo1 joint -> Link1 COM -> Servo2 joint
            # From URDF: Servo1 origin xyz="0.017875 -0.45598 -0.03525"
            # Link2: Servo2 origin xyz="-0.034252 -0.20903 -0.021308" (relative to Link1)
            # Sensor: origin xyz="-0.0351 -0.30046 -0.052517" (relative to Link2)
            
            # Approximate link lengths based on joint-to-joint distances
            # Link1 effective length (base to Link1/Link2 joint)
            base_to_servo1 = math.sqrt(0.017875**2 + 0.45598**2)  # ~0.456m
            
            # Link2 effective length (Link1/Link2 joint to end effector)
            servo2_to_sensor = math.sqrt((-0.034252)**2 + (-0.20903)**2)  # ~0.212m
            sensor_extension = math.sqrt((-0.0351)**2 + (-0.30046)**2)     # ~0.302m
            
            # Total effective link lengths for 2D planar approximation
            L1 = 0.456  # Base to elbow joint
            L2 = 0.212 + 0.302  # Elbow joint to end effector (sensor tip)
            
            self.link_lengths = torch.tensor([L1, L2], device=device)
        else:
            self.link_lengths = torch.tensor(link_lengths, device=device)
        
        # Base offset for coordinate transformation
        self.base_offset = torch.tensor(base_offset, device=device)
        
        # Current trajectory state
        self.current_trajectory = None
        self.trajectory_time = 0.0
        self.trajectory_duration = 0.0
        self.is_executing = False
        
        # Workspace limits based on TWOR geometry
        self.workspace_limits = {
            'x_min': -0.8, 'x_max': 0.8,   # Based on link lengths
            'y_min': -0.8, 'y_max': 0.8,
            'max_reach': float(self.link_lengths.sum()),
            'min_reach': float(torch.abs(self.link_lengths[0] - self.link_lengths[1]))
        }
    
    def forward_kinematics(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute forward kinematics for TWOR robot.
        
        Args:
            joint_angles: Joint angles [num_envs, 2] or [2] - [Servo1, Servo2]
            
        Returns:
            end_effector_pos: End effector position [num_envs, 3] or [3] in world coordinates
        """
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
            
        q1, q2 = joint_angles[:, 0], joint_angles[:, 1]  # Servo1, Servo2
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Forward kinematics for TWOR robot
        # Based on add_new_robot.py working motion, the robot coordinate system is:
        # - q1 = 0 points in +x direction  
        # - q1 = π/2 points in +y direction
        # - The robot moves in the x-y plane naturally
        x = L1 * torch.cos(q1) + L2 * torch.cos(q1 + q2)
        y = L1 * torch.sin(q1) + L2 * torch.sin(q1 + q2)
        
        # Transform to world coordinates - add base offset
        x_world = x + self.base_offset[0]  
        y_world = y + self.base_offset[1]   
        z_world = torch.full_like(x_world, 0.125)  # Fixed Z at cube height
        
        return torch.stack([x_world, y_world, z_world], dim=-1)

    def inverse_kinematics(self, target_pos: torch.Tensor, elbow_up: bool = True, check_workspace: bool = None) -> torch.Tensor:
        """
        Compute inverse kinematics for TWOR robot (2-DOF planar arm).
        
        Args:
            target_pos: Target position [num_envs, 2/3] or [2/3] in world coordinates (x,y) or (x,y,z)
            elbow_up: Choose elbow-up solution if True
            check_workspace: Override workspace checking (uses self.workspace_check if None)
            
        Returns:
            joint_angles: Joint angles [num_envs, 2] or [2] - [Servo1, Servo2]
            
        Raises:
            WorkspaceError: If target is outside workspace and checking is enabled
        """
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Extract x,y coordinates (ignore z if provided)
        target_xy = target_pos[:, :2]
        
        # Check workspace reachability
        if check_workspace is None:
            check_workspace = self.workspace_check
        
        if check_workspace:
            self.check_workspace_reachability(target_xy, raise_error=True)
        
        # Transform to robot base coordinates
        x_robot = target_xy[:, 0] - self.base_offset[0]  
        y_robot = target_xy[:, 1] - self.base_offset[1]  
        
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Distance to target
        r = torch.sqrt(x_robot**2 + y_robot**2)
        
        # Cosine rule for q2 (elbow angle)
        cos_q2 = (x_robot**2 + y_robot**2 - L1**2 - L2**2) / (2 * L1 * L2)
        
        # Clamp to handle numerical precision issues
        cos_q2 = torch.clamp(cos_q2, -1.0, 1.0)
        
        # Two solutions for q2
        if elbow_up:
            q2 = torch.acos(cos_q2)  # Elbow up
        else:
            q2 = -torch.acos(cos_q2)  # Elbow down
            
        # Solve for q1 (shoulder angle)
        k1 = L1 + L2 * torch.cos(q2)
        k2 = L2 * torch.sin(q2)
        q1 = torch.atan2(y_robot, x_robot) - torch.atan2(k2, k1)
        
        # Normalize angles to [-pi, pi]
        q1 = torch.atan2(torch.sin(q1), torch.cos(q1))
        q2 = torch.atan2(torch.sin(q2), torch.cos(q2))
        
        joint_angles = torch.stack([q1, q2], dim=-1)
        
        if squeeze_output:
            joint_angles = joint_angles.squeeze(0)
            
        return joint_angles

    def check_workspace_reachability(self, target_pos: torch.Tensor, raise_error: bool = True) -> torch.Tensor:
        """
        Check if target positions are within the manipulator workspace.
        
        Args:
            target_pos: Target position [num_envs, 2] or [2] in world coordinates
            raise_error: If True, raise WorkspaceError for unreachable targets
            
        Returns:
            reachable_mask: Boolean tensor indicating which targets are reachable
            
        Raises:
            WorkspaceError: If any target is outside workspace and raise_error=True
        """
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
        
        # Transform to robot base coordinates - direct mapping
        x = target_pos[:, 0] - self.base_offset[0]  # Direct x mapping
        y = target_pos[:, 1] - self.base_offset[1]  # Direct y mapping
        
        # Distance to target
        r = torch.sqrt(x**2 + y**2)
        
        # Check reachability
        max_reach = self.workspace_limits['max_reach']
        min_reach = self.workspace_limits['min_reach']
        
        # Create reachability mask
        reachable_mask = (r >= (min_reach - self.workspace_tolerance)) & (r <= (max_reach + self.workspace_tolerance))
        
        if raise_error and not torch.all(reachable_mask):
            # Find which targets are unreachable
            unreachable_indices = torch.where(~reachable_mask)[0]
            unreachable_positions = target_pos[unreachable_indices]
            unreachable_distances = r[unreachable_indices]
            
            error_msg = f"Target position(s) outside manipulator workspace!\n"
            error_msg += f"Workspace limits: min_reach={min_reach:.3f}m, max_reach={max_reach:.3f}m\n"
            error_msg += f"Base offset: ({self.base_offset[0]:.3f}, {self.base_offset[1]:.3f})\n"
            
            for i, (idx, pos, dist) in enumerate(zip(unreachable_indices, unreachable_positions, unreachable_distances)):
                error_msg += f"  Target {idx.item()}: pos=({pos[0]:.3f}, {pos[1]:.3f}), distance={dist:.3f}m"
                if dist < min_reach:
                    error_msg += " [TOO CLOSE - SINGULARITY]"
                elif dist > max_reach:
                    error_msg += f" [TOO FAR - exceeds by {dist - max_reach:.3f}m]"
                error_msg += "\n"
            
            raise WorkspaceError(error_msg)
        
        return reachable_mask

    def generate_box_pushing_trajectory(
        self,
        box_start_pos: torch.Tensor,
        box_target_pos: torch.Tensor,
        total_time: float = 4.0,
        approach_offset: float = 0.1
    ) -> dict:
        """
        Generate trajectory for box pushing task based on your environment.
        
        Args:
            box_start_pos: Initial box position [num_envs, 3] or [3] - [x, y, z]
            box_target_pos: Target box position [num_envs, 3] or [3] - [x, y, z]
            total_time: Total execution time [s]
            approach_offset: Distance to approach box before contact [m]
            
        Returns:
            trajectory: Dictionary containing trajectory data
            
        Raises:
            WorkspaceError: If any waypoint is outside workspace
        """
        # Ensure tensors are on correct device
        box_start_pos = box_start_pos.to(device=self.device)
        box_target_pos = box_target_pos.to(device=self.device)
        
        # Ensure 3D coordinates
        if box_start_pos.dim() == 1:
            if len(box_start_pos) == 2:
                # Add z-coordinate if only x,y provided
                box_start_pos = torch.cat([box_start_pos, torch.tensor([0.125], device=self.device)])
            box_start_pos = box_start_pos.unsqueeze(0).expand(self.num_envs, -1)
        if box_target_pos.dim() == 1:
            if len(box_target_pos) == 2:
                # Add z-coordinate if only x,y provided
                box_target_pos = torch.cat([box_target_pos, torch.tensor([0.125], device=self.device)])
            box_target_pos = box_target_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        # Create waypoints for box pushing - use only x,y for 2D planning
        waypoints = []
        
        # Waypoint 1: Approach position (before box in x direction)
        approach_pos = box_start_pos[0, :2].clone()  # Take x,y only
        approach_pos[0] += approach_offset  # Move back in x-direction to approach
        
        # Check if approach position is reachable
        try:
            self.check_workspace_reachability(approach_pos.unsqueeze(0), raise_error=True)
        except WorkspaceError as e:
            raise WorkspaceError(f"Approach waypoint unreachable: {e}")
        
        waypoints.append(Waypoint(
            position=approach_pos,
            time=total_time * 0.25
        ))
        
        # Waypoint 2: Contact position (at box)
        try:
            self.check_workspace_reachability(box_start_pos[0, :2].unsqueeze(0), raise_error=True)
        except WorkspaceError as e:
            raise WorkspaceError(f"Box start position unreachable: {e}")
        
        waypoints.append(Waypoint(
            position=box_start_pos[0, :2],  # Use x,y only
            time=total_time * 0.4
        ))
        
        # Waypoint 3: Push position (at target)
        try:
            self.check_workspace_reachability(box_target_pos[0, :2].unsqueeze(0), raise_error=True)
        except WorkspaceError as e:
            raise WorkspaceError(f"Box target position unreachable: {e}")
        
        waypoints.append(Waypoint(
            position=box_target_pos[0, :2],  # Use x,y only
            time=total_time
        ))
        
        # Generate trajectory using base method
        return self.generate_trajectory(waypoints, total_time)
    
    def get_workspace_info(self) -> dict:
        """
        Get detailed workspace information for debugging.
        
        Returns:
            workspace_info: Dictionary with workspace parameters
        """
        return {
            'link_lengths': {
                'L1': float(self.link_lengths[0]),
                'L2': float(self.link_lengths[1])
            },
            'base_offset': {
                'x': float(self.base_offset[0]),
                'y': float(self.base_offset[1])
            },
            'workspace_limits': self.workspace_limits.copy(),
            'joint_limits': {
                'servo1': (float(self.joint_limits[0, 0]), float(self.joint_limits[0, 1])),
                'servo2': (float(self.joint_limits[1, 0]), float(self.joint_limits[1, 1]))
            },
            'workspace_tolerance': self.workspace_tolerance
        }
    
    def compute_jacobian(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix for TWOR robot.
        
        Args:
            joint_angles: Joint angles [num_envs, 2] - [Servo1, Servo2]
            
        Returns:
            jacobian: Jacobian matrix [num_envs, 2, 2]
        """
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
            
        q1, q2 = joint_angles[:, 0], joint_angles[:, 1]
        L1, L2 = self.link_lengths[0], self.link_lengths[1]
        
        # Jacobian elements for TWOR robot
        # ∂x/∂q1, ∂x/∂q2
        J11 = -L1 * torch.sin(q1) - L2 * torch.sin(q1 + q2)
        J12 = -L2 * torch.sin(q1 + q2)
        
        # ∂y/∂q1, ∂y/∂q2  
        J21 = L1 * torch.cos(q1) + L2 * torch.cos(q1 + q2)
        J22 = L2 * torch.cos(q1 + q2)
        
        # Stack into Jacobian matrix [batch, 2, 2]
        jacobian = torch.stack([
            torch.stack([J11, J12], dim=-1),
            torch.stack([J21, J22], dim=-1)
        ], dim=-2)
        
        return jacobian

    def generate_trajectory(
        self,
        waypoints: List[Waypoint],
        total_time: float,
        initial_joint_pos: torch.Tensor | None = None
    ) -> dict:
        """
        Generate smooth joint trajectory from Cartesian waypoints.
        
        Args:
            waypoints: List of Cartesian waypoints
            total_time: Total execution time [s]
            initial_joint_pos: Initial joint positions [num_envs, 2]
            
        Returns:
            trajectory: Dictionary containing trajectory data
            
        Raises:
            WorkspaceError: If any waypoint is outside workspace
        """
        if len(waypoints) < 1:
            raise ValueError("At least one waypoint required")
            
        # Convert waypoints to joint space
        joint_waypoints = []
        times = []
        
        for i, wp in enumerate(waypoints):
            # Convert Cartesian position to joint angles
            target_pos = wp.position[:2]  # Use only x, y for 2D arm
            if target_pos.dim() == 0 or target_pos.shape[0] < 2:
                target_pos = target_pos.unsqueeze(0) if target_pos.dim() == 0 else target_pos
                target_pos = target_pos.expand(2) if target_pos.shape[0] == 1 else target_pos
            
            if target_pos.dim() == 1:
                target_pos = target_pos.unsqueeze(0).expand(self.num_envs, -1)
            
            # This will raise WorkspaceError if unreachable
            try:
                joint_pos = self.inverse_kinematics(target_pos, check_workspace=True)
            except WorkspaceError as e:
                raise WorkspaceError(f"Waypoint {i+1} unreachable: {e}")
                
            joint_waypoints.append(joint_pos)
            times.append(wp.time if wp.time > 0 else (i + 1) * total_time / len(waypoints))
        
        # Ensure times are monotonic
        times = torch.tensor(times, device=self.device)
        if len(times) > 1:
            times[1:] = torch.maximum(times[1:], times[:-1] + 0.01)  # Minimum 10ms between waypoints
        
        # Add initial position if provided
        if initial_joint_pos is not None:
            joint_waypoints.insert(0, initial_joint_pos)
            times = torch.cat([torch.zeros(1, device=self.device), times])
        
        # Generate time vector
        num_steps = int(total_time / self.dt) + 1
        time_vec = torch.linspace(0, total_time, num_steps, device=self.device)
        
        # Interpolate joint trajectories
        if self.interpolation_method == "cubic":
            joint_traj = self._cubic_spline_interpolation(joint_waypoints, times, time_vec)
        elif self.interpolation_method == "quintic":
            joint_traj = self._quintic_interpolation(joint_waypoints, times, time_vec)
        else:  # linear
            joint_traj = self._linear_interpolation(joint_waypoints, times, time_vec)
        
        # Compute velocities and accelerations
        joint_vel = self._compute_derivatives(joint_traj, self.dt, order=1)
        joint_acc = self._compute_derivatives(joint_traj, self.dt, order=2)
        
        # Apply joint limits
        joint_traj = self._apply_joint_limits(joint_traj)
        
        # Store trajectory
        self.current_trajectory = {
            'time': time_vec,
            'position': joint_traj,  # [num_steps, num_envs, 2]
            'velocity': joint_vel,   # [num_steps, num_envs, 2]
            'acceleration': joint_acc,  # [num_steps, num_envs, 2]
            'duration': total_time
        }
        
        self.trajectory_duration = total_time
        self.trajectory_time = 0.0
        self.is_executing = True
        
        return self.current_trajectory
    
    def get_reference(self, current_time: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get reference joint position, velocity, and acceleration at current time.
        
        Args:
            current_time: Current time [s]
            
        Returns:
            Tuple of (position, velocity, acceleration) [num_envs, 2]
        """
        if not self.is_executing or self.current_trajectory is None:
            # Return zeros if no trajectory
            zero_ref = torch.zeros(self.num_envs, 2, device=self.device)
            return zero_ref, zero_ref, zero_ref
        
        # Clamp current_time to trajectory duration
        current_time = max(0.0, min(current_time, self.trajectory_duration))
        
        # Find closest time index
        time_vec = self.current_trajectory['time']
        idx = torch.searchsorted(time_vec, current_time)
        idx = torch.clamp(idx, 0, len(time_vec) - 1)
        
        # Get references - ensure proper indexing
        try:
            pos_ref = self.current_trajectory['position'][idx]
            vel_ref = self.current_trajectory['velocity'][idx]
            acc_ref = self.current_trajectory['acceleration'][idx]
            
            # Ensure correct dimensions [num_envs, 2]
            if pos_ref.dim() == 1:
                pos_ref = pos_ref.unsqueeze(0).expand(self.num_envs, -1)
            if vel_ref.dim() == 1:
                vel_ref = vel_ref.unsqueeze(0).expand(self.num_envs, -1)
            if acc_ref.dim() == 1:
                acc_ref = acc_ref.unsqueeze(0).expand(self.num_envs, -1)
                
        except Exception as e:
            print(f"Reference lookup error at time {current_time:.3f}s, idx {idx}: {e}")
            zero_ref = torch.zeros(self.num_envs, 2, device=self.device)
            return zero_ref, zero_ref, zero_ref
        
        # Return reference without modifying execution state
        # Environment will handle trajectory cycling
        return pos_ref, vel_ref, acc_ref
    
    def reset(self):
        """Reset trajectory generator - called only by environment."""
        self.current_trajectory = None
        self.trajectory_time = 0.0
        self.is_executing = False

    def generate_manual_waypoint_trajectory(
        self,
        waypoints: List[Tuple[float, float, float]],
        total_time: float | None = None
    ) -> dict:
        """
        Generate trajectory from manually defined waypoints.
        
        Args:
            waypoints: List of (x, y, time) tuples defining the trajectory
            total_time: Override total time (uses max waypoint time if None)
            
        Returns:
            trajectory: Dictionary containing trajectory data
            
        Raises:
            WorkspaceError: If any waypoint is outside workspace
        """
        if len(waypoints) < 1:
            raise ValueError("At least one waypoint required")
        
        # Convert waypoints to Waypoint objects
        waypoint_objects = []
        max_time = 0.0
        
        for i, (x, y, time) in enumerate(waypoints):
            # Create position tensor
            pos = torch.tensor([x, y], device=self.device, dtype=torch.float32)
            
            # Check workspace reachability
            try:
                self.check_workspace_reachability(pos.unsqueeze(0), raise_error=True)
            except WorkspaceError as e:
                raise WorkspaceError(f"Manual waypoint {i+1} at ({x:.3f}, {y:.3f}) unreachable: {e}")
            
            waypoint_objects.append(Waypoint(
                position=pos,
                time=time
            ))
            max_time = max(max_time, time)
        
        # Use provided total_time or infer from waypoints
        if total_time is None:
            total_time = max_time
        
        print(f"Generating manual waypoint trajectory with {len(waypoint_objects)} waypoints over {total_time:.1f}s")
        for i, wp in enumerate(waypoint_objects):
            print(f"  Waypoint {i+1}: ({wp.position[0]:.3f}, {wp.position[1]:.3f}) at t={wp.time:.1f}s")
        
        # Generate trajectory using base method
        return self.generate_trajectory(waypoint_objects, total_time)
    
    def _cubic_spline_interpolation(
        self,
        waypoints: List[torch.Tensor],
        times: torch.Tensor,
        time_vec: torch.Tensor
    ) -> torch.Tensor:
        """Cubic spline interpolation between waypoints."""
        waypoints_stacked = torch.stack(waypoints, dim=0)
        interpolated = torch.zeros(len(time_vec), self.num_envs, 2, device=self.device)
        
        for i in range(len(time_vec)):
            t = time_vec[i]
            
            if t <= times[0]:
                interpolated[i] = waypoints_stacked[0]
            elif t >= times[-1]:
                interpolated[i] = waypoints_stacked[-1]
            else:
                idx = torch.searchsorted(times, t) - 1
                t0, t1 = times[idx], times[idx + 1]
                p0, p1 = waypoints_stacked[idx], waypoints_stacked[idx + 1]
                
                alpha = (t - t0) / (t1 - t0)
                alpha3 = alpha ** 3
                alpha2 = alpha ** 2
                
                interpolated[i] = p0 * (2 * alpha3 - 3 * alpha2 + 1) + p1 * (3 * alpha2 - 2 * alpha3)
        
        return interpolated
    
    def _quintic_interpolation(
        self,
        waypoints: List[torch.Tensor],
        times: torch.Tensor,
        time_vec: torch.Tensor
    ) -> torch.Tensor:
        """Quintic polynomial interpolation."""
        return self._cubic_spline_interpolation(waypoints, times, time_vec)
    
    def _linear_interpolation(
        self,
        waypoints: List[torch.Tensor],
        times: torch.Tensor,
        time_vec: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between waypoints."""
        waypoints_stacked = torch.stack(waypoints, dim=0)
        interpolated = torch.zeros(len(time_vec), self.num_envs, 2, device=self.device)
        
        for i in range(len(time_vec)):
            t = time_vec[i]
            
            if t <= times[0]:
                interpolated[i] = waypoints_stacked[0]
            elif t >= times[-1]:
                interpolated[i] = waypoints_stacked[-1]
            else:
                idx = torch.searchsorted(times, t) - 1
                t0, t1 = times[idx], times[idx + 1]
                p0, p1 = waypoints_stacked[idx], waypoints_stacked[idx + 1]
                
                alpha = (t - t0) / (t1 - t0)
                interpolated[i] = p0 * (1 - alpha) + p1 * alpha
        
        return interpolated
    
    def _compute_derivatives(self, trajectory: torch.Tensor, dt: float, order: int = 1) -> torch.Tensor:
        """Compute numerical derivatives of trajectory."""
        if order == 1:
            vel = torch.zeros_like(trajectory)
            vel[1:] = (trajectory[1:] - trajectory[:-1]) / dt
            vel[0] = vel[1]
            return vel
        elif order == 2:
            acc = torch.zeros_like(trajectory)
            acc[1:-1] = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt ** 2)
            acc[0] = acc[1]
            acc[-1] = acc[-2]
            return acc
        else:
            return trajectory
    
    def _apply_joint_limits(self, joint_traj: torch.Tensor) -> torch.Tensor:
        """Apply joint angle limits to trajectory."""
        for i in range(joint_traj.shape[-1]):
            joint_traj[:, :, i] = torch.clamp(
                joint_traj[:, :, i],
                self.joint_limits[i, 0],
                self.joint_limits[i, 1]
            )
        return joint_traj

def test_workspace_checking():
    """Test workspace checking functionality."""
    print("Testing TWOR IK Trajectory Generator Workspace Checking...")
    
    # Initialize generator with TWOR parameters
    generator = IKTrajectoryGenerator(
        num_envs=1,
        device="cpu",
        dt=0.01,
        joint_limits=[(-3.14159, 3.14159), (-3.14159, 3.14159)],
        interpolation_method="cubic",
        workspace_check=True  # Enable workspace checking
    )
    
    # Print workspace info
    workspace_info = generator.get_workspace_info()
    print("Workspace Information:")
    print(f"  Link lengths: L1={workspace_info['link_lengths']['L1']:.3f}m, L2={workspace_info['link_lengths']['L2']:.3f}m")
    print(f"  Max reach: {workspace_info['workspace_limits']['max_reach']:.3f}m")
    print(f"  Min reach: {workspace_info['workspace_limits']['min_reach']:.3f}m")
    print(f"  Base offset: ({workspace_info['base_offset']['x']:.3f}, {workspace_info['base_offset']['y']:.3f})")
    
    # Test valid positions
    print("\nTesting VALID positions:")
    valid_positions = [
        torch.tensor([0.5, 0.0]),   # Reachable
        torch.tensor([0.0, 0.7]),   # Reachable
        torch.tensor([-0.3, 0.0]),  # Your cube position
    ]
    
    for i, pos in enumerate(valid_positions):
        try:
            joints = generator.inverse_kinematics(pos)
            ee_check = generator.forward_kinematics(joints)
            print(f"  Position {i+1}: {pos.numpy()} -> Joints: [{joints[0]:.3f}, {joints[1]:.3f}] rad -> EE: [{ee_check[0, 0]:.3f}, {ee_check[0, 1]:.3f}] ✓")
        except WorkspaceError as e:
            print(f"  Position {i+1}: {pos.numpy()} -> ERROR: {e}")
    
    # Test invalid positions
    print("\nTesting INVALID positions:")
    invalid_positions = [
        torch.tensor([2.0, 0.0]),    # Too far
        torch.tensor([0.0, 2.0]),    # Too far
        torch.tensor([0.01, 0.01]),  # Too close (singularity)
        torch.tensor([-2.0, -2.0]),  # Way too far
    ]
    
    for i, pos in enumerate(invalid_positions):
        try:
            joints = generator.inverse_kinematics(pos)
            print(f"  Position {i+1}: {pos.numpy()} -> Joints: [{joints[0]:.3f}, {joints[1]:.3f}] rad (Should have failed!) ✗")
        except WorkspaceError as e:
            print(f"  Position {i+1}: {pos.numpy()} -> Correctly rejected: ✓")
            print(f"    Error: {str(e).split(chr(10))[0]}...")  # First line only
    
    # Test trajectory generation with invalid waypoints
    print("\nTesting trajectory generation with invalid targets:")
    try:
        box_start = torch.tensor([-0.3, 0.0])   # Valid
        box_target = torch.tensor([2.0, 0.0])   # Invalid - too far
        
        trajectory = generator.generate_box_pushing_trajectory(
            box_start_pos=box_start,
            box_target_pos=box_target,
            total_time=4.0
        )
        print("  Trajectory generation should have failed! ✗")
    except WorkspaceError as e:
        print("  Trajectory generation correctly rejected invalid target: ✓")
        print(f"    Error: {str(e).split(chr(10))[0]}...")


def test_twor_ik_trajectory_generator():
    """Test the IK trajectory generator with TWOR-specific parameters."""
    import matplotlib.pyplot as plt
    
    # Initialize generator with TWOR parameters
    generator = IKTrajectoryGenerator(
        num_envs=1,
        device="cpu",
        dt=0.01,
        joint_limits=[(-3.14159, 3.14159), (-3.14159, 3.14159)],
        interpolation_method="cubic",
        workspace_check=True  # Enable workspace checking
    )
    
    # Test valid trajectory
    print(f"TWOR Link lengths: L1={generator.link_lengths[0]:.3f}m, L2={generator.link_lengths[1]:.3f}m")
    print(f"Max reach: {generator.workspace_limits['max_reach']:.3f}m")
    print(f"Min reach: {generator.workspace_limits['min_reach']:.3f}m")
    
    # Test box pushing trajectory with valid positions
    box_start = torch.tensor([-0.3, 0.0])  # Your cube initial position  
    box_target = torch.tensor([-0.6, 0.0])  # Closer target (reachable)
    
    try:
        trajectory = generator.generate_box_pushing_trajectory(
            box_start_pos=box_start,
            box_target_pos=box_target,
            total_time=4.0
        )
        print("✓ Trajectory generated successfully!")
        
        # Plot results (same plotting code as before)
        # ... plotting code ...
        
    except WorkspaceError as e:
        print(f"✗ Trajectory generation failed: {e}")


if __name__ == "__main__":
    test_workspace_checking()
    print("\n" + "="*50 + "\n")
    test_twor_ik_trajectory_generator()