# File: twor_external_v0/twor_external_v0_env.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch 
import wandb
import isaaclab.sim as sim_utils
from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyMaterialCfg
)

from .twor_external_v0_env_cfg import TworExternalV0EnvCfg
from .impedance_position_generator import ImpedancePositionGenerator
from .impedance_filter import ImpedanceFilter
from .ik_trajectory_generator import IKTrajectoryGenerator, Waypoint, WorkspaceError

# =============================================================================
# TworExternalV0 Environment
# =============================================================================

class TworExternalV0Env(DirectRLEnv):
    """
    Custom RL environment for the TworExternalV0 task using joint-space
    impedance control.
    """
    cfg: TworExternalV0EnvCfg

    def __init__(self, cfg: TworExternalV0EnvCfg, render_mode=None, **kwargs):
        # ---------------------------------------------------------------------
        # Initialize logging
        # ---------------------------------------------------------------------
        wandb.init(project="twor-rewards", reinit=True)
        self._wb_step = 0

        # ---------------------------------------------------------------------
        # Parent initialization
        # ---------------------------------------------------------------------
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ---------------------------------------------------------------------
        # Joint setup
        # ---------------------------------------------------------------------
        self._servo1_idx, _ = self.robot.find_joints(cfg.servo1_dof_name)
        self._servo2_idx, _ = self.robot.find_joints(cfg.servo2_dof_name)
        self._joint_ids     = [*self._servo1_idx, *self._servo2_idx]

        # ---------------------------------------------------------------------
        # Sensor body index (pick first match)
        # ---------------------------------------------------------------------
        body_ids, _ = self.robot.find_bodies(cfg.sensor_body_name)
        self._sensor_body_idx = int(body_ids[0])

        # ---------------------------------------------------------------------
        # State buffers
        # ---------------------------------------------------------------------
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        idxs            = self._joint_ids
        default_pos     = self.robot.data.default_joint_pos[:, idxs].clone()
        self._desired_pos       = default_pos.clone()
        self._offset            = default_pos.clone()
        self._prev_vel          = torch.zeros_like(self.robot.data.joint_vel[:, idxs])
        self._prev_desired_pos  = default_pos.clone()
        self._prev_desired_vel  = torch.zeros_like(self._prev_vel)
        # counter for hardcoded trajectory
        self._count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._max_count = cfg.trajectory_max_count

        # ---------------------------------------------------------------------
        # Joint-space impedance generator
        # ---------------------------------------------------------------------
        self.imp_gen = ImpedancePositionGenerator(
            M_d = torch.tensor(cfg.imp_M, device=self.device),
            D_d = torch.zeros(len(idxs), device=self.device),
            K_d = torch.zeros(len(idxs), device=self.device),
            dt  = cfg.sim.dt,
            x0  = self._offset[0]
        )
        
        # Action scaling tensors
        self.action_scale  = torch.tensor(cfg.action_scale,
                                          device=self.device,
                                          dtype=torch.float32)
        self.action_offset = torch.tensor(cfg.action_offset,
                                          device=self.device,
                                          dtype=torch.float32)

        # Initialize impedance filter
        self.impedance_filter = ImpedanceFilter(
            M_v=torch.tensor(cfg.virtual_mass, device=self.device),
            B_v=torch.tensor(cfg.virtual_damping, device=self.device),
            K_v=torch.tensor(cfg.virtual_stiffness, device=self.device),
            dt=cfg.sim.dt,
            num_envs=self.num_envs,
            num_joints=2,  # Servo1, Servo2
            device=self.device,
            method=cfg.impedance_method
        )
        
        # Reference trajectory
        self._x_des = torch.zeros(self.num_envs, 2, device=self.device)

        # IK Trajectory Generator for box pushing
        # Enable strict workspace checking
        self.ik_trajectory_generator = IKTrajectoryGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=cfg.sim.dt,
            workspace_check=True,  # Enable checking
            workspace_tolerance=0.01  # 1cm tolerance
        )

        # Generate trajectory based on configuration
        if cfg.use_manual_waypoints:
            try:
                self.trajectory = self.ik_trajectory_generator.generate_manual_waypoint_trajectory(
                    waypoints=cfg.manual_waypoints,
                    total_time=cfg.trajectory_total_time
                )
                print("✓ Manual waypoint trajectory generated successfully!")
                print(f"  Waypoints: {len(cfg.manual_waypoints)}")
                print(f"  Total time: {cfg.trajectory_total_time:.1f}s")
            except WorkspaceError as e:
                print(f"✗ Cannot generate manual waypoint trajectory: {e}")
                print("  Falling back to automatic box pushing trajectory")
                cfg.use_manual_waypoints = False
        
        if not cfg.use_manual_waypoints:
            # Fallback to automatic box pushing trajectory
            box_start = torch.tensor(cfg.box_start_pos, device=self.device) 
            box_target = torch.tensor(cfg.box_target_pos, device=self.device)
            
            try:
                self.trajectory = self.ik_trajectory_generator.generate_box_pushing_trajectory(
                    box_start_pos=box_start,
                    box_target_pos=box_target,
                    total_time=cfg.trajectory_total_time,
                    approach_offset=cfg.trajectory_approach_offset
                )
                print("✓ Automatic box pushing trajectory generated successfully!")
            except WorkspaceError as e:
                print(f"✗ Cannot generate trajectory: {e}")
                # Continue with fallback (circular trajectory)

        # Get workspace info for debugging
        workspace_info = self.ik_trajectory_generator.get_workspace_info()
        print(f"Max reach: {workspace_info['workspace_limits']['max_reach']:.3f}m")

    def _setup_scene(self) -> None:
        """
        Build simulation scene: robot, ground plane, cube, sensors.
        """
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments must come after creating all scene elements
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions if on CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[r"/World/envs/env_.*/.*"])
            
        # Register robot with scene
        self.scene.articulations["twor"] = self.robot

        # Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Cube object - position it at the box start position from config
        cube_cfg = RigidObjectCfg(
            prim_path = "/World/envs/env_.*/Cube",
            spawn     = sim_utils.CuboidCfg(
                size             = (0.25,0.25,0.25),
                rigid_props      = RigidBodyPropertiesCfg(),
                mass_props       = MassPropertiesCfg(mass=10.0),  # Reduced mass for easier pushing
                collision_props  = CollisionPropertiesCfg(),
                physics_material = RigidBodyMaterialCfg(
                    static_friction=0.3,    # Add friction for realistic contact
                    dynamic_friction=0.2,
                    restitution=0.1
                )
            ),
            init_state= RigidObjectCfg.InitialStateCfg(
                pos=(self.cfg.box_start_pos[0], self.cfg.box_start_pos[1], self.cfg.box_start_pos[2])  # Use config position
            )
        )
        self.scene.rigid_objects["Cube"] = RigidObject(cube_cfg)

        # Contact sensor - following the working example from add_new_robot.py
        sensor_cfg = ContactSensorCfg(
            prim_path              = "/World/envs/env_.*/Twor/Link2",  # Sensor on Link2 (end effector)
            update_period          = 0.0,  # Update every step
            history_length         = 1,
            debug_vis              = False,   # Disable debug visualization for now
            filter_prim_paths_expr = ["/World/envs/env_.*/Cube"]  # Only detect contact with cube
        )
        self.scene.sensors["contact_L2"] = ContactSensor(sensor_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Cache actions for the upcoming physics step.
        """
        # Scale actions from [0,1] to actual ranges
        self._actions = actions * (self.action_scale - self.action_offset) + self.action_offset

    def _apply_action(self) -> None:
        """
        Use IK trajectory generator for reference positions.
        """
        # Unpack RL actions
        k1, d1, k2, d2 = torch.unbind(self._actions, dim=-1)
        
        # Update impedance filter parameters
        K_v = torch.stack([k1, k2], dim=-1)
        B_v = torch.stack([d1, d2], dim=-1)
        self.impedance_filter.set_parameters(B_v=B_v, K_v=K_v)
        
        # Get external forces from contact sensor
        contact_sensor = self.scene.sensors["contact_L2"]
        F_ext_cart = contact_sensor.data.net_forces_w  # [B, 3]
        
        # Ensure we have the right dimensions
        if F_ext_cart.dim() == 3:
            # If shape is [B, N, 3], sum over contact points
            F_ext_cart = F_ext_cart.sum(dim=1)  # [B, 3]
        elif F_ext_cart.dim() == 2 and F_ext_cart.shape[1] != 3:
            # If shape is wrong, try alternative data sources
            if hasattr(contact_sensor.data, 'force_matrix_w') and contact_sensor.data.force_matrix_w.numel() > 0:
                F_ext_cart = contact_sensor.data.force_matrix_w
                if F_ext_cart.dim() == 3:
                    F_ext_cart = F_ext_cart.sum(dim=1)  # Sum over contact points
            else:
                # Fallback to zeros if no contact data
                F_ext_cart = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Ensure F_ext_cart is [B, 3]
        if F_ext_cart.dim() == 1:
            F_ext_cart = F_ext_cart.unsqueeze(0)
        
        # Map Cartesian forces to joint torques via Jacobian
        J_all = self.robot.root_physx_view.get_jacobians()
        J_lin = J_all[:, self._sensor_body_idx, 0:3, :]  # [B, 3, num_dofs]
        J = J_lin[:, :, self._joint_ids]  # [B, 3, 2]
        
        # Fix the einsum operation - F_ext_cart should be [B, 3], not [B, j]
        tau_ext = torch.einsum("bij,bi->bj", J, F_ext_cart)  # [B, 2]
        
        # Compute end effector position using forward kinematics
        current_joint_pos = self.joint_pos[:, self._joint_ids]  # [B, 2]
        end_effector_pos_ik = self.ik_trajectory_generator.forward_kinematics(current_joint_pos)  # [B, 3]
        
        # Get actual sensor position from simulation
        sensor_pos_actual = self.robot.data.body_pos_w[:, self._sensor_body_idx]  # [B, 3]
        
        # Get box position
        box_pos = self.scene.rigid_objects["Cube"].data.root_state_w[:, 0:3]  # [B, 3] - only need x,y,z
        
        # Update per-environment counters
        self._count += 1
        
        # Check if trajectory has completed (mark for episode termination)
        trajectory_completed = (self._count % self._max_count == 0)
        self._trajectory_completed = trajectory_completed  # Store for _get_dones()
        
        # Get reference from IK trajectory - use modulo for continuous cycling within episode
        trajectory_time = ((self._count % self._max_count).float() * self.cfg.sim.dt)
        current_time = trajectory_time.max().item()  # Use max for synchronization
        
        # Ensure current_time is within bounds
        current_time = max(0.0, min(current_time, self.cfg.trajectory_total_time))
        
        try:
            pos_ref, _, _ = self.ik_trajectory_generator.get_reference(current_time)
            self._x_des = pos_ref  # Use IK reference as the nominal trajectory
            
            # Debug logging
            if self._count[0] % 100 == 0:  # Log every 100 steps for first env
                env_idx = 0
                print(f"\n--- Step {self._count[env_idx].item()} Debug Info ---")
                print(f"EE Position (IK):      [{end_effector_pos_ik[env_idx, 0]:.4f}, {end_effector_pos_ik[env_idx, 1]:.4f}, {end_effector_pos_ik[env_idx, 2]:.4f}] m")
                print(f"EE Position (Actual):  [{sensor_pos_actual[env_idx, 0]:.4f}, {sensor_pos_actual[env_idx, 1]:.4f}, {sensor_pos_actual[env_idx, 2]:.4f}] m")
                print(f"Box Position:          [{box_pos[env_idx, 0]:.4f}, {box_pos[env_idx, 1]:.4f}, {box_pos[env_idx, 2]:.4f}] m")
                print(f"Contact Force (Cart):  [{F_ext_cart[env_idx, 0]:.2f}, {F_ext_cart[env_idx, 1]:.2f}, {F_ext_cart[env_idx, 2]:.2f}] N")
                print(f"Contact Force (Joint): [{tau_ext[env_idx, 0]:.2f}, {tau_ext[env_idx, 1]:.2f}] Nm")
                print(f"Joint Positions:       [{current_joint_pos[env_idx, 0]:.3f}, {current_joint_pos[env_idx, 1]:.3f}] rad")
                print(f"Trajectory Time:       {current_time:.2f}s / {self.cfg.trajectory_total_time:.2f}s")
                print(f"Reference Joints:      [{pos_ref[0,0]:.3f}, {pos_ref[0,1]:.3f}] rad")
                print(f"Trajectory Executing:  {self.ik_trajectory_generator.is_executing}")
                print(f"Count % Max:           {self._count[env_idx].item() % self._max_count}")
                print(f"Trajectory Completed:  {trajectory_completed[env_idx].item()}")
                
                # Show reference end effector position
                ref_ee_pos = self.ik_trajectory_generator.forward_kinematics(pos_ref[0:1])
                print(f"Reference EE Position: [{ref_ee_pos[0, 0]:.4f}, {ref_ee_pos[0, 1]:.4f}, {ref_ee_pos[0, 2]:.4f}] m")
                
                if self.cfg.use_manual_waypoints:
                    print(f"Using manual waypoints: {len(self.cfg.manual_waypoints)} points")
                    
                # Additional contact sensor debugging
                try:
                    if hasattr(contact_sensor.data, 'pos_w') and contact_sensor.data.pos_w is not None:
                        if contact_sensor.data.pos_w.numel() > 0:
                            num_contacts = contact_sensor.data.pos_w.shape[1] if contact_sensor.data.pos_w.dim() > 1 else 0
                            print(f"Number of contacts:    {num_contacts}")
                        else:
                            print(f"Number of contacts:    0 (empty tensor)")
                    else:
                        print(f"Number of contacts:    0 (pos_w is None)")
                except Exception as debug_error:
                    print(f"Contact debug error:   {debug_error}")
                
        except Exception as e:
            print(f"IK trajectory error: {e}")
            # Fallback to the working motion pattern from add_new_robot.py
            self._use_fallback_trajectory()

        # Update impedance filter to get modified reference
        x_ref = self.impedance_filter.update(tau_ext, self._x_des)
        
        # Apply reference as joint position targets
        self.robot.set_joint_position_target(x_ref, joint_ids=self._joint_ids)
        self._desired_pos = x_ref.clone()

    def _generate_new_trajectory(self):
        """Generate new trajectory based on configuration."""
        if self.cfg.use_manual_waypoints:
            try:
                self.ik_trajectory_generator.generate_manual_waypoint_trajectory(
                    waypoints=self.cfg.manual_waypoints,
                    total_time=self.cfg.trajectory_total_time
                )
                print("✓ Manual waypoint trajectory regenerated")
            except WorkspaceError as e:
                print(f"Manual waypoint trajectory generation failed: {e}")
        else:
            box_start = torch.tensor(self.cfg.box_start_pos, device=self.device)
            box_target = torch.tensor(self.cfg.box_target_pos, device=self.device)
            
            try:
                self.ik_trajectory_generator.generate_box_pushing_trajectory(
                    box_start_pos=box_start,
                    box_target_pos=box_target,
                    total_time=self.cfg.trajectory_total_time,
                    approach_offset=self.cfg.trajectory_approach_offset
                )
                print("✓ Box pushing trajectory regenerated")
            except WorkspaceError as e:
                print(f"Box pushing trajectory generation failed: {e}")

    def _use_fallback_trajectory(self):
        """Use fallback trajectory when IK fails."""
        frac = (self._count % self._max_count).float() / self._max_count
        for env_idx in range(self.num_envs):
            env_frac = frac[env_idx].item()
            # Use the exact pattern from add_new_robot.py that works
            self._x_des[env_idx, 0] = math.pi/2 * env_frac - math.pi/8       # Servo1
            self._x_des[env_idx, 1] = -math.pi/2 * env_frac + math.pi/2 + math.pi/8  # Servo2
        print(f"Using fallback trajectory, frac: {frac[0].item():.3f}")

    def _reset_internal_buffers(self, env_ids: torch.Tensor):
        """Reset internal state buffers for specified environments."""
        idxs = self._joint_ids
        default_pos = self.robot.data.default_joint_pos[env_ids][:, idxs].clone()
        zeros_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids][:, idxs])
        
        self._desired_pos[env_ids] = default_pos
        self._prev_desired_pos[env_ids] = default_pos
        self._prev_vel[env_ids] = zeros_vel
        self._prev_desired_vel[env_ids] = zeros_vel

    def _reset_all_components(self, env_ids: torch.Tensor):
        """Reset all control components for specified environments."""
        idxs = self._joint_ids
        default_pos = self.robot.data.default_joint_pos[env_ids][:, idxs].clone()
        
        # Reset impedance filter
        self.impedance_filter.reset(env_ids)
        
        # Reset impedance generator
        if hasattr(self, 'imp_gen'):
            self.imp_gen.reset(env_ids, x0=default_pos)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Build the 15-D policy observation vector.
        """
        idxs = self._joint_ids
        dt   = self.cfg.sim.dt

        # Safe contact force reading with fallback
        try:
            contact_data = self.scene.sensors["contact_L2"].data.net_forces_w
            if contact_data.dim() == 3:
                forces = contact_data.squeeze(1)  # [B, N, 3] -> [B, 3]
            elif contact_data.dim() == 2:
                forces = contact_data  # Already [B, 3]
            else:
                forces = torch.zeros(self.num_envs, 3, device=self.device)
        except (AttributeError, RuntimeError):
            # Fallback if sensor data not available during reset
            forces = torch.zeros(self.num_envs, 3, device=self.device)

        pos     = self.joint_pos[:, idxs]
        des     = self._desired_pos
        vel     = self.joint_vel[:, idxs]
        des_vel = (des - self._prev_desired_pos) / dt
        act_acc = (vel - self._prev_vel) / dt
        des_acc = (des_vel - self._prev_desired_vel) / dt

        # Update history buffers
        self._prev_vel            = vel.clone()
        self._prev_desired_pos    = des.clone()
        self._prev_desired_vel    = des_vel.clone()

        obs = torch.cat([forces, pos, des, vel, des_vel, act_acc, des_acc], dim=-1)
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        MAIN RESET METHOD - Reset specified environments for episode termination.
        This is the ONLY place where resets should be coordinated.
        """
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        print(f"Resetting environments {env_ids.tolist()} - starting new episodes")

        # Reset trajectory counters - this starts fresh trajectories
        self._count[env_ids] = 0
        
        # Clear trajectory completion flags
        if hasattr(self, '_trajectory_completed'):
            self._trajectory_completed[env_ids] = False

        # Reset robot state in simulation
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids
        )

        # Reset cube to initial position with proper state
        if "Cube" in self.scene.rigid_objects:
            cube = self.scene.rigid_objects["Cube"]
            
            # Get default cube state and apply to specified environments
            cube_state = cube.data.default_root_state[env_ids].clone()
            
            # Set position to configured box start position
            cube_state[:, 0] = self.cfg.box_start_pos[0]  # x
            cube_state[:, 1] = self.cfg.box_start_pos[1]  # y  
            cube_state[:, 2] = self.cfg.box_start_pos[2]  # z
            
            # Reset velocity to zero
            cube_state[:, 7:] = 0.0  # linear and angular velocities
            
            # Apply the reset state
            cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
            cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids)
            
            print(f"Reset cube position to: [{self.cfg.box_start_pos[0]:.3f}, {self.cfg.box_start_pos[1]:.3f}, {self.cfg.box_start_pos[2]:.3f}]")

        # Reset internal state buffers
        self._reset_internal_buffers(env_ids)
        
        # Reset all control components
        self._reset_all_components(env_ids)
        
        # Reset progress tracking for box pushing task
        if hasattr(self, '_last_dist'):
            spawn_x = torch.full((len(env_ids),), self.cfg.box_start_pos[0], device=self.device)
            target_x = spawn_x - self.cfg.target_pos_x
            cube_x = torch.full((len(env_ids),), self.cfg.box_start_pos[0], device=self.device)
            self._last_dist[env_ids] = torch.abs(cube_x - target_x)

        # Generate fresh trajectory for new episodes
        self._generate_new_trajectory()
        
        # Force update scene to ensure sensor data is available
        self.scene.update(self.cfg.sim.dt)

    def _get_rewards(self) -> torch.Tensor:
        """
        Sum reward components and subtract penalties.
        """
        comps = self._get_reward_components()
        return (comps["r_dist"]
                + comps["r_track"]
                + comps["r_prog"]
                + comps["r_term"]
               ) - (comps["p_k"] + comps["p_d"])

    def _get_reward_components(self) -> dict[str, torch.Tensor]:
        """
        Compute individual reward components and penalties.
        """
        spawn_x   = self.scene.rigid_objects["Cube"].data.default_root_state[:, 0]
        target_x  = spawn_x - self.cfg.target_pos_x
        cube_x    = self.scene.rigid_objects["Cube"].data.root_state_w[:, 0]
        term_mask = (self.episode_length_buf >= self.max_episode_length - 1).float()

        # Distance reward
        dist    = torch.abs(cube_x - target_x)
        initial = (spawn_x - target_x).abs()
        r_dist  = self.cfg.w_pos * (initial - dist) / initial

        # Tracking reward
        q_err   = torch.abs(self.joint_pos[:, self._joint_ids] - self._desired_pos)
        r_track = self.cfg.w_tracking * (self.cfg.max_joint_pos - q_err).clamp(min=0).sum(dim=-1)

        # Progress bonus
        prog    = (getattr(self, "_last_dist", dist) - dist).clamp(min=0)
        r_prog  = self.cfg.w_prog * prog
        self._last_dist = dist

        # Terminal bonus
        r_term  = self.cfg.w_terminal * (initial - dist) / initial * term_mask

        # Penalties on K, D (use unscaled actions)
        actions_unscaled = self._actions
        k1, d1, k2, d2 = torch.unbind(actions_unscaled, dim=-1)
        p_k            = self.cfg.w_stiffness * (k1.pow(2) + k2.pow(2))
        p_d            = self.cfg.w_damping   * (d1.pow(2) + d2.pow(2))

        return {
            "r_dist":  r_dist,
            "r_track": r_track,
            "r_prog":  r_prog,
            "r_term":  r_term,
            "p_k":     p_k,
            "p_d":     p_d,
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute done and truncation masks.
        """
        # Time-based episode termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Force violation termination
        try:
            contact_data = self.scene.sensors["contact_L2"].data.net_forces_w
            if contact_data.dim() == 3:
                forces = contact_data.squeeze(1)  # [B, N, 3] -> [B, 3]
            elif contact_data.dim() == 2:
                forces = contact_data  # Already [B, 3]
            else:
                forces = torch.zeros(self.num_envs, 3, device=self.device)
        except (AttributeError, RuntimeError):
            forces = torch.zeros(self.num_envs, 3, device=self.device)
        
        force_violation = torch.norm(forces, dim=-1) > self.cfg.max_allowed_force
        
        # Trajectory completion termination (NEW)
        trajectory_completed = getattr(self, '_trajectory_completed', torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
        # Episode ends on: timeout, force violation, OR trajectory completion
        done = time_out | force_violation | trajectory_completed
        
        return done, time_out

    def step(self, actions: torch.Tensor):
        """
        Perform one environment step, log metrics, and return Gym API tuple.
        """
        obs, reward, terminated, truncated, info = super().step(actions)

        comps = self._get_reward_components()
        wandb.log({
            "reward/dist":   comps["r_dist"].mean().item(),
            "reward/track":  comps["r_track"].mean().item(),
            "reward/prog":   comps["r_prog"].mean().item(),
            "reward/term":   comps["r_term"].mean().item(),
            "penalty/stiff": comps["p_k"].mean().item(),
            "penalty/damp":  comps["p_d"].mean().item(),
            "reward/total":  reward.mean().item(),
        }, step=self._wb_step)
        self._wb_step += 1

        info.update({k: v.mean().item() for k, v in comps.items()})
        info["reward/total"] = reward.mean().item()
        return obs, reward, terminated, truncated, info

