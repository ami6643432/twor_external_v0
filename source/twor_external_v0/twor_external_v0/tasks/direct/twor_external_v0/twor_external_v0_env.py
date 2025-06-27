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
from .impedance_position_generator       import ImpedancePositionGenerator

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
        self._count = 0
        self._max_count = 1000

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

    def _setup_scene(self) -> None:
        """
        Build simulation scene: robot, ground plane, cube, sensors.
        """
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[r"/World/envs/env_.*/.*"])
        self.scene.articulations["twor"] = self.robot

        # Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Cube object
        cube_cfg = RigidObjectCfg(
            prim_path = "/World/envs/env_.*/Cube",
            spawn     = sim_utils.CuboidCfg(
                size             = (0.25,0.25,0.25),
                rigid_props      = RigidBodyPropertiesCfg(),
                mass_props       = MassPropertiesCfg(mass=100.0),
                collision_props  = CollisionPropertiesCfg(),
                physics_material = RigidBodyMaterialCfg()
            ),
            init_state= RigidObjectCfg.InitialStateCfg(pos=(-0.3,0,0.0))
        )
        self.scene.rigid_objects["Cube"] = RigidObject(cube_cfg)

        # Contact sensor
        sensor_cfg = ContactSensorCfg(
            prim_path              = "/World/envs/env_.*/Twor/Link2",
            update_period          = 0.0,
            history_length         = 1,
            debug_vis              = False,
            filter_prim_paths_expr = ["/World/envs/env_.*/Cube"]
        )
        self.scene.sensors["contact_L2"] = ContactSensor(sensor_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Cache actions for the upcoming physics step.
        """
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        """
        1) Unpack K_d, D_d from actions
        2) Map Cartesian contact force → joint torques
        3) Pull true inertia M(q) → virtual mass
        4) Run impedance law → new q_d
        5) Apply q_d as joint position target
        """
        # ---------------------------------------------------------------------
        # 1) Unpack and set gains
        # ---------------------------------------------------------------------
        k1, d1, k2, d2    = torch.unbind(self._actions, dim=-1)
        self.imp_gen.K_d = torch.stack([k1, k2], dim=-1)
        self.imp_gen.D_d = torch.stack([d1, d2], dim=-1)

        # ---------------------------------------------------------------------
        # 2) Cartesian force → joint torques via PhysX Jacobians
        # ---------------------------------------------------------------------
        F_ext_cart = self.scene["contact_L2"].data.net_forces_w.squeeze(1)  # [B,3]
        J_all      = self.robot.root_physx_view.get_jacobians()
        # select our body and linear rows
        J_lin      = J_all[:, self._sensor_body_idx, 0:3, :]                # [B,3,num_dofs]
        J          = J_lin[:, :, self._joint_ids]                          # [B,3,2]
        tau_ext    = torch.einsum("bji,bj->bi", J, F_ext_cart)             # [B,2]

        # ---------------------------------------------------------------------
        # 3) Pull true inertia M(q) and set virtual mass
        # ---------------------------------------------------------------------
        # Get the instantaneous joint-space mass matrix from PhysX
        # This returns [B, num_dofs, num_dofs], so we index for our joints
        mass_matrix_full = self.robot.root_physx_view.get_generalized_mass_matrices()  # [B, total_dofs, total_dofs]
        M_q = mass_matrix_full[:, self._joint_ids, :][:, :, self._joint_ids]           # [B, 2, 2]
        self.imp_gen.M_d = torch.diagonal(M_q, dim1=-2, dim2=-1)                       # [B,2]

        # ---------------------------------------------------------------------
        # 4) Impedance update → q_d
        # ---------------------------------------------------------------------

        # compute hardcoded linear trajectory
        frac = (self._count % self._max_count) / self._max_count
        q1 = (math.pi/2) * frac - (math.pi/8)
        q2 = - (math.pi/2) * frac + (math.pi/2) + (math.pi/8)
        # x_r[:, 0] = q1
        # x_r[:, 1] = q2

        q_d = self.imp_gen.update(F_ext=tau_ext, x_r = [q1,q2])

        # ---------------------------------------------------------------------
        # 5) Apply as joint position target
        # ---------------------------------------------------------------------
        self.robot.set_joint_position_target(q_d, joint_ids=self._joint_ids)
        self.robot.write_data_to_sim()
        self._desired_pos = q_d.clone()


    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        Build the 15-D policy observation vector.
        """
        idxs = self._joint_ids
        dt   = self.cfg.sim.dt

        forces  = self.scene["contact_L2"].data.net_forces_w.squeeze(1)
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

        # Penalties on K, D
        k1, d1, k2, d2 = torch.unbind(self._actions, dim=-1)
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute done and truncation masks.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        force_violation = torch.norm(
            self.scene["contact_L2"].data.net_forces_w.squeeze(1), dim=-1
        ) > self.cfg.max_allowed_force
        done = time_out | force_violation
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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specified environments: robot, cube, and internal buffers.
        """
        # Reset history buffers
        idxs        = self._joint_ids
        default_pos = self.robot.data.default_joint_pos[:, idxs].clone()
        self._desired_pos      = default_pos.clone()
        self._prev_desired_pos = default_pos.clone()
        self._prev_vel         = torch.zeros_like(self.robot.data.joint_vel[:, idxs])
        self._prev_desired_vel = torch.zeros_like(self._prev_vel)

        # Parent reset
        super()._reset_idx(env_ids)

        # Reset robot root state
        root_states = self.robot.data.default_root_state[env_ids].clone()
        root_states[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_states[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_states[:, 7:], env_ids)

        # Reset robot joint state
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids
        )

        # Reset cube state
        if "Cube" in self.scene.rigid_objects:
            obj   = self.scene.rigid_objects["Cube"]
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            obj.write_root_pose_to_sim(state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(state[:, 7:], env_ids)

        # Write all data to sim
        self.scene.write_data_to_sim()
