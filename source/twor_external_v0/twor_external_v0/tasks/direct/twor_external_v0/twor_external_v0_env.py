# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import torch
import isaaclab.sim as sim_utils
from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.envs import DirectRLEnv

# from isaaclab.sim import UsdFileCfg, spawn_from_usd
from isaaclab.sim import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg, RigidBodyMaterialCfg


from .twor_external_v0_env_cfg import TworExternalV0EnvCfg

class TworExternalV0Env(DirectRLEnv):
    cfg: TworExternalV0EnvCfg

    def __init__(self, cfg: TworExternalV0EnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # joint indices for controlled joints
        self._servo1_idx, _ = self.robot.find_joints(self.cfg.servo1_dof_name)
        self._servo2_idx, _ = self.robot.find_joints(self.cfg.servo2_dof_name)
        self._joint_ids = [*self._servo1_idx, *self._servo2_idx]
        # state references
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        # default joint offsets
        idxs = self._joint_ids
        default_pos = self.robot.data.default_joint_pos[:, idxs].clone()
        self._desired_pos = default_pos.clone()
        self._offset = default_pos.clone()
        # buffers for finite-difference
        self._prev_vel = torch.zeros_like(self.joint_vel[:, idxs])
        self._prev_desired_pos = default_pos.clone()
        self._prev_desired_vel = torch.zeros_like(self._prev_vel)
        # counter for hardcoded trajectory
        self._count = 0
        self._max_count = 1000

    def _setup_scene(self) -> None:
        # spawn articulation & ground
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[r"/World/envs/env_.*/.*"])
        self.scene.articulations["twor"] = self.robot
        # lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # spawn rigid object (cube)
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.25),
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=100.0),
                collision_props=CollisionPropertiesCfg(),
                physics_material=RigidBodyMaterialCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.3, 0, 0.25)),
        )
        self.scene.rigid_objects["Cube"] = RigidObject(cube_cfg)
        # contact sensor
        sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Twor/Link2",
            update_period=0.0, history_length=1, debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
        )

        # attach contact sensor to Link2 of Twor
        self.scene.sensors["contact_L2"] = ContactSensor(sensor_cfg)

        



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: [k1, d1, k2, d2]
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        # unpack impedance gains
        k1, d1, k2, d2 = torch.unbind(self._actions, dim=-1)
        stiff = torch.stack([k1, k2], dim=-1)
        damp = torch.stack([d1, d2], dim=-1)
        # compute hardcoded linear trajectory
        frac = (self._count % self._max_count) / self._max_count
        q1 = (math.pi/2) * frac - (math.pi/8)
        q2 = - (math.pi/2) * frac + (math.pi/2) + (math.pi/8)
        q_des = self._offset.clone()
        q_des[:, 0] = q1
        q_des[:, 1] = q2
        # apply impedance via direct joint stiffness/damping and position target

        # self.robot.write_joint_stiffness_to_sim(stiffness=stiff, joint_ids=self._joint_ids)
        # self.robot.write_joint_damping_to_sim(damping=damp, joint_ids=self._joint_ids)
        self.robot.set_joint_position_target(q_des, joint_ids=self._joint_ids)
        self.robot.write_data_to_sim()
        # update for next step
        self._desired_pos = q_des.clone()
        self._count += 1

    def _get_observations(self) -> dict:
        idxs = self._joint_ids
        dt = self.cfg.sim.dt
        forces = self.scene["contact_L2"].data.net_forces_w.squeeze(1)
        pos = self.joint_pos[:, idxs]
        des = self._desired_pos
        vel = self.joint_vel[:, idxs]
        # finite differences
        des_vel = (des - self._prev_desired_pos) / dt
        act_acc = (vel - self._prev_vel) / dt
        des_acc = (des_vel - self._prev_desired_vel) / dt
        # update buffers
        self._prev_vel = vel.clone()
        self._prev_desired_pos = des.clone()
        self._prev_desired_vel = des_vel.clone()
        obs = torch.cat([forces, pos, des, vel, des_vel, act_acc, des_acc], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # --- 1) Terminal position cost (only on last step) -------------------
        # spawn x from default state (world frame)
        spawn_x = self.scene.rigid_objects["Cube"].data.default_root_state[:, 0]
        # target x is 1m negative from spawn
        target_x = spawn_x - 1.0
        # current cube x-position (world frame)
        cube_x = self.scene.rigid_objects["Cube"].data.root_state_w[:, 0]
        # squared distance to target
        dist2 = (cube_x - target_x).pow(2)
        # mask so terminal cost applies only at end of episode
        terminal_mask = (self.episode_length_buf >= self.max_episode_length - 1).float()
        c_pos = self.cfg.w_pos * dist2 * terminal_mask

        # --- 2) Impedance tracking cost (per-step) -----------------------------
        q = self.joint_pos[:, self._joint_ids]               # [N_env,2]
        q_ref = self._desired_pos                            # [N_env,2]
        track_cost = (q - q_ref).pow(2).sum(dim=-1)          # [N_env]
        c_track = self.cfg.w_tracking * track_cost

        # --- 3) Stiffness penalty (per-step) -----------------------------------
        k1, _, k2, _ = torch.unbind(self._actions, dim=-1)
        stiff = torch.stack([k1, k2], dim=-1)                # [N_env,2]
        stiff_cost = stiff.pow(2).sum(dim=-1)                # [N_env]
        c_k = self.cfg.w_stiffness * stiff_cost

        # total negative cost = reward
        reward = - (c_pos + c_track + c_k)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        force_violation = torch.norm(
            self.scene["contact_L2"].data.net_forces_w.squeeze(1), dim=-1
        ) > self.cfg.max_allowed_force
        done = time_out | force_violation
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # reset trajectory buffers and counter
        idxs = self._joint_ids
        default_pos = self.robot.data.default_joint_pos[:, idxs].clone()
        self._desired_pos = default_pos.clone()
        self._prev_desired_pos = default_pos.clone()
        self._prev_vel = torch.zeros_like(self.joint_vel[:, idxs])
        self._prev_desired_vel = torch.zeros_like(self._prev_vel)
        self._count = 0

        # reset robot base pose & velocity
        root_states = self.robot.data.default_root_state[env_ids].clone()
        root_states[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_states[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_states[:, 7:], env_ids)

        # reset all joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids
        )

        # reset cube object
        if "Cube" in self.scene.rigid_objects:
            obj = self.scene.rigid_objects["Cube"]
            default_state = obj.data.default_root_state[env_ids].clone()
            default_state[:, :3] += self.scene.env_origins[env_ids]
            obj.write_root_pose_to_sim(default_state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # push updated cube state into sim immediately
        self.scene.write_data_to_sim()
