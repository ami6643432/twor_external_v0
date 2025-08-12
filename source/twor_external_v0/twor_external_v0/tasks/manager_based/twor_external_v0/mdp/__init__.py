# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight functional MDP utilities for IsaacLab 4.5 (no RewardTerm/ObservationTerm subclasses)."""

from __future__ import annotations

import torch

# Attempt to import built-in Isaac Lab MDP helpers (joint_pos_rel, etc.)
try:  # noqa: F401,F403
    from isaaclab.envs.mdp import *  # type: ignore
except Exception as _base_mdp_err:  # pragma: no cover
    print(f"[WARN] Base MDP import skipped: {_base_mdp_err}")

# Expose variable impedance action term config if still desired (kept optional)
try:  # noqa: F401,F403
    from .actions import (
        VariableImpedanceActionTerm,
        VariableImpedanceActionTermCfg,
    )
except Exception as _act_err:  # pragma: no cover
    print(f"[WARN] VariableImpedanceActionTerm unavailable: {_act_err}")

# ---------------------------------------------------------------------------
# Functional observation / reward helpers
# ---------------------------------------------------------------------------

def _reduce_contact_forces(forces: torch.Tensor) -> torch.Tensor:
    """Utility: sum over contact points until shape becomes [B,3]."""
    while forces.ndim > 2:
        forces = forces.sum(dim=1)
    return forces

def contact_force_vector(env, sensor_cfg):
    """Return raw (summed) contact force vector [num_envs,3].

    Parameters:
        env: Manager-based env (provides scene & device)
        sensor_cfg: SceneEntityCfg passed by IsaacLab containing .name
    """
    sensor = env.scene[sensor_cfg.name]
    f = _reduce_contact_forces(sensor.data.net_forces_w)
    return f  # [B,3]

def contact_force_norm(env, sensor_cfg):
    """Return contact force magnitude as shape [B,1]."""
    f = contact_force_vector(env, sensor_cfg)
    return f.norm(dim=-1, keepdim=True)

def contact_force_magnitude(env, sensor_cfg):
    """Return contact force magnitude as shape [B]. Suitable for reward terms."""
    return contact_force_norm(env, sensor_cfg).squeeze(-1)

def joint_effort_norm(env, asset_name: str = "robot", joint_names=("Servo1","Servo2")):
    """Return norm of specified joint efforts [B]."""
    asset = env.scene[asset_name]
    joint_ids = []
    for jn in joint_names:
        idxs, _ = asset.find_joints(jn)
        joint_ids.extend(idxs)
    efforts = asset.data.applied_torque[:, joint_ids]
    return efforts.norm(dim=-1)

def joint_pos_vel(env, asset_name: str = "robot", joint_names=("Servo1","Servo2")):
    """Concatenate joint pos & vel -> shape [B, 2*J]."""
    asset = env.scene[asset_name]
    ids = []
    for jn in joint_names:
        idxs, _ = asset.find_joints(jn)
        ids.extend(idxs)
    pos = asset.data.joint_pos[:, ids]
    vel = asset.data.joint_vel[:, ids]
    return torch.cat([pos, vel], dim=-1)

__all__ = [
    # action (optional)
    "VariableImpedanceActionTerm",
    "VariableImpedanceActionTermCfg",
    # observation helpers
    "contact_force_vector",
    "contact_force_norm",
    "joint_pos_vel",
    # reward helpers
    "contact_force_magnitude",
    "joint_effort_norm",
]