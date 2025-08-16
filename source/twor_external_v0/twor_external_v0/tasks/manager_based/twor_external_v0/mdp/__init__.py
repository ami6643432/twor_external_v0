# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for TwoR variable impedance control."""

from __future__ import annotations

import torch

# Import built-in Isaac Lab MDP helpers
try:
    from isaaclab.envs.mdp import *
except Exception as _base_mdp_err:
    print(f"[WARN] Base MDP import skipped: {_base_mdp_err}")

# Import custom command and action terms
from .commands import JointPositionCommandCfg, JointPositionCommand
from .actions import VariableImpedanceActionCfg, VariableImpedanceAction

# Import managers for type hints
from isaaclab.managers import SceneEntityCfg

def contact_force_norm(env, sensor_cfg):
    """Return contact force magnitude as shape [B,1]."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    # Reduce to [num_envs, 3] if needed
    while forces.ndim > 2:
        forces = forces.sum(dim=1)
    return forces.norm(dim=-1, keepdim=True)

def contact_force_magnitude(env, sensor_cfg):
    """Return contact force magnitude as shape [B]. Suitable for reward terms."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    # Reduce to [num_envs, 3] if needed
    while forces.ndim > 2:
        forces = forces.sum(dim=1)
    return torch.norm(forces, dim=-1)

__all__ = [
    "JointPositionCommandCfg",
    "JointPositionCommand", 
    "VariableImpedanceActionCfg",
    "VariableImpedanceAction",
    "contact_force_norm",
    "contact_force_magnitude",
]