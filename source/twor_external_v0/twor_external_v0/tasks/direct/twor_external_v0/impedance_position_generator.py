# File: twor_external_v0/impedance_position_generator.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch

# =============================================================================
# Joint-Space Impedance Position Generator
# =============================================================================

class ImpedancePositionGenerator:
    """
    Simple joint-space impedance generator:
      M_d·(ẍ_d) + D_d·(ẋ_d - ẋ_r) + K_d·(x_d - x_r) = F_ext
    implemented via forward-Euler integration.
    """

    def __init__(
        self,
        M_d: torch.Tensor,
        D_d: torch.Tensor,
        K_d: torch.Tensor,
        dt: float,
        x0: torch.Tensor | None = None
    ):
        """
        Args:
            M_d: virtual mass per joint (shape [n_joints])
            D_d: virtual damping per joint
            K_d: virtual stiffness per joint
            dt:  control timestep (s)
            x0:  initial commanded joint positions
        """
        self.M_d = M_d
        self.D_d = D_d
        self.K_d = K_d
        self.dt  = dt

        # Initialize with proper batch dimensions
        if x0 is None:
            # Create default zero tensors with proper shape
            self.x_d_prev2 = torch.zeros(1, len(M_d), device=M_d.device, dtype=M_d.dtype)
            self.x_d_prev1 = torch.zeros(1, len(M_d), device=M_d.device, dtype=M_d.dtype)
        else:
            # Ensure x0 has batch dimension [batch, n_joints]
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)
            self.x_d_prev2 = x0.clone()
            self.x_d_prev1 = x0.clone()

    def update(self, F_ext: torch.Tensor, x_r) -> torch.Tensor:
        """
        Compute next commanded joint positions.

        Args:
            F_ext: external joint torques [B, n_joints]
            x_r:   reference joint positions [B, n_joints] or list/tuple
        Returns:
            x_d: new commanded joint positions [B, n_joints]
        """
        # Ensure x_r has proper dimensions and device
        if isinstance(x_r, (list, tuple)):
            x_r = torch.tensor(x_r, dtype=self.x_d_prev1.dtype, device=self.x_d_prev1.device)
        elif not isinstance(x_r, torch.Tensor):
            x_r = torch.tensor(x_r, dtype=self.x_d_prev1.dtype, device=self.x_d_prev1.device)
        else:
            x_r = x_r.to(dtype=self.x_d_prev1.dtype, device=self.x_d_prev1.device)

        # Ensure proper batch dimensions
        if x_r.dim() == 1:
            x_r = x_r.unsqueeze(0)
        if F_ext.dim() == 1:
            F_ext = F_ext.unsqueeze(0)
            
        # Expand x_r to match F_ext batch size if needed
        if x_r.shape[0] == 1 and F_ext.shape[0] > 1:
            x_r = x_r.expand(F_ext.shape[0], -1)

        # velocity estimate
        v_d = (self.x_d_prev1 - self.x_d_prev2) / self.dt

        # Expand parameters to match batch size
        if self.M_d.dim() == 1:
            M_d = self.M_d.unsqueeze(0).expand(F_ext.shape[0], -1)
            D_d = self.D_d.unsqueeze(0).expand(F_ext.shape[0], -1)
            K_d = self.K_d.unsqueeze(0).expand(F_ext.shape[0], -1)
        else:
            M_d = self.M_d
            D_d = self.D_d
            K_d = self.K_d

        # Expand history to match batch size if needed
        if self.x_d_prev1.shape[0] != F_ext.shape[0]:
            self.x_d_prev1 = self.x_d_prev1.expand(F_ext.shape[0], -1)
            self.x_d_prev2 = self.x_d_prev2.expand(F_ext.shape[0], -1)
            v_d = v_d.expand(F_ext.shape[0], -1)

        # virtual mass-damper-spring acceleration
        a_k = (F_ext - D_d * v_d - K_d * (self.x_d_prev1 - x_r)) / M_d

        # forward-Euler update
        x_d = 2*self.x_d_prev1 - self.x_d_prev2 + a_k*(self.dt**2)

        # shift history
        self.x_d_prev2 = self.x_d_prev1.clone()
        self.x_d_prev1 = x_d.clone()
        return x_d

    def reset(self, env_ids: torch.Tensor | None = None, x0: torch.Tensor | None = None):
        """Reset the impedance generator state - called only by environment."""
        if x0 is None:
            if env_ids is None:
                # Reset all environments
                self.x_d_prev2.zero_()
                self.x_d_prev1.zero_()
            else:
                # Reset specific environments
                self.x_d_prev2[env_ids] = 0.0
                self.x_d_prev1[env_ids] = 0.0
        else:
            # Ensure x0 has proper dimensions
            if x0.dim() == 1:
                x0 = x0.unsqueeze(0)
                
            if env_ids is None:
                # Reset all environments to x0
                self.x_d_prev2 = x0.clone()
                self.x_d_prev1 = x0.clone()
            else:
                # Reset specific environments to x0
                if x0.shape[0] == 1:
                    x0 = x0.expand(len(env_ids), -1)
                self.x_d_prev2[env_ids] = x0
                self.x_d_prev1[env_ids] = x0

    def compute_servo_torque(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
        q_d: torch.Tensor,
        q_dot_d: torch.Tensor,
        q_ddot_d: torch.Tensor,
        M_func,
        C_func,
        g_func,
        Kp: torch.Tensor,
        Kd: torch.Tensor
    ) -> torch.Tensor:
        """
        Model-based joint torque:
        τ = M(q)·q̈_d + C(q,q̇)·q̇_d + g(q)
            + Kp*(q_d−q) + Kd*(q̇_d−q̇)
        """
        M = M_func(q)
        C = C_func(q, q_dot)
        g = g_func(q)
        tau = (M.matmul(q_ddot_d.unsqueeze(-1)).squeeze(-1)
               + C.matmul(q_dot_d.unsqueeze(-1)).squeeze(-1)
               + g
               + Kp*(q_d - q)
               + Kd*(q_dot_d - q_dot))
        return tau