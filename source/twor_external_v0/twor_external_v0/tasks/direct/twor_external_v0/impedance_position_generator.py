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

        # Two‐step history for integration
        if x0 is None:
            self.x_d_prev2 = torch.zeros_like(M_d)
            self.x_d_prev1 = torch.zeros_like(M_d)
        else:
            self.x_d_prev2 = x0.clone()
            self.x_d_prev1 = x0.clone()

    def update(self, F_ext: torch.Tensor, x_r) -> torch.Tensor:
        """
        Compute next commanded joint positions.

        Args:
            F_ext: external joint torques [B, n_joints]
            x_r:   reference joint positions [B, n_joints] or compatible
        Returns:
            x_d: new commanded joint positions [B, n_joints]
        """
        # Ensure x_r is a torch.Tensor on the correct device and dtype
        if not isinstance(x_r, torch.Tensor):
            x_r = torch.tensor(x_r, dtype=self.x_d_prev1.dtype, device=self.x_d_prev1.device)
        else:
            x_r = x_r.to(dtype=self.x_d_prev1.dtype, device=self.x_d_prev1.device)

        # velocity estimate
        v_d = (self.x_d_prev1 - self.x_d_prev2) / self.dt

        # virtual mass-damper-spring accel.
        a_k = (F_ext
               - self.D_d * v_d
               - self.K_d * (self.x_d_prev1 - x_r)
              ) / self.M_d

        # forward-Euler update
        x_d = 2*self.x_d_prev1 - self.x_d_prev2 + a_k*(self.dt**2)

        # shift history
        self.x_d_prev2, self.x_d_prev1 = self.x_d_prev1, x_d
        return x_d

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
        tau = (M.matmul(q_ddot_d)
               + C.matmul(q_dot_d)
               + g
               + Kp*(q_d - q)
               + Kd*(q_dot_d - q_dot))
        return tau
