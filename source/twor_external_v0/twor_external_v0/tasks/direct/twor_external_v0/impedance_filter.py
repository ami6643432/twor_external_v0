"""
Discrete-time second-order impedance filter implementation.

The filter implements:
M_v·Δ̈x + B_v·Δ̇x + K_v·Δx = F_ext

Output: x_ref = x_des + Δx
"""

from __future__ import annotations
import torch 
import numpy as np
from typing import Literal


class ImpedanceFilter:
    """
    Discrete-time second-order impedance filter that warps position reference based on external force.
    
    The filter obeys: M_v·Δ̈x + B_v·Δ̇x + K_v·Δx = F_ext
    Z-domain: H(z) = 1/(M_v·(z-1)²/T² + B_v·(z-1)/T + K_v)
    
    Output: x_ref = x_des + Δx
    """
    
    def __init__(
        self,
        M_v: torch.Tensor | float,
        B_v: torch.Tensor | float, 
        K_v: torch.Tensor | float,
        dt: float,
        num_envs: int = 1,
        num_joints: int = 2,
        device: str = "cuda",
        method: Literal["euler", "iir"] = "euler"
    ):
        """
        Initialize impedance filter.
        
        Args:
            M_v: Virtual mass [kg] or [kg·m²] for rotational joints
            B_v: Virtual damping [N·s/m] or [N·m·s/rad] 
            K_v: Virtual stiffness [N/m] or [N·m/rad]
            dt: Sample time [s]
            num_envs: Number of parallel environments
            num_joints: Number of joints per environment
            device: Device for computations
            method: Integration method ("euler" or "iir")
        """
        self.dt = dt
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device
        self.method = method
        
        # Convert parameters to tensors with proper shape [num_envs, num_joints]
        self.M_v = self._to_tensor(M_v)
        self.B_v = self._to_tensor(B_v)
        self.K_v = self._to_tensor(K_v)
        
        # Initialize state variables
        self.reset()
        
        # Pre-compute IIR filter coefficients if using IIR method
        if method == "iir":
            self._compute_iir_coefficients()
    
    def _to_tensor(self, value: torch.Tensor | float) -> torch.Tensor:
        """Convert value to properly shaped tensor."""
        if isinstance(value, (int, float)):
            return torch.full((self.num_envs, self.num_joints), value, 
                            device=self.device, dtype=torch.float32)
        else:
            value = value.to(device=self.device, dtype=torch.float32)
            if value.dim() == 1:
                value = value.unsqueeze(0).expand(self.num_envs, -1)
            return value
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset filter state - called only by environment."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # For Euler method: position and velocity history
        if not hasattr(self, 'delta_x'):
            self.delta_x = torch.zeros(self.num_envs, self.num_joints, device=self.device)
            self.delta_x_prev = torch.zeros(self.num_envs, self.num_joints, device=self.device)
            self.delta_x_dot = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        
        self.delta_x[env_ids] = 0.0
        self.delta_x_prev[env_ids] = 0.0
        self.delta_x_dot[env_ids] = 0.0
        
        # For IIR method: input/output history
        if self.method == "iir":
            if not hasattr(self, 'u_hist'):
                self.u_hist = torch.zeros(self.num_envs, self.num_joints, 3, device=self.device)
                self.y_hist = torch.zeros(self.num_envs, self.num_joints, 3, device=self.device)
            
            self.u_hist[env_ids] = 0.0
            self.y_hist[env_ids] = 0.0
    
    def _compute_iir_coefficients(self):
        """Pre-compute IIR filter coefficients for H(z) = b0/(a0 + a1*z^-1 + a2*z^-2)"""
        dt2 = self.dt ** 2
        
        # Denominator coefficients: M_v*(z-1)²/T² + B_v*(z-1)/T + K_v
        # Expanding: M_v/T²*(z² - 2z + 1) + B_v/T*(z - 1) + K_v
        # = M_v/T²*z² + (-2*M_v/T² + B_v/T)*z + (M_v/T² - B_v/T + K_v)
        # In standard form: a0*z² + a1*z + a2 = 0
        # For difference equation: a0*y[k] + a1*y[k-1] + a2*y[k-2] = b0*u[k]
        
        self.a0 = self.M_v / dt2  # coefficient of z²
        self.a1 = -2 * self.M_v / dt2 + self.B_v / self.dt  # coefficient of z¹
        self.a2 = self.M_v / dt2 - self.B_v / self.dt + self.K_v  # coefficient of z⁰
        self.b0 = torch.ones_like(self.a0)  # numerator coefficient
    
    def update_euler(self, F_ext: torch.Tensor, x_des: torch.Tensor) -> torch.Tensor:
        """
        Update filter using forward Euler integration.
        
        Implements: M_v·Δ̈x + B_v·Δ̇x + K_v·Δx = F_ext
        """
        # Compute acceleration: Δ̈x = (F_ext - B_v·Δ̇x - K_v·Δx) / M_v
        delta_x_ddot = (F_ext - self.B_v * self.delta_x_dot - self.K_v * self.delta_x) / self.M_v
        
        # Forward Euler integration
        # Δ̇x[k+1] = Δ̇x[k] + Δ̈x[k] * dt
        # Δx[k+1] = Δx[k] + Δ̇x[k] * dt
        self.delta_x_dot = self.delta_x_dot + delta_x_ddot * self.dt
        self.delta_x = self.delta_x + self.delta_x_dot * self.dt
        
        # Output reference: x_ref = x_des + Δx
        return x_des + self.delta_x
    
    def update_iir(self, F_ext: torch.Tensor, x_des: torch.Tensor) -> torch.Tensor:
        """
        Update filter using direct IIR implementation.
        
        H(z) = 1/(M_v·(z-1)²/T² + B_v·(z-1)/T + K_v)
        """
        # Shift history
        self.u_hist[:, :, 2] = self.u_hist[:, :, 1].clone()  # u[k-2] = u[k-1]
        self.u_hist[:, :, 1] = self.u_hist[:, :, 0].clone()  # u[k-1] = u[k]
        self.u_hist[:, :, 0] = F_ext  # u[k] = F_ext
        
        self.y_hist[:, :, 2] = self.y_hist[:, :, 1].clone()  # y[k-2] = y[k-1]
        self.y_hist[:, :, 1] = self.y_hist[:, :, 0].clone()  # y[k-1] = y[k]
        
        # Compute output: a0*y[k] + a1*y[k-1] + a2*y[k-2] = b0*u[k]
        # Therefore: y[k] = (b0*u[k] - a1*y[k-1] - a2*y[k-2]) / a0
        self.y_hist[:, :, 0] = (
            self.b0 * self.u_hist[:, :, 0] 
            - self.a1 * self.y_hist[:, :, 1] 
            - self.a2 * self.y_hist[:, :, 2]
        ) / self.a0
        
        self.delta_x = self.y_hist[:, :, 0]
        
        # Output reference: x_ref = x_des + Δx
        return x_des + self.delta_x
    
    def update(self, F_ext: torch.Tensor, x_des: torch.Tensor) -> torch.Tensor:
        """
        Update impedance filter.
        
        Args:
            F_ext: External force [num_envs, num_joints]
            x_des: Desired position [num_envs, num_joints]
            
        Returns:
            x_ref: Modified reference position [num_envs, num_joints]
        """
        # Ensure inputs have correct shape
        if F_ext.dim() == 1:
            F_ext = F_ext.unsqueeze(0)
        if x_des.dim() == 1:
            x_des = x_des.unsqueeze(0)
        
        if self.method == "euler":
            return self.update_euler(F_ext, x_des)
        elif self.method == "iir":
            return self.update_iir(F_ext, x_des)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def set_parameters(self, M_v: torch.Tensor | None = None, 
                      B_v: torch.Tensor | None = None,
                      K_v: torch.Tensor | None = None):
        """Update filter parameters (useful for variable impedance)."""
        if M_v is not None:
            self.M_v = self._to_tensor(M_v)
        if B_v is not None:
            self.B_v = self._to_tensor(B_v)
        if K_v is not None:
            self.K_v = self._to_tensor(K_v)
        
        # Recompute IIR coefficients if using IIR method
        if self.method == "iir":
            self._compute_iir_coefficients()


def test_impedance_filter():
    """Unit test for impedance filter with step force input."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
        plt = None
    
    # Test parameters
    dt = 0.01  # 100 Hz
    duration = 2.0
    steps = int(duration / dt)
    
    M_v = 1.0   # 1 kg virtual mass
    B_v = 10.0  # 10 N·s/m damping
    K_v = 100.0 # 100 N/m stiffness
    
    device = "cpu"
    
    # Create filters for both methods
    filter_euler = ImpedanceFilter(M_v, B_v, K_v, dt, num_envs=1, num_joints=1, 
                                  device=device, method="euler")
    filter_iir = ImpedanceFilter(M_v, B_v, K_v, dt, num_envs=1, num_joints=1, 
                                device=device, method="iir")
    
    # Test signals
    time = torch.arange(0, duration, dt, device=device)
    x_des = torch.zeros(steps, 1, 1, device=device)  # Zero desired position
    
    # Step force at t=0.5s
    F_ext = torch.zeros(steps, 1, 1, device=device)
    step_start = int(0.5 / dt)
    step_end = int(1.5 / dt)
    F_ext[step_start:step_end, 0, 0] = 10.0  # 10 N step force
    
    # Simulate
    x_ref_euler = torch.zeros(steps, 1, 1, device=device)
    x_ref_iir = torch.zeros(steps, 1, 1, device=device)
    
    for k in range(steps):
        x_ref_euler[k] = filter_euler.update(F_ext[k], x_des[k])
        x_ref_iir[k] = filter_iir.update(F_ext[k], x_des[k])
    
    # Plot results only if matplotlib is available
    if plt is not None:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(time.cpu(), F_ext[:, 0, 0].cpu(), 'r-', linewidth=2, label='External Force')
        plt.ylabel('Force [N]')
        plt.legend()
        plt.grid(True)
        plt.title('Impedance Filter Test: Step Force Response')
        
        plt.subplot(3, 1, 2)
        plt.plot(time.cpu(), x_ref_euler[:, 0, 0].cpu(), 'b-', linewidth=2, label='Euler Method')
        plt.plot(time.cpu(), x_ref_iir[:, 0, 0].cpu(), 'g--', linewidth=2, label='IIR Method')
        plt.ylabel('Position [m]')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        error = torch.abs(x_ref_euler[:, 0, 0] - x_ref_iir[:, 0, 0])
        plt.plot(time.cpu(), error.cpu(), 'k-', linewidth=1, label='|Euler - IIR|')
        plt.ylabel('Error [m]')
        plt.xlabel('Time [s]')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        try:
            plt.savefig('impedance_filter_test.png', dpi=150)
            plt.show(block=False)  # Non-blocking show
            plt.pause(2)  # Display for 2 seconds
            plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"Plot display failed: {e}")
    
    print(f"Max error between methods: {error.max().item():.6f} m")
    print(f"Steady-state displacement: {x_ref_euler[-1, 0, 0].item():.6f} m")
    print(f"Expected steady-state (F/K): {10.0/K_v:.6f} m")


if __name__ == "__main__":
    test_impedance_filter()