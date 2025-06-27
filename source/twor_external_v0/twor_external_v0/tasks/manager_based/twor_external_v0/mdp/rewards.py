from isaaclab.managers import RewardTerm, RewardTermCfg

class AdaptiveImpedanceRewardTerm(RewardTerm):
    """Reward term for learning adaptive impedance control."""
    
    def __init__(self, cfg: "AdaptiveImpedanceRewardTermCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._action_term = env.action_manager._terms[cfg.action_term_name]
        
        # Track performance metrics
        self._tracking_error = torch.zeros(env.num_envs, device=env.device)
        self._force_error = torch.zeros(env.num_envs, device=env.device)
        self._stability_metric = torch.zeros(env.num_envs, device=env.device)
    
    def compute(self, dt: float) -> torch.Tensor:
        """Compute reward based on task performance and stability."""
        rewards = torch.zeros(self._env.num_envs, device=self._env.device)
        
        # Task-specific reward (e.g., tracking)
        if self.cfg.reward_tracking:
            target_pos = self._env.command_manager.get_command("ee_pose")[:, :3]
            current_pos = self._asset.data.body_pos_w[:, self.cfg.ee_body_idx]
            self._tracking_error = torch.norm(target_pos - current_pos, dim=-1)
            rewards += self.cfg.tracking_weight * torch.exp(-self._tracking_error)
        
        # Force regulation reward
        if self.cfg.reward_force:
            target_force = self._env.command_manager.get_command("ee_force")
            current_force = self._get_ee_force()
            self._force_error = torch.norm(target_force - current_force, dim=-1)
            rewards += self.cfg.force_weight * torch.exp(-self._force_error)
        
        # Stability reward (penalize oscillations)
        if self.cfg.reward_stability:
            joint_acc = self._estimate_joint_acceleration()
            self._stability_metric = torch.norm(joint_acc, dim=-1)
            rewards -= self.cfg.stability_weight * self._stability_metric
        
        # Effort penalty
        if self.cfg.penalize_effort:
            torques = self._asset.data.joint_torque_target
            effort = torch.sum(torques ** 2, dim=-1)
            rewards -= self.cfg.effort_weight * effort
        
        # Impedance adaptation reward (encourage appropriate stiffness)
        if self.cfg.reward_adaptation:
            # Reward lower stiffness when no contact, higher when in contact
            contact_forces = torch.norm(self._external_torques, dim=-1)
            stiffness_mean = torch.mean(self._action_term._processed_stiffness, dim=-1)
            
            # Normalize
            contact_normalized = torch.tanh(contact_forces / self.cfg.contact_threshold)
            stiffness_normalized = stiffness_mean / self.cfg.stiffness_max
            
            # Reward matching: high contact -> high stiffness, low contact -> low stiffness
            adaptation_reward = 1.0 - torch.abs(contact_normalized - stiffness_normalized)
            rewards += self.cfg.adaptation_weight * adaptation_reward
        
        return rewards
    
    def _estimate_joint_acceleration(self) -> torch.Tensor:
        """Estimate joint accelerations for stability metric."""
        # Simple finite difference
        if not hasattr(self, '_prev_joint_vel'):
            self._prev_joint_vel = self._asset.data.joint_vel.clone()
            return torch.zeros_like(self._asset.data.joint_vel)
        
        dt = self._env.physics_dt
        joint_acc = (self._asset.data.joint_vel - self._prev_joint_vel) / dt
        self._prev_joint_vel = self._asset.data.joint_vel.clone()
        
        return joint_acc

@configclass
class AdaptiveImpedanceRewardTermCfg(RewardTermCfg):
    """Configuration for adaptive impedance reward."""
    
    class_type: type = AdaptiveImpedanceRewardTerm
    
    asset_name: str = MISSING
    action_term_name: str = "variable_impedance"
    ee_body_idx: int = -1
    
    # Reward components
    reward_tracking: bool = True
    tracking_weight: float = 1.0
    
    reward_force: bool = False
    force_weight: float = 0.5
    
    reward_stability: bool = True
    stability_weight: float = 0.1
    
    penalize_effort: bool = True
    effort_weight: float = 0.01
    
    reward_adaptation: bool = True
    adaptation_weight: float = 0.5
    contact_threshold: float = 10.0
    stiffness_max: float = 1000.0