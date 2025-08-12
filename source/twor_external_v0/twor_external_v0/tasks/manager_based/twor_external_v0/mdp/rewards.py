from .mdp.rewards import (
    TrackingPerformanceRewardCfg, tracking_performance_reward,
    ForceStabilityRewardCfg,     force_stability_reward,
    ImpedanceParameterRegularizationCfg, impedance_parameter_regularization,
    ContactQualityRewardCfg,     contact_quality_reward,
    EffortEfficiencyRewardCfg,   effort_efficiency_reward,
    TaskCompletionRewardCfg,     task_completion_reward,
)

class RewardsCfg:
    track_perf = RewTerm(
        func=tracking_performance_reward,
        weight=1.0,
        params=TrackingPerformanceRewardCfg().to_dict(),   # or inline dict
    )
    force_stab = RewTerm(
        func=force_stability_reward,
        weight=1.0,
        params=ForceStabilityRewardCfg().to_dict(),
    )
    imp_reg = RewTerm(
        func=impedance_parameter_regularization,
        weight=1.0,
        params=ImpedanceParameterRegularizationCfg().to_dict(),
    )
    contact_quality = RewTerm(
        func=contact_quality_reward,
        weight=1.0,
        params=ContactQualityRewardCfg().to_dict(),
    )
    effort_eff = RewTerm(
        func=effort_efficiency_reward,
        weight=0.1,
        params=EffortEfficiencyRewardCfg().to_dict(),
    )
    task_done = RewTerm(
        func=task_completion_reward,
        weight=10.0,
        params=TaskCompletionRewardCfg().to_dict(),
    )
