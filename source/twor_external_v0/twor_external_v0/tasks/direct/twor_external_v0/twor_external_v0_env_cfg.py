# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from gymnasium.spaces import Box
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg

from twor_external_v0.robots.twor import TWOR_CONFIG

# module‐level constants
Kmax = 500.0    # max stiffness
Dmax = 20.0     # max damping

@configclass
class TworExternalV0EnvCfg(DirectRLEnvCfg):
    # ————— Run settings —————
    decimation       = 1
    episode_length_s = 10.0

    # ————— Spaces —————
    # 4 actions: [k1, d1, k2, d2]
    action_space: Box = Box(
        low = np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float32),
        high= np.array([Kmax, Dmax, Kmax, Dmax], dtype=np.float32),
        dtype=np.float32
    )
    # 15 obs: [Fx, Fy, Fz, q1, q2, q1_des, q2_des,
    #          q1_dot, q2_dot, q1_dot_des, q2_dot_des,
    #          q1_ddot, q2_ddot, q1_ddot_des, q2_ddot_des]
    observation_space: Box = Box(
        low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
    )

    # ————— Physics —————
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation
    )

    # ————— Robot —————
    robot_cfg: ArticulationCfg = TWOR_CONFIG.replace(
        prim_path="/World/envs/env_.*/Twor",
        spawn=TWOR_CONFIG.spawn.replace(activate_contact_sensors=True)
    )

    # ————— Scene —————
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50, env_spacing=2.0, replicate_physics=True
    )

    # ————— Joint names —————
    servo1_dof_name = "Servo1"
    servo2_dof_name = "Servo2"

    # —— optional reward/done scales & limits ——
    force_scale:      float = 1.0       # penalty weight for contact force
    max_joint_pos:    float = 1.5708    # rad, 90°
    max_allowed_force:float = 200.0     # N

    # no extra “state” vector
    state_space: int = 0

    # in TworExternalV0EnvCfg
    target_pos_x: float = -1                    # desired pos for the cube
    w_pos: float = 10.0                       # weight on terminal position error
    w_tracking: float = 1000.0                    # joint‐tracking weight
    w_stiffness: float = 1e-6                  # stiffness penalty (small scale)

    # ————— Asset paths —————
    nvidia_cube_usd_path: str = "./Props/Blocks/nvidia_cube.usd"

