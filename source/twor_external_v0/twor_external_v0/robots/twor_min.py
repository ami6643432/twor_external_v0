import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import numpy as np

TWOR_MIN_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/amitabh/twor_external_v0/source/twor_external_v0/twor_external_v0/robots/usd/twor/urdf/Arm_min/Arm_min.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Servo1": - np.pi/8,
            "Servo2": np.pi / 2.0 - np.pi/8,
        },
        pos=(0.0, 0.0, 0.36),
    ),
    actuators={
        "servo1_act": ImplicitActuatorCfg(
            joint_names_expr=["Servo1"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            stiffness=100.0,
            damping=30.0,
        ),
        "servo2_act": ImplicitActuatorCfg(
            joint_names_expr=["Servo2"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=10.0,
            stiffness=100.0,
            damping=30.0,
        ),
    },
)
