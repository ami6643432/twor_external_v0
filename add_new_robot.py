# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot (TWOR) to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg



# Import the TWOR_CONFIG from your twor.py file
from twor_external_v0.robots.twor import TWOR_CONFIG


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Twor = TWOR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Twor")

    # cone object at (-0.25, 0, 0)
    # Rigid Object
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.25, 0, 0.25)),
    )

    # contact_forces_H = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Twor/Link2",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=True,
    # )

    


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    max = 100

    while simulation_app.is_running():
        # reset
        if count % max == 0:
            count = 0
            # Reset Twor robot
            root_twor_state = scene["Twor"].data.default_root_state.clone()
            root_twor_state[:, :3] += scene.env_origins
            scene["Twor"].write_root_pose_to_sim(root_twor_state[:, :7])
            scene["Twor"].write_root_velocity_to_sim(root_twor_state[:, 7:])
            joint_pos, joint_vel = (
                scene["Twor"].data.default_joint_pos.clone(),
                scene["Twor"].data.default_joint_vel.clone(),
            )
            scene["Twor"].write_joint_state_to_sim(joint_pos, joint_vel)
            # # Reset cone object
            # if "Cone" in scene:
            #     root_cone_state = scene["Cone"].data.default_root_state.clone()
            #     scene["Cone"].write_root_pose_to_sim(root_cone_state[:, :7])
            #     scene["Cone"].write_root_velocity_to_sim(root_cone_state[:, 7:])
            #     scene["Cone"].reset()
            # scene.reset()
            # print("[INFO]: Resetting Twor and Cone state...")

        # Example: oscillate Servo1 and Servo2, keep Clamp at custom initial value
        wave_action = scene["Twor"].data.default_joint_pos.clone()
        wave_action[:, 0] =  np.pi/2 * (count % max)/max - np.pi/8
        wave_action[:, 1] = -np.pi/2 * (count % max)/max + np.pi / 2.0 + np.pi/8
        scene["Twor"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()