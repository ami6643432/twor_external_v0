#!/usr/bin/env python3
import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser("Run manager-based Twor scene and print contact forces.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run")
parser.add_argument("--no-gui", action="store_true", help="Force headless stepping irrespective of app.is_running()")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def main():
    # Launch Omniverse app BEFORE importing heavy isaaclab modules
    app = AppLauncher(args).app
    print("[DEBUG] App launched, entering setup...")
    try:
        # Import after app launch so isaacsim is available
        import sys, time, importlib
        print("[DEBUG] (A) Starting manager-based imports..."); sys.stdout.flush()
        try:
            print("[DEBUG] (B) Importing isaaclab.envs..."); sys.stdout.flush()
            ManagerBasedRLEnv_mod = importlib.import_module("isaaclab.envs")
            ManagerBasedRLEnv = getattr(ManagerBasedRLEnv_mod, "ManagerBasedRLEnv")
            print("[DEBUG] (C) Imported ManagerBasedRLEnv"); sys.stdout.flush()
        except BaseException as e:
            import traceback
            print("[ERROR] Failed importing isaaclab.envs:", repr(e))
            traceback.print_exc()
            return
        try:
            print("[DEBUG] (D) Importing env cfg module..."); sys.stdout.flush()
            cfg_mod_path = "twor_external_v0.tasks.manager_based.twor_external_v0.twor_external_v0_env_cfg"
            env_cfg_mod = importlib.import_module(cfg_mod_path)
            TworExternalV0EnvCfg = getattr(env_cfg_mod, "TworExternalV0EnvCfg")
            print("[DEBUG] (E) Imported TworExternalV0EnvCfg"); sys.stdout.flush()
        except BaseException as e:
            import traceback
            print("[ERROR] Failed importing env cfg module:", repr(e))
            traceback.print_exc()
            return
        print("[DEBUG] (F) Creating env config..."); sys.stdout.flush()
        cfg = TworExternalV0EnvCfg()
        cfg.scene.num_envs = args.num_envs
        print("[DEBUG] (G) Instantiating ManagerBasedRLEnv..."); sys.stdout.flush()
        try:
            env = ManagerBasedRLEnv(cfg)
        except Exception as e:
            import traceback
            print("[ERROR] Exception during env construction:", e)
            traceback.print_exc()
            return
        print("[DEBUG] (H) Env created. Resetting..."); sys.stdout.flush()
        try:
            obs, _ = env.reset()
        except Exception as e:
            import traceback
            print("[ERROR] Exception during env.reset():", e)
            traceback.print_exc()
            return
        print("[DEBUG] (I) Reset complete. Scene assets:", list(env.scene.keys())); sys.stdout.flush()
        
        # Check for contact sensor
        if "contact_sensor" not in env.scene:
            print("[WARN] 'contact_sensor' not found in scene. Available keys:", list(env.scene.keys()))
        else:
            try:
                print("[DEBUG] Contact sensor prim:", env.scene["contact_sensor"].prim_path)
            except Exception as e:
                print("[WARN] Could not fetch contact sensor prim:", e)

        t = 0
        period = 1000

        def step_once(tick: int):
            # Simple oscillation of Servo1 & Servo2 (position targets)
            wave = env.scene["robot"].data.default_joint_pos.clone()
            frac = (t % period) / period
            wave[:, 0] = np.pi / 2.0 * frac - np.pi / 8.0
            wave[:, 1] = -np.pi / 2.0 * frac + np.pi / 2.0 + np.pi / 8.0
            env.scene["robot"].set_joint_position_target(wave)
            env.scene.write_data_to_sim()

            # zero actions (we use joint_effort term set to zero)
            action_dim = env.action_space.shape[0]
            actions = torch.zeros((env.num_envs, action_dim), device=env.device)

            try:
                obs, rew, terminated, truncated, info = env.step(actions)
            except Exception as e:
                import traceback, sys
                print(f"[ERROR] Exception during env.step at t={t}:", e)
                traceback.print_exc()
                sys.exit(1)

            # print contact forces
            forces = env.scene["contact_sensor"].data.net_forces_w
            if forces.ndim == 3 and forces.shape[1] == 1:
                forces_to_print = forces.squeeze(1)
            else:
                forces_to_print = forces
            if (t % 10) == 0:
                print(f"[DEBUG] t={t} forces=", forces_to_print)

        # Prefer UI loop when available; otherwise run a fixed-step fallback
        max_steps = args.steps
        ran_any = False
        for _ in range(max_steps):
            if (not args.no_gui) and (not app.is_running()):
                # If GUI expected but app not running yet, yield once then continue
                import time
                time.sleep(0.01)
                if not app.is_running():
                    break
            step_once(t)
            t += 1
            ran_any = True

        if not ran_any:
            print("[WARN] No simulation steps executed. Possible causes: headless mode detected app.is_running()==False, typo in CLI args, or early shutdown. Retry with: python -u scripts/test_manager_contact.py --num_envs 1 --steps 200 --no-gui")
            sys.stdout.flush()
    finally:
        app.close()


if __name__ == "__main__":
    main()


