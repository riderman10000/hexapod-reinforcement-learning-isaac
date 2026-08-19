# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the runtime policy contract of an Isaac Lab environment as JSON."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump the ordered policy interface used by a trained environment.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import hexpod_rl_lab.tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """Create one environment and print the resolved policy-facing order."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    direct_env = env.unwrapped

    contract = {
        "task": args_cli.task,
        "observation_size": int(direct_env.cfg.observation_space),
        "action_size": int(direct_env.cfg.action_space),
        "physics_dt": float(direct_env.cfg.sim.dt),
        "decimation": int(direct_env.cfg.decimation),
        "policy_dt": float(direct_env.step_dt),
        "action_scale": float(direct_env.cfg.action_scale),
        "action_clip": float(direct_env.cfg.action_clip),
        "joint_names": list(direct_env._joint_names),
        "body_names": list(direct_env.robot.body_names),
    }
    print("POLICY_INTERFACE_BEGIN")
    print(json.dumps(contract, indent=2))
    print("POLICY_INTERFACE_END")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
