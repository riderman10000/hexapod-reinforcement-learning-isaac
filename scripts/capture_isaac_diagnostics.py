# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture deterministic Isaac Lab data for cross-simulator diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Template-Hexpod-Rl-Lab-Direct-v0")
parser.add_argument("--policy", type=Path, required=True, help="Exported policy.onnx used for raw-action parity.")
parser.add_argument("--interface", type=Path, default=Path("sim2sim/configs/policy_interface.yaml"))
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--joint-delta", type=float, default=0.05)
parser.add_argument("--response-seconds", type=float, default=0.5)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This diagnostic uses one environment on one device. Explicitly disable Kit's
# renderer multi-GPU mode while leaving CUDA visibility untouched so Vulkan and
# CUDA retain the same physical-device numbering. Isaac may still print its
# non-fatal IOMMU/P2P startup diagnostic on a multi-GPU workstation.
single_gpu_kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1"
)
args_cli.kit_args = f"{args_cli.kit_args} {single_gpu_kit_args}".strip()
app_launcher = AppLauncher(args_cli, multi_gpu=False)
simulation_app = app_launcher.app

import gymnasium as gym
import hexpod_rl_lab.tasks  # noqa: F401
import numpy as np
import torch

from isaaclab.utils.math import quat_apply_inverse

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from sim2sim.hexapod_mujoco.config import PolicyInterface, load_policy_interface
from sim2sim.hexapod_mujoco.policy import OnnxPolicy


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _observation_terms(observation: np.ndarray, interface: PolicyInterface) -> dict[str, list[float]]:
    terms: dict[str, list[float]] = {}
    start = 0
    for name, size in interface.observation_layout:
        terms[name] = observation[start : start + size].astype(float).tolist()
        start += size
    return terms


def _leg_index(joint_name: str) -> int:
    match = re.search(r"(?:body_leg_|leg_)(\d)", joint_name)
    if match is None:
        raise ValueError(f"Cannot determine leg for joint {joint_name!r}")
    return int(match.group(1))


def _reset_neutral(env) -> None:
    robot = env.robot
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins
    root_state[:, 7:] = 0.0
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    env._actions.zero_()
    env._previous_actions.zero_()
    env._processed_actions.copy_(joint_pos[:, env._joint_ids])
    env._gait_phase_offset.zero_()
    env.episode_length_buf.zero_()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def _foot_position_in_base(env, foot_body_id: int) -> torch.Tensor:
    robot = env.robot
    return quat_apply_inverse(
        robot.data.root_link_quat_w[0],
        robot.data.body_pos_w[0, foot_body_id] - robot.data.root_link_pos_w[0],
    )


def _capture_neutral_snapshot(env, interface: PolicyInterface, policy: OnnxPolicy, output_dir: Path) -> None:
    _reset_neutral(env)
    observation = env._get_observations()["policy"][0].detach().cpu().numpy().astype(np.float32)
    raw_action = policy(observation)
    _write_json(
        output_dir / "neutral_snapshot.json",
        {
            "schema_version": 1,
            "simulator": "isaac_physx",
            "joint_names": list(env._joint_names),
            "observation": observation.astype(float).tolist(),
            "observation_terms": _observation_terms(observation, interface),
            "raw_action": raw_action.astype(float).tolist(),
            "clipped_action": np.clip(raw_action, -interface.action_clip, interface.action_clip).astype(float).tolist(),
        },
    )


def _capture_joint_sweep(
    env,
    interface: PolicyInterface,
    output_dir: Path,
    joint_delta: float,
    response_seconds: float,
) -> None:
    sample_count = max(1, math.ceil(response_seconds / env.physics_dt))
    fieldnames = [
        "simulator",
        "joint_index",
        "joint_name",
        "time",
        "command_delta",
        "position_delta",
        "velocity",
        "actuator_force",
        "foot_dx",
        "foot_dy",
        "foot_dz",
    ]
    rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str | list[float]]] = []
    robot = env.robot
    if tuple(env._joint_names) != interface.policy_joint_names:
        raise ValueError("Isaac runtime joint order differs from the checked-in policy interface")

    for joint_index, joint_name in enumerate(env._joint_names):
        _reset_neutral(env)
        leg_index = _leg_index(joint_name)
        foot_ids, _ = robot.find_bodies(f"dummy_eef_{leg_index}", preserve_order=True)
        if len(foot_ids) != 1:
            raise RuntimeError(f"Expected one lower-leg body for leg {leg_index}, found {foot_ids}")
        neutral_foot = _foot_position_in_base(env, foot_ids[0]).clone()
        targets = robot.data.default_joint_pos.clone()
        robot_joint_index = env._joint_ids[joint_index]
        targets[0, robot_joint_index] += joint_delta
        robot.set_joint_position_target(targets)

        for sample_index in range(sample_count + 1):
            if sample_index > 0:
                robot.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            foot_delta = _foot_position_in_base(env, foot_ids[0]) - neutral_foot
            rows.append(
                {
                    "simulator": "isaac_physx",
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "time": sample_index * env.physics_dt,
                    "command_delta": joint_delta,
                    "position_delta": float(
                        robot.data.joint_pos[0, robot_joint_index]
                        - robot.data.default_joint_pos[0, robot_joint_index]
                    ),
                    "velocity": float(robot.data.joint_vel[0, robot_joint_index]),
                    "actuator_force": float(robot.data.applied_torque[0, robot_joint_index]),
                    "foot_dx": float(foot_delta[0]),
                    "foot_dy": float(foot_delta[1]),
                    "foot_dz": float(foot_delta[2]),
                }
            )

        final = rows[-1]
        summaries.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "command_delta": joint_delta,
                "final_position_delta": final["position_delta"],
                "final_foot_delta": [final["foot_dx"], final["foot_dy"], final["foot_dz"]],
            }
        )

    with (output_dir / "joint_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "joint_sweep_summary.json",
        {
            "schema_version": 1,
            "simulator": "isaac_physx",
            "physics_dt": env.physics_dt,
            "joint_delta": joint_delta,
            "response_seconds": response_seconds,
            "gravity_disabled": True,
            "joints": summaries,
        },
    )


def main() -> None:
    if args_cli.joint_delta <= 0.0 or args_cli.response_seconds <= 0.0:
        raise ValueError("Joint delta and response duration must be positive")
    interface = load_policy_interface(args_cli.interface)
    policy = OnnxPolicy(args_cli.policy, interface)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.reset_position_noise = 0.0
    env_cfg.reset_velocity_noise = 0.0
    env_cfg.reset_yaw_noise = 0.0
    env_cfg.robot_cfg.spawn.rigid_props.disable_gravity = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _capture_neutral_snapshot(env, interface, policy, args_cli.output_dir)
        _capture_joint_sweep(
            env,
            interface,
            args_cli.output_dir,
            args_cli.joint_delta,
            args_cli.response_seconds,
        )
        print(f"Isaac diagnostics: {args_cli.output_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
