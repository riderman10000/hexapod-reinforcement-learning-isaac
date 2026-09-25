# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture a deterministic Isaac/PhysX policy rollout in the shared gait schema."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Template-Hexpod-Rl-Lab-Direct-v0")
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--interface", type=Path, default=Path("sim2sim/configs/policy_interface.yaml"))
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--phase-offset", type=float, default=0.0)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

single_gpu_kit_args = (
    "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"
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

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.policy import OnnxPolicy
from sim2sim.rollout_data import (
    SCHEMA_VERSION,
    ArrayRecorder,
    desired_tripod_contacts,
    file_sha256,
    phase_at_step,
    summarize_policy_rollout,
    write_json,
)


class IsaacStateSampler:
    """Read matched robot, foot, and normal-contact data from one Isaac environment."""

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.robot
        self.foot_marker_ids = []
        for leg_index in range(6):
            ids, names = self.robot.find_bodies(f"dummy_eef_{leg_index}", preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(f"Expected dummy_eef_{leg_index}, found {names}")
            self.foot_marker_ids.append(ids[0])

    @staticmethod
    def _numpy(value) -> np.ndarray:
        return value.detach().cpu().numpy().copy()

    def sample(self, time_seconds: float) -> dict[str, np.ndarray | float]:
        robot = self.robot
        base_position = self._numpy(robot.data.root_link_pos_w[0])
        base_quaternion = self._numpy(robot.data.root_link_quat_w[0])
        foot_position_world = self._numpy(robot.data.body_link_pos_w[0, self.foot_marker_ids])
        foot_position_body = self._numpy(
            quat_apply_inverse(
                robot.data.root_link_quat_w[0].expand(6, -1),
                robot.data.body_link_pos_w[0, self.foot_marker_ids] - robot.data.root_link_pos_w[0],
            )
        )
        foot_force = self._numpy(self.env.contact_sensor.data.net_forces_w[0, self.env._foot_ids])
        foot_contact = np.linalg.norm(foot_force, axis=1) > self.env.cfg.foot_contact_force_threshold
        return {
            "time": float(time_seconds),
            "base_position": base_position,
            "base_quaternion_wxyz": base_quaternion,
            "world_linear_velocity": self._numpy(robot.data.root_lin_vel_w[0]),
            "body_linear_velocity": self._numpy(robot.data.root_lin_vel_b[0]),
            "body_angular_velocity": self._numpy(robot.data.root_ang_vel_b[0]),
            "joint_position": self._numpy(robot.data.joint_pos[0, self.env._joint_ids]),
            "joint_velocity": self._numpy(robot.data.joint_vel[0, self.env._joint_ids]),
            "joint_force": self._numpy(robot.data.applied_torque[0, self.env._joint_ids]),
            "foot_position_world": foot_position_world,
            "foot_position_body": foot_position_body,
            "foot_contact": foot_contact,
            "foot_contact_force_world": foot_force,
        }


def _prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}{name}": value for name, value in values.items()}


def _reset_neutral(env, phase_offset: float) -> None:
    robot = env.robot
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += env.scene.env_origins
    root_state[:, 7:] = 0.0
    joint_position = robot.data.default_joint_pos.clone()
    joint_velocity = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    robot.write_data_to_sim()
    env._actions.zero_()  # noqa: SLF001
    env._previous_actions.zero_()  # noqa: SLF001
    env._processed_actions.copy_(joint_position[:, env._joint_ids])  # noqa: SLF001
    env._gait_phase_offset.fill_(phase_offset)  # noqa: SLF001
    env.episode_length_buf.zero_()
    env.common_step_counter = 0
    env.sim.forward()
    env.scene.update(dt=0.0)


def _manual_policy_step(env, action: torch.Tensor, control_step: int, sampler, recorder, target) -> None:
    env._pre_physics_step(action)  # noqa: SLF001
    for substep in range(env.cfg.decimation):
        env._sim_step_counter += 1  # noqa: SLF001
        env._apply_action()  # noqa: SLF001
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        recorder.append(
            policy_step=control_step,
            substep=substep,
            joint_target=target,
            **sampler.sample((control_step * env.cfg.decimation + substep + 1) * env.physics_dt),
        )
    env.episode_length_buf += 1
    env.common_step_counter += 1


def main() -> None:
    if args_cli.duration <= 0.0:
        raise ValueError("Duration must be positive")
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
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        if tuple(env._joint_names) != interface.policy_joint_names:  # noqa: SLF001
            raise ValueError("Isaac runtime joint order differs from policy_interface.yaml")
        _reset_neutral(env, args_cli.phase_offset)
        sampler = IsaacStateSampler(env)
        policy_recorder = ArrayRecorder()
        physics_recorder = ArrayRecorder()
        observation = env._get_observations()["policy"][0].detach().cpu().numpy().astype(np.float32)  # noqa: SLF001
        maximum_steps = math.ceil(args_cli.duration / env.step_dt)
        fall_reason = ""

        for control_step in range(maximum_steps):
            state = sampler.sample(control_step * env.step_dt)
            raw_action = policy(observation)
            applied_action = np.clip(raw_action, -interface.action_clip, interface.action_clip).astype(np.float32)
            # Use runtime neutral values so the recorded control contract remains explicit.
            neutral = env.robot.data.default_joint_pos[0, env._joint_ids].detach().cpu().numpy()  # noqa: SLF001
            lower = env.robot.data.soft_joint_pos_limits[0, env._joint_ids, 0].detach().cpu().numpy()  # noqa: SLF001
            upper = env.robot.data.soft_joint_pos_limits[0, env._joint_ids, 1].detach().cpu().numpy()  # noqa: SLF001
            target = np.clip(neutral + interface.action_scale * applied_action, lower, upper)
            action_tensor = torch.as_tensor(raw_action, dtype=torch.float32, device=env.device).unsqueeze(0)
            phase = phase_at_step(
                control_step,
                env.step_dt,
                env.cfg.gait_frequency,
                args_cli.phase_offset,
            )
            _manual_policy_step(env, action_tensor, control_step, sampler, physics_recorder, target)
            terminated_tensor, timeout_tensor = env._get_dones()  # noqa: SLF001
            terminated = bool(terminated_tensor[0] or timeout_tensor[0])
            next_state = sampler.sample((control_step + 1) * env.step_dt)
            policy_recorder.append(
                step=control_step,
                phase=phase,
                observation=observation,
                raw_action=raw_action,
                applied_action=applied_action,
                joint_target=target,
                desired_foot_contact=desired_tripod_contacts(phase),
                terminated=terminated,
                **state,
                **_prefixed("next_", next_state),
            )
            observation = env._get_observations()["policy"][0].detach().cpu().numpy().astype(np.float32)  # noqa: SLF001
            if terminated:
                fall_reason = "environment_termination"
                break

        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        policy_recorder.save(args_cli.output_dir / "policy_steps.npz")
        physics_recorder.save(args_cli.output_dir / "physics_steps.npz")
        policy_data = policy_recorder.arrays()
        summary = summarize_policy_rollout(policy_data, env.cfg.target_forward_velocity)
        summary["fall_reason"] = fall_reason
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "simulator": "isaac_physx",
            "mode": "closed_loop_policy",
            "task": args_cli.task,
            "policy": str(args_cli.policy.resolve()),
            "policy_sha256": file_sha256(args_cli.policy),
            "interface": str(args_cli.interface.resolve()),
            "joint_names": list(env._joint_names),  # noqa: SLF001
            "foot_names": [f"dummy_eef_{index}" for index in range(6)],
            "contact_body_names": list(env._foot_names),  # noqa: SLF001
            "physics_dt": env.physics_dt,
            "policy_dt": env.step_dt,
            "decimation": env.cfg.decimation,
            "phase_offset": args_cli.phase_offset,
            "command_world": env._commands[0].detach().cpu().numpy().astype(float).tolist(),  # noqa: SLF001
            "contact_force_threshold": env.cfg.foot_contact_force_threshold,
            "policy_steps": policy_recorder.sample_count,
            "physics_steps": physics_recorder.sample_count,
        }
        write_json(args_cli.output_dir / "metadata.json", metadata)
        write_json(args_cli.output_dir / "summary.json", summary)
        print(f"Isaac rollout: {args_cli.output_dir.resolve()}")
        print(f"Policy steps: {policy_recorder.sample_count}; physics steps: {physics_recorder.sample_count}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
