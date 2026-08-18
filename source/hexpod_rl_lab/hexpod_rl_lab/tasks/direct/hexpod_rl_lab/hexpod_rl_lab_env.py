# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .hexpod_rl_lab_env_cfg import HexpodRlLabEnvCfg


class HexpodRlLabEnv(DirectRLEnv):
    cfg: HexpodRlLabEnvCfg

    def __init__(self, cfg: HexpodRlLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, self._joint_names = self.robot.find_joints(".*")
        if len(self._joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                f"Expected {self.cfg.action_space} actuated joints, found {len(self._joint_ids)}: {self._joint_names}"
            )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = self.robot.data.default_joint_pos[:, self._joint_ids].clone()

        # Body-frame command: forward velocity, lateral velocity, and yaw rate.
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands[:, 0] = self.cfg.target_forward_velocity
        self._commands[:, 1] = self.cfg.target_lateral_velocity
        self._commands[:, 2] = self.cfg.target_yaw_velocity

        self._base_ids, base_names = self.contact_sensor.find_bodies("base_link")
        if len(self._base_ids) != 1:
            raise RuntimeError(f"Expected one base_link body in the contact sensor, found {base_names}")
        self._undesired_contact_ids, undesired_contact_names = self.contact_sensor.find_bodies("leg_._[12]")
        if len(self._undesired_contact_ids) != 12:
            raise RuntimeError(
                f"Expected the 12 proximal/middle leg bodies in the contact sensor, found {undesired_contact_names}"
            )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "track_forward_velocity",
                "track_yaw_velocity",
                "alive",
                "lateral_velocity_l2",
                "vertical_velocity_l2",
                "angular_velocity_xy_l2",
                "flat_orientation_l2",
                "joint_torque_l2",
                "joint_acceleration_l2",
                "action_rate_l2",
                "undesired_contacts",
                "terminated",
            )
        }

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=self.cfg.ground_material),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip))

        default_joint_pos = self.robot.data.default_joint_pos[:, self._joint_ids]
        self._processed_actions = default_joint_pos + self.cfg.action_scale * self._actions

        # Keep position targets inside the USD joint limits even if action parameters change later.
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=joint_limits[..., 0],
            max=joint_limits[..., 1],
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        joint_pos_rel = (
            self.robot.data.joint_pos[:, self._joint_ids] - self.robot.data.default_joint_pos[:, self._joint_ids]
        )
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                joint_pos_rel,
                self.robot.data.joint_vel[:, self._joint_ids],
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b

        forward_velocity_error = torch.square(self._commands[:, 0] - root_lin_vel_b[:, 0])
        track_forward_velocity = torch.exp(-forward_velocity_error / self.cfg.velocity_tracking_sigma)

        yaw_velocity_error = torch.square(self._commands[:, 2] - root_ang_vel_b[:, 2])
        track_yaw_velocity = torch.exp(-yaw_velocity_error / self.cfg.velocity_tracking_sigma)

        lateral_velocity_error = torch.square(self._commands[:, 1] - root_lin_vel_b[:, 1])
        vertical_velocity_error = torch.square(root_lin_vel_b[:, 2])
        angular_velocity_xy_error = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
        flat_orientation_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        joint_torque = torch.sum(torch.square(self.robot.data.applied_torque[:, self._joint_ids]), dim=1)
        joint_acceleration = torch.sum(torch.square(self.robot.data.joint_acc[:, self._joint_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_ids]
        peak_contact_force = torch.max(torch.norm(contact_forces, dim=-1), dim=1).values
        undesired_contacts = torch.sum(
            peak_contact_force > self.cfg.undesired_contact_force_threshold,
            dim=1,
        )

        rewards = {
            "track_forward_velocity": (track_forward_velocity * self.cfg.rew_track_forward_velocity * self.step_dt),
            "track_yaw_velocity": track_yaw_velocity * self.cfg.rew_track_yaw_velocity * self.step_dt,
            "alive": (1.0 - self.reset_terminated.float()) * self.cfg.rew_alive * self.step_dt,
            "lateral_velocity_l2": lateral_velocity_error * self.cfg.rew_lateral_velocity * self.step_dt,
            "vertical_velocity_l2": vertical_velocity_error * self.cfg.rew_vertical_velocity * self.step_dt,
            "angular_velocity_xy_l2": (angular_velocity_xy_error * self.cfg.rew_angular_velocity_xy * self.step_dt),
            "flat_orientation_l2": (flat_orientation_error * self.cfg.rew_flat_orientation * self.step_dt),
            "joint_torque_l2": joint_torque * self.cfg.rew_joint_torque * self.step_dt,
            "joint_acceleration_l2": (joint_acceleration * self.cfg.rew_joint_acceleration * self.step_dt),
            "action_rate_l2": action_rate * self.cfg.rew_action_rate * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.rew_undesired_contacts * self.step_dt,
            # Keep the terminal penalty independent of control frequency.
            "terminated": self.reset_terminated.float() * self.cfg.rew_terminated,
        }

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(tuple(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        too_low = base_height < self.cfg.termination_height

        # For an upright robot projected gravity is [0, 0, -1]. This test is
        # quaternion-safe and catches combined roll/pitch falls.
        tilted = self.robot.data.projected_gravity_b[:, 2] > -math.cos(self.cfg.termination_tilt)

        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self._base_ids]
        peak_base_force = torch.max(torch.norm(contact_forces, dim=-1), dim=1).values
        base_contact = torch.any(peak_base_force > self.cfg.base_contact_force_threshold, dim=1)

        terminated = too_low | tilted | base_contact
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids, 0] = self.cfg.target_forward_velocity
        self._commands[env_ids, 1] = self.cfg.target_lateral_velocity
        self._commands[env_ids, 2] = self.cfg.target_yaw_velocity

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos += sample_uniform(
            -self.cfg.reset_position_noise,
            self.cfg.reset_position_noise,
            joint_pos.shape,
            joint_pos.device,
        )
        joint_vel += sample_uniform(
            -self.cfg.reset_velocity_noise,
            self.cfg.reset_velocity_noise,
            joint_vel.shape,
            joint_vel.device,
        )

        joint_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, min=joint_limits[..., 0], max=joint_limits[..., 1])

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._processed_actions[env_ids] = joint_pos[:, self._joint_ids]

        extras = {}
        for key in self._episode_sums:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras
