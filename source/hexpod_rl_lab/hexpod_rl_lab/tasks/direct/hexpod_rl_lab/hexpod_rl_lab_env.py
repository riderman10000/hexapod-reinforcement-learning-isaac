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
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform, yaw_quat

from hexpod_rl_lab.robots.joints import FOOT_LINKS

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

        # World-frame planar velocity command and body yaw-rate command. Tracking
        # the planar vector in world coordinates prevents a rotating robot from
        # receiving full "forward" reward while tracing a circle.
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands[:, 0] = self.cfg.target_forward_velocity
        self._commands[:, 1] = self.cfg.target_lateral_velocity
        self._commands[:, 2] = self.cfg.target_yaw_velocity

        target_planar_speed = math.hypot(self.cfg.target_forward_velocity, self.cfg.target_lateral_velocity)
        if target_planar_speed < 1.0e-6:
            raise ValueError("The straight-line task requires a non-zero planar velocity command.")
        self._desired_heading_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_heading_w[:, 0] = self.cfg.target_forward_velocity / target_planar_speed
        self._desired_heading_w[:, 1] = self.cfg.target_lateral_velocity / target_planar_speed

        self._base_ids, base_names = self.contact_sensor.find_bodies("base_link")
        if len(self._base_ids) != 1:
            raise RuntimeError(f"Expected one base_link body in the contact sensor, found {base_names}")
        self._undesired_contact_ids, undesired_contact_names = self.contact_sensor.find_bodies("leg_._[12]")
        if len(self._undesired_contact_ids) != 12:
            raise RuntimeError(
                f"Expected the 12 proximal/middle leg bodies in the contact sensor, found {undesired_contact_names}"
            )
        self._foot_ids, self._foot_names = self.contact_sensor.find_bodies(FOOT_LINKS, preserve_order=True)
        if len(self._foot_ids) != 6:
            raise RuntimeError(f"Expected six foot bodies in the contact sensor, found {self._foot_names}")
        self._tripod_a_mask = torch.tensor(
            [True, False, True, False, True, False],
            dtype=torch.bool,
            device=self.device,
        )
        self._gait_phase_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "track_forward_velocity",
                "forward_progress",
                "forward_speed_shortfall",
                "gait_contact",
                "heading_error",
                "yaw_rate_l2",
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
        self._episode_metric_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "world_forward_velocity",
                "abs_world_lateral_velocity",
                "abs_yaw_rate",
                "heading_alignment",
                "yaw_tracking_quality",
                "gait_contact_match",
            )
        }
        self._episode_step_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_foot_contact_steps = torch.zeros(
            self.num_envs,
            len(self._foot_ids),
            dtype=torch.float,
            device=self.device,
        )
        self._episode_path_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_world_displacement = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self._previous_root_pos_w = self.robot.data.root_pos_w.clone()

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

    def _get_gait_phase_features(self) -> torch.Tensor:
        phase = (
            2.0 * math.pi * self.cfg.gait_frequency * self.episode_length_buf.float() * self.step_dt
            + self._gait_phase_offset
        )
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)

    def _get_observations(self) -> dict:
        joint_pos_rel = (
            self.robot.data.joint_pos[:, self._joint_ids] - self.robot.data.default_joint_pos[:, self._joint_ids]
        )
        desired_heading_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), self._desired_heading_w)
        gait_phase = self._get_gait_phase_features()
        obs = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                desired_heading_b[:, :2],
                gait_phase,
                joint_pos_rel,
                self.robot.data.joint_vel[:, self._joint_ids],
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        desired_heading_b = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), self._desired_heading_w)

        planar_velocity_error = torch.sum(torch.square(self._commands[:, :2] - root_lin_vel_w[:, :2]), dim=1)
        raw_velocity_tracking = torch.exp(-planar_velocity_error / self.cfg.linear_velocity_tracking_sigma)

        # A Gaussian centered on the command has a positive value at zero
        # velocity. Subtract and normalize that value so standing still earns
        # exactly zero forward-tracking reward while retaining a smooth slope.
        stationary_planar_error = torch.sum(torch.square(self._commands[:, :2]), dim=1)
        stationary_velocity_tracking = torch.exp(-stationary_planar_error / self.cfg.linear_velocity_tracking_sigma)
        track_forward_velocity = torch.clamp(
            (raw_velocity_tracking - stationary_velocity_tracking)
            / torch.clamp(1.0 - stationary_velocity_tracking, min=1.0e-6),
            min=0.0,
            max=1.0,
        )

        commanded_forward_speed = torch.clamp(self._commands[:, 0], min=1.0e-6)
        forward_progress = torch.clamp(root_lin_vel_w[:, 0] / commanded_forward_speed, min=0.0, max=1.0)
        minimum_forward_speed = self.cfg.minimum_forward_speed_ratio * commanded_forward_speed
        forward_speed_shortfall = torch.relu(minimum_forward_speed - root_lin_vel_w[:, 0])

        yaw_velocity_error = torch.square(self._commands[:, 2] - root_ang_vel_b[:, 2])
        track_yaw_velocity = torch.exp(-yaw_velocity_error / self.cfg.yaw_velocity_tracking_sigma)

        # The heading term corrects accumulated yaw drift, while the quadratic
        # yaw-rate term remains informative even when the Gaussian reward is
        # nearly zero. Together they remove the old stable circling solution.
        heading_error = 1.0 - desired_heading_b[:, 0]
        yaw_rate_error = torch.square(root_ang_vel_b[:, 2])
        lateral_velocity_error = torch.square(self._commands[:, 1] - root_lin_vel_w[:, 1])
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
        foot_contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self._foot_ids]
        peak_foot_contact_force = torch.max(torch.norm(foot_contact_forces, dim=-1), dim=1).values
        foot_contacts = peak_foot_contact_force > self.cfg.foot_contact_force_threshold

        # Alternating tripod contact target: legs 0/2/4 oppose legs 1/3/5.
        # Its signed formulation gives all-feet-down and all-feet-up a zero
        # baseline, so the robot cannot earn it by merely standing in place.
        tripod_a_stance = self._get_gait_phase_features()[:, 0] >= 0.0
        desired_foot_contacts = torch.where(
            self._tripod_a_mask.unsqueeze(0),
            tripod_a_stance.unsqueeze(1),
            ~tripod_a_stance.unsqueeze(1),
        )
        gait_contact_match = torch.mean(
            (2.0 * foot_contacts.float() - 1.0) * (2.0 * desired_foot_contacts.float() - 1.0),
            dim=1,
        )

        rewards = {
            "track_forward_velocity": (track_forward_velocity * self.cfg.rew_track_forward_velocity * self.step_dt),
            "forward_progress": forward_progress * self.cfg.rew_forward_progress * self.step_dt,
            "forward_speed_shortfall": (forward_speed_shortfall * self.cfg.rew_forward_speed_shortfall * self.step_dt),
            "gait_contact": gait_contact_match * self.cfg.rew_gait_contact * self.step_dt,
            "heading_error": heading_error * self.cfg.rew_heading_error * self.step_dt,
            "yaw_rate_l2": yaw_rate_error * self.cfg.rew_yaw_rate * self.step_dt,
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

        self._episode_metric_sums["world_forward_velocity"] += root_lin_vel_w[:, 0]
        self._episode_metric_sums["abs_world_lateral_velocity"] += torch.abs(root_lin_vel_w[:, 1])
        self._episode_metric_sums["abs_yaw_rate"] += torch.abs(root_ang_vel_b[:, 2])
        self._episode_metric_sums["heading_alignment"] += desired_heading_b[:, 0]
        self._episode_metric_sums["yaw_tracking_quality"] += track_yaw_velocity
        self._episode_metric_sums["gait_contact_match"] += gait_contact_match
        self._episode_foot_contact_steps += foot_contacts.float()
        self._episode_step_count += 1.0

        root_pos_w = self.robot.data.root_pos_w
        planar_step = root_pos_w[:, :2] - self._previous_root_pos_w[:, :2]
        self._episode_path_length += torch.norm(planar_step, dim=1)
        self._episode_world_displacement += planar_step
        self._previous_root_pos_w.copy_(root_pos_w)

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
        self._gait_phase_offset[env_ids] = sample_uniform(
            0.0,
            2.0 * math.pi,
            (len(env_ids),),
            self.device,
        )

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
        reset_yaw = sample_uniform(
            -self.cfg.reset_yaw_noise,
            self.cfg.reset_yaw_noise,
            (len(env_ids),),
            default_root_state.device,
        )
        zeros = torch.zeros_like(reset_yaw)
        default_root_state[:, 3:7] = quat_mul(
            quat_from_euler_xyz(zeros, zeros, reset_yaw),
            default_root_state[:, 3:7],
        )

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

        episode_steps = torch.clamp(self._episode_step_count[env_ids], min=1.0)
        for key in self._episode_metric_sums:
            extras[f"Episode_Metric/{key}"] = torch.mean(self._episode_metric_sums[key][env_ids] / episode_steps)
            self._episode_metric_sums[key][env_ids] = 0.0

        foot_contact_fraction = self._episode_foot_contact_steps[env_ids] / episode_steps.unsqueeze(1)
        for foot_index, foot_name in enumerate(self._foot_names):
            extras[f"Episode_Metric/foot_contact_fraction_{foot_name}"] = torch.mean(
                foot_contact_fraction[:, foot_index]
            )

        path_length = torch.clamp(self._episode_path_length[env_ids], min=1.0e-6)
        forward_path_efficiency = self._episode_world_displacement[env_ids, 0] / path_length
        extras["Episode_Metric/forward_path_efficiency"] = torch.mean(
            torch.clamp(forward_path_efficiency, min=-1.0, max=1.0)
        )

        self._episode_step_count[env_ids] = 0.0
        self._episode_foot_contact_steps[env_ids] = 0.0
        self._episode_path_length[env_ids] = 0.0
        self._episode_world_displacement[env_ids] = 0.0
        self._previous_root_pos_w[env_ids] = default_root_state[:, :3]
        extras["Episode_Termination/fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras
