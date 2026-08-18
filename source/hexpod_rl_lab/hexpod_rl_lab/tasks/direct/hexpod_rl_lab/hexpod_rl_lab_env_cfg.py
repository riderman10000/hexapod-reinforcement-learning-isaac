# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from hexpod_rl_lab.robots.hexapod import HEXAPOD_CFG


@configclass
class HexpodRlLabEnvCfg(DirectRLEnvCfg):
    # Environment
    decimation = 4
    episode_length_s = 20.0
    action_space = 18
    # base linear/angular velocity (6), projected gravity (3), command (3),
    # relative joint position/velocity (36), and previous action (18)
    observation_space = 66
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    ground_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=sim.dt,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # Position actions: target = default joint position + scale * clipped action.
    action_scale = 0.3  # [rad]
    action_clip = 1.0

    # Reset randomization
    reset_position_noise = 0.02  # [rad]
    reset_velocity_noise = 0.05  # [rad/s]

    # Desired body-frame command: forward, lateral, and yaw velocity.
    target_forward_velocity = 0.5  # [m/s]
    target_lateral_velocity = 0.0  # [m/s]
    target_yaw_velocity = 0.0  # [rad/s]
    velocity_tracking_sigma = 0.25

    # Reward scales. Non-terminal terms are multiplied by the environment step time.
    rew_track_forward_velocity = 2.0
    rew_track_yaw_velocity = 0.5
    rew_alive = 0.1
    rew_lateral_velocity = -1.0
    rew_vertical_velocity = -2.0
    rew_angular_velocity_xy = -0.05
    rew_flat_orientation = -5.0
    rew_joint_torque = -2.5e-5
    rew_joint_acceleration = -2.5e-7
    rew_action_rate = -0.01
    rew_undesired_contacts = -1.0
    rew_terminated = -5.0

    # Fall detection. The upright zero-angle stance settles near 0.128 m.
    termination_height = 0.075  # [m]
    termination_tilt = 0.8  # [rad], approximately 46 degrees
    base_contact_force_threshold = 1.0  # [N]
    undesired_contact_force_threshold = 1.0  # [N]
