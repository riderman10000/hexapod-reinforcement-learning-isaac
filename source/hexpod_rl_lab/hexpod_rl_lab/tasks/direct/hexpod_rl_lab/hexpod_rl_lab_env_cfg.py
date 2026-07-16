# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from hexpod_rl_lab.robots.hexapod import HEXAPOD_CFG


@configclass
class HexpodRlLabEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_space = 18 #rl: 18 controllable joints 
    observation_space = 48 #rl: TODO: update after implementation of _get_observations()
    state_space = 0 #rl: no privileged observation yet  

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, 
        render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = HEXAPOD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, # num_envs=4096, 
        env_spacing=3.0, # env_spacing=4.0, 
        replicate_physics=True, # replicate_physics=True
        )

    # custom parameters/scales
    # - controllable joint

    # - action scale
    action_scale = 0.5 # 100.0  # [N]

    #rl: reset randomization 
    reset_position_noise = 0.02 
    reset_velocity_noise = 0.05 

    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -10.0
    rew_forward_velocity = 2.0 
    rew_height = 1.0
    rew_orientation = 1.0 
    rew_joint_limit = -0.2 
    rew_joint_velocity = -.002 
    rew_action_rate = -0.01
    # - reset states/conditions
    termination_height = 0.10 # rl  
    termination_roll = 1.2  # rl    
    termination_pitch = 1.2 # rl
    # desired command 
    target_forward_velocity = 0.5 