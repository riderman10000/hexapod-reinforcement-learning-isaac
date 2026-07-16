# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .hexpod_rl_lab_env_cfg import HexpodRlLabEnvCfg


class HexpodRlLabEnv(DirectRLEnv):
    cfg: HexpodRlLabEnvCfg


    def _hexapod_inspect(self):
        # rl: method to inspect the hexapod apis and test 
        print("\n========== HEXAPOD INFORMATION ==========")
        print("Robot type:", type(self.robot))
        print("Robot data type:", type(self.robot.data))
        print("Number of joints:", self.robot.num_joints)
        print("Joint names:", self.robot.joint_names)
        print("Joint position shape:", self.robot.data.joint_pos.shape)
        print("Joint velocity shape:", self.robot.data.joint_vel.shape)
        print("Default joint position shape:", self.robot.data.default_joint_pos.shape)
        print("=========================================\n")

    def __init__(self, cfg: HexpodRlLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # setup of the cart pole 
        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        self._hexapod_inspect() 

        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel # the .data is from the class Articulation so can check out that for the required attributes associated 

        # this matches every articulation joint . 
        self._joint_ids, self._joint_names = self.robot.find_joints(
            ".*"
        )
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel 

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids], 
                self.joint_vel[:, self._joint_ids],
            ),
            dim=-1,
        )
        print(f"[OBS] {obs.shape}")
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids] # lists all the joints ?? 
        joint_vel = self.robot.data.default_joint_vel[env_ids] 

        # rl
        print("--------------")
        print(f"[env_ids] ", env_ids)
        print(f"[joint pos] {joint_pos} and {joint_pos.shape}")
        print(self.cfg.robot_cfg.init_state.pos)
        print(f"reset randomization {self.cfg.reset_position_noise}")
        print(f"reset randomization {self.cfg.reset_velocity_noise}")
        
        # filter through the index and list the necessary joints only  
        # joint_pos[:, self._joint_ids] += sample_uniform(
        #     self.cfg.robot_cfg.init_state.pos , 
        #     self.cfg.robot_cfg.init_state.pos , 
        #     joint_pos[:, self._joint_ids].shape,
        #     joint_pos.device,
        # )

        # print(joint_pos)
        # b.update({key: c[key] for key in a if key in c})
        assert(len(self._joint_ids) == len(self._joint_names)), f"_joint_ids :  {len(self._joint_ids)} and _joint_name {len(self._joint_names)} not of same size"
        for joint_idx, joint_name in zip(self._joint_ids, self._joint_names): 
            joint_pos[:, joint_idx] = self.cfg.robot_cfg.init_state.joint_pos[joint_name]
            joint_vel[:, joint_idx] = self.cfg.robot_cfg.init_state.joint_vel[joint_name]

        # ---- 

        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        print(f"[RESET] success")


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward