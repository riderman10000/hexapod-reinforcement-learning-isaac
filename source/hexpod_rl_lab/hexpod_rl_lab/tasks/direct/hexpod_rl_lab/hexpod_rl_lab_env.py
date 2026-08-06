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
        self.debug_counter = 0 
        self.debug_interval = 200 
        self.DEBUG  = True 

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
        # print(f"[ACTION] {self.actions} and size {self.actions.shape}")

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale, 
            joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # Count how many simulation steps have elapsed
        self.debug_counter += 1
        
        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids], 
                self.joint_vel[:, self._joint_ids], 
            ),
            dim=-1,
        )
        # print(f"[OBS] {obs.shape}")
        observations = {"policy": obs}
        self._debug(obs = obs)

        return observations

    def _get_rewards(self) -> torch.Tensor:

        forward_velocity = self.robot.data.root_state_w[:, 7]

        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_forward_velocity, 
            forward_velocity,
            self.reset_terminated,
        )

        self._debug(
            reward = total_reward 
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # print(f"[robot pose]")
        # print(f"world_poses {self.robot.data.__dir__()}")

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._joint_ids]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        
        # 1. Get the full root state in world frame
        root_state = self.robot.data.root_state_w  # Shape: (num_envs, 13)
        # 2. Extract position (X, Y, Z)
        position = root_state[:, 0:3]         # Shape: (num_envs, 3)
        base_height = root_state[:, 2]
        # 3. Extract orientation quaternion (w, x, y, z)
        orientation = root_state[:, 3:7]      # Shape: (num_envs, 4)
        
        # print(f"robot state : position: {position} and orientation: {orientation}") 

        # register as out of bounds if it is near the ground 
        # print(f"condition {(base_height < self.cfg.termination_height)} {(base_height < self.cfg.termination_height).shape}")
        out_of_bounds = base_height < self.cfg.termination_height
        # print(f"position : {position}")

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # default position of the robots 
        joint_pos = self.robot.data.default_joint_pos[env_ids] # lists all the joints ?? 
        joint_vel = self.robot.data.default_joint_vel[env_ids] 

        # rl
        # print("--------------")
        # print(f"[env_ids] ", env_ids)
        # print(f"[joint pos] {joint_pos} and {joint_pos.shape}")
        # print(self.cfg.robot_cfg.init_state.pos)
        # print(f"reset randomization {self.cfg.reset_position_noise}")
        # print(f"reset randomization {self.cfg.reset_velocity_noise}")


        joint_pos += sample_uniform(
            -self.cfg.reset_position_noise,
            self.cfg.reset_position_noise,
            joint_pos.shape,
            joint_pos.device,
        )

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # print(f"[RESET] success")

    
    def _debug(
        self,
        obs: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
    ):
        """Print useful information every N simulation steps."""

        if not self.DEBUG: # if you don't want to print any of these stuff 
            return 

        # Only print every 200 steps
        if self.debug_counter % 200 != 0:
            return

        print("\n" + "=" * 70)
        print(f"DEBUG STEP {self.debug_counter}")
        print("=" * 70)

        # -----------------------
        # Base state
        # -----------------------
        root_state = self.robot.data.root_state_w

        base_pos = root_state[:, 0:3]
        base_quat = root_state[:, 3:7]
        base_lin_vel = root_state[:, 7:10]
        base_ang_vel = root_state[:, 10:13]

        print(f"Base height           : {base_pos[:,2].mean():.3f} m")
        print(f"Base linear vel max   : {base_lin_vel.abs().max():.3f}")
        print(f"Base angular vel max  : {base_ang_vel.abs().max():.3f}")

        # -----------------------
        # Joint information
        # -----------------------
        joint_deg = torch.rad2deg(self.joint_pos)

        print(f"Joint position max    : {self.joint_pos.abs().max():.3f} rad")
        print(f"Joint position max    : {joint_deg.abs().max():.1f} deg")
        print(f"Joint velocity max    : {self.joint_vel.abs().max():.3f}")

        # -----------------------
        # Actions
        # -----------------------
        if hasattr(self, "actions"):
            print(
                f"Action range          : "
                f"{self.actions.min():.3f} -> {self.actions.max():.3f}"
            )

        # -----------------------
        # Observations
        # -----------------------
        if obs is not None:
            print(
                f"Observation range     : "
                f"{obs.min():.3f} -> {obs.max():.3f}"
            )

            if torch.isnan(obs).any():
                print("WARNING: NaN detected in observations!")

            if torch.isinf(obs).any():
                print("WARNING: Inf detected in observations!")

        # -----------------------
        # Rewards
        # -----------------------
        if reward is not None:
            print(
                f"Reward range          : "
                f"{reward.min():.3f} -> {reward.max():.3f}"
            )

            if torch.isnan(reward).any():
                print("WARNING: NaN detected in rewards!")

        print("=" * 70)
    


@torch.jit.script
def compute_rewards(
    rew_alive_scale: float,
    rew_terminated_scale: float,
    rew_forward_scale: float,
    forward_velocity: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_alive_scale * (1.0 - reset_terminated.float())
    rew_terminated = rew_terminated_scale * reset_terminated.float()
    rew_forward = rew_forward_scale * forward_velocity
    total_reward = (
        rew_alive
        + rew_terminated
        + rew_forward
    )
    return total_reward