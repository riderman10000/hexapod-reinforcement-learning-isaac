"""MuJoCo control loop matching the Isaac Lab timing and action contract."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import PolicyInterface
from .joint_mapping import JointMapping
from .metrics import RolloutMetrics
from .observations import ObservationBuilder
from .policy import OnnxPolicy


@dataclass(frozen=True)
class RolloutOptions:
    duration_seconds: float = 20.0
    settle_seconds: float = 0.0
    ramp_seconds: float = 0.0
    real_time: bool = False
    phase_offset: float = 0.0


class HexapodSimulation:
    """Own the model, policy, observation builder, and policy-rate loop."""

    def __init__(
        self,
        model_path: str | Path,
        policy_path: str | Path,
        interface: PolicyInterface,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.interface = interface
        if not np.isclose(self.model.opt.timestep, interface.physics_dt, atol=1.0e-12):
            raise ValueError(
                f"MuJoCo timestep {self.model.opt.timestep} does not match policy contract {interface.physics_dt}"
            )
        self.joints = JointMapping.from_model(self.model, interface.policy_joint_names)
        self.policy = OnnxPolicy(policy_path, interface)
        self.base_body_id = self.model.body("base_link").id
        self.free_joint_id = self.model.joint("base_free_joint").id
        self.free_qpos_address = int(self.model.jnt_qposadr[self.free_joint_id])
        self.free_dof_address = int(self.model.jnt_dofadr[self.free_joint_id])
        self.base_geom_id = self.model.geom("base_collision").id
        self.ground_geom_id = self.model.geom("ground").id

    def contract_summary(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "observation_size": self.interface.observation_size,
            "action_size": self.interface.action_size,
            "physics_dt": float(self.model.opt.timestep),
            "decimation": self.interface.decimation,
            "policy_dt": self.interface.policy_dt,
            "joint_velocity_limit": self.interface.joint_velocity_limit,
            "policy_joint_order": list(self.interface.policy_joint_names),
            "mujoco_joint_order": list(self.joints.mujoco_names),
        }

    def reset(self) -> None:
        home_key = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        self.joints.write_targets(self.data, self.joints.neutral_positions)
        mujoco.mj_forward(self.model, self.data)

    def _base_ground_contact(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self.base_geom_id, self.ground_geom_id}:
                return True
        return False

    def _fall_reason(self, projected_gravity_body: np.ndarray) -> str | None:
        base_height = float(self.data.xpos[self.base_body_id, 2])
        if base_height < self.interface.minimum_base_height:
            return "base_height"
        if projected_gravity_body[2] > -math.cos(self.interface.maximum_tilt_radians):
            return "base_tilt"
        if self._base_ground_contact():
            return "base_contact"
        return None

    def _step_physics(self, viewer: Any | None) -> None:
        for _ in range(self.interface.decimation):
            mujoco.mj_step(self.model, self.data)
            joint_velocity = self.data.qvel[self.joints.dof_addresses]
            clipped_velocity = np.clip(
                joint_velocity,
                -self.interface.joint_velocity_limit,
                self.interface.joint_velocity_limit,
            )
            if not np.array_equal(joint_velocity, clipped_velocity):
                self.data.qvel[self.joints.dof_addresses] = clipped_velocity
                mujoco.mj_forward(self.model, self.data)
        if viewer is not None:
            viewer.sync()

    def run(
        self,
        *,
        options: RolloutOptions,
        metrics: RolloutMetrics,
        viewer: Any | None = None,
    ) -> dict[str, float | bool | str]:
        self.reset()
        observation_builder = ObservationBuilder(
            self.model,
            self.interface,
            self.joints,
            phase_offset=options.phase_offset,
        )

        settle_steps = round(options.settle_seconds / self.interface.physics_dt)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

        policy_start_time = float(self.data.time)
        maximum_control_steps = math.ceil(options.duration_seconds / self.interface.policy_dt)
        fall_reason: str | None = None
        elapsed = 0.0

        for control_step in range(maximum_control_steps):
            wall_start = time.perf_counter()
            observation = observation_builder.build(self.data, control_step)
            raw_action = np.clip(
                self.policy(observation),
                -self.interface.action_clip,
                self.interface.action_clip,
            )
            if options.ramp_seconds > 0.0:
                ramp = min(1.0, (control_step + 1) * self.interface.policy_dt / options.ramp_seconds)
            else:
                ramp = 1.0
            applied_action = (ramp * raw_action).astype(np.float32)
            targets = self.joints.clip_targets(
                self.joints.neutral_positions + self.interface.action_scale * applied_action
            )
            self.joints.write_targets(self.data, targets)
            self._step_physics(viewer)
            observation_builder.previous_action[:] = applied_action

            elapsed = float(self.data.time) - policy_start_time
            base_position = self.data.qpos[self.free_qpos_address : self.free_qpos_address + 3].copy()
            base_quaternion = self.data.qpos[self.free_qpos_address + 3 : self.free_qpos_address + 7].copy()
            world_linear_velocity = self.data.qvel[self.free_dof_address : self.free_dof_address + 3].copy()
            body_angular_velocity = self.data.qvel[self.free_dof_address + 3 : self.free_dof_address + 6].copy()
            rotation_body_to_world = self.data.xmat[self.base_body_id].reshape(3, 3)
            body_linear_velocity = rotation_body_to_world.T @ world_linear_velocity
            projected_gravity_body = rotation_body_to_world.T @ np.asarray([0.0, 0.0, -1.0])
            heading_alignment = float(math.cos(math.atan2(rotation_body_to_world[1, 0], rotation_body_to_world[0, 0])))
            fall_reason = self._fall_reason(projected_gravity_body)

            metrics.record(
                time_seconds=elapsed,
                base_position=base_position,
                base_quaternion=base_quaternion,
                world_linear_velocity=world_linear_velocity,
                body_linear_velocity=body_linear_velocity,
                body_angular_velocity=body_angular_velocity,
                heading_alignment=heading_alignment,
                joint_position=self.joints.positions(self.data),
                joint_velocity=self.joints.velocities(self.data),
                action=applied_action,
                target=targets,
                actuator_force=self.joints.actuator_forces(self.data),
                lower_limits=self.joints.lower_limits,
                upper_limits=self.joints.upper_limits,
                contact_count=int(self.data.ncon),
                fall_reason=fall_reason,
            )
            if fall_reason is not None:
                break
            if viewer is not None and not viewer.is_running():
                break
            if options.real_time:
                sleep_seconds = self.interface.policy_dt - (time.perf_counter() - wall_start)
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)

        summary = metrics.close(duration=elapsed, fall_reason=fall_reason)
        print(json.dumps(summary, indent=2))
        return summary
