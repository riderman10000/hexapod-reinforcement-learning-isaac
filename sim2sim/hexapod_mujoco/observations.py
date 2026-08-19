"""Reproduce the 70-value Isaac Lab policy observation in MuJoCo."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import PolicyInterface
from .joint_mapping import JointMapping


class ObservationBuilder:
    """Stateful observation builder; previous action is part of the policy input."""

    def __init__(
        self,
        model: mujoco.MjModel,
        interface: PolicyInterface,
        joint_mapping: JointMapping,
        phase_offset: float = 0.0,
    ) -> None:
        self.model = model
        self.interface = interface
        self.joint_mapping = joint_mapping
        self.phase_offset = float(phase_offset)
        self.previous_action = np.zeros(interface.action_size, dtype=np.float32)
        self.base_body_id = model.body("base_link").id
        self.linear_velocity_sensor = model.sensor("base_linear_velocity_body").id
        self.angular_velocity_sensor = model.sensor("base_angular_velocity_body").id

    def reset(self) -> None:
        self.previous_action.fill(0.0)

    def build(self, data: mujoco.MjData, control_step: int) -> np.ndarray:
        rotation_body_to_world = data.xmat[self.base_body_id].reshape(3, 3)
        projected_gravity_body = rotation_body_to_world.T @ np.asarray([0.0, 0.0, -1.0])

        yaw = math.atan2(rotation_body_to_world[1, 0], rotation_body_to_world[0, 0])
        commanded_heading_world = self.interface.command_world[:2].astype(np.float64)
        heading_norm = np.linalg.norm(commanded_heading_world)
        if heading_norm < 1.0e-9:
            raise ValueError("The straight-line policy requires a non-zero planar command")
        commanded_heading_world /= heading_norm
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        desired_heading_body = np.asarray(
            [
                cos_yaw * commanded_heading_world[0] + sin_yaw * commanded_heading_world[1],
                -sin_yaw * commanded_heading_world[0] + cos_yaw * commanded_heading_world[1],
            ]
        )

        phase = (
            2.0 * math.pi * self.interface.gait_frequency_hz * control_step * self.interface.policy_dt
            + self.phase_offset
        )
        gait_phase = np.asarray([math.sin(phase), math.cos(phase)])

        base_linear_velocity_body = data.sensor(self.linear_velocity_sensor).data.copy()
        base_angular_velocity_body = data.sensor(self.angular_velocity_sensor).data.copy()
        joint_position_relative = self.joint_mapping.positions(data) - self.joint_mapping.neutral_positions
        joint_velocity = self.joint_mapping.velocities(data)

        observation = np.concatenate(
            (
                base_linear_velocity_body,
                base_angular_velocity_body,
                projected_gravity_body,
                self.interface.command_world,
                desired_heading_body,
                gait_phase,
                joint_position_relative,
                joint_velocity,
                self.previous_action,
            )
        ).astype(np.float32)
        if observation.shape != (self.interface.observation_size,):
            raise RuntimeError(f"Expected observation {(self.interface.observation_size,)}, got {observation.shape}")
        if not np.isfinite(observation).all():
            raise FloatingPointError("Observation contains NaN or infinity")
        return observation
