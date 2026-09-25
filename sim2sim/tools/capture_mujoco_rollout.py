"""Capture a deterministic MuJoCo policy rollout in the shared gait schema."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.observations import ObservationBuilder
from sim2sim.hexapod_mujoco.simulation import HexapodSimulation, RolloutOptions
from sim2sim.rollout_data import (
    SCHEMA_VERSION,
    ArrayRecorder,
    desired_tripod_contacts,
    file_sha256,
    phase_at_step,
    summarize_policy_rollout,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"


class MujocoStateSampler:
    """Read matched robot, foot, and normal-contact data from MuJoCo."""

    def __init__(self, simulation: HexapodSimulation, contact_threshold: float) -> None:
        self.simulation = simulation
        self.model = simulation.model
        self.data = simulation.data
        self.contact_threshold = float(contact_threshold)
        self.linear_velocity_sensor = self.model.sensor("base_linear_velocity_body").id
        self.angular_velocity_sensor = self.model.sensor("base_angular_velocity_body").id
        self.foot_site_ids = np.asarray([self.model.site(f"eef_{index}").id for index in range(6)])
        self.foot_body_ids = np.asarray([self.model.body(f"leg_{index}_3").id for index in range(6)])
        self.body_to_foot = {int(body_id): index for index, body_id in enumerate(self.foot_body_ids)}

    def _foot_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((6, 3), dtype=float)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.simulation.ground_geom_id not in {geom1, geom2}:
                continue
            foot_geom = geom2 if geom1 == self.simulation.ground_geom_id else geom1
            foot_body = int(self.model.geom_bodyid[foot_geom])
            foot_index = self.body_to_foot.get(foot_body)
            if foot_index is None:
                continue
            local_force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, contact_index, local_force)
            normal = np.asarray(contact.frame[:3], dtype=float)
            if normal[2] < 0.0:
                normal = -normal
            forces[foot_index] += abs(float(local_force[0])) * normal
        contact = np.linalg.norm(forces, axis=1) > self.contact_threshold
        return contact, forces

    def sample(self, time_seconds: float | None = None) -> dict[str, np.ndarray | float]:
        base_position = self.data.xpos[self.simulation.base_body_id].copy()
        base_quaternion = self.data.xquat[self.simulation.base_body_id].copy()
        rotation_body_to_world = self.data.xmat[self.simulation.base_body_id].reshape(3, 3).copy()
        foot_position_world = self.data.site_xpos[self.foot_site_ids].copy()
        foot_position_body = (rotation_body_to_world.T @ (foot_position_world - base_position).T).T
        foot_contact, foot_contact_force = self._foot_contacts()
        body_linear_velocity = self.data.sensor(self.linear_velocity_sensor).data.copy()
        body_angular_velocity = self.data.sensor(self.angular_velocity_sensor).data.copy()
        world_linear_velocity = self.data.qvel[
            self.simulation.free_dof_address : self.simulation.free_dof_address + 3
        ].copy()
        return {
            "time": float(self.data.time if time_seconds is None else time_seconds),
            "base_position": base_position,
            "base_quaternion_wxyz": base_quaternion,
            "world_linear_velocity": world_linear_velocity,
            "body_linear_velocity": body_linear_velocity,
            "body_angular_velocity": body_angular_velocity,
            "joint_position": self.simulation.joints.positions(self.data),
            "joint_velocity": self.simulation.joints.velocities(self.data),
            "joint_force": self.simulation.joints.actuator_forces(self.data),
            "foot_position_world": foot_position_world,
            "foot_position_body": foot_position_body,
            "foot_contact": foot_contact,
            "foot_contact_force_world": foot_contact_force,
        }


def _prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}{name}": value for name, value in values.items()}


def capture(args: argparse.Namespace) -> dict:
    interface = load_policy_interface(args.interface)
    simulation = HexapodSimulation(args.model, args.policy, interface)
    simulation.reset()
    sampler = MujocoStateSampler(simulation, args.contact_threshold)
    observation_builder = ObservationBuilder(
        simulation.model,
        interface,
        simulation.joints,
        phase_offset=args.phase_offset,
    )
    options = RolloutOptions(
        duration_seconds=args.duration,
        phase_offset=args.phase_offset,
        velocity_limit_mode=args.velocity_limit_mode,
    )
    policy_recorder = ArrayRecorder()
    physics_recorder = ArrayRecorder()
    maximum_steps = math.ceil(args.duration / interface.policy_dt)
    fall_reason = ""

    for control_step in range(maximum_steps):
        state = sampler.sample()
        observation = observation_builder.build(simulation.data, control_step)
        raw_action = simulation.policy(observation)
        applied_action = np.clip(raw_action, -interface.action_clip, interface.action_clip).astype(np.float32)
        target = simulation.joints.clip_targets(
            simulation.joints.neutral_positions + interface.action_scale * applied_action
        )
        simulation.joints.write_targets(simulation.data, target)
        phase = phase_at_step(
            control_step,
            interface.policy_dt,
            interface.gait_frequency_hz,
            args.phase_offset,
        )

        def record_physics(substep: int) -> None:
            physics_recorder.append(
                policy_step=control_step,
                substep=substep,
                joint_target=target,
                **sampler.sample(),
            )

        simulation.step_physics(None, options, record_physics)
        observation_builder.previous_action[:] = applied_action
        next_state = sampler.sample()
        rotation_body_to_world = simulation.data.xmat[simulation.base_body_id].reshape(3, 3)
        projected_gravity = rotation_body_to_world.T @ np.asarray([0.0, 0.0, -1.0])
        reason = simulation._fall_reason(projected_gravity)  # noqa: SLF001
        terminated = reason is not None
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
        if terminated:
            fall_reason = reason or "unknown"
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "policy_steps.npz"
    physics_path = args.output_dir / "physics_steps.npz"
    policy_recorder.save(policy_path)
    physics_recorder.save(physics_path)
    policy_data = policy_recorder.arrays()
    summary = summarize_policy_rollout(policy_data, interface.command_world[0])
    summary["fall_reason"] = fall_reason
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "simulator": "mujoco",
        "mode": "closed_loop_policy",
        "policy": str(args.policy.resolve()),
        "policy_sha256": file_sha256(args.policy),
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "interface": str(args.interface.resolve()),
        "joint_names": list(interface.policy_joint_names),
        "foot_names": [f"dummy_eef_{index}" for index in range(6)],
        "physics_dt": interface.physics_dt,
        "policy_dt": interface.policy_dt,
        "decimation": interface.decimation,
        "phase_offset": args.phase_offset,
        "command_world": interface.command_world.astype(float).tolist(),
        "contact_force_threshold": args.contact_threshold,
        "velocity_limit_mode": args.velocity_limit_mode,
        "policy_steps": policy_recorder.sample_count,
        "physics_steps": physics_recorder.sample_count,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "summary.json", summary)
    print(f"MuJoCo rollout: {args.output_dir.resolve()}")
    print(f"Policy steps: {policy_recorder.sample_count}; physics steps: {physics_recorder.sample_count}")
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--phase-offset", type=float, default=0.0)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--velocity-limit-mode", choices=("hard", "soft", "none"), default="hard")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", help="Accepted for command symmetry; capture is headless.")
    args = parser.parse_args()
    if args.duration <= 0.0 or args.contact_threshold < 0.0:
        raise ValueError("Duration must be positive and contact threshold cannot be negative")
    return args


if __name__ == "__main__":
    capture(_arguments())
