"""Replay an Isaac joint-target sequence open-loop in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.simulation import HexapodSimulation, RolloutOptions
from sim2sim.rollout_data import (
    SCHEMA_VERSION,
    ArrayRecorder,
    file_sha256,
    load_rollout,
    summarize_policy_rollout,
    write_json,
)
from sim2sim.tools.capture_mujoco_rollout import MujocoStateSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"


def _load_metadata(directory: Path) -> dict:
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def set_mujoco_state_from_rollout(simulation: HexapodSimulation, source: dict[str, np.ndarray], index: int) -> None:
    """Set MuJoCo generalized state from a shared-schema pre-action state."""
    simulation.reset()
    data = simulation.data
    qpos_address = simulation.free_qpos_address
    dof_address = simulation.free_dof_address
    data.qpos[qpos_address : qpos_address + 3] = source["base_position"][index]
    data.qpos[qpos_address + 3 : qpos_address + 7] = source["base_quaternion_wxyz"][index]
    data.qpos[simulation.joints.qpos_addresses] = source["joint_position"][index]
    data.qvel[dof_address : dof_address + 3] = source["world_linear_velocity"][index]
    # MuJoCo free-joint angular qvel is body-local, matching the recorded body angular velocity.
    data.qvel[dof_address + 3 : dof_address + 6] = source["body_angular_velocity"][index]
    data.qvel[simulation.joints.dof_addresses] = source["joint_velocity"][index]
    data.time = float(source["time"][index])
    simulation.joints.write_targets(data, source["joint_target"][index])
    mujoco.mj_forward(simulation.model, data)


def replay(args: argparse.Namespace) -> dict:
    source_dir = args.isaac_rollout
    source_meta = _load_metadata(source_dir)
    if source_meta["simulator"] != "isaac_physx":
        raise ValueError("Open-loop source must be an Isaac/PhysX capture")
    source = load_rollout(source_dir / "policy_steps.npz")
    interface = load_policy_interface(args.interface)
    if source_meta["joint_names"] != list(interface.policy_joint_names):
        raise ValueError("Source joint order differs from the policy interface")
    simulation = HexapodSimulation(args.model, Path(source_meta["policy"]), interface)
    sampler = MujocoStateSampler(simulation, source_meta["contact_force_threshold"])
    set_mujoco_state_from_rollout(simulation, source, 0)
    options = RolloutOptions(
        duration_seconds=None,
        phase_offset=float(source_meta["phase_offset"]),
        velocity_limit_mode=args.velocity_limit_mode,
    )
    policy_recorder = ArrayRecorder()
    physics_recorder = ArrayRecorder()
    fall_reason = ""

    for control_step in range(len(source["step"])):
        state = sampler.sample(float(source["time"][control_step]))
        target = source["joint_target"][control_step]
        simulation.joints.write_targets(simulation.data, target)

        def record_physics(substep: int) -> None:
            physics_recorder.append(
                policy_step=control_step,
                substep=substep,
                joint_target=target,
                **sampler.sample(),
            )

        simulation.step_physics(None, options, record_physics)
        next_state = sampler.sample(float(source["next_time"][control_step]))
        rotation_body_to_world = simulation.data.xmat[simulation.base_body_id].reshape(3, 3)
        projected_gravity = rotation_body_to_world.T @ np.asarray([0.0, 0.0, -1.0])
        reason = simulation._fall_reason(projected_gravity)  # noqa: SLF001
        terminated = reason is not None
        policy_recorder.append(
            step=source["step"][control_step],
            phase=source["phase"][control_step],
            observation=source["observation"][control_step],
            raw_action=source["raw_action"][control_step],
            applied_action=source["applied_action"][control_step],
            joint_target=target,
            desired_foot_contact=source["desired_foot_contact"][control_step],
            terminated=terminated,
            **state,
            **{f"next_{name}": value for name, value in next_state.items()},
        )
        if terminated:
            fall_reason = reason or "unknown"
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_recorder.save(args.output_dir / "policy_steps.npz")
    physics_recorder.save(args.output_dir / "physics_steps.npz")
    result = policy_recorder.arrays()
    summary = summarize_policy_rollout(result, interface.command_world[0])
    summary["fall_reason"] = fall_reason
    metadata = {
        **source_meta,
        "schema_version": SCHEMA_VERSION,
        "simulator": "mujoco",
        "mode": "open_loop_isaac_target_replay",
        "source_rollout": str(source_dir.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "velocity_limit_mode": args.velocity_limit_mode,
        "policy_steps": policy_recorder.sample_count,
        "physics_steps": physics_recorder.sample_count,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "summary.json", summary)
    print(f"MuJoCo open-loop replay: {args.output_dir.resolve()}")
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-rollout", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--velocity-limit-mode", choices=("hard", "soft", "none"), default="hard")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    replay(_arguments())
