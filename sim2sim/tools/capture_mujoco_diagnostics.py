"""Capture deterministic MuJoCo data for cross-simulator diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.hexapod_mujoco.config import PolicyInterface, load_policy_interface
from sim2sim.hexapod_mujoco.joint_mapping import JointMapping
from sim2sim.hexapod_mujoco.observations import ObservationBuilder
from sim2sim.hexapod_mujoco.policy import OnnxPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _observation_terms(observation: np.ndarray, interface: PolicyInterface) -> dict[str, list[float]]:
    terms: dict[str, list[float]] = {}
    start = 0
    for name, size in interface.observation_layout:
        terms[name] = observation[start : start + size].astype(float).tolist()
        start += size
    return terms


def _leg_index(joint_name: str) -> int:
    match = re.search(r"(?:body_leg_|leg_)(\d)", joint_name)
    if match is None:
        raise ValueError(f"Cannot determine leg for joint {joint_name!r}")
    return int(match.group(1))


def _foot_position_in_base(model: mujoco.MjModel, data: mujoco.MjData, leg_index: int) -> np.ndarray:
    base_id = model.body("base_link").id
    foot_position_world = data.site_xpos[model.site(f"eef_{leg_index}").id]
    rotation_body_to_world = data.xmat[base_id].reshape(3, 3)
    return rotation_body_to_world.T @ (foot_position_world - data.xpos[base_id])


def _capture_neutral_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: JointMapping,
    interface: PolicyInterface,
    policy: OnnxPolicy,
    output_dir: Path,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mapping.write_targets(data, mapping.neutral_positions)
    mujoco.mj_forward(model, data)
    observation = ObservationBuilder(model, interface, mapping).build(data, control_step=0)
    raw_action = policy(observation)
    _write_json(
        output_dir / "neutral_snapshot.json",
        {
            "schema_version": 1,
            "simulator": "mujoco",
            "joint_names": list(interface.policy_joint_names),
            "observation": observation.astype(float).tolist(),
            "observation_terms": _observation_terms(observation, interface),
            "raw_action": raw_action.astype(float).tolist(),
            "clipped_action": np.clip(raw_action, -interface.action_clip, interface.action_clip).astype(float).tolist(),
        },
    )


def _capture_joint_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: JointMapping,
    interface: PolicyInterface,
    output_dir: Path,
    joint_delta: float,
    response_seconds: float,
) -> None:
    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    sample_count = max(1, math.ceil(response_seconds / interface.physics_dt))
    fieldnames = [
        "simulator",
        "joint_index",
        "joint_name",
        "time",
        "command_delta",
        "position_delta",
        "velocity",
        "actuator_force",
        "foot_dx",
        "foot_dy",
        "foot_dz",
    ]
    summaries: list[dict[str, float | int | str | list[float]]] = []
    rows: list[dict[str, float | int | str]] = []
    try:
        for joint_index, joint_name in enumerate(interface.policy_joint_names):
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            mapping.write_targets(data, mapping.neutral_positions)
            mujoco.mj_forward(model, data)
            leg_index = _leg_index(joint_name)
            neutral_foot = _foot_position_in_base(model, data, leg_index)
            targets = mapping.neutral_positions.copy()
            targets[joint_index] += joint_delta
            targets = mapping.clip_targets(targets)
            mapping.write_targets(data, targets)

            for sample_index in range(sample_count + 1):
                if sample_index > 0:
                    mujoco.mj_step(model, data)
                position_delta = mapping.positions(data)[joint_index] - mapping.neutral_positions[joint_index]
                foot_delta = _foot_position_in_base(model, data, leg_index) - neutral_foot
                row: dict[str, float | int | str] = {
                    "simulator": "mujoco",
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "time": sample_index * interface.physics_dt,
                    "command_delta": joint_delta,
                    "position_delta": float(position_delta),
                    "velocity": float(mapping.velocities(data)[joint_index]),
                    "actuator_force": float(mapping.actuator_forces(data)[joint_index]),
                    "foot_dx": float(foot_delta[0]),
                    "foot_dy": float(foot_delta[1]),
                    "foot_dz": float(foot_delta[2]),
                }
                rows.append(row)

            final = rows[-1]
            summaries.append(
                {
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "command_delta": joint_delta,
                    "final_position_delta": final["position_delta"],
                    "final_foot_delta": [final["foot_dx"], final["foot_dy"], final["foot_dz"]],
                }
            )
    finally:
        model.opt.gravity[:] = original_gravity

    csv_path = output_dir / "joint_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "joint_sweep_summary.json",
        {
            "schema_version": 1,
            "simulator": "mujoco",
            "physics_dt": interface.physics_dt,
            "joint_delta": joint_delta,
            "response_seconds": response_seconds,
            "gravity_disabled": True,
            "joints": summaries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-delta", type=float, default=0.05)
    parser.add_argument("--response-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if args.joint_delta <= 0.0 or args.response_seconds <= 0.0:
        raise ValueError("Joint delta and response duration must be positive")

    interface = load_policy_interface(args.interface)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    mapping = JointMapping.from_model(model, interface.policy_joint_names)
    policy = OnnxPolicy(args.policy, interface)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _capture_neutral_snapshot(model, data, mapping, interface, policy, args.output_dir)
    _capture_joint_sweep(
        model,
        data,
        mapping,
        interface,
        args.output_dir,
        args.joint_delta,
        args.response_seconds,
    )
    print(f"MuJoCo diagnostics: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
