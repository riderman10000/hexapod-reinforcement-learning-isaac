"""Evaluate MuJoCo one-policy-step predictions from recorded Isaac states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hexapod-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from sim2sim.hexapod_mujoco.config import load_policy_interface
from sim2sim.hexapod_mujoco.simulation import HexapodSimulation, RolloutOptions
from sim2sim.rollout_data import load_rollout, quaternion_angle_error, write_json
from sim2sim.tools.capture_mujoco_rollout import MujocoStateSampler
from sim2sim.tools.replay_mujoco_targets import set_mujoco_state_from_rollout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_INTERFACE = PROJECT_ROOT / "sim2sim" / "configs" / "policy_interface.yaml"


def _load_metadata(directory: Path) -> dict:
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def _select_phase_indices(phase: np.ndarray, sample_count: int) -> list[int]:
    targets = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    selected = []
    for target in targets:
        circular_error = np.abs(np.angle(np.exp(1j * (phase - target))))
        for index in np.argsort(circular_error):
            if int(index) not in selected:
                selected.append(int(index))
                break
    return sorted(selected)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def compare(args: argparse.Namespace) -> dict:
    source_meta = _load_metadata(args.isaac_rollout)
    if source_meta["simulator"] != "isaac_physx":
        raise ValueError("One-step source must be an Isaac/PhysX capture")
    source = load_rollout(args.isaac_rollout / "policy_steps.npz")
    interface = load_policy_interface(args.interface)
    simulation = HexapodSimulation(args.model, Path(source_meta["policy"]), interface)
    sampler = MujocoStateSampler(simulation, source_meta["contact_force_threshold"])
    options = RolloutOptions(
        duration_seconds=None,
        phase_offset=float(source_meta["phase_offset"]),
        velocity_limit_mode=args.velocity_limit_mode,
    )
    indices = _select_phase_indices(source["phase"], args.sample_count)
    rows = []
    for index in indices:
        set_mujoco_state_from_rollout(simulation, source, index)
        simulation.joints.write_targets(simulation.data, source["joint_target"][index])
        simulation.step_physics(None, options)
        predicted = sampler.sample(float(source["next_time"][index]))
        row = {
            "source_index": index,
            "time": float(source["time"][index]),
            "phase": float(source["phase"][index]),
            "isaac_contact_count": int(np.count_nonzero(source["next_foot_contact"][index])),
            "mujoco_contact_count": int(np.count_nonzero(predicted["foot_contact"])),
            "base_position_error_m": float(
                np.linalg.norm(predicted["base_position"] - source["next_base_position"][index])
            ),
            "base_orientation_error_rad": quaternion_angle_error(
                predicted["base_quaternion_wxyz"], source["next_base_quaternion_wxyz"][index]
            ),
            "world_linear_velocity_rmse_m_s": _rmse(
                predicted["world_linear_velocity"] - source["next_world_linear_velocity"][index]
            ),
            "body_angular_velocity_rmse_rad_s": _rmse(
                predicted["body_angular_velocity"] - source["next_body_angular_velocity"][index]
            ),
            "joint_position_rmse_rad": _rmse(predicted["joint_position"] - source["next_joint_position"][index]),
            "joint_velocity_rmse_rad_s": _rmse(predicted["joint_velocity"] - source["next_joint_velocity"][index]),
            "foot_position_world_rmse_m": _rmse(
                predicted["foot_position_world"] - source["next_foot_position_world"][index]
            ),
            "foot_contact_agreement": float(np.mean(predicted["foot_contact"] == source["next_foot_contact"][index])),
        }
        rows.append(row)

    metrics = [
        "base_position_error_m",
        "base_orientation_error_rad",
        "world_linear_velocity_rmse_m_s",
        "body_angular_velocity_rmse_rad_s",
        "joint_position_rmse_rad",
        "joint_velocity_rmse_rad_s",
        "foot_position_world_rmse_m",
        "foot_contact_agreement",
    ]
    aggregate = {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "maximum": float(np.max([row[name] for row in rows])),
        }
        for name in metrics
    }
    result = {
        "source_rollout": str(args.isaac_rollout.resolve()),
        "model": str(args.model.resolve()),
        "velocity_limit_mode": args.velocity_limit_mode,
        "sample_count": len(rows),
        "aggregate": aggregate,
        "samples": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "one_step_dynamics.json", result)
    with (args.output_dir / "one_step_dynamics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ordered = sorted(rows, key=lambda row: row["phase"])
    phases = np.asarray([row["phase"] for row in ordered])
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(phases, [row["base_position_error_m"] for row in ordered], "o-", label="base position")
    axes[0].plot(
        phases,
        [row["foot_position_world_rmse_m"] for row in ordered],
        "o-",
        label="foot position",
    )
    axes[0].set_ylabel("error [m]")
    axes[1].plot(
        phases,
        [row["joint_position_rmse_rad"] for row in ordered],
        "o-",
        label="joint position",
    )
    axes[1].plot(
        phases,
        [row["joint_velocity_rmse_rad_s"] for row in ordered],
        "o-",
        label="joint velocity",
    )
    axes[1].set_ylabel("rad, rad/s")
    axes[2].plot(
        phases,
        [row["foot_contact_agreement"] for row in ordered],
        "o-",
        label="contact agreement",
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("fraction")
    axes[2].set_xlabel("gait phase [rad]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.suptitle("One-policy-step MuJoCo prediction error from Isaac states")
    figure.tight_layout()
    figure.savefig(args.output_dir / "one_step_dynamics.png", dpi=160)
    plt.close(figure)

    report = [
        "# One-step dynamics comparison",
        "",
        (
            "Each sample resets MuJoCo to a recorded Isaac pre-action state, applies the same joint target "
            "for one policy interval, and compares the resulting state."
        ),
        "",
        "| Metric | Mean | Maximum |",
        "|---|---:|---:|",
    ]
    for name, values in aggregate.items():
        report.append(f"| {name} | {values['mean']:.6g} | {values['maximum']:.6g} |")
    report.extend(
        [
            "",
            (
                "Large swing-phase errors point toward actuator dynamics. Errors concentrated at touchdown "
                "or stance point toward contact, friction, or compliance."
            ),
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"One-step dynamics comparison: {args.output_dir.resolve()}")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-rollout", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--interface", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--velocity-limit-mode", choices=("hard", "soft", "none"), default="hard")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("Sample count must be positive")
    return args


if __name__ == "__main__":
    compare(_arguments())
