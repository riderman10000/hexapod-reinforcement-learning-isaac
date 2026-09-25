"""Compare matched Isaac and MuJoCo gait rollouts and generate diagnostic figures."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hexapod-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from sim2sim.rollout_data import load_rollout, quaternion_wxyz_to_euler, write_json


def _metadata(directory: Path) -> dict:
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _first_divergence_time(time: np.ndarray, error: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(error > threshold)
    return float(time[indices[0]]) if len(indices) else None


def _phase_profile(phase: np.ndarray, values: np.ndarray, bin_count: int = 50) -> tuple[np.ndarray, np.ndarray]:
    centers = (np.arange(bin_count) + 0.5) * 2.0 * np.pi / bin_count
    bins = np.floor((phase % (2.0 * np.pi)) * bin_count / (2.0 * np.pi)).astype(int)
    profile = np.full((bin_count, *values.shape[1:]), np.nan, dtype=float)
    for index in range(bin_count):
        selected = values[bins == index]
        if len(selected):
            profile[index] = np.mean(selected, axis=0)
    return centers, profile


def _euler_series(quaternions: np.ndarray) -> np.ndarray:
    return np.asarray([quaternion_wxyz_to_euler(quaternion) for quaternion in quaternions])


def _plot_xy(isaac: dict, mujoco: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(isaac["next_base_position"][:, 0], isaac["next_base_position"][:, 1], label="Isaac/PhysX")
    axis.plot(mujoco["next_base_position"][:, 0], mujoco["next_base_position"][:, 1], label="MuJoCo")
    maximum_x = max(float(np.max(isaac["next_base_position"][:, 0])), float(np.max(mujoco["next_base_position"][:, 0])))
    axis.plot([0.0, maximum_x], [0.0, 0.0], "k--", linewidth=1, label="desired +X")
    for data, marker in ((isaac, "o"), (mujoco, "s")):
        axis.scatter(data["next_base_position"][0, 0], data["next_base_position"][0, 1], marker=marker, s=40)
        axis.scatter(data["next_base_position"][-1, 0], data["next_base_position"][-1, 1], marker="x", s=60)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.set_xlabel("world X [m]")
    axis.set_ylabel("world Y [m]")
    axis.set_title("Base XY trajectory")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_base_motion(isaac: dict, mujoco: dict, output: Path) -> None:
    count = min(len(isaac["next_time"]), len(mujoco["next_time"]))
    time = isaac["next_time"][:count]
    isaac_euler = _euler_series(isaac["next_base_quaternion_wxyz"][:count])
    mujoco_euler = _euler_series(mujoco["next_base_quaternion_wxyz"][:count])
    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(time, isaac["next_base_position"][:count, 2], label="Isaac z")
    axes[0].plot(time, mujoco["next_base_position"][:count, 2], label="MuJoCo z")
    axes[0].set_ylabel("height [m]")
    axes[1].plot(time, isaac_euler[:, 0], label="Isaac roll")
    axes[1].plot(time, isaac_euler[:, 1], label="Isaac pitch")
    axes[1].plot(time, mujoco_euler[:, 0], "--", label="MuJoCo roll")
    axes[1].plot(time, mujoco_euler[:, 1], "--", label="MuJoCo pitch")
    axes[1].set_ylabel("tilt [rad]")
    axes[2].plot(time, isaac["next_world_linear_velocity"][:count, 0], label="Isaac vx")
    axes[2].plot(time, mujoco["next_world_linear_velocity"][:count, 0], label="MuJoCo vx")
    axes[2].plot(time, isaac["next_world_linear_velocity"][:count, 1], label="Isaac vy")
    axes[2].plot(time, mujoco["next_world_linear_velocity"][:count, 1], label="MuJoCo vy")
    axes[2].axhline(0.5, color="black", linestyle=":", linewidth=1, label="command")
    axes[2].set_ylabel("velocity [m/s]")
    axes[3].plot(time, isaac_euler[:, 2], label="Isaac yaw")
    axes[3].plot(time, mujoco_euler[:, 2], label="MuJoCo yaw")
    axes[3].plot(time, isaac["next_body_angular_velocity"][:count, 2], label="Isaac yaw rate")
    axes[3].plot(time, mujoco["next_body_angular_velocity"][:count, 2], label="MuJoCo yaw rate")
    axes[3].set_ylabel("rad, rad/s")
    axes[3].set_xlabel("time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    figure.suptitle("Base motion and stability")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_contacts(isaac: dict, mujoco: dict, output: Path) -> None:
    count = min(len(isaac["next_time"]), len(mujoco["next_time"]))
    blocks = [
        ("desired", isaac["desired_foot_contact"][:count]),
        ("Isaac", isaac["next_foot_contact"][:count]),
        ("MuJoCo", mujoco["next_foot_contact"][:count]),
    ]
    image = np.concatenate([values.T.astype(float) for _, values in blocks], axis=0)
    figure, axis = plt.subplots(figsize=(12, 7))
    extent = [float(isaac["next_time"][0]), float(isaac["next_time"][count - 1]), len(image), 0]
    axis.imshow(image, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0, extent=extent)
    for boundary in (6, 12):
        axis.axhline(boundary, color="tab:red", linewidth=1.5)
    labels = [f"{group} L{leg}" for group, _ in blocks for leg in range(6)]
    axis.set_yticks(np.arange(0.5, 18.0, 1.0), labels=labels)
    axis.set_xlabel("time [s]")
    axis.set_title("Foot-contact gait diagram (black = stance)")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_joint_phase(isaac: dict, mujoco: dict, output: Path) -> None:
    isaac_phase, isaac_profile = _phase_profile(isaac["phase"], isaac["next_joint_position"])
    mujoco_phase, mujoco_profile = _phase_profile(mujoco["phase"], mujoco["next_joint_position"])
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    groups = (("hip", 0), ("thigh", 6), ("knee", 12))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 6))
    for axis, (group_name, start) in zip(axes, groups, strict=True):
        for leg_index, color in enumerate(colors):
            axis.plot(isaac_phase, isaac_profile[:, start + leg_index], color=color, label=f"L{leg_index} Isaac")
            axis.plot(
                mujoco_phase, mujoco_profile[:, start + leg_index], "--", color=color, label=f"L{leg_index} MuJoCo"
            )
        axis.set_ylabel(f"{group_name} [rad]")
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
        labels=["0", "π/2", "π", "3π/2", "2π"],
    )
    axes[-1].set_xlabel("gait phase")
    figure.suptitle("Mean joint angles over normalized gait phase")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_foot_trajectories(isaac: dict, mujoco: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for leg_index, axis in enumerate(axes.flat):
        axis.plot(
            isaac["next_foot_position_body"][:, leg_index, 0],
            isaac["next_foot_position_body"][:, leg_index, 2],
            label="Isaac",
        )
        axis.plot(
            mujoco["next_foot_position_body"][:, leg_index, 0],
            mujoco["next_foot_position_body"][:, leg_index, 2],
            label="MuJoCo",
        )
        axis.set_title(f"leg {leg_index}")
        axis.set_xlabel("base-frame X [m]")
        axis.set_ylabel("base-frame Z [m]")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()
    figure.suptitle("Foot sagittal trajectories")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_actuator_tracking(isaac: dict, mujoco: dict, joint_names: list[str], output: Path) -> None:
    selected = [0, 6, 12]
    count = min(len(isaac["next_time"]), len(mujoco["next_time"]))
    time = isaac["next_time"][:count]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, joint_index in zip(axes, selected, strict=True):
        axis.plot(time, isaac["joint_target"][:count, joint_index], label="Isaac target")
        axis.plot(time, isaac["next_joint_position"][:count, joint_index], label="Isaac actual")
        axis.plot(time, mujoco["joint_target"][:count, joint_index], "--", label="MuJoCo target")
        axis.plot(time, mujoco["next_joint_position"][:count, joint_index], "--", label="MuJoCo actual")
        axis.set_ylabel("angle [rad]")
        axis.set_title(joint_names[joint_index])
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Target-to-joint tracking for leg 0")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def compare(isaac_dir: Path, mujoco_dir: Path, output_dir: Path) -> dict:
    isaac_meta = _metadata(isaac_dir)
    mujoco_meta = _metadata(mujoco_dir)
    errors = []
    for field in (
        "policy_sha256",
        "joint_names",
        "physics_dt",
        "policy_dt",
        "decimation",
        "phase_offset",
        "command_world",
    ):
        if isaac_meta[field] != mujoco_meta[field]:
            errors.append(f"{field}: Isaac={isaac_meta[field]!r}, MuJoCo={mujoco_meta[field]!r}")
    if errors:
        raise ValueError("Rollouts are not matched:\n  " + "\n  ".join(errors))
    isaac = load_rollout(isaac_dir / "policy_steps.npz")
    mujoco = load_rollout(mujoco_dir / "policy_steps.npz")
    count = min(len(isaac["time"]), len(mujoco["time"]))
    if count < 2:
        raise ValueError("Need at least two common rollout samples")
    time = isaac["next_time"][:count]
    base_error = np.linalg.norm(isaac["next_base_position"][:count] - mujoco["next_base_position"][:count], axis=1)
    joint_error = np.sqrt(
        np.mean(np.square(isaac["next_joint_position"][:count] - mujoco["next_joint_position"][:count]), axis=1)
    )
    action_error = np.sqrt(np.mean(np.square(isaac["raw_action"][:count] - mujoco["raw_action"][:count]), axis=1))
    comparison = {
        "isaac_mode": isaac_meta["mode"],
        "mujoco_mode": mujoco_meta["mode"],
        "common_samples": count,
        "common_duration_seconds": float(time[-1]),
        "base_position_rmse_m": _rmse(isaac["next_base_position"][:count], mujoco["next_base_position"][:count]),
        "joint_position_rmse_rad": _rmse(isaac["next_joint_position"][:count], mujoco["next_joint_position"][:count]),
        "joint_velocity_rmse_rad_s": _rmse(isaac["next_joint_velocity"][:count], mujoco["next_joint_velocity"][:count]),
        "raw_action_rmse": _rmse(isaac["raw_action"][:count], mujoco["raw_action"][:count]),
        "foot_position_body_rmse_m": _rmse(
            isaac["next_foot_position_body"][:count], mujoco["next_foot_position_body"][:count]
        ),
        "foot_contact_agreement": float(
            np.mean(isaac["next_foot_contact"][:count] == mujoco["next_foot_contact"][:count])
        ),
        "first_base_divergence_over_2cm_s": _first_divergence_time(time, base_error, 0.02),
        "first_joint_rmse_over_0_05rad_s": _first_divergence_time(time, joint_error, 0.05),
        "first_action_rmse_over_0_1_s": _first_divergence_time(time, action_error, 0.1),
        "initial_observation_max_abs_error": float(np.max(np.abs(isaac["observation"][0] - mujoco["observation"][0]))),
        "initial_raw_action_max_abs_error": float(np.max(np.abs(isaac["raw_action"][0] - mujoco["raw_action"][0]))),
        "initial_joint_target_max_abs_error": float(
            np.max(np.abs(isaac["joint_target"][0] - mujoco["joint_target"][0]))
        ),
        "initial_joint_state_max_abs_error": float(
            np.max(np.abs(isaac["joint_position"][0] - mujoco["joint_position"][0]))
        ),
        "first_step_joint_position_rmse_rad": float(joint_error[0]),
        "isaac_contact_duty_factor": np.mean(isaac["next_foot_contact"][:count], axis=0).tolist(),
        "mujoco_contact_duty_factor": np.mean(mujoco["next_foot_contact"][:count], axis=0).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    _plot_xy(isaac, mujoco, figures / "xy_trajectory.png")
    _plot_base_motion(isaac, mujoco, figures / "base_motion.png")
    _plot_contacts(isaac, mujoco, figures / "gait_contact_diagram.png")
    _plot_joint_phase(isaac, mujoco, figures / "joint_phase_profiles.png")
    _plot_foot_trajectories(isaac, mujoco, figures / "foot_trajectories.png")
    _plot_actuator_tracking(isaac, mujoco, isaac_meta["joint_names"], figures / "actuator_tracking.png")
    write_json(output_dir / "comparison.json", comparison)
    report = [
        "# Matched gait and trajectory comparison",
        "",
        "The two captures use the same policy hash, command, gait phase, joint order, and control timing.",
        "",
        f"MuJoCo capture mode: `{comparison['mujoco_mode']}`.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Common duration | {comparison['common_duration_seconds']:.3f} s |",
        f"| Base-position RMSE | {comparison['base_position_rmse_m']:.5f} m |",
        f"| Joint-position RMSE | {comparison['joint_position_rmse_rad']:.5f} rad |",
        f"| Joint-velocity RMSE | {comparison['joint_velocity_rmse_rad_s']:.5f} rad/s |",
        f"| Raw-action RMSE | {comparison['raw_action_rmse']:.5f} |",
        f"| Body-frame foot-position RMSE | {comparison['foot_position_body_rmse_m']:.5f} m |",
        f"| Per-foot contact agreement | {comparison['foot_contact_agreement']:.2%} |",
        "",
        "## Initial-state parity",
        "",
        f"- Observation maximum error: `{comparison['initial_observation_max_abs_error']:.6g}`.",
        f"- Raw-action maximum error: `{comparison['initial_raw_action_max_abs_error']:.6g}`.",
        f"- Joint-target maximum error: `{comparison['initial_joint_target_max_abs_error']:.6g}`.",
        f"- Joint-state maximum error: `{comparison['initial_joint_state_max_abs_error']:.6g}`.",
        f"- Joint RMSE after the first policy interval: `{comparison['first_step_joint_position_rmse_rad']:.6g} rad`.",
        "",
        "## First divergence",
        "",
        f"- Base error above 2 cm: `{comparison['first_base_divergence_over_2cm_s']}` seconds.",
        f"- Joint RMSE above 0.05 rad: `{comparison['first_joint_rmse_over_0_05rad_s']}` seconds.",
        f"- Raw-action RMSE above 0.1: `{comparison['first_action_rmse_over_0_1_s']}` seconds.",
        "",
        (
            "Use early divergence to locate the cause. Long-horizon pointwise equality is not expected "
            "after contact changes the closed-loop observations."
        ),
        "",
        "## Figures",
        "",
        "- `figures/xy_trajectory.png`: commanded path, drift, and turning.",
        "- `figures/base_motion.png`: hopping, tilt, velocity, and yaw.",
        "- `figures/gait_contact_diagram.png`: desired and actual stance timing.",
        "- `figures/joint_phase_profiles.png`: gait shape normalized by phase.",
        "- `figures/foot_trajectories.png`: foot swing loops and clearance.",
        "- `figures/actuator_tracking.png`: target versus actual angles.",
        "",
    ]
    if mujoco_meta["mode"] == "open_loop_isaac_target_replay":
        report.insert(
            -1,
            (
                "In open-loop replay, observations and actions are copied from Isaac by design; "
                "zero action error does not test inference."
            ),
        )
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Gait comparison: {output_dir.resolve()}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.isaac_dir, args.mujoco_dir, args.output_dir)


if __name__ == "__main__":
    main()
