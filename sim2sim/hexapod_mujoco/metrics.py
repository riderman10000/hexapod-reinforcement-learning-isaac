"""CSV logging, summary statistics, and optional rollout plots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


def quaternion_to_euler_wxyz(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Return intrinsic roll, pitch, yaw for a MuJoCo wxyz quaternion."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class RolloutMetrics:
    """Stream detailed policy-step metrics to CSV and build a compact summary."""

    def __init__(self, output_dir: str | Path, joint_names: tuple[str, ...]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "rollout.csv"
        self.summary_path = self.output_dir / "summary.json"
        self.joint_names = joint_names
        self._stream = self.csv_path.open("w", newline="", encoding="utf-8")

        base_fields = [
            "time",
            "base_x",
            "base_y",
            "base_z",
            "roll",
            "pitch",
            "yaw",
            "world_vx",
            "world_vy",
            "world_vz",
            "body_vx",
            "body_vy",
            "body_vz",
            "body_wx",
            "body_wy",
            "body_wz",
            "heading_alignment",
            "path_efficiency",
            "max_abs_action",
            "max_abs_actuator_force",
            "minimum_joint_limit_margin",
            "contact_count",
            "fall_reason",
        ]
        joint_fields = [
            f"{prefix}_{name}"
            for prefix in ("position", "velocity", "action", "target", "force")
            for name in joint_names
        ]
        self._writer = csv.DictWriter(self._stream, fieldnames=base_fields + joint_fields)
        self._writer.writeheader()

        self._sample_count = 0
        self._sum_world_velocity = np.zeros(3)
        self._sum_abs_lateral_velocity = 0.0
        self._sum_abs_yaw_rate = 0.0
        self._max_abs_roll = 0.0
        self._max_abs_pitch = 0.0
        self._max_abs_force = 0.0
        self._max_abs_joint_velocity = 0.0
        self._saturated_steps = 0
        self._path_length = 0.0
        self._initial_position: np.ndarray | None = None
        self._previous_position: np.ndarray | None = None
        self._final_position: np.ndarray | None = None

    def record(
        self,
        *,
        time_seconds: float,
        base_position: np.ndarray,
        base_quaternion: np.ndarray,
        world_linear_velocity: np.ndarray,
        body_linear_velocity: np.ndarray,
        body_angular_velocity: np.ndarray,
        heading_alignment: float,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        action: np.ndarray,
        target: np.ndarray,
        actuator_force: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        contact_count: int,
        fall_reason: str | None,
    ) -> None:
        roll, pitch, yaw = quaternion_to_euler_wxyz(base_quaternion)
        if self._initial_position is None:
            self._initial_position = base_position.copy()
            self._previous_position = base_position.copy()
        assert self._previous_position is not None
        self._path_length += float(np.linalg.norm(base_position[:2] - self._previous_position[:2]))
        self._previous_position = base_position.copy()
        self._final_position = base_position.copy()

        displacement = base_position[:2] - self._initial_position[:2]
        path_efficiency = float(displacement[0] / max(self._path_length, 1.0e-9))
        limit_margin = np.minimum(joint_position - lower_limits, upper_limits - joint_position)
        max_abs_action = float(np.max(np.abs(action)))
        max_abs_force = float(np.max(np.abs(actuator_force)))

        row: dict[str, float | int | str] = {
            "time": time_seconds,
            "base_x": base_position[0],
            "base_y": base_position[1],
            "base_z": base_position[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "world_vx": world_linear_velocity[0],
            "world_vy": world_linear_velocity[1],
            "world_vz": world_linear_velocity[2],
            "body_vx": body_linear_velocity[0],
            "body_vy": body_linear_velocity[1],
            "body_vz": body_linear_velocity[2],
            "body_wx": body_angular_velocity[0],
            "body_wy": body_angular_velocity[1],
            "body_wz": body_angular_velocity[2],
            "heading_alignment": heading_alignment,
            "path_efficiency": path_efficiency,
            "max_abs_action": max_abs_action,
            "max_abs_actuator_force": max_abs_force,
            "minimum_joint_limit_margin": float(np.min(limit_margin)),
            "contact_count": contact_count,
            "fall_reason": fall_reason or "",
        }
        for index, name in enumerate(self.joint_names):
            row[f"position_{name}"] = joint_position[index]
            row[f"velocity_{name}"] = joint_velocity[index]
            row[f"action_{name}"] = action[index]
            row[f"target_{name}"] = target[index]
            row[f"force_{name}"] = actuator_force[index]
        self._writer.writerow(row)

        self._sample_count += 1
        self._sum_world_velocity += world_linear_velocity
        self._sum_abs_lateral_velocity += abs(float(world_linear_velocity[1]))
        self._sum_abs_yaw_rate += abs(float(body_angular_velocity[2]))
        self._max_abs_roll = max(self._max_abs_roll, abs(roll))
        self._max_abs_pitch = max(self._max_abs_pitch, abs(pitch))
        self._max_abs_force = max(self._max_abs_force, max_abs_force)
        self._max_abs_joint_velocity = max(self._max_abs_joint_velocity, float(np.max(np.abs(joint_velocity))))
        self._saturated_steps += int(max_abs_action >= 0.999)

    def close(self, *, duration: float, fall_reason: str | None) -> dict[str, float | bool | str]:
        self._stream.close()
        samples = max(self._sample_count, 1)
        initial = self._initial_position if self._initial_position is not None else np.zeros(3)
        final = self._final_position if self._final_position is not None else initial
        summary: dict[str, float | bool | str] = {
            "duration_seconds": float(duration),
            "samples": self._sample_count,
            "fell": fall_reason is not None,
            "fall_reason": fall_reason or "",
            "forward_displacement": float(final[0] - initial[0]),
            "lateral_displacement": float(final[1] - initial[1]),
            "path_length": self._path_length,
            "forward_path_efficiency": float((final[0] - initial[0]) / max(self._path_length, 1.0e-9)),
            "mean_world_forward_velocity": float(self._sum_world_velocity[0] / samples),
            "mean_abs_world_lateral_velocity": float(self._sum_abs_lateral_velocity / samples),
            "mean_abs_yaw_rate": float(self._sum_abs_yaw_rate / samples),
            "maximum_abs_roll": self._max_abs_roll,
            "maximum_abs_pitch": self._max_abs_pitch,
            "maximum_abs_actuator_force": self._max_abs_force,
            "maximum_abs_joint_velocity": self._max_abs_joint_velocity,
            "action_saturation_fraction": float(self._saturated_steps / samples),
        }
        with self.summary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        return summary


def plot_rollout(csv_path: str | Path, output_path: str | Path) -> None:
    """Create a compact diagnostic plot from a rollout CSV."""
    import matplotlib.pyplot as plt

    with Path(csv_path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows])

    time_values = values("time")
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time_values, values("world_vx"), label="forward vx")
    axes[0].plot(time_values, values("world_vy"), label="lateral vy")
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="command")
    axes[0].legend()
    axes[0].set_ylabel("velocity [m/s]")
    axes[1].plot(time_values, values("body_wz"), label="yaw rate")
    axes[1].plot(time_values, values("yaw"), label="yaw")
    axes[1].legend()
    axes[1].set_ylabel("rad, rad/s")
    axes[2].plot(time_values, values("base_z"), label="base height")
    axes[2].plot(time_values, values("roll"), label="roll")
    axes[2].plot(time_values, values("pitch"), label="pitch")
    axes[2].legend()
    axes[3].plot(time_values, values("max_abs_action"), label="max |action|")
    axes[3].plot(time_values, values("max_abs_actuator_force"), label="max |force|")
    axes[3].legend()
    axes[3].set_xlabel("policy time [s]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
