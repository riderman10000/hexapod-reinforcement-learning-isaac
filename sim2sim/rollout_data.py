"""Shared storage and math helpers for matched simulator rollouts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
FOOT_NAMES = tuple(f"foot_{index}" for index in range(6))


class ArrayRecorder:
    """Collect consistently shaped named arrays and save a compressed NPZ."""

    def __init__(self) -> None:
        self._values: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

    def append(self, **values) -> None:
        if not values:
            raise ValueError("At least one value is required")
        if self._values and set(values) != set(self._values):
            missing = sorted(set(self._values) - set(values))
            extra = sorted(set(values) - set(self._values))
            raise ValueError(f"Recorder fields changed. Missing: {missing}; extra: {extra}")
        for name, value in values.items():
            array = np.asarray(value)
            if name not in self._values:
                self._values[name] = []
                self._shapes[name] = array.shape
            elif array.shape != self._shapes[name]:
                raise ValueError(f"Field {name!r} changed shape from {self._shapes[name]} to {array.shape}")
            self._values[name].append(array.copy())

    @property
    def sample_count(self) -> int:
        if not self._values:
            return 0
        return len(next(iter(self._values.values())))

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.stack(values, axis=0) for name, values in self._values.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())


def load_rollout(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ into independent arrays so the file can close immediately."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_at_step(step: int, policy_dt: float, gait_frequency_hz: float, offset: float) -> float:
    return float((2.0 * math.pi * gait_frequency_hz * step * policy_dt + offset) % (2.0 * math.pi))


def desired_tripod_contacts(phase: float) -> np.ndarray:
    tripod_a_stance = math.sin(phase) >= 0.0
    tripod_a = np.asarray([True, False, True, False, True, False])
    return tripod_a if tripod_a_stance else ~tripod_a


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_wxyz_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quaternion_angle_error(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(left, right)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def summarize_policy_rollout(data: dict[str, np.ndarray], target_forward_velocity: float) -> dict:
    if len(data["time"]) == 0:
        raise ValueError("Cannot summarize an empty rollout")
    position = data["next_base_position"]
    velocity = data["next_world_linear_velocity"]
    angular_velocity = data["next_body_angular_velocity"]
    displacement = position[-1] - data["base_position"][0]
    path_segments = np.linalg.norm(np.diff(np.vstack((data["base_position"][0, :2], position[:, :2])), axis=0), axis=1)
    path_length = float(np.sum(path_segments))
    contact = data["next_foot_contact"].astype(bool)
    desired = data["desired_foot_contact"].astype(bool)
    applied_action = data["applied_action"]
    force = data["next_joint_force"]
    stance_slip = np.zeros(6)
    foot_positions = data["next_foot_position_world"]
    for index in range(1, len(foot_positions)):
        common_stance = contact[index - 1] & contact[index]
        stance_slip += common_stance * np.linalg.norm(foot_positions[index] - foot_positions[index - 1], axis=1)
    return {
        "duration_seconds": float(data["next_time"][-1]),
        "samples": int(len(data["time"])),
        "fell": bool(np.any(data["terminated"])),
        "forward_displacement": float(displacement[0]),
        "lateral_displacement": float(displacement[1]),
        "path_length": path_length,
        "forward_path_efficiency": float(displacement[0] / max(path_length, 1.0e-9)),
        "mean_world_forward_velocity": float(np.mean(velocity[:, 0])),
        "forward_velocity_rmse": float(np.sqrt(np.mean(np.square(velocity[:, 0] - target_forward_velocity)))),
        "mean_abs_world_lateral_velocity": float(np.mean(np.abs(velocity[:, 1]))),
        "mean_abs_yaw_rate": float(np.mean(np.abs(angular_velocity[:, 2]))),
        "gait_contact_agreement": float(np.mean(contact == desired)),
        "foot_contact_duty_factor": np.mean(contact, axis=0).astype(float).tolist(),
        "stance_marker_motion_distance_per_foot": stance_slip.astype(float).tolist(),
        "mean_stance_marker_motion_distance": float(np.mean(stance_slip)),
        "action_component_saturation_fraction": float(np.mean(np.abs(applied_action) >= 0.999)),
        "maximum_abs_joint_force": float(np.max(np.abs(force))),
    }
