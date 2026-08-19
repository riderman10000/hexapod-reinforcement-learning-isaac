"""Load and validate the simulator-independent policy contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class PolicyInterface:
    observation_size: int
    action_size: int
    input_name: str
    output_name: str
    physics_dt: float
    decimation: int
    policy_dt: float
    action_scale: float
    action_clip: float
    joint_velocity_limit: float
    gait_frequency_hz: float
    command_world: np.ndarray
    minimum_base_height: float
    maximum_tilt_radians: float
    policy_joint_names: tuple[str, ...]
    observation_layout: tuple[tuple[str, int], ...]


def load_policy_interface(path: str | Path) -> PolicyInterface:
    """Load a YAML policy contract and reject incompatible dimensions early."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    policy = raw["policy"]
    timing = raw["timing"]
    control = raw["control"]
    termination = raw["termination"]
    joint_names = tuple(str(name) for name in raw["policy_joint_names"])
    observation_layout = tuple((str(term["name"]), int(term["size"])) for term in raw["observation_layout"])

    interface = PolicyInterface(
        observation_size=int(policy["observation_size"]),
        action_size=int(policy["action_size"]),
        input_name=str(policy["input_name"]),
        output_name=str(policy["output_name"]),
        physics_dt=float(timing["physics_dt"]),
        decimation=int(timing["decimation"]),
        policy_dt=float(timing["policy_dt"]),
        action_scale=float(control["action_scale"]),
        action_clip=float(control["action_clip"]),
        joint_velocity_limit=float(control["joint_velocity_limit"]),
        gait_frequency_hz=float(control["gait_frequency_hz"]),
        command_world=np.asarray(
            [
                control["target_forward_velocity"],
                control["target_lateral_velocity"],
                control["target_yaw_velocity"],
            ],
            dtype=np.float32,
        ),
        minimum_base_height=float(termination["minimum_base_height"]),
        maximum_tilt_radians=float(termination["maximum_tilt_radians"]),
        policy_joint_names=joint_names,
        observation_layout=observation_layout,
    )

    if interface.observation_size != sum(size for _, size in observation_layout):
        raise ValueError(
            f"Observation layout totals {sum(size for _, size in observation_layout)}, "
            f"but policy expects {interface.observation_size}"
        )
    if len(joint_names) != interface.action_size or len(set(joint_names)) != len(joint_names):
        raise ValueError("policy_joint_names must contain each of the 18 action joints exactly once")
    expected_policy_dt = interface.physics_dt * interface.decimation
    if not np.isclose(interface.policy_dt, expected_policy_dt, atol=1.0e-10):
        raise ValueError(f"policy_dt={interface.policy_dt} does not equal dt*decimation={expected_policy_dt}")
    if interface.action_clip <= 0.0 or interface.action_scale <= 0.0 or interface.joint_velocity_limit <= 0.0:
        raise ValueError("Action clip, action scale, and joint velocity limit must be positive")
    return interface
