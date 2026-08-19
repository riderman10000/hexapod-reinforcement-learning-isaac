"""Name-based mapping between the policy contract and MuJoCo storage."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointMapping:
    policy_names: tuple[str, ...]
    mujoco_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    neutral_positions: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel, policy_names: tuple[str, ...]) -> JointMapping:
        mujoco_names = tuple(
            name
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
            and (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)) is not None
        )
        if set(mujoco_names) != set(policy_names):
            missing = sorted(set(policy_names) - set(mujoco_names))
            extra = sorted(set(mujoco_names) - set(policy_names))
            raise ValueError(f"Joint-name mismatch. Missing in MuJoCo: {missing}; extra in MuJoCo: {extra}")

        joint_ids = np.asarray([model.joint(name).id for name in policy_names], dtype=np.int32)
        qpos_addresses = model.jnt_qposadr[joint_ids].astype(np.int32)
        dof_addresses = model.jnt_dofadr[joint_ids].astype(np.int32)
        actuator_ids = np.asarray([model.actuator(f"{name}_position").id for name in policy_names], dtype=np.int32)
        limits = model.jnt_range[joint_ids]
        return cls(
            policy_names=policy_names,
            mujoco_names=mujoco_names,
            joint_ids=joint_ids,
            qpos_addresses=qpos_addresses,
            dof_addresses=dof_addresses,
            actuator_ids=actuator_ids,
            lower_limits=limits[:, 0].copy(),
            upper_limits=limits[:, 1].copy(),
            neutral_positions=model.qpos0[qpos_addresses].copy(),
        )

    def positions(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_addresses].copy()

    def velocities(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.dof_addresses].copy()

    def actuator_forces(self, data: mujoco.MjData) -> np.ndarray:
        return data.actuator_force[self.actuator_ids].copy()

    def clip_targets(self, targets: np.ndarray) -> np.ndarray:
        return np.clip(targets, self.lower_limits, self.upper_limits)

    def write_targets(self, data: mujoco.MjData, targets: np.ndarray) -> None:
        if targets.shape != (len(self.policy_names),):
            raise ValueError(f"Expected {len(self.policy_names)} joint targets, got {targets.shape}")
        data.ctrl[self.actuator_ids] = targets
