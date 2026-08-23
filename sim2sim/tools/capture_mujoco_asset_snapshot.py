"""Capture the resolved MuJoCo model properties used for asset verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "sim2sim" / "results" / "assets" / "mujoco_asset_snapshot.json"


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def _rotation_from_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _capture(model_path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    bodies = []
    for body_id in range(1, model.nbody):
        body_rotation = data.xmat[body_id].reshape(3, 3).copy()
        inertial_rotation = _rotation_from_quat_wxyz(model.body_iquat[body_id])
        world_inertial_rotation = body_rotation @ inertial_rotation
        world_inertia = world_inertial_rotation @ np.diag(model.body_inertia[body_id]) @ world_inertial_rotation.T
        world_com = data.xpos[body_id] + body_rotation @ model.body_ipos[body_id]
        bodies.append(
            {
                "index": body_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "mass": float(model.body_mass[body_id]),
                "inertia_principal": model.body_inertia[body_id].astype(float).tolist(),
                "inertial_position_body": model.body_ipos[body_id].astype(float).tolist(),
                "inertial_quaternion_body_wxyz": model.body_iquat[body_id].astype(float).tolist(),
                "link_position_world": data.xpos[body_id].astype(float).tolist(),
                "link_quaternion_world_wxyz": data.xquat[body_id].astype(float).tolist(),
                "com_position_world": world_com.astype(float).tolist(),
                "inertia_world": world_inertia.astype(float).tolist(),
            }
        )

    actuator_by_joint = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_by_joint[joint_id] = actuator_id

    joints = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        actuator_id = actuator_by_joint.get(joint_id)
        joint = {
            "index": joint_id,
            "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
            "axis": model.jnt_axis[joint_id].astype(float).tolist(),
            "position_range": model.jnt_range[joint_id].astype(float).tolist(),
            "armature": float(model.dof_armature[model.jnt_dofadr[joint_id]]),
            "damping": float(model.dof_damping[model.jnt_dofadr[joint_id]]),
            "friction": float(model.dof_frictionloss[model.jnt_dofadr[joint_id]]),
        }
        if actuator_id is not None:
            joint.update(
                {
                    "actuator_name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
                    "stiffness": float(model.actuator_gainprm[actuator_id, 0]),
                    "actuator_damping": float(-model.actuator_biasprm[actuator_id, 2]),
                    "actuator_bias_parameters": model.actuator_biasprm[actuator_id].astype(float).tolist(),
                    "control_range": model.actuator_ctrlrange[actuator_id].astype(float).tolist(),
                    "force_range": model.actuator_forcerange[actuator_id].astype(float).tolist(),
                }
            )
        joints.append(joint)

    geoms = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        geoms.append(
            {
                "index": geom_id,
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "type": mujoco.mjtGeom(model.geom_type[geom_id]).name,
                "size": model.geom_size[geom_id].astype(float).tolist(),
                "position_body": model.geom_pos[geom_id].astype(float).tolist(),
                "quaternion_body_wxyz": model.geom_quat[geom_id].astype(float).tolist(),
                "position_world": data.geom_xpos[geom_id].astype(float).tolist(),
                "orientation_world_matrix": data.geom_xmat[geom_id].reshape(3, 3).astype(float).tolist(),
                "friction": model.geom_friction[geom_id].astype(float).tolist(),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )

    robot_geoms = [geom for geom in geoms if geom["body"] != "world"]
    self_collision_disabled = all(
        (left["contype"] & right["conaffinity"]) == 0 and (right["contype"] & left["conaffinity"]) == 0
        for index, left in enumerate(robot_geoms)
        for right in robot_geoms[index + 1 :]
    )
    ground = next((geom for geom in geoms if geom["name"] == "ground"), None)
    return {
        "schema_version": 1,
        "simulator": "mujoco",
        "model": str(model_path.resolve()),
        "physics_dt": float(model.opt.timestep),
        "total_mass": float(sum(body["mass"] for body in bodies)),
        "self_collisions_enabled": not self_collision_disabled,
        "ground_friction": ground["friction"] if ground else None,
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = _capture(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
    print(f"MuJoCo asset snapshot: {args.output.resolve()}")
    print(
        f"Captured {len(snapshot['bodies'])} bodies, {len(snapshot['joints'])} joints, "
        f"and {len(snapshot['geoms'])} geoms."
    )


if __name__ == "__main__":
    main()
