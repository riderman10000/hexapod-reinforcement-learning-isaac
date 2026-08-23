# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the resolved Isaac/PhysX robot assets for sim-to-sim verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Template-Hexpod-Rl-Lab-Direct-v0")
parser.add_argument("--output", type=Path, default=Path("sim2sim/results/assets/isaac_asset_snapshot.json"))
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

single_gpu_kit_args = (
    "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1"
)
args_cli.kit_args = f"{args_cli.kit_args} {single_gpu_kit_args}".strip()
app_launcher = AppLauncher(args_cli, multi_gpu=False)
simulation_app = app_launcher.app

import gymnasium as gym
import hexpod_rl_lab.tasks  # noqa: F401
import numpy as np

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _array(value) -> list:
    """Convert a tensor or NumPy-compatible object to JSON lists."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).astype(float).tolist()


def _quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _json_value(value):
    """Best-effort conversion of authored USD values to JSON."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "path"):
        return str(value.path)
    try:
        array = np.asarray(value)
        if array.dtype.kind in "biuf":
            return array.astype(float).tolist()
    except (TypeError, ValueError):
        pass
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _capture_collision_prims(stage: Usd.Stage, robot_path: str) -> list[dict]:
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {robot_path}")

    collisions = []
    xform_cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        attributes = {
            attribute.GetName(): _json_value(attribute.Get())
            for attribute in prim.GetAttributes()
            if attribute.HasAuthoredValueOpinion()
        }
        material_bindings = []
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith("material:binding"):
                material_bindings.extend(str(target) for target in relationship.GetTargets())
        collisions.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "attributes": attributes,
                "material_bindings": sorted(set(material_bindings)),
                "local_to_world_matrix": np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=float).tolist(),
            }
        )
    return collisions


def _capture_material_prims(stage: Usd.Stage) -> list[dict]:
    materials = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        api = UsdPhysics.MaterialAPI(prim)
        materials.append(
            {
                "path": str(prim.GetPath()),
                "static_friction": _json_value(api.GetStaticFrictionAttr().Get()),
                "dynamic_friction": _json_value(api.GetDynamicFrictionAttr().Get()),
                "restitution": _json_value(api.GetRestitutionAttr().Get()),
            }
        )
    return materials


def _capture_snapshot(env, env_cfg) -> dict:
    robot = env.robot
    robot.reset()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    masses = robot.data.default_mass[0].detach().cpu().numpy()
    inertias = robot.data.default_inertia[0].detach().cpu().numpy().reshape(-1, 3, 3)
    link_positions = robot.data.body_link_pos_w[0].detach().cpu().numpy()
    link_quaternions = robot.data.body_link_quat_w[0].detach().cpu().numpy()
    com_positions = robot.data.body_com_pos_w[0].detach().cpu().numpy()

    bodies = []
    for body_index, body_name in enumerate(robot.body_names):
        rotation = _quat_wxyz_to_matrix(link_quaternions[body_index])
        world_inertia = rotation @ inertias[body_index] @ rotation.T
        bodies.append(
            {
                "index": body_index,
                "name": body_name,
                "mass": float(masses[body_index]),
                "inertia_link_frame": inertias[body_index].astype(float).tolist(),
                "link_position_world": link_positions[body_index].astype(float).tolist(),
                "link_quaternion_world_wxyz": link_quaternions[body_index].astype(float).tolist(),
                "com_position_world": com_positions[body_index].astype(float).tolist(),
                "inertia_world": world_inertia.astype(float).tolist(),
            }
        )

    joints = []
    for joint_index, joint_name in enumerate(robot.joint_names):
        joints.append(
            {
                "index": joint_index,
                "name": joint_name,
                "position_range": _array(robot.data.joint_pos_limits[0, joint_index]),
                "velocity_limit": float(robot.data.joint_vel_limits[0, joint_index]),
                "effort_limit": float(robot.data.joint_effort_limits[0, joint_index]),
                "stiffness": float(robot.data.default_joint_stiffness[0, joint_index]),
                "damping": float(robot.data.default_joint_damping[0, joint_index]),
                "armature": float(robot.data.default_joint_armature[0, joint_index]),
                "friction": float(robot.data.default_joint_friction[0, joint_index]),
            }
        )

    stage = omni.usd.get_context().get_stage()
    xform_cache = UsdGeom.XformCache()
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    ground = env_cfg.ground_material
    return {
        "schema_version": 1,
        "simulator": "isaac_physx",
        "task": args_cli.task,
        "physics_dt": float(env.physics_dt),
        "total_mass": float(np.sum(masses)),
        "self_collisions_enabled": bool(env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions),
        "ground_material_config": {
            "static_friction": float(ground.static_friction),
            "dynamic_friction": float(ground.dynamic_friction),
            "restitution": float(ground.restitution),
            "friction_combine_mode": str(ground.friction_combine_mode),
            "restitution_combine_mode": str(ground.restitution_combine_mode),
        },
        "bodies": bodies,
        "joints": joints,
        "collision_prims": _capture_collision_prims(stage, "/World/envs/env_0/Robot"),
        "physics_material_prims": _capture_material_prims(stage),
        "robot_prim_world_matrix": np.asarray(xform_cache.GetLocalToWorldTransform(robot_prim), dtype=float).tolist(),
    }


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.reset_position_noise = 0.0
    env_cfg.reset_velocity_noise = 0.0
    env_cfg.reset_yaw_noise = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        snapshot = _capture_snapshot(env, env_cfg)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        with args_cli.output.open("w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2)
            stream.write("\n")
        print(f"Isaac asset snapshot: {args_cli.output.resolve()}")
        print(
            f"Captured {len(snapshot['bodies'])} bodies, {len(snapshot['joints'])} joints, "
            f"and {len(snapshot['collision_prims'])} collision prims."
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
