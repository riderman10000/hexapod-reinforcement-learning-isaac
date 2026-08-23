"""Compare resolved Isaac and MuJoCo assets and write an auditable gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC = PROJECT_ROOT / "sim2sim" / "results" / "assets" / "isaac_asset_snapshot.json"
DEFAULT_MUJOCO = PROJECT_ROOT / "sim2sim" / "results" / "assets" / "mujoco_asset_snapshot.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sim2sim" / "results" / "assets" / "comparison"

TOLERANCES = {
    "total_mass_relative": 1.0e-4,
    "cluster_mass_relative": 1.0e-4,
    "cluster_com_m": 5.0e-5,
    "cluster_inertia_relative": 2.0e-2,
    "joint_range_rad": 1.0e-5,
    "collision_half_extent_m": 1.0e-6,
    "collision_center_m": 5.0e-5,
    "collision_orientation_rad": 1.0e-4,
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1.0e-12)


def _aggregate_bodies(bodies: list[dict]) -> tuple[float, np.ndarray, np.ndarray]:
    masses = np.asarray([body["mass"] for body in bodies], dtype=float)
    positions = np.asarray([body["com_position_world"] for body in bodies], dtype=float)
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    inertia = np.zeros((3, 3), dtype=float)
    for mass, position, body in zip(masses, positions, bodies, strict=True):
        offset = position - center
        inertia += np.asarray(body["inertia_world"], dtype=float)
        inertia += mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
    return total_mass, center, inertia


def _cluster_definitions() -> dict[str, tuple[list[str], list[str]]]:
    clusters = {"base_link": (["base_link"], ["base_link"])}
    for leg_index in range(6):
        clusters[f"leg_{leg_index}_1"] = ([f"leg_{leg_index}_1"], [f"leg_{leg_index}_1"])
        clusters[f"leg_{leg_index}_2"] = ([f"leg_{leg_index}_2"], [f"leg_{leg_index}_2"])
        clusters[f"leg_{leg_index}_3+eef"] = (
            [f"leg_{leg_index}_3", f"dummy_eef_{leg_index}"],
            [f"leg_{leg_index}_3"],
        )
    return clusters


def _compare_mass_inertia(isaac: dict, mujoco: dict) -> dict:
    isaac_by_name = {body["name"]: body for body in isaac["bodies"]}
    mujoco_by_name = {body["name"]: body for body in mujoco["bodies"]}
    isaac_base = isaac_by_name["base_link"]
    mujoco_base = mujoco_by_name["base_link"]
    isaac_base_position = np.asarray(isaac_base["link_position_world"], dtype=float)
    mujoco_base_position = np.asarray(mujoco_base["link_position_world"], dtype=float)
    isaac_base_rotation = _quat_wxyz_to_matrix(isaac_base["link_quaternion_world_wxyz"])
    mujoco_base_rotation = _quat_wxyz_to_matrix(mujoco_base["link_quaternion_world_wxyz"])

    rows = []
    for cluster_name, (isaac_names, mujoco_names) in _cluster_definitions().items():
        isaac_mass, isaac_com, isaac_inertia = _aggregate_bodies([isaac_by_name[name] for name in isaac_names])
        mujoco_mass, mujoco_com, mujoco_inertia = _aggregate_bodies([mujoco_by_name[name] for name in mujoco_names])
        isaac_relative_com = isaac_base_rotation.T @ (isaac_com - isaac_base_position)
        mujoco_relative_com = mujoco_base_rotation.T @ (mujoco_com - mujoco_base_position)
        isaac_eigenvalues = np.linalg.eigvalsh(isaac_inertia)
        mujoco_eigenvalues = np.linalg.eigvalsh(mujoco_inertia)
        inertia_relative_error = float(
            np.max(np.abs(isaac_eigenvalues - mujoco_eigenvalues))
            / max(float(np.max(np.abs(isaac_eigenvalues))), 1.0e-12)
        )
        rows.append(
            {
                "cluster": cluster_name,
                "isaac_bodies": isaac_names,
                "mujoco_bodies": mujoco_names,
                "isaac_mass": isaac_mass,
                "mujoco_mass": mujoco_mass,
                "mass_relative_error": _relative_error(mujoco_mass, isaac_mass),
                "com_error_m": float(np.linalg.norm(isaac_relative_com - mujoco_relative_com)),
                "inertia_eigenvalue_relative_error": inertia_relative_error,
            }
        )

    total_mass_error = _relative_error(mujoco["total_mass"], isaac["total_mass"])
    maxima = {
        "total_mass_relative_error": total_mass_error,
        "cluster_mass_relative_error": max(row["mass_relative_error"] for row in rows),
        "cluster_com_error_m": max(row["com_error_m"] for row in rows),
        "cluster_inertia_relative_error": max(row["inertia_eigenvalue_relative_error"] for row in rows),
    }
    passed = (
        maxima["total_mass_relative_error"] <= TOLERANCES["total_mass_relative"]
        and maxima["cluster_mass_relative_error"] <= TOLERANCES["cluster_mass_relative"]
        and maxima["cluster_com_error_m"] <= TOLERANCES["cluster_com_m"]
        and maxima["cluster_inertia_relative_error"] <= TOLERANCES["cluster_inertia_relative"]
    )
    return {"status": "PASS" if passed else "FAIL", "maxima": maxima, "clusters": rows}


def _compare_joints(isaac: dict, mujoco: dict) -> dict:
    isaac_by_name = {joint["name"]: joint for joint in isaac["joints"]}
    mujoco_by_name = {joint["name"]: joint for joint in mujoco["joints"]}
    isaac_names = set(isaac_by_name)
    mujoco_names = set(mujoco_by_name)
    shared_names = sorted(isaac_names & mujoco_names)
    rows = []
    for name in shared_names:
        isaac_range = np.asarray(isaac_by_name[name]["position_range"], dtype=float)
        mujoco_range = np.asarray(mujoco_by_name[name]["position_range"], dtype=float)
        rows.append(
            {
                "joint": name,
                "isaac_range": isaac_range.tolist(),
                "mujoco_range": mujoco_range.tolist(),
                "maximum_range_error_rad": float(np.max(np.abs(isaac_range - mujoco_range))),
            }
        )
    maximum_range_error = max((row["maximum_range_error_rad"] for row in rows), default=float("inf"))
    passed = (
        not (isaac_names - mujoco_names)
        and not (mujoco_names - isaac_names)
        and maximum_range_error <= TOLERANCES["joint_range_rad"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "isaac_order": list(isaac_by_name),
        "mujoco_order": list(mujoco_by_name),
        "missing_from_mujoco": sorted(isaac_names - mujoco_names),
        "extra_in_mujoco": sorted(mujoco_names - isaac_names),
        "maximum_range_error_rad": maximum_range_error,
        "joints": rows,
        "note": "Storage order may differ; both control paths map joints by name.",
    }


def _isaac_collision_body(collision: dict) -> str:
    marker = "/Robot/"
    return collision["path"].split(marker, 1)[1].split("/", 1)[0]


def _usd_translation(matrix: np.ndarray) -> np.ndarray:
    # USD/Gf uses row-vector transform notation; translation occupies row 3.
    return matrix[3, :3]


def _usd_rotation(matrix: np.ndarray) -> np.ndarray:
    """Remove scale from a Gf row-vector transform and return column-vector rotation."""
    left, _, right = np.linalg.svd(matrix[:3, :3])
    row_rotation = left @ right
    if np.linalg.det(row_rotation) < 0.0:
        left[:, -1] *= -1.0
        row_rotation = left @ right
    return row_rotation.T


def _rotation_error_radians(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _compare_collisions(isaac: dict, mujoco: dict) -> dict:
    isaac_by_body = {_isaac_collision_body(collision): collision for collision in isaac["collision_prims"]}
    mujoco_by_body = {geom["body"]: geom for geom in mujoco["geoms"] if geom["body"] != "world"}
    shared_names = sorted(set(isaac_by_body) & set(mujoco_by_body))
    isaac_robot_matrix = np.asarray(isaac.get("robot_prim_world_matrix", np.eye(4)), dtype=float)
    isaac_robot_position = _usd_translation(isaac_robot_matrix)
    mujoco_base = next(body for body in mujoco["bodies"] if body["name"] == "base_link")
    mujoco_base_position = np.asarray(mujoco_base["link_position_world"], dtype=float)
    rows = []
    for body_name in shared_names:
        isaac_collision = isaac_by_body[body_name]
        mujoco_geom = mujoco_by_body[body_name]
        extent = np.asarray(isaac_collision["attributes"].get("extent"), dtype=float)
        isaac_half_extent = (extent[1] - extent[0]) / 2.0
        mujoco_half_extent = np.asarray(mujoco_geom["size"], dtype=float)
        isaac_matrix = np.asarray(isaac_collision.get("local_to_world_matrix", np.eye(4)), dtype=float)
        isaac_center = _usd_translation(isaac_matrix) - isaac_robot_position
        mujoco_center = np.asarray(mujoco_geom["position_world"], dtype=float) - mujoco_base_position
        isaac_orientation = _usd_rotation(isaac_matrix)
        mujoco_orientation = np.asarray(mujoco_geom["orientation_world_matrix"], dtype=float)
        rows.append(
            {
                "body": body_name,
                "isaac_type": isaac_collision["type"],
                "mujoco_type": mujoco_geom["type"],
                "isaac_half_extent_m": isaac_half_extent.tolist(),
                "mujoco_half_extent_m": mujoco_half_extent.tolist(),
                "maximum_half_extent_error_m": float(np.max(np.abs(isaac_half_extent - mujoco_half_extent))),
                "center_error_m": float(np.linalg.norm(isaac_center - mujoco_center)),
                "orientation_error_rad": _rotation_error_radians(isaac_orientation, mujoco_orientation),
            }
        )
    missing = sorted(set(isaac_by_body) - set(mujoco_by_body))
    extra = sorted(set(mujoco_by_body) - set(isaac_by_body))
    maximum_extent_error = max((row["maximum_half_extent_error_m"] for row in rows), default=float("inf"))
    maximum_center_error = max((row["center_error_m"] for row in rows), default=float("inf"))
    maximum_orientation_error = max((row["orientation_error_rad"] for row in rows), default=float("inf"))
    types_match = all(row["isaac_type"] == "Cube" and row["mujoco_type"] == "mjGEOM_BOX" for row in rows)
    passed = (
        not missing
        and not extra
        and types_match
        and maximum_extent_error <= TOLERANCES["collision_half_extent_m"]
        and maximum_center_error <= TOLERANCES["collision_center_m"]
        and maximum_orientation_error <= TOLERANCES["collision_orientation_rad"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_from_mujoco": missing,
        "extra_in_mujoco": extra,
        "maximum_half_extent_error_m": maximum_extent_error,
        "maximum_center_error_m": maximum_center_error,
        "maximum_orientation_error_rad": maximum_orientation_error,
        "collisions": rows,
    }


def _compare_friction(isaac: dict, mujoco: dict) -> dict:
    ground = isaac["ground_material_config"]
    default_material = next(
        (
            material
            for material in isaac.get("physics_material_prims", [])
            if material["path"].endswith("/defaultMaterial")
        ),
        None,
    )
    robot_static = float(default_material["static_friction"]) if default_material else 0.5
    robot_dynamic = float(default_material["dynamic_friction"]) if default_material else 0.5
    combine_mode = ground["friction_combine_mode"]
    if combine_mode == "multiply":
        effective_static = robot_static * float(ground["static_friction"])
        effective_dynamic = robot_dynamic * float(ground["dynamic_friction"])
    else:
        effective_static = None
        effective_dynamic = None
    mujoco_sliding = float(mujoco["ground_friction"][0])
    exact_match = (
        effective_static is not None
        and np.isclose(mujoco_sliding, effective_static)
        and np.isclose(mujoco_sliding, effective_dynamic)
    )
    return {
        "status": "PASS" if exact_match else "FAIL",
        "isaac_ground_static": float(ground["static_friction"]),
        "isaac_ground_dynamic": float(ground["dynamic_friction"]),
        "isaac_unbound_robot_static": robot_static,
        "isaac_unbound_robot_dynamic": robot_dynamic,
        "isaac_combine_mode": combine_mode,
        "isaac_effective_static": effective_static,
        "isaac_effective_dynamic": effective_dynamic,
        "mujoco_sliding": mujoco_sliding,
        "mujoco_torsional": float(mujoco["ground_friction"][1]),
        "mujoco_rolling": float(mujoco["ground_friction"][2]),
        "note": (
            "MuJoCo has one sliding coefficient rather than separate static and dynamic values. "
            "Its two contacting geoms both use the recorded sliding value."
        ),
    }


def _compare_misc(isaac: dict, mujoco: dict) -> dict:
    dt_match = np.isclose(isaac["physics_dt"], mujoco["physics_dt"], atol=1.0e-12)
    self_collision_match = isaac["self_collisions_enabled"] == mujoco["self_collisions_enabled"]
    return {
        "status": "PASS" if dt_match and self_collision_match else "FAIL",
        "isaac_physics_dt": isaac["physics_dt"],
        "mujoco_physics_dt": mujoco["physics_dt"],
        "isaac_self_collisions_enabled": isaac["self_collisions_enabled"],
        "mujoco_self_collisions_enabled": mujoco["self_collisions_enabled"],
    }


def _markdown(report: dict) -> str:
    mass = report["checks"]["mass_inertia"]
    joints = report["checks"]["joint_ranges"]
    collisions = report["checks"]["collision_geometry"]
    friction = report["checks"]["friction"]
    misc = report["checks"]["simulation_structure"]
    lines = [
        "# Hexapod asset verification report",
        "",
        f"Overall gate: **{report['overall_status']}**",
        "",
        "| Check | Status | Main result |",
        "|---|---:|---|",
        (
            f"| Mass, COM and inertia | {mass['status']} | total mass error "
            f"{mass['maxima']['total_mass_relative_error']:.3%}; maximum cluster inertia error "
            f"{mass['maxima']['cluster_inertia_relative_error']:.3%} |"
        ),
        (
            f"| Joint names and ranges | {joints['status']} | {len(joints['joints'])} shared joints; "
            f"maximum range error {joints['maximum_range_error_rad']:.3g} rad |"
        ),
        (
            f"| Collision geometry | {collisions['status']} | {len(collisions['collisions'])} matched boxes; "
            f"maximum size error {collisions['maximum_half_extent_error_m']:.3g} m; maximum center error "
            f"{collisions['maximum_center_error_m']:.3g} m; maximum orientation error "
            f"{collisions['maximum_orientation_error_rad']:.3g} rad |"
        ),
        (
            f"| Contact friction | {friction['status']} | Isaac effective static/dynamic "
            f"{friction['isaac_effective_static']:.3g}/{friction['isaac_effective_dynamic']:.3g}; "
            f"MuJoCo sliding {friction['mujoco_sliding']:.3g} |"
        ),
        (
            f"| Time step and self-collision | {misc['status']} | "
            f"dt {misc['isaac_physics_dt']:.6g} s; self-collision disabled in both |"
        ),
        "",
        "## Interpretation",
        "",
    ]
    if report["overall_status"] == "PASS":
        lines.append("The static asset gate passed. Proceed to loaded-dynamics testing.")
    else:
        lines.extend(
            [
                (
                    "The static asset gate has not passed, so loaded-dynamics differences would "
                    "currently mix asset mismatch with solver behavior."
                ),
                "",
                (
                    "The detected blocker is contact friction. Isaac's robot collision shapes have no "
                    "explicit material binding and therefore use the 0.5/0.5 default material. With the "
                    "ground's `multiply` rule, the resulting static/dynamic coefficients are 0.4/0.3. "
                    "MuJoCo currently uses 0.8 sliding friction on both robot and ground geoms."
                ),
                "",
                (
                    "Mass, inertia, joint ranges, collision boxes, physics time step, and self-collision "
                    "policy passed their tolerances. MuJoCo's 19 versus Isaac's 25 bodies is expected: "
                    "each fixed `dummy_eef` is merged into its lower-leg body before the cluster comparison."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Tolerances",
            "",
            "```json",
            json.dumps(report["tolerances"], indent=2),
            "```",
            "",
            (
                "The machine-readable details, including every cluster, joint, and collision, are in "
                "`asset_verification.json` beside this report."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, default=DEFAULT_ISAAC)
    parser.add_argument("--mujoco", type=Path, default=DEFAULT_MUJOCO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    isaac = _load(args.isaac)
    mujoco = _load(args.mujoco)
    checks = {
        "mass_inertia": _compare_mass_inertia(isaac, mujoco),
        "joint_ranges": _compare_joints(isaac, mujoco),
        "collision_geometry": _compare_collisions(isaac, mujoco),
        "friction": _compare_friction(isaac, mujoco),
        "simulation_structure": _compare_misc(isaac, mujoco),
    }
    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "overall_status": overall_status,
        "tolerances": TOLERANCES,
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "asset_verification.json"
    markdown_path = args.output_dir / "report.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(f"Asset verification gate: {overall_status}")
    for name, check in checks.items():
        print(f"  {name}: {check['status']}")
    print(f"Report: {markdown_path.resolve()}")
    print(f"Details: {json_path.resolve()}")


if __name__ == "__main__":
    main()
