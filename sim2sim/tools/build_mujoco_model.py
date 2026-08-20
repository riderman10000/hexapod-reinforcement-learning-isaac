"""Convert the source URDF into the floating-base MJCF used for sim-to-sim.

MuJoCo imports an unconnected URDF root as a fixed world body. This utility uses
MuJoCo's own URDF importer, then wraps the imported robot in a floating
``base_link``, restores the base inertia, adds a ground plane, and installs the
same nominal position-drive parameters used by Isaac Lab.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "assets" / "urdf" / "hexapod.urdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "assets" / "mujoco" / "hexapod.xml"


def _numbers(text: str | None, expected: int, default: tuple[float, ...]) -> list[float]:
    if text is None:
        return list(default)
    values = [float(value) for value in text.split()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {values}")
    return values


def _base_inertial(urdf_path: Path) -> dict[str, str]:
    urdf_root = ET.parse(urdf_path).getroot()
    inertial = urdf_root.find("./link[@name='base_link']/inertial")
    if inertial is None:
        raise ValueError("URDF base_link must have an inertial element")

    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError("URDF base_link inertial must specify mass and inertia")

    xyz = _numbers(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0))
    ixx = float(inertia.get("ixx", "0"))
    iyy = float(inertia.get("iyy", "0"))
    izz = float(inertia.get("izz", "0"))
    ixy = float(inertia.get("ixy", "0"))
    ixz = float(inertia.get("ixz", "0"))
    iyz = float(inertia.get("iyz", "0"))
    return {
        "pos": " ".join(f"{value:.9g}" for value in xyz),
        "mass": mass.get("value", "0"),
        "fullinertia": f"{ixx:.9g} {iyy:.9g} {izz:.9g} {ixy:.9g} {ixz:.9g} {iyz:.9g}",
    }


def _joint_names_in_tree(worldbody: ET.Element) -> list[str]:
    return [joint.get("name", "") for joint in worldbody.iter("joint") if joint.get("name")]


def _drive_gains(joint_name: str) -> tuple[float, float]:
    if joint_name.startswith("body_leg_"):
        return 40.0, 2.0
    if joint_name.endswith("_1_2"):
        return 50.0, 2.5
    if joint_name.endswith("_2_3"):
        return 60.0, 3.0
    raise ValueError(f"No actuator group for joint {joint_name!r}")


def build_model(urdf_path: Path, output_path: Path) -> None:
    """Build and validate the standalone MuJoCo model."""
    urdf_path = urdf_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imported_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    with tempfile.TemporaryDirectory(prefix="hexapod-mujoco-") as temp_dir:
        imported_xml = Path(temp_dir) / "imported.xml"
        mujoco.mj_saveLastXML(str(imported_xml), imported_model)
        tree = ET.parse(imported_xml)

    root = tree.getroot()
    root.set("model", "hexapod_sim2sim")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo's URDF conversion did not create a worldbody")
    imported_children = list(worldbody)
    imported_joint_names = _joint_names_in_tree(worldbody)
    if len(imported_joint_names) != 18:
        raise ValueError(f"Expected 18 imported joints, found {imported_joint_names}")
    for child in imported_children:
        worldbody.remove(child)

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(list(root).index(worldbody), asset)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "ground_grid_texture",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": "0.16 0.20 0.24",
            "rgb2": "0.72 0.76 0.80",
            "mark": "edge",
            "markrgb": "0.05 0.05 0.05",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "ground_grid",
            "texture": "ground_grid_texture",
            "texrepeat": "5 5",
            "texuniform": "true",
            "reflectance": "0.05",
        },
    )

    option = ET.Element(
        "option",
        {
            "timestep": f"{1.0 / 120.0:.16g}",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "100",
        },
    )
    root.insert(list(root).index(worldbody), option)

    statistic = ET.Element("statistic", {"center": "0 0 0.08", "extent": "0.8"})
    root.insert(list(root).index(worldbody), statistic)
    visual = ET.Element("visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    root.insert(list(root).index(worldbody), visual)

    ET.SubElement(
        worldbody,
        "light",
        {"name": "sun", "pos": "0 0 2", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "0 0 0.05",
            "material": "ground_grid",
            "friction": "0.8 0.005 0.0001",
            "contype": "0",
            "conaffinity": "1",
        },
    )
    robot = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": "0 0 0.15"})
    ET.SubElement(robot, "freejoint", {"name": "base_free_joint"})
    ET.SubElement(robot, "inertial", _base_inertial(urdf_path))
    ET.SubElement(robot, "site", {"name": "imu_site", "pos": "0 0 0", "size": "0.004", "rgba": "0 1 0 1"})
    ET.SubElement(
        robot,
        "camera",
        {"name": "robot_follow", "mode": "trackcom", "pos": "-0.6 -0.6 0.35", "xyaxes": "0.707 -0.707 0 0.3 0.3 0.906"},
    )

    base_geom_seen = False
    for child in imported_children:
        robot.append(child)
        if child.tag == "geom" and not base_geom_seen:
            child.set("name", "base_collision")
            base_geom_seen = True
    if not base_geom_seen:
        raise ValueError("MuJoCo's URDF conversion did not produce the base collision geom")

    # Isaac currently disables all robot self-collisions. Robot geoms collide
    # with the ground (conaffinity=1), but not with one another.
    for geom_index, geom in enumerate(robot.iter("geom")):
        if not geom.get("name"):
            geom.set("name", f"robot_collision_{geom_index}")
        geom.set("contype", "1")
        geom.set("conaffinity", "0")
        geom.set("friction", "0.8 0.005 0.0001")

    body_elements = {body.get("name"): body for body in robot.iter("body")}
    for leg_index in range(6):
        lower_leg = body_elements[f"leg_{leg_index}_3"]
        lateral_sign = 1.0 if leg_index < 3 else -1.0
        ET.SubElement(
            lower_leg,
            "site",
            {
                "name": f"eef_{leg_index}",
                "pos": f"0 {lateral_sign * 0.0883883476:.10g} -0.0883883476",
                "size": "0.004",
                "rgba": "0 1 0 1",
            },
        )

    actuator = ET.SubElement(root, "actuator")
    joint_elements = {joint.get("name"): joint for joint in robot.iter("joint")}
    for joint_name in imported_joint_names:
        joint = joint_elements[joint_name]
        # Reflected inertia is especially important for MuJoCo's light distal
        # links. This conservative nominal value suppresses solver-specific
        # impulse spikes without hiding it in the policy controller.
        joint.set("armature", "0.0001")
        lower, upper = _numbers(joint.get("range"), 2, (-0.3491, 0.3491))
        kp, kv = _drive_gains(joint_name)
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_position",
                "joint": joint_name,
                "kp": f"{kp:g}",
                "kv": f"{kv:g}",
                "ctrllimited": "true",
                "ctrlrange": f"{lower:.9g} {upper:.9g}",
                "forcelimited": "true",
                "forcerange": "-1.961 1.961",
            },
        )

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", {"name": "base_position", "objtype": "xbody", "objname": "base_link"})
    ET.SubElement(sensor, "framequat", {"name": "base_orientation", "objtype": "xbody", "objname": "base_link"})
    ET.SubElement(sensor, "velocimeter", {"name": "base_linear_velocity_body", "site": "imu_site"})
    ET.SubElement(sensor, "gyro", {"name": "base_angular_velocity_body", "site": "imu_site"})

    keyframe = ET.SubElement(root, "keyframe")
    qpos = [0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0] + [0.0] * 18
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": " ".join(f"{value:g}" for value in qpos),
            "ctrl": " ".join("0" for _ in range(18)),
        },
    )

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    if (model.nq, model.nv, model.nu) != (25, 24, 18):
        raise ValueError(f"Unexpected final model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Wrote {output_path}")
    print(f"Model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("MuJoCo hinge order:")
    for name in imported_joint_names:
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_model(args.urdf, args.output)


if __name__ == "__main__":
    main()
