"""
Configuration for the custom Hexapod robot.

This module defines the ArticulationCfg used by Isaac Lab.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)

from .joints import (
    ALL_JOINTS,
    DEFAULT_JOINT_POSITIONS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
HEXAPOD_USD = str(PROJECT_ROOT / "assets" / "urdf" / "hexapod" / "hexapod.usd")

# Robot configuration
HEXAPOD_CFG = ArticulationCfg(
    # Spawn configuration
    spawn=sim_utils.UsdFileCfg(
        usd_path=HEXAPOD_USD,
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # Keep contact correction impulses reasonable for this small robot.
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    # Initial state
    init_state=ArticulationCfg.InitialStateCfg(
        # At zero joint angles the lowest collision point is 0.1275 m below
        # the root. Spawning at 0.15 m provides clearance before settling.
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_JOINT_POSITIONS,
        joint_vel={joint: 0.0 for joint in ALL_JOINTS},
    ),
    # Actuators
    actuators={
        # Hip joints
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[
                "body_leg_.*",
            ],
            effort_limit_sim=1.961,
            velocity_limit_sim=5.236,
            stiffness=40.0,
            damping=2.0,
        ),
        # Thigh joints
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_._1_2",
            ],
            effort_limit_sim=1.961,
            velocity_limit_sim=5.236,
            stiffness=50.0,
            damping=2.5,
        ),
        # Knee joints
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_._2_3",
            ],
            effort_limit_sim=1.961,
            velocity_limit_sim=5.236,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)
