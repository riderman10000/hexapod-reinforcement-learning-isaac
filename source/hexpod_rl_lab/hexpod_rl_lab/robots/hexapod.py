"""
Configuration for the custom Hexapod robot.

This module defines the ArticulationCfg used by Isaac Lab.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)

from .joints import (
    ALL_JOINTS,
    DEFAULT_JOINT_POSITIONS,
)

# Replace this with the location of your robot USD.
# Example:
# usd_path="/home/user/projects/hexpod/assets/hexapod.usd"
# usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Hexapod/hexapod.usd"
HEXAPOD_USD = "/home/rlwagun/files/hexapod-sim-real-proj/hexpod_rl_lab/assets/urdf/hexapod/hexapod.usd"

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
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    # Initial state
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DEFAULT_JOINT_POSITIONS,
        joint_vel={
            joint: 0.0
            for joint in ALL_JOINTS
        },
    ),
    # Actuators
    actuators={
        # Hip joints
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[
                "body_leg_.*",
            ],
            effort_limit=2.0,
            velocity_limit=6.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # Thigh joints
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_._1_2",
            ],
            effort_limit=2.0,
            velocity_limit=6.0,
            stiffness=50.0,
            damping=2.5,
        ),
        # Knee joints
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_._2_3",
            ],
            effort_limit=2.0,
            velocity_limit=6.0,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)