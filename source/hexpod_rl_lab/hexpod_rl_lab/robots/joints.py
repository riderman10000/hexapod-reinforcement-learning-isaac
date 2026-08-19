"""
Joint definitions for the custom Hexapod robot.

These constants are shared throughout the project so that joint names
are only defined in one location.
"""

# Number of legs / joints
NUM_LEGS = 6
JOINTS_PER_LEG = 3
NUM_DOFS = NUM_LEGS * JOINTS_PER_LEG

# Hip joints
HIP_JOINTS = [
    "body_leg_0",
    "body_leg_1",
    "body_leg_2",
    "body_leg_3",
    "body_leg_4",
    "body_leg_5",
]

# Thigh joints
THIGH_JOINTS = [
    "leg_0_1_2",
    "leg_1_1_2",
    "leg_2_1_2",
    "leg_3_1_2",
    "leg_4_1_2",
    "leg_5_1_2",
]

# Knee joints
KNEE_JOINTS = [
    "leg_0_2_3",
    "leg_1_2_3",
    "leg_2_2_3",
    "leg_3_2_3",
    "leg_4_2_3",
    "leg_5_2_3",
]


# All actuated joints
ALL_JOINTS = HIP_JOINTS + THIGH_JOINTS + KNEE_JOINTS

# Ground-contact links. The dummy_eef links are visual markers without
# collision geometry; contact forces are reported on these lower-leg links.
FOOT_LINKS = [
    "leg_0_3",
    "leg_1_3",
    "leg_2_3",
    "leg_3_3",
    "leg_4_3",
    "leg_5_3",
]

# Individual leg definitions
LEG0 = [
    "body_leg_0",
    "leg_0_1_2",
    "leg_0_2_3",
]

LEG1 = [
    "body_leg_1",
    "leg_1_1_2",
    "leg_1_2_3",
]

LEG2 = [
    "body_leg_2",
    "leg_2_1_2",
    "leg_2_2_3",
]

LEG3 = [
    "body_leg_3",
    "leg_3_1_2",
    "leg_3_2_3",
]

LEG4 = [
    "body_leg_4",
    "leg_4_1_2",
    "leg_4_2_3",
]

LEG5 = [
    "body_leg_5",
    "leg_5_1_2",
    "leg_5_2_3",
]


LEGS = [
    LEG0,
    LEG1,
    LEG2,
    LEG3,
    LEG4,
    LEG5,
]


# Neutral standing configuration (radians). The lower-leg geometry already
# contains the downward bend, so zero is the centered pose for all 20-degree
# joints in the current URDF/USD asset.
DEFAULT_HIP_ANGLE = 0.0

DEFAULT_THIGH_ANGLE = 0.0

DEFAULT_KNEE_ANGLE = 0.0


DEFAULT_JOINT_POSITIONS = {
    # Leg 0
    "body_leg_0": DEFAULT_HIP_ANGLE,
    "leg_0_1_2": DEFAULT_THIGH_ANGLE,
    "leg_0_2_3": DEFAULT_KNEE_ANGLE,
    # Leg 1
    "body_leg_1": DEFAULT_HIP_ANGLE,
    "leg_1_1_2": DEFAULT_THIGH_ANGLE,
    "leg_1_2_3": DEFAULT_KNEE_ANGLE,
    # Leg 2
    "body_leg_2": DEFAULT_HIP_ANGLE,
    "leg_2_1_2": DEFAULT_THIGH_ANGLE,
    "leg_2_2_3": DEFAULT_KNEE_ANGLE,
    # Leg 3
    "body_leg_3": DEFAULT_HIP_ANGLE,
    "leg_3_1_2": DEFAULT_THIGH_ANGLE,
    "leg_3_2_3": DEFAULT_KNEE_ANGLE,
    # Leg 4
    "body_leg_4": DEFAULT_HIP_ANGLE,
    "leg_4_1_2": DEFAULT_THIGH_ANGLE,
    "leg_4_2_3": DEFAULT_KNEE_ANGLE,
    # Leg 5
    "body_leg_5": DEFAULT_HIP_ANGLE,
    "leg_5_1_2": DEFAULT_THIGH_ANGLE,
    "leg_5_2_3": DEFAULT_KNEE_ANGLE,
}
