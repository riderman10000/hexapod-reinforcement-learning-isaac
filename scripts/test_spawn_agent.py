from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

from hexpod_rl_lab.robots.hexapod import HEXAPOD_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = sim_utils.SimulationContext(sim_cfg)

robot_cfg = HEXAPOD_CFG.replace(prim_path="/World/Robot")
robot = Articulation(robot_cfg)

# Ground plane
ground_cfg = sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0,
    )
)
ground_cfg.func("/World/GroundPlane", ground_cfg)

# Dome light
light_cfg = sim_utils.DomeLightCfg(
    intensity=3000.0,
    color=(1.0, 1.0, 1.0),
)
light_cfg.func("/World/Light", light_cfg)


sim.reset()

# ArticulationCfg.init_state defines the desired reset state, but a standalone
# scene must explicitly write it to PhysX (DirectRLEnv normally does this).
root_state = robot.data.default_root_state.clone()
joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
robot.write_root_pose_to_sim(root_state[:, :7])
robot.write_root_velocity_to_sim(root_state[:, 7:])
robot.write_joint_state_to_sim(joint_pos, joint_vel)
robot.set_joint_position_target(joint_pos)

while simulation_app.is_running():
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())

simulation_app.close()
