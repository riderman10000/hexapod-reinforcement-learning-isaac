from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from hexpod_rl_lab.robots.hexapod import HEXAPOD_CFG

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = sim_utils.SimulationContext(sim_cfg)

robot_cfg = HEXAPOD_CFG.replace(
    prim_path="/World/Robot"
)

robot = robot_cfg.spawn.func(robot_cfg.prim_path, robot_cfg.spawn)



# Ground plane
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/GroundPlane", ground_cfg)

# Dome light
light_cfg = sim_utils.DomeLightCfg(
    intensity=3000.0,
    color=(1.0, 1.0, 1.0),
)
light_cfg.func("/World/Light", light_cfg)


sim.reset()

while simulation_app.is_running():
    sim.step()

simulation_app.close()