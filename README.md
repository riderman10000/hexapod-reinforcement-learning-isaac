# Hexapod RL Lab

An Isaac Lab project for training a 6-legged (18 DOF) hexapod robot to walk using reinforcement learning.
Built from the Isaac Lab external-project template, developed against **Isaac Lab 2.3.2** / **Isaac Sim 5.1.0**.

## Robot assets

| Path | What it is |
|---|---|
| [`assets/urdf/hexapod.urdf`](assets/urdf/hexapod.urdf) | Original URDF source for the robot. |
| [`assets/urdf/hexapod/hexapod.usd`](assets/urdf/hexapod/hexapod.usd) | The USD actually loaded into simulation — referenced by `HEXAPOD_USD` in [`robots/hexapod.py`](source/hexpod_rl_lab/hexpod_rl_lab/robots/hexapod.py). |
| [`assets/urdf/hexapod/configuration/`](assets/urdf/hexapod/configuration/) | Split-out USD layers (`hexapod_base`, `hexapod_physics`, `hexapod_robot`, `hexapod_sensor`) referenced by the main USD. |
| [`assets/usd/hexapod-all.usd`](assets/usd/hexapod-all.usd) | A combined/standalone USD variant. |
| [`assets/mujoco/hexapod.xml`](assets/mujoco/hexapod.xml) | Floating-base MJCF used for standalone MuJoCo sim-to-sim evaluation. |

The `ArticulationCfg` (actuator gains, effort/velocity limits, spawn pose, initial joint angles) lives in
[`source/hexpod_rl_lab/hexpod_rl_lab/robots/hexapod.py`](source/hexpod_rl_lab/hexpod_rl_lab/robots/hexapod.py).
Joint name groupings (6 legs × 3 joints each: hip/`body_leg_*`, thigh/`leg_*_1_2`, knee/`leg_*_2_3`) are centralized in
[`robots/joints.py`](source/hexpod_rl_lab/hexpod_rl_lab/robots/joints.py) so they aren't redefined elsewhere.

`HEXAPOD_USD` is resolved relative to the repository root, so the project can be moved or cloned without editing
the robot configuration.

To sanity-check that the asset loads and looks right on its own (no RL, just a spawn), use:

```bash
python scripts/test_spawn_agent.py
```

## Environment setup

1. Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   (conda or uv recommended). This project was built/tested against Isaac Lab 2.3.2 + Isaac Sim 5.1.0.

2. Install this project in editable mode, using the Python that has Isaac Lab installed:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab isn't on a venv/conda python
    python -m pip install -e source/hexpod_rl_lab
    ```

3. **Pin `rsl-rl-lib` to the version this Isaac Lab checkout expects.** The default/older `rsl-rl-lib` on some
   environments (e.g. `3.1.2`) is missing config fields (`optimizer`, `share_cnn_encoders`, ...) that
   `isaaclab_rl`'s `RslRlPpoAlgorithmCfg` now always sets, and will fail with
   `TypeError: PPO.__init__() got an unexpected keyword argument ...` when the runner is constructed. Check
   what your Isaac Lab checkout actually requires (`source/isaaclab_rl/setup.py`, `"rsl-rl"` extra — `5.0.1`
   in this checkout) and install that exact version:

    ```bash
    python -m pip install "rsl-rl-lib==5.0.1" "onnxscript>=0.5"
    ```

4. Verify the task is registered:

    ```bash
    python scripts/list_envs.py
    ```

   You should see `Template-Hexpod-Rl-Lab-Direct-v0` in the list (registered in
   [`tasks/direct/hexpod_rl_lab/__init__.py`](source/hexpod_rl_lab/hexpod_rl_lab/tasks/direct/hexpod_rl_lab/__init__.py)).

5. Sanity-check the environment itself with dummy agents before touching RL — useful for confirming
   observation/action shapes and that the robot doesn't immediately explode:

    ```bash
    python scripts/zero_agent.py --task=Template-Hexpod-Rl-Lab-Direct-v0     # holds zero action
    python scripts/random_agent.py --task=Template-Hexpod-Rl-Lab-Direct-v0   # random actions
    ```

## Where the task is configured

- Environment logic (observations, rewards, resets, termination): [`hexpod_rl_lab_env.py`](source/hexpod_rl_lab/hexpod_rl_lab/tasks/direct/hexpod_rl_lab/hexpod_rl_lab_env.py)
- Environment/scene/reward/termination constants: [`hexpod_rl_lab_env_cfg.py`](source/hexpod_rl_lab/hexpod_rl_lab/tasks/direct/hexpod_rl_lab/hexpod_rl_lab_env_cfg.py)
  (`num_envs`, `action_scale`, reward scales, `termination_height`, etc.)
- PPO/runner hyperparameters (network sizes, learning rate, epochs, `max_iterations`, ...):
  [`agents/rsl_rl_ppo_cfg.py`](source/hexpod_rl_lab/hexpod_rl_lab/tasks/direct/hexpod_rl_lab/agents/rsl_rl_ppo_cfg.py)

## Training

Trainer used: `rsl_rl` (PPO). Other library scaffolds also exist under `scripts/` (`rl_games`, `skrl`, `sb3`) but
are not the ones wired up for this robot — use `rsl_rl`.

**Always smoke-test before a full run** — use a few environments and iterations, then inspect the TensorBoard
`Episode_Reward/*` and `Episode_Termination/*` series. A high fall count or a reward dominated by penalties means
the spawn pose, actuator gains, or reward scales still need tuning:

```bash
python scripts/rsl_rl/train.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --headless --num_envs=64 --max_iterations=50
```

If that looks healthy, run the full training job (uses `num_envs`/`max_iterations` from the cfg files above
unless overridden on the CLI):

```bash
python scripts/rsl_rl/train.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --headless
```

Each run writes to `logs/rsl_rl/hexapod_direct/<timestamp>[_<run_name>]/`, including periodic checkpoints
(`model_<iter>.pt`) and the resolved `env.yaml`/`agent.yaml` for that run.

The current task uses 70 observations and position-target actions. Checkpoints trained with the previous
36/48-observation effort-action tasks, the 66-observation body-frame task, or the 68-observation straight-line task
are incompatible and must not be resumed; start a fresh run after this change.

Straight-line runs expose additional `Episode_Metric/*` TensorBoard series. A healthy result should approach
`world_forward_velocity=0.5`, `abs_yaw_rate=0`, `heading_alignment=1`, `gait_contact_match=1`, and
`forward_path_efficiency=1`. The six `foot_contact_fraction_leg_*_3` curves make an inactive leg visible even when
the total reward looks healthy.

[`scripts/train.sh`](scripts/train.sh) starts a fresh corrected run. Resume only checkpoints created with this
70-observation straight-line tripod-gait position-action task:

```bash
python scripts/rsl_rl/train.py --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --resume --load_run <run_dir_name> --checkpoint <checkpoint_file>.pt --run_name <new_run_suffix>
```

## Inference / playback

Visualize a trained checkpoint (drop `--headless`, i.e. this opens a viewer). Note the exact task name —
`Template-Hexpod-Rl-Lab-Direct-v0` — matches the package name (`hexpod_rl_lab`), not "Hexapod".

`--checkpoint`, if given, is resolved as a literal path (relative to your cwd, or absolute) and is **not**
joined with `--load_run` — see [`play.py`](scripts/rsl_rl/play.py). So either:

```bash
# A) give the full path to the checkpoint
python scripts/rsl_rl/play.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --resume \
    --checkpoint logs/rsl_rl/hexapod_direct/<run_dir_name>/<checkpoint_file>.pt
```

```bash
# B) omit --checkpoint and let it auto-pick the latest checkpoint in the run
python scripts/rsl_rl/play.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --resume --load_run <run_dir_name>
```

[`scripts/play.sh`](scripts/play.sh) is a convenience wrapper around option A. Pass the corrected checkpoint path:

```bash
./scripts/play.sh logs/rsl_rl/hexapod_direct/<run_dir_name>/<checkpoint_file>.pt
```

## Sim-to-sim transfer with MuJoCo

The standalone MuJoCo runner replays the exported ONNX policy in a different physics engine before hardware
deployment. It intentionally lives under [`sim2sim/`](sim2sim/) and does not import Isaac Lab, so simulator-specific
dependencies and coordinate conversions remain visible and testable.

### 1. Create the MuJoCo environment

Use Python 3.11 and keep this environment separate from the Isaac Lab environment. The native MuJoCo archive under
`~/.mujoco/` provides command-line tools, but it does not install the Python bindings used by this runner.

```bash
conda create -n hexapod_mujoco python=3.11 -y
conda activate hexapod_mujoco
python -m pip install -r sim2sim/requirements.txt
```

Confirm the versions and run the contract tests:

```bash
python -c "import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
python -m pytest sim2sim/validation -q
```

The model is pinned to MuJoCo 3.3.6. ONNX Runtime uses the CPU provider; the small actor easily meets the 30 Hz policy
rate without CUDA.

### 2. Build or refresh the MJCF asset

The source URDF imports into native MuJoCo as a fixed-base model without a ground plane or actuators. The builder uses
MuJoCo's URDF importer and then adds the missing transfer contract: a free base, restored base inertia, ground contact,
18 position drives, sensors, a home keyframe, and the Isaac nominal timestep and gains.

```bash
python sim2sim/tools/build_mujoco_model.py
```

This regenerates [`assets/mujoco/hexapod.xml`](assets/mujoco/hexapod.xml) and validates `nq=25`, `nv=24`, and `nu=18`.
Run it again whenever the URDF geometry, inertia, limits, or joint names change. Self-collisions remain disabled to
match the current Isaac asset. A `0.0001 kg m^2` joint armature provides reflected inertia for MuJoCo's very light
distal links, following Isaac Lab's recommendation to model armature explicitly during solver transfer. It is a
solver-stability starting point that should eventually be replaced by a motor/gear measurement. MuJoCo has one
sliding-friction coefficient rather than separate static/dynamic values, so the model uses `0.8` as the nominal
contact value and records that approximation as part of the transfer gap.

### 3. Verify the trained policy interface

Joint order is part of the checkpoint, even though joint names are not embedded in ONNX. On a terminal where Isaac Sim
and this project are configured, print the actual PhysX order:

```bash
python scripts/dump_policy_interface.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --headless --num_envs=1
```

If a standalone Isaac Sim installation is used with conda, source its environment first:

```bash
conda activate env_232
source <ISAAC_SIM_PATH>/setup_conda_env.sh
python scripts/dump_policy_interface.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --headless --num_envs=1
```

Compare `joint_names` in the output with `policy_joint_names` in
[`sim2sim/configs/policy_interface.yaml`](sim2sim/configs/policy_interface.yaml). Update the YAML list if they differ;
do not reorder the trained policy or change the Isaac environment to make it match. The MuJoCo runner reads and writes
all joints by name and performs the required permutation.

The checked-in contract has now been verified against the runtime articulation: Isaac orders all six hip joints,
then all six thigh joints, then all six knee joints. MuJoCo's native model remains organized leg-by-leg. These two
printed orders are therefore expected to differ; the name-based mapping converts between them.

The same YAML records the complete transfer contract: 70 observations, 18 actions, `dt=1/120 s`, decimation 4,
30 Hz position targets, action scale `0.3 rad`, clipping to `[-1, 1]`, the `5.236 rad/s` Isaac simulation velocity
limit, the 1.5 Hz gait phase, and the 0.5 m/s world-X command.

### 4. Export and validate the policy

Playing an RSL-RL checkpoint through [`scripts/rsl_rl/play.py`](scripts/rsl_rl/play.py) automatically writes
`exported/policy.onnx` and `exported/policy.pt` beside the checkpoint. The ONNX graph already includes the learned
observation normalizer; do not normalize the 70-value input again.

Validate the model, policy shapes, timing, and joint mapping without stepping physics:

```bash
python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/<run_name>/exported/policy.onnx \
    --validate-only
```

The expected interface is `obs [1,70] -> actions [1,18]`. Validation fails immediately for a wrong checkpoint,
missing joint, duplicated joint, changed timestep, or incompatible input/output name.

### 5. Run inference with the viewer

Run from the repository root in a graphical desktop session:

```bash
python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/<run_name>/exported/policy.onnx
```

The viewer follows `base_link` and playback is synchronized to real time. Close the viewer to stop early. To start
from a randomized phase like Isaac resets, add `--random-phase --seed 42`. The defaults use no settling delay and no
action ramp because those defaults reproduce the training contract. `--settle-seconds` and `--ramp-seconds` are
available as explicit diagnostics, but results produced with them should be labeled as modified reset/control tests.

The viewer uses a high-contrast tiled ground so translation and turning are easier to judge. A normal interactive run
lasts up to 60 seconds. Press `Q` or `Esc` to stop early and still write the CSV and summary. To remove the time limit:

```bash
python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/<run_name>/exported/policy.onnx \
    --until-closed
```

An unlimited run still terminates if the robot meets a configured fall condition; headless evaluation always uses a
finite duration.

### 6. Run repeatable headless evaluation

```bash
python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/<run_name>/exported/policy.onnx \
    --headless --duration 20 --random-phase --seed 42 --plot
```

Each run creates `sim2sim/results/<timestamp>/` containing:

- `rollout.csv`: base pose and velocity, heading, contact count, all joint positions/velocities, policy actions,
  position targets, and actuator forces at every 30 Hz policy step.
- `summary.json`: falls, displacement, mean forward/lateral velocity, mean absolute yaw rate, path efficiency,
  orientation extrema, force maximum, and action-saturation fraction.
- `rollout.png`: velocity, heading/yaw, orientation/height, action, and force plots when `--plot` is supplied.

Use several fixed seeds and compare these results with Isaac's `Episode_Metric/*` values. A useful initial acceptance
gate is a 20-second rollout without falling, world-forward velocity near `0.5 m/s`, small lateral velocity/yaw rate,
high forward-path efficiency, and no persistent action or force saturation. Do not tune MuJoCo merely to make one
trajectory visually match Isaac; first verify names, frames, units, timing, actuator limits, and contact assumptions.

### Execution flow

At every policy step, the runner performs the same interface operations as the Isaac task:

1. Read the floating-base state and named joint state from MuJoCo.
2. Build the 70-value observation in the exact Isaac order: body linear/angular velocity, projected gravity,
   world-frame command, body-frame desired heading, gait phase, relative joint positions, joint velocities, and the
   previous clipped action.
3. Run the ONNX actor once on CPU, reject NaN/Inf, and clip its 18 outputs to `[-1, 1]`.
4. Calculate `target = neutral + 0.3 * action`, clamp it to joint limits, and map it to MuJoCo actuators by joint name.
5. Hold that target for four `1/120 s` physics steps, enforcing the same `5.236 rad/s` numerical joint-velocity limit
   and giving the same 30 Hz policy period as Isaac.
6. Evaluate fall conditions and write one metrics row before constructing the next observation.

The current runner is a nominal sim-to-sim baseline. If the policy contract is correct but dynamics differ, follow the
[Isaac Lab transfer workflow](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html):
identify actuator/contact discrepancies, measure source and target metrics, then retrain with physically plausible
randomization of mass/inertia, friction, drive gains, joint offsets, latency, and observation noise. Keep a
deterministic nominal evaluation alongside randomized runs.

### Current verified-order baseline result

The `straight_line_v2` ONNX policy was smoke-tested with MuJoCo 3.3.6 for 20 seconds using `--random-phase --seed 42`
after correcting the policy order from the runtime Isaac dump. It completed the rollout without falling, traveled
`4.526 m` forward and `1.040 m` laterally, and achieved `0.227 m/s` mean world-forward velocity with `0.481` forward
path efficiency. The mapping correction therefore produced a substantial improvement over the invalid leg-by-leg
baseline.

The transfer is not yet deployment-ready: mean absolute lateral velocity was `0.278 m/s`, mean absolute yaw rate was
`1.426 rad/s`, and action, effort, and numerical velocity limits were still reached. Treat this as the valid nominal
baseline for the next diagnostics: joint-axis signs, actuator response and velocity limiting, then contact/friction
and physically plausible domain-randomized retraining.

### 8. Run the cross-simulator diagnostic suite

For a standalone explanation of every test, its purpose, commands, figures, current results, limitations, and next
steps, read [`sim2sim/DIAGNOSTICS_GUIDE.md`](sim2sim/DIAGNOSTICS_GUIDE.md).

The diagnostics deliberately separate three failure classes instead of changing several physics parameters at once:

1. **Joint sweep:** command `+0.05 rad` on one joint at a time with robot gravity disabled, record its position,
   velocity, actuator effort, and the downstream end-effector displacement in the base frame. This checks both the
   joint-coordinate response and the physical direction of the leg tip.
2. **Observation/action parity:** reset to a deterministic neutral state, save all 70 observations by named term, and
   run the same ONNX actor without clipping its raw output. This identifies the first term at which the two policy
   interfaces diverge.
3. **Velocity-limit A/B:** run the same MuJoCo initial condition and phase with the legacy hard state clip, an
   experimental smooth brake, and no explicit speed limiter. This isolates the effect of velocity handling.

Capture the Isaac reference in the GPU-enabled Isaac environment:

```bash
conda activate env_232
source /home/rlwagun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh

python scripts/capture_isaac_diagnostics.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/isaac \
    --headless
```

This diagnostic disables explicit and automatic renderer multi-GPU use because it only needs one environment. CUDA
visibility is deliberately left unchanged because filtering it can make CUDA and Vulkan assign different indices to
the same physical GPU. GPU 0 is selected by default; use `--device cuda:N` to select another GPU. On an IOMMU-enabled
multi-GPU workstation, Isaac Sim 5.1 may still print a `p2pBandwidthLatencyTest` "peer access is already enabled"
message. In this workflow it is a non-fatal startup diagnostic; success is determined by the final
`Isaac diagnostics: ...` line and the three generated files.

The capture script changes only its private diagnostic environment: reset noise is zero and gravity is disabled on
the robot so contacts and falling do not hide a joint-axis error. It does not modify training configuration files.

Capture the matching MuJoCo data:

```bash
conda activate hexapod_mujoco

python -m sim2sim.tools.capture_mujoco_diagnostics \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/mujoco
```

Compare the captures and generate the report and plots:

```bash
python -m sim2sim.tools.compare_diagnostics \
    --isaac-dir sim2sim/results/diagnostics/isaac \
    --mujoco-dir sim2sim/results/diagnostics/mujoco \
    --output-dir sim2sim/results/diagnostics/comparison
```

The comparison directory contains:

- `report.md` and `comparison.json`: per-term observation errors, raw-action errors, and one verdict per joint.
- `joint_sweep_comparison.png`: a 6-by-3 grid overlaying the Isaac and MuJoCo position response. A downstream-foot
  direction cosine near `+1` means the physical axis directions agree; near `-1` indicates a reversed axis.
- `observation_action_comparison.png`: observation error by named term and all 18 raw actor outputs.

For a neutral snapshot, exact command, heading, phase, joint position, joint velocity, and previous-action terms should
match to numerical tolerance. Investigate the first non-zero term rather than compensating for it later in the
controller. The report marks a joint as passing when both simulators respond positively to `+0.05 rad` and their
end-effector displacement directions have cosine similarity of at least `0.8`.

Run the velocity treatment comparison only after observation and axis checks pass:

```bash
python -m sim2sim.tools.compare_velocity_modes \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/velocity_ab \
    --duration 20 \
    --random-phase --seed 42
```

This writes one rollout directory per mode, `velocity_mode_comparison.json`, and
`velocity_mode_comparison.png`. The modes are:

- `hard`: the original post-step `qvel` overwrite and current default, retained as the control case.
- `soft`: a continuous smoothstep braking torque that activates above 80% of the configured speed limit. It does not
  overwrite simulator state, but remains an experimental approximation until motor parameters are identified.
- `none`: no explicit MuJoCo speed treatment; effort-limited position actuators remain active.

Run multiple fixed phase seeds before choosing a mode. The summary now reports component-level action saturation,
action sign changes, actuator-force saturation, joint-speed-limit activity, and maximum speed/limit ratio. A smoother
mode is preferable only if it reduces chatter and lateral/yaw error without causing falls or uncontrolled overspeed.

To inspect one mode in the viewer, use for example `--velocity-limit-mode soft` with the normal inference command.

## Known limitations (as of this writing)

These don't block training but are worth knowing before trusting the resulting gait:

- The URDF/USD joint limits are currently ±20 degrees for all 18 joints. Confirm these ranges against the physical
  servos before sim-to-real training; changing the limits also requires retuning `action_scale` and the neutral pose.
- Self-collisions remain disabled because adjacent box collisions overlap near several joint pivots. Enable them only
  after replacing those collision shapes with hardware-accurate simplified geometry.
- The target command is fixed at 0.5 m/s forward. Randomized velocity/yaw commands and dynamics randomization should
  be added after a stable forward gait is learned.

## Set up IDE (Optional)

- Run VSCode Tasks (`Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env`), and provide the absolute path to
  your Isaac Sim installation when prompted. This generates `.vscode/.python.env` with paths to all Isaac
  Sim/Omniverse Python modules, for editor autocomplete/indexing.

## Setup as Omniverse Extension (Optional)

An example UI extension is provided in `source/hexpod_rl_lab/hexpod_rl_lab/ui_extension_example.py`. To enable it:

1. In Omniverse, go to `Window` → `Extensions` → hamburger icon → `Settings`.
2. Under `Extension Search Paths`, add the absolute path to this repo's `source` directory (and, if not
   already present, `IsaacLab/source`).
3. Hamburger icon → `Refresh`, then find this extension under `Third Party` and enable it.

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Pylance missing indexing of extensions

Add the path to this project in `.vscode/settings.json` under `"python.analysis.extraPaths"`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/hexpod_rl_lab"
    ]
}
```

### Pylance crash

Usually caused by indexing too many Omniverse packages and running out of memory. Comment out unused package
paths under `"python.analysis.extraPaths"` in `.vscode/settings.json`, e.g.:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
```



---
RL setup 
```bash 
 conda create --no-default-packages python=3.11 -n env_232

#  232 is the isaaclab version 
```

```bash 
conda activate env_232

python scripts/rsl_rl/train.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --headless --num_envs=64 --max_iterations=50


python scripts/rsl_rl/play.py --task Template-Hexpod-Rl-Lab-Direct-v0 --resume --checkpoint logs/rsl_rl/hexapod_direct/2026-08-06_16-00-17/model_49.pt

```
