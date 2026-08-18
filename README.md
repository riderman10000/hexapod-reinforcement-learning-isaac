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

The current task uses 66 observations and position-target actions. Checkpoints trained with the previous
36/48-observation effort-action task are incompatible and must not be resumed; start a fresh run after this change.

[`scripts/train.sh`](scripts/train.sh) starts a fresh corrected run. Resume only checkpoints created with this
66-observation position-action task:

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