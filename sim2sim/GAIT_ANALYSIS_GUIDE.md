# Matched gait, trajectory, replay, and one-step analysis

This guide explains how to record the same ONNX policy in Isaac/PhysX and MuJoCo, visualize both gaits, replay
Isaac's targets open-loop in MuJoCo, and measure one-policy-step dynamics errors. Run commands from the repository
root.

The examples use:

```text
logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx
```

Substitute a different exported policy path when evaluating another checkpoint.

## Why these tests are separate

A closed-loop comparison answers whether the complete transferred controller behaves similarly. It does not isolate
the cause: after a small physical difference, the next observation changes, the policy emits a different action, and
the future trajectories naturally diverge.

The three tests answer different questions:

| Test | Commands supplied to MuJoCo | Question answered |
|---|---|---|
| Matched closed-loop rollout | MuJoCo policy outputs | Does the complete controller transfer? |
| Open-loop target replay | Recorded Isaac targets | Do equal commands produce equal physical motion? |
| One-step comparison | One recorded Isaac target from a copied state | Where does the first local dynamics error occur? |

## 1. Rebuild and verify the active MJCF

Changing `sim2sim/tools/build_mujoco_model.py` does not alter the generated XML until the builder is run:

```bash
conda activate hexapod_mujoco

python -m sim2sim.tools.build_mujoco_model
python -m sim2sim.tools.capture_mujoco_asset_snapshot
python -m sim2sim.tools.compare_asset_snapshots
```

Check the runtime friction values:

```bash
grep 'friction="' assets/mujoco/hexapod.xml | head
```

The current model contains sliding friction `0.4` on the ground and robot geoms. The static asset report still marks
friction as a strict mismatch because one MuJoCo sliding coefficient cannot simultaneously equal Isaac's effective
`0.4` static and `0.3` dynamic values. Record the selected mapping as an approximation.

## 2. Capture the matched Isaac rollout

Activate the Isaac environment and load the standalone Isaac Sim environment variables:

```bash
conda activate env_232
source /home/rlwagun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh
```

Then run:

```bash
python scripts/capture_isaac_rollout.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --phase-offset 0 \
    --duration 20 \
    --output-dir sim2sim/results/gait/phase_0/isaac \
    --device cuda:0 --headless
```

The capture uses one environment, zero reset noise, a zero-yaw neutral state, zero previous action, and the requested
gait phase. It runs the exported ONNX directly so both simulators evaluate the same policy file.

## 3. Capture the matched MuJoCo rollout

```bash
conda activate hexapod_mujoco

python -m sim2sim.tools.capture_mujoco_rollout \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --phase-offset 0 \
    --duration 20 \
    --velocity-limit-mode hard \
    --output-dir sim2sim/results/gait/phase_0/mujoco \
    --headless
```

Keep `phase-offset`, `duration`, policy, command, timing, and velocity-limit treatment unchanged while comparing
physics parameters.

## 4. Generate the closed-loop figures and report

```bash
python -m sim2sim.tools.compare_gait_rollouts \
    --isaac-dir sim2sim/results/gait/phase_0/isaac \
    --mujoco-dir sim2sim/results/gait/phase_0/mujoco \
    --output-dir sim2sim/results/gait/phase_0/comparison
```

Open:

- `sim2sim/results/gait/phase_0/comparison/report.md`
- `sim2sim/results/gait/phase_0/comparison/comparison.json`
- the PNG files under `sim2sim/results/gait/phase_0/comparison/figures/`

The comparison rejects captures with different policy hashes, joint names, timing, phase offsets, or commands.

### XY trajectory

`xy_trajectory.png` shows the top-down base paths with equal axis scaling. Look for lateral drift, curvature, backward
motion, and unequal final displacement. It describes the outcome but does not identify the mechanism.

### Base motion

`base_motion.png` shows height, roll/pitch, forward/lateral velocity, yaw, and yaw rate. Periodic height peaks at
touchdown indicate hopping or excessive vertical contact impulses.

### Gait contact diagram

`gait_contact_diagram.png` has 18 rows: six desired contacts, six Isaac contacts, and six MuJoCo contacts. Black means
stance. Alternating tripods should switch together. Early touchdown, long double-stance, missing contacts, or extra
contacts indicate gait/contact differences.

### Joint phase profiles

`joint_phase_profiles.png` bins all samples by gait phase and plots mean hip, thigh, and knee angles. Phase
normalization separates gait shape from accumulated timing drift. Solid curves are Isaac and dashed curves are
MuJoCo.

### Foot trajectories

`foot_trajectories.png` plots each `dummy_eef` marker's base-frame X-Z loop. Compare step length, clearance, and loop
shape. The marker is attached to the lower leg; it is not the exact changing contact point on the box collision.

### Actuator tracking

`actuator_tracking.png` overlays target and actual hip, thigh, and knee angles for leg 0. If targets initially match
but actual angles immediately differ, investigate loaded gains, damping, armature, force limiting, and contact before
changing the policy.

## Shared capture files

Each simulator directory contains:

| File | Contents |
|---|---|
| `metadata.json` | Simulator, policy/model hashes, joint order, command, phase, timing, and thresholds |
| `policy_steps.npz` | Observations, raw/applied actions, targets, states, foot data, and next states at 30 Hz |
| `physics_steps.npz` | Joint/base/foot/contact data at 120 Hz |
| `summary.json` | Locomotion, contact, saturation, and marker-motion metrics |

Use NumPy to inspect a signal:

```python
import numpy as np

data = np.load("sim2sim/results/gait/phase_0/isaac/policy_steps.npz")
print(data.files)
print(data["observation"].shape)             # [steps, 70]
print(data["raw_action"].shape)              # [steps, 18]
print(data["next_foot_contact"].shape)       # [steps, 6]
print(data["next_foot_position_body"].shape) # [steps, 6, 3]
```

`stance_marker_motion_distance` is a useful relative proxy, not a direct foot-slip measurement. A true slip test
must track the actual ground contact points because the collision is a lower-leg box while `dummy_eef` is only a
marker.

## 5. Replay Isaac targets open-loop in MuJoCo

```bash
python -m sim2sim.tools.replay_mujoco_targets \
    --isaac-rollout sim2sim/results/gait/phase_0/isaac \
    --velocity-limit-mode hard \
    --output-dir sim2sim/results/gait/phase_0/mujoco_replay
```

Generate the replay comparison:

```bash
python -m sim2sim.tools.compare_gait_rollouts \
    --isaac-dir sim2sim/results/gait/phase_0/isaac \
    --mujoco-dir sim2sim/results/gait/phase_0/mujoco_replay \
    --output-dir sim2sim/results/gait/phase_0/replay_comparison
```

The replay copies Isaac's observation, raw action, clipped action, and target arrays for reporting, but only the joint
targets are applied to MuJoCo. Therefore raw-action error is zero by construction. Physical divergence with zero
target error is evidence of dynamics/contact mismatch rather than network inference.

## 6. Run the one-step dynamics comparison

```bash
python -m sim2sim.tools.compare_one_step_dynamics \
    --isaac-rollout sim2sim/results/gait/phase_0/isaac \
    --sample-count 8 \
    --velocity-limit-mode hard \
    --output-dir sim2sim/results/gait/phase_0/one_step
```

For eight gait phases, the tool resets MuJoCo to the recorded Isaac pre-action base/joint state, applies the recorded
target for four physics steps, and compares against Isaac's recorded next state. Outputs are:

- `report.md`
- `one_step_dynamics.json`
- `one_step_dynamics.csv`
- `one_step_dynamics.png`

Errors during low/no-contact swing phases point more strongly toward actuator dynamics. Errors concentrated near
touchdown and stance point more strongly toward contact, friction, or compliance. One-step state transfer reconstructs
contacts from geometry and does not transfer solver warm-start caches, so interpret contact-phase errors as a combined
local model discrepancy.

## 7. Evaluate several initial gait phases

Use the same phase offsets in both simulators:

| Name | Offset |
|---|---:|
| `phase_0` | `0` |
| `phase_1` | `0.7853981634` |
| `phase_2` | `1.5707963268` |
| `phase_3` | `2.3561944902` |
| `phase_4` | `3.1415926536` |
| `phase_5` | `3.9269908170` |
| `phase_6` | `4.7123889804` |
| `phase_7` | `5.4977871438` |

Change both `--phase-offset` and output directory for each capture. Tune with a defined validation subset, then use
the remaining phases as the final held-out check. Compare fall rate, worst phase, median forward tracking, yaw,
lateral drift, contact agreement, marker motion, and saturation instead of selecting one visually favorable run.

## Current five-second validation result

The checked workspace example used phase `0`, MuJoCo sliding friction `0.4`, and hard velocity limiting.

- Initial 70-value observation error: `0`.
- Initial raw-action error: `0`.
- Initial target and joint-state error: `0`.
- Initial foot-marker position error: about `2.1e-7 m`.
- Joint-position RMSE after the first 33 ms interval: about `0.0678 rad`.
- Closed-loop raw actions first exceeded RMSE `0.1` at about `0.0667 s`.
- Closed-loop base error first exceeded `2 cm` at about `0.133 s`.
- Five-second closed-loop contact agreement was about `43.8%`.
- Open-loop replay retained raw-action RMSE `0` but still diverged physically.
- Across eight one-step samples, mean joint-position RMSE was about `0.0660 rad` and mean joint-velocity RMSE was
  about `1.78 rad/s`.

This result verifies the rollout interface and shows that the earliest remaining disagreement is loaded physical
response. It does not prove friction is the only cause. The next tuning order is loaded actuator response, contact
timing/friction, damping/armature, and velocity-limit behavior, changing one family at a time.

## Suggested pass gates

Treat these as initial engineering gates and refine them using measured hardware behavior:

1. Initial observation, action, target, and state errors below numerical tolerance.
2. No falls for every held-out phase in either simulator.
3. No immediate one-step joint or base error caused by an interface mismatch.
4. Similar contact duty factors and high per-foot contact agreement.
5. Similar stride-level joint and foot phase profiles.
6. Comparable forward tracking with low lateral velocity and yaw.
7. Saturation low enough that the controller is not dominated by action, effort, or speed limits.

Do not require long-horizon pointwise trajectories to remain identical. Contact-rich closed-loop systems amplify small
differences. Compare early local errors and long-horizon stride statistics together.
