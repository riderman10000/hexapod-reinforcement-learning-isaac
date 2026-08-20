# Isaac–MuJoCo sim-to-sim diagnostics guide

This document explains the sim-to-sim tests in this repository: what each test checks, why it exists, how to run it,
how to read its outputs, and what the current results do and do not prove. It is written for someone who did not
participate in the original debugging.

The example policy used below is:

```text
logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx
```

Run every command from the repository root. The names `env_232` and `hexapod_mujoco` are the environments used on the
development machine; substitute your own environment names when necessary.

## Current result at a glance

| Question | Current answer | Meaning |
|---|---|---|
| Does the checkpoint joint order match the recorded contract? | Yes | Isaac's grouped order is mapped to MuJoCo by joint name. |
| Do positive joint commands move the legs in the same direction? | Yes, `18/18` axes passed | No reversed joint axis was found. |
| Do the unloaded joint-step responses match? | Nearly exactly | The basic small-signal position response is similar. |
| Do both simulators build the same neutral observation? | Yes, zero error | The neutral 70-value interface is correct. |
| Does the same observation produce the same network output? | Yes, zero error | ONNX inference and output ordering agree. |
| Is the policy output well behaved? | No | About 79% of action components saturate during the tested rollout. |
| Should the hard speed clip be removed now? | No | Soft is calmer but too slow; the `none` mode overspeeds. |

These results verify the policy interface and kinematic direction. They do **not** prove that contact, loaded actuator
dynamics, mass/inertia, or friction match during locomotion.

## Test 0: policy contract and joint-order validation

### What it is

An exported ONNX policy contains tensor shapes, but it does not contain the semantic joint names associated with each
output. The checkpoint therefore depends on the exact joint order used during training.

The current Isaac policy order is:

| Output indices | Isaac policy joints |
|---|---|
| `0:6` | `body_leg_0` through `body_leg_5` |
| `6:12` | `leg_0_1_2` through `leg_5_1_2` |
| `12:18` | `leg_0_2_3` through `leg_5_2_3` |

MuJoCo stores its joints leg-by-leg. That difference is acceptable because the runner converts between the two orders
by joint name. The authoritative checked-in policy order is
[`configs/policy_interface.yaml`](configs/policy_interface.yaml).

### Why we run it

A valid network connected to the wrong joint order can move the wrong legs while producing no software exception.
This was an actual source of poor behavior in the original transfer.

### How to run it

In the Isaac environment:

```bash
conda activate env_232
source /home/rlwagun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh

python scripts/dump_policy_interface.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --headless --num_envs=1
```

In the MuJoCo environment, validate the complete contract without stepping physics:

```bash
conda activate hexapod_mujoco

python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --validate-only
```

### What to look for

- ONNX input shape: `[1, 70]`.
- ONNX output shape: `[1, 18]`.
- Isaac runtime names equal `policy_joint_names` in the YAML.
- MuJoCo contains each name exactly once.
- `physics_dt=1/120 s`, decimation `4`, and policy rate `30 Hz`.

Different printed Isaac and MuJoCo storage orders are not themselves an error. Missing, duplicated, or incorrectly
mapped names are errors.

## Test 1: repeatable closed-loop MuJoCo rollout

### What it is

This is the integrated smoke test. It runs the complete loop—state reading, observation construction, ONNX inference,
action conversion, physics stepping, termination checks, and metric recording—for a fixed phase seed.

### Why we run it

The smaller diagnostics below isolate individual mechanisms. This rollout answers the higher-level question: does the
complete transferred controller remain upright and make useful forward progress?

### How to run it

```bash
conda activate hexapod_mujoco

python -m sim2sim.hexapod_mujoco.run \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --headless --duration 20 --random-phase --seed 42 --plot
```

Each run writes a timestamped directory under `sim2sim/results/`:

- `rollout.csv`: one row per 30 Hz policy step.
- `summary.json`: displacement, velocity, yaw, tilt, saturation, and fall metrics.
- `rollout.png`: time-series plots when `--plot` is used.

### What to look for

| Metric | Desired behavior |
|---|---|
| `fell` | `false` |
| Mean world-forward velocity | Near the `0.5 m/s` command |
| Mean absolute lateral velocity | Near zero |
| Mean absolute yaw rate | Near zero |
| Forward path efficiency | Near `1.0` |
| Action and force saturation | Low and not persistent |
| Maximum tilt | Safely below the termination threshold |

Use the same seed when comparing code changes. Later, repeat several seeds so a favorable initial gait phase does not
hide a regression.

## Tests 2 and 3: create matched Isaac and MuJoCo captures

The joint-axis and neutral policy-interface checks share one capture workflow.

### Capture Isaac

```bash
conda activate env_232
source /home/rlwagun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh

python scripts/capture_isaac_diagnostics.py \
    --device cuda:0 \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/isaac \
    --headless
```

Success is indicated by the final `Isaac diagnostics: ...` line. Isaac Sim 5.1 may print a non-fatal IOMMU/P2P
message on a multi-GPU workstation. Use `--device cuda:N` to select a different physical GPU; do not remap it with
`CUDA_VISIBLE_DEVICES` because CUDA and Vulkan may then disagree about device numbering.

### Capture MuJoCo

```bash
conda activate hexapod_mujoco

python -m sim2sim.tools.capture_mujoco_diagnostics \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/mujoco
```

### Generate the comparison

```bash
python -m sim2sim.tools.compare_diagnostics \
    --isaac-dir sim2sim/results/diagnostics/isaac \
    --mujoco-dir sim2sim/results/diagnostics/mujoco \
    --output-dir sim2sim/results/diagnostics/comparison
```

The main human-readable output is
[`results/diagnostics/comparison/report.md`](results/diagnostics/comparison/report.md).

## Test 2: `+0.05 rad` joint-axis and actuator-response sweep

### What it is

Starting from a neutral pose with robot gravity disabled, the test commands one joint at a time by `+0.05 rad`
(`2.86 degrees`) for 0.5 seconds. It records joint position, velocity, effort, and the downstream foot displacement in
the base frame. All other joints remain at their neutral targets.

### Why we run it

Matching names are not enough. A joint can have the correct name but a reversed axis, or a nominally correct actuator
can respond at a very different rate. Either error scrambles a locomotion policy.

### How to interpret foot-direction cosine

For one joint, let `d_isaac` and `d_mujoco` be the three-dimensional foot displacement vectors. The comparison uses:

```text
cosine = dot(d_isaac, d_mujoco) / (length(d_isaac) * length(d_mujoco))
```

| Cosine | Interpretation |
|---:|---|
| `+1.0` | Both feet move in the same physical direction. |
| Near `0.0` | Motions are approximately perpendicular or one displacement is poorly defined. |
| `-1.0` | Motions are opposite; a reversed axis/sign is likely. |
| `0.8` to `1.0` | Passing direction agreement for this test. |

The test also requires both final joint-position changes to be positive. A high cosine alone cannot pass a joint that
responds with the wrong coordinate sign.

### Figure

![Isaac and MuJoCo single-joint step responses](results/diagnostics/comparison/joint_sweep_comparison.png)

For every panel:

- Blue is Isaac/PhysX.
- Orange is MuJoCo.
- The dashed line is the requested `0.05 rad` change.
- Overlapping curves indicate similar unloaded position response.
- `foot cosine=1.00` indicates matching downstream-foot direction.

### Current result

- `18/18` axes passed.
- Final motion was approximately `+0.05 rad` in both simulators.
- Every foot-direction cosine was effectively `1.0`.
- The step-response curves are very close.

This rules out reversed axes and basic unloaded response as the primary remaining problem. It does not verify behavior
while supporting body weight, at high speed, at the effort limit, or during foot contact.

## Test 3: neutral observation and raw-policy-output parity

### What it is

Both simulators are placed in the same deterministic neutral state. Each one constructs the complete 70-value policy
observation and runs the same ONNX actor once before action clipping.

### Why we run it

A policy can receive the correct tensor shape but incorrect values because of a changed field order, coordinate frame,
unit, joint permutation, gait phase, or previous-action convention. Comparing named terms locates the first mismatch.

### The 70 network inputs

| Indices | Size | Term |
|---|---:|---|
| `0:3` | 3 | Base linear velocity in body coordinates |
| `3:6` | 3 | Base angular velocity in body coordinates |
| `6:9` | 3 | Gravity direction projected into the body frame |
| `9:12` | 3 | World command: forward, lateral, yaw rate |
| `12:14` | 2 | Desired world heading expressed in the body frame |
| `14:16` | 2 | Gait phase: sine and cosine |
| `16:34` | 18 | Joint position minus neutral position |
| `34:52` | 18 | Joint velocity |
| `52:70` | 18 | Previously applied clipped action |

At the tested neutral state, the meaningful non-zero terms are gravity `[0, 0, -1]`, command `[0.5, 0, 0]`, desired
heading `[1, 0]`, and gait phase `[0, 1]`. Most other values should be zero.

### How raw outputs become joint targets

The network output is not mathematically restricted to `[-1, 1]`. The controller applies the trained contract:

```python
clipped_action = clip(raw_network_output, -1.0, 1.0)
target = neutral_joint_position + 0.3 * clipped_action
target = clip(target, lower_joint_limit, upper_joint_limit)
```

A raw output of `5.27` therefore becomes `1.0`; it does not command `5.27 rad`. Raw values outside the clip range are
valid numerically, but frequent clipping removes fine control because `1.1`, `5`, and `20` all produce the same action.

Two saturation metrics must not be confused:

- `action_saturation_fraction`: fraction of policy steps where **at least one** of 18 actions was clipped.
- `action_component_saturation_fraction`: fraction of all individual action values that were clipped.

### Figure

![Neutral observation and raw-action comparison](results/diagnostics/comparison/observation_action_comparison.png)

The observation panel should show zero error for every named term. The action panel should show matching Isaac and
MuJoCo bars; bars beyond `+1` or `-1` indicate clipping, not an inference mismatch.

### Current result

- Observation RMSE: `0.0`.
- Maximum observation error: `0.0`.
- Raw-action RMSE: `0.0`.
- Maximum raw-action error: `0.0`.
- In the neutral snapshot, 16 of 18 raw outputs exceed `[-1, 1]`.
- During the hard-mode rollout, about 79.2% of individual action components saturated.

This verifies the neutral interface and ONNX inference. It also reveals an overly saturated policy. It does not yet
test moving-state observations, where velocity-frame signs, tilted projected gravity, and phase synchronization become
observable.

Do not simply increase `action_clip`, `action_scale`, or joint limits to accommodate the raw outputs. The policy was
trained with the existing transformation, and changing it changes the controller semantics and can cause leg
collisions. Such a change requires a new safety review and normally retraining.

## Test 4: velocity-limit A/B comparison

### What it is

This runs the same policy, initial gait phase, model, and 20-second duration under three MuJoCo velocity treatments:

- `hard`: overwrite joint velocity after every physics step so it cannot exceed `5.236 rad/s`.
- `soft`: apply a continuous braking torque above 80% of the speed limit without overwriting simulator state.
- `none`: use only the effort-limited position actuators, with no additional speed treatment.

### Why we run it

The hard state overwrite is numerically effective but non-physical. The A/B test measures whether a smoother model can
reduce chatter, yaw, lateral movement, and force saturation without causing falls, overspeed, or loss of forward motion.

### How to run it

```bash
conda activate hexapod_mujoco

python -m sim2sim.tools.compare_velocity_modes \
    --policy logs/rsl_rl/hexapod_direct/2026-08-18_17-07-24_straight_line_v2/exported/policy.onnx \
    --output-dir sim2sim/results/diagnostics/velocity_ab \
    --duration 20 \
    --random-phase --seed 42
```

### What to look for

| Metric | Better direction | Why |
|---|---|---|
| Mean world-forward velocity | Higher, toward `0.5 m/s` | Measures command tracking. |
| Mean absolute lateral velocity | Lower, toward zero | Measures sideways drift. |
| Mean absolute yaw rate | Lower, toward zero | Measures unwanted turning. |
| Maximum speed/configured limit | At or below `1.0` | Values above one exceed the intended speed. |
| Action sign-change fraction | Lower, if motion remains responsive | A proxy for command chatter. |
| Force-saturation fraction | Lower | Persistent maximum effort suggests actuator mismatch. |
| `fell` | `false` | Required for any acceptable candidate. |

Higher is not universally better. Higher forward velocity is useful; higher lateral velocity, yaw, overspeed, or force
saturation is undesirable.

### Figure

![Hard, soft, and unrestricted velocity comparison](results/diagnostics/velocity_ab/velocity_mode_comparison.png)

### Current single-seed result

| Mode | Forward velocity | Absolute lateral velocity | Absolute yaw rate | Max speed/limit | Force-component saturation |
|---|---:|---:|---:|---:|---:|
| Hard | `0.227 m/s` | `0.278 m/s` | `1.426 rad/s` | `1.000` | `30.5%` |
| Soft | `0.092 m/s` | `0.228 m/s` | `1.059 rad/s` | `1.052` | `26.0%` |
| None | `0.184 m/s` | `0.286 m/s` | `1.423 rad/s` | `1.520` | `33.8%` |

All three completed without falling. Hard produced the best forward motion and enforced the limit, but it had large
drift/yaw and frequent limiting. Soft reduced drift, yaw, chatter, force saturation, and tilt, but removed too much
forward speed. None exceeded the configured joint-speed limit by 52% and had the worst force saturation.

The hard mode therefore remains the current baseline. The soft implementation is an experiment, not a calibrated
motor model and not the default. One seed is insufficient for a permanent choice; repeat several fixed seeds and
aggregate the results first.

## What has been completed

| Recommended debugging item | Status |
|---|---|
| Verify runtime Isaac joint order | Completed and corrected |
| Run `+0.05 rad` single-joint sweeps in both simulators | Completed; `18/18` passed |
| Compare 70 observations and 18 raw network outputs at neutral | Completed; exact match |
| Implement and test a smoother alternative to hard clipping | Experimental soft mode implemented and tested |
| Permanently replace hard clipping | Not done; current evidence does not support it |

## What remains to be tested

The remaining transfer gap is most likely dynamic rather than an interface or axis error. The next work should be:

1. **Physical asset audit:** compare resolved USD, URDF, and MJCF mass, inertia, joint ranges, collision geometry, and
   contact friction.
2. **Loaded actuator tests:** repeat controlled movements while the legs support body weight; compare position,
   velocity, effort, settling time, and overshoot.
3. **Contact tests:** compare standing foot-force distribution, vertical drop/bounce, and commanded foot slip.
4. **Nominal model identification:** tune physically defensible effort limits, stiffness, damping, armature, friction,
   and actuator delay. Monitor action sign changes and force saturation.
5. **Multi-seed evaluation:** repeat nominal and velocity A/B rollouts over several fixed gait phases.
6. **Retraining:** randomize plausible ranges of gains, armature, mass/inertia, friction, actuator response, latency,
   and observation noise. Address policy saturation with appropriate action regularization or a bounded actor design.

Do not tune the target simulator solely until one trajectory looks like Isaac. Establish a measured nominal model,
then randomize around physically credible ranges.

## How to keep project notes

This guide should be the reproducible reference for stable commands, interface facts, acceptance criteria, and verified
results. A separate personal project notebook is still valuable because it preserves reasoning that does not belong in
a permanent guide.

For each experiment, record:

```text
Date and run/checkpoint ID:
Question:
Hypothesis:
Single change being tested:
Fixed parameters and seed:
Exact command:
Output directory:
Measured result:
Interpretation:
Decision: keep / reject / repeat
Next action:
Status: idea / in progress / verified / disproved
```

Keep facts, hypotheses, and actions visibly separate. Link to the YAML, JSON, CSV, plot, or commit instead of copying
configuration lists that can become stale. In particular, older notes that grouped policy outputs leg-by-leg are no
longer authoritative; the verified checkpoint order groups all hips, then all thighs, then all knees.
