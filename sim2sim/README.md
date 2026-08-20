# Standalone MuJoCo transfer

This directory runs the exported 70-observation/18-action RSL-RL policy without importing Isaac Lab. The authoritative
setup, validation, viewer, headless evaluation, and metrics instructions are in the repository's main
[`README.md`](../README.md#sim-to-sim-transfer-with-mujoco).

The checked-in contract is [`configs/policy_interface.yaml`](configs/policy_interface.yaml). Keep its observation
layout, joint names, control rate, action scale, and clipping identical to the Isaac task that produced the policy.
The current joint order was verified with `scripts/dump_policy_interface.py`: Isaac exposes the six hips, six thighs,
and six knees in groups, and the runner maps that order to MuJoCo's leg-by-leg storage by joint name.

The beginner-readable [`DIAGNOSTICS_GUIDE.md`](DIAGNOSTICS_GUIDE.md) documents every completed test, exact commands,
generated figures, interpretation, verified conclusions, limitations, and remaining work. The main README retains the
shorter operational reference.
