# Standalone MuJoCo transfer

This directory runs the exported 70-observation/18-action RSL-RL policy without importing Isaac Lab. The authoritative
setup, validation, viewer, headless evaluation, and metrics instructions are in the repository's main
[`README.md`](../README.md#sim-to-sim-transfer-with-mujoco).

The checked-in contract is [`configs/policy_interface.yaml`](configs/policy_interface.yaml). Keep its observation
layout, joint names, control rate, action scale, and clipping identical to the Isaac task that produced the policy.
Use `scripts/dump_policy_interface.py` on the training machine to verify the runtime PhysX joint order.

