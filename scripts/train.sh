#!/bin/bash

# Zero-at-rest progress rewards and alternating-tripod phase observations
# change the task semantics, so this must start from a fresh policy.
python scripts/rsl_rl/train.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --headless \
    --num_envs=64 \
    --max_iterations=5000 \
    --run_name straight_line_v2
