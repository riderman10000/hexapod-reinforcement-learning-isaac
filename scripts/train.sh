#!/bin/bash

# The corrected task has new observation and action semantics, so it must start
# from a fresh policy rather than resuming an effort-control checkpoint.
python scripts/rsl_rl/train.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --run_name corrected_locomotion
