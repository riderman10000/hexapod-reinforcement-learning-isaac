#!/bin/bash 

python scripts/rsl_rl/train.py --task=Template-Hexpod-Rl-Lab-Direct-v0 --resume --load_run 2026-07-17_14-13-47 --checkpoint model_0.pt --run_name hexapod_test1

