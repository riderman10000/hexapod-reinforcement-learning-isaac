#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-corrected-model.pt>"
    exit 2
fi

python scripts/rsl_rl/play.py \
    --task=Template-Hexpod-Rl-Lab-Direct-v0 \
    --resume \
    --checkpoint "$1"
