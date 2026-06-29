#!/bin/bash
set -e

# Edit options/gensirr.yml before running:
# - network_g.model_path
# - network_g.vae_path
# - all dataset datadir/fns fields
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

python3 xreflection/tools/train.py --config options/gensirr.yml "$@"
