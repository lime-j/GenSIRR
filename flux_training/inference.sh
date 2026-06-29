#!/bin/bash
set -e

# Example:
#   ./inference.sh /path/to/last.ckpt /path/to/input_images inference_results 768
#
# Edit options/gensirr.yml first so network_g.model_path and network_g.vae_path
# point to your local model assets.

CHECKPOINT=${1:?Usage: ./inference.sh CHECKPOINT INPUT [OUTPUT] [SHORT_SIDE]}
INPUT=${2:?Usage: ./inference.sh CHECKPOINT INPUT [OUTPUT] [SHORT_SIDE]}
OUTPUT=${3:-inference_results}
SHORT_SIDE=${4:-768}

python3 xreflection/tools/infer_flux.py \
  --config options/gensirr.yml \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --steps 28 \
  --short-side "$SHORT_SIDE"
