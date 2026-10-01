#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
torchrun --standalone --nnodes=1 --nproc_per_node="${PIPER_GPUS:-4}" \
  -m starVLA.training.train_piper --config scripts/configs/vlajepa_piper_ft.yaml "$@"
