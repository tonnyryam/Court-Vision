#!/bin/bash
#SBATCH --job-name=rfdetr-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

echo "============================="
echo "Job: ${SLURM_JOB_NAME}"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
date
echo "============================="

# ---- (OPTIONAL) Activate conda env ----
# If you already have torch working in your current env, keep using it.
# If you want to use conda, uncomment and set your env name.
#
# source /opt/anaconda3/2023.07-2/etc/profile.d/conda.sh
# /opt/anaconda3/2023.07-2/condabin/conda activate rfdetr

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__); print('CUDA?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "---- nvidia-smi (before) ----"
nvidia-smi || true

# ---- DATASET COPY TO LOCAL SCRATCH ----
DATA_SRC="test.v1-test.coco"   # <-- CHANGE THIS (your dataset root)

SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_SCRATCH="${SCRATCH_BASE}/${USER}_rfdetr_${SLURM_JOB_ID}"
DATA_DST="${JOB_SCRATCH}/dataset"

echo "SLURM_TMPDIR=${SLURM_TMPDIR-<unset>}"
echo "Using scratch: $JOB_SCRATCH"
echo "Copying dataset from: $DATA_SRC"
echo "Copying dataset to:   $DATA_DST"

mkdir -p "$JOB_SCRATCH"

# Faster + safer than cp for large dirs (keeps permissions, shows progress-ish)
rsync -a --info=progress2 "$DATA_SRC"/ "$DATA_DST"/

export DATASET_DIR="$DATA_DST"

echo "Dataset staged. Listing top-level:"
ls -la "$DATASET_DIR" | head -n 50

# ---- TRAIN ----
# Assumes train_rfdetr.py is in the directory you submit from
echo "Starting training..."
date
python train_rfdetr.py
date
echo "Training finished."

echo "---- nvidia-smi (after) ----"
nvidia-smi || true

# ---- COPY RESULTS BACK ----
OUT_SRC="runs/rfdetr_waterpolo"
OUT_DST="$PWD/runs/rfdetr_waterpolo_${SLURM_JOB_ID}"

echo "Copying outputs from: $OUT_SRC"
echo "Copying outputs to:   $OUT_DST"

mkdir -p "$OUT_DST"
if [ -d "$OUT_SRC" ]; then
  rsync -a --info=progress2 "$OUT_SRC"/ "$OUT_DST"/
else
  echo "WARNING: $OUT_SRC not found. Check training script export path."
fi

# ---- CLEANUP ----
echo "Cleaning up scratch: $JOB_SCRATCH"
rm -rf "$JOB_SCRATCH"

echo "Done."