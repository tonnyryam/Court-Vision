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

# Silence albumentations update check warning (optional)
export NO_ALBUMENTATIONS_UPDATE=1

# ---- (OPTIONAL) Activate conda env ----
# If you already have torch + rfdetr installed in the env SLURM uses, you can leave this commented.
# Otherwise, uncomment and set your env name.
#
# source /opt/anaconda3/2023.07-2/etc/profile.d/conda.sh
# /opt/anaconda3/2023.07-2/condabin/conda activate test   # <-- change env name if needed

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

# rsync is safer/faster for big datasets than cp -r
rsync -a --info=progress2 "$DATA_SRC"/ "$DATA_DST"/

export DATASET_DIR="$DATA_DST"

echo "Dataset staged. Top-level:"
ls -la "$DATASET_DIR" | head -n 80

# ---- DDP env vars for single-GPU (fixes MASTER_ADDR error) ----
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# ---- NCCL safety (prevents some HPC hangs) ----
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "DDP env:"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"

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
  echo "WARNING: $OUT_SRC not found. Check model.export() path in train_rfdetr.py"
fi

# ---- CLEANUP ----
echo "Cleaning up scratch: $JOB_SCRATCH"
rm -rf "$JOB_SCRATCH"

echo "Done."