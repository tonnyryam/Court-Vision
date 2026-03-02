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

echo "Node: $(hostname)"
nvidia-smi

# If you use conda, activate it here (optional)
# source /opt/anaconda3/2023.07-2/etc/profile.d/conda.sh
# /opt/anaconda3/2023.07-2/condabin/conda activate rfdetr

python -c "import torch; print(torch.__version__); print('CUDA?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

# ---- DATASET COPY TO LOCAL SCRATCH ----
DATA_SRC="test.v1-test.coco"              # <-- CHANGE THIS
DATA_DST="${SLURM_TMPDIR}/dataset"

echo "Copying dataset to $DATA_DST"
cp -r "$DATA_SRC" "$DATA_DST"

export DATASET_DIR="$DATA_DST"

# Train
python train_rfdetr.py

# Copy results back
OUT_SRC="runs/rfdetr_waterpolo"
OUT_DST="$PWD/runs/rfdetr_waterpolo_${SLURM_JOB_ID}"
mkdir -p "$OUT_DST"
cp -r "$OUT_SRC"/* "$OUT_DST"/

echo "Saved outputs to $OUT_DST"