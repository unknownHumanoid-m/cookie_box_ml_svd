#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=phase_single_train
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# ***** Edit these to point at the data / models you want *****
# The values below are passed to train_phase_single.py as CLI flags.
# ----------------------------------------------------------------------------

# ':'-separated list of h5 files (or directories) to train on.
# Default: autoencoder-denoised training set (Ypdf_denoised alongside Ximg/Ypdf).
export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/train"

# Which per-shot h5 dataset to feed the regressor. Should match the how_many
# INPUT_KEY so the classifier gate at eval-time sees the same data.
export INPUT_KEY="Ypdf_denoised"

# Deep-MLP output bins (softmax-weighted decoder over [0, 2*pi]).
export NUM_CLASSES=2000

# Training hyperparameters.
export BATCH_SIZE=32
export LR=0.001
export EPOCHS=50
export PATIENCE=5
export VAL_FRAC=0.2
export NUM_WORKERS=4

# Where to save the .pth.
export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/phase_single_svd1d_r8/"
export MODEL_IDENTIFIER="phase_single_svd1d_r8"

# ----------------------------------------------------------------------------

echo starting run 1 at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions_1D_SVD

echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
echo "INPUT_KEY=$INPUT_KEY  NUM_CLASSES=$NUM_CLASSES"
echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR  MODEL_IDENTIFIER=$MODEL_IDENTIFIER"
echo "BATCH_SIZE=$BATCH_SIZE  LR=$LR  EPOCHS=$EPOCHS  PATIENCE=$PATIENCE"

mkdir -p "${MODEL_SAVE_DIR}"

CMD=(python3 train_phase_single.py
    --data_dirs "${TRAIN_DATA_DIRS}"
    --input_key "${INPUT_KEY}"
    --num_classes "${NUM_CLASSES}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --val_frac "${VAL_FRAC}"
    --num_workers "${NUM_WORKERS}"
    --save_dir "${MODEL_SAVE_DIR}"
    --save_model "${MODEL_IDENTIFIER}.pth"
)
if [ -n "${FIGURES_DIR:-}" ]; then
    CMD+=(--figures_dir "${FIGURES_DIR}")
fi

cd "$REGRESSION_DIR" && "${CMD[@]}"

echo Finished at: `date`
