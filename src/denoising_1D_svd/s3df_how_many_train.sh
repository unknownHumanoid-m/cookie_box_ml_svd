#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=how_many_train
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml


# ----------------------------------------------------------------------------
# ***** Edit these to point at the data / models you want *****
# The values below are passed to train_how_many.py as CLI flags.
# ----------------------------------------------------------------------------

# ':'-separated list of h5 files (or directories) to train on.
# Default: autoencoder-denoised training set (mirrors 2d_svd_train, adds
# Ypdf_denoised alongside Ximg / Ypdf).
# Passing the directory pulls in every .h5 file inside (load_how_many_h5 expands it).
export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/train"

# Which per-shot h5 dataset to feed the classifier.
#   Ypdf          -> clean truth (upper-bound run)
#   Ximg          -> raw SVD input (lower-bound run)
#   Ypdf_denoised -> autoencoder output (default here; realistic end-to-end)
export INPUT_KEY="Ypdf_denoised"

# Pulse-count range the classifier learns over. Labels are remapped to
# 0..(MAX_PULSES - MIN_PULSES) for CrossEntropyLoss.
export MIN_PULSES=1
export MAX_PULSES=4

# Training hyperparameters.
export BATCH_SIZE=128
export LR=0.001
export EPOCHS=50
export PATIENCE=5
export VAL_FRAC=0.2
export NUM_WORKERS=4

# Where to save the .pth (identifier + directory).
export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/how_many_denoised_svd1d_r8/"
export MODEL_IDENTIFIER="how_many_denoised_svd1d_r8"

# Optional override for figure output; defaults to <repo>/src/denoising/figures/.
# export FIGURES_DIR="/some/other/figures/"

# ----------------------------------------------------------------------------
# End of user config.
# ----------------------------------------------------------------------------

echo starting run 1 at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

COOKIE_DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising_1D_svd

echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
echo "INPUT_KEY=$INPUT_KEY"
echo "MIN_PULSES=$MIN_PULSES  MAX_PULSES=$MAX_PULSES"
echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "MODEL_IDENTIFIER=$MODEL_IDENTIFIER"
echo "BATCH_SIZE=$BATCH_SIZE  LR=$LR  EPOCHS=$EPOCHS  PATIENCE=$PATIENCE"

mkdir -p "${MODEL_SAVE_DIR}"

CMD=(python3 train_how_many.py
    --data_dirs "${TRAIN_DATA_DIRS}"
    --input_key "${INPUT_KEY}"
    --min_pulses "${MIN_PULSES}"
    --max_pulses "${MAX_PULSES}"
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

cd "$COOKIE_DENOISING_DIR" && "${CMD[@]}"

echo Finished at: `date`
