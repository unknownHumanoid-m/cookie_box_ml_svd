#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=phase_two_train
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=1-12:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# Retrain the 2D two-pulse phase-diff regressor over every (rank, branch) pair
# that still holds pre-Option-A ckpts:
#     r=2 denoised, r=2 noisy, r=4 denoised, r=4 noisy, r=8 noisy
# r=8 denoised was already retrained at 14:54 today and is skipped.
# ----------------------------------------------------------------------------

BASE_DATA_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data"
BASE_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"

export NUM_CLASSES=1000
export BATCH_SIZE=32
export LR=0.001
export EPOCHS=50
export PATIENCE=5
export VAL_FRAC=0.2
export NUM_WORKERS=4

# rank | branch | input_key
CONFIGS=(
    "2|denoised|Ypdf_denoised"
    "2|noisy|Ximg"
    "4|denoised|Ypdf_denoised"
    "4|noisy|Ximg"
    "8|noisy|Ximg"
)

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions
cd "$REGRESSION_DIR" || exit 1

run_one () {
    local rank=$1
    local branch=$2
    local input_key=$3

    local train_dir="${BASE_DATA_DIR}/svd2d_r${rank}_autoencoder/train"

    local identifier
    if [ "$branch" = "denoised" ]; then
        identifier="phase_two_svd2d_r${rank}"
    else
        identifier="phase_two_noisy_svd2d_r${rank}"
    fi
    local save_dir="${BASE_SAVE_DIR}/${identifier}/"

    echo "============================================================"
    echo "[$identifier] TRAIN_DATA_DIRS=$train_dir"
    echo "[$identifier] INPUT_KEY=$input_key  NUM_CLASSES=$NUM_CLASSES"
    echo "[$identifier] MODEL_SAVE_DIR=$save_dir"
    echo "============================================================"

    mkdir -p "$save_dir"

    local cmd=(python3 train_phase_two.py
        --data_dirs "${train_dir}"
        --input_key "${input_key}"
        --num_classes "${NUM_CLASSES}"
        --batch_size "${BATCH_SIZE}"
        --lr "${LR}"
        --epochs "${EPOCHS}"
        --patience "${PATIENCE}"
        --val_frac "${VAL_FRAC}"
        --num_workers "${NUM_WORKERS}"
        --save_dir "${save_dir}"
        --save_model "${identifier}.pth"
    )
    if [ -n "${FIGURES_DIR:-}" ]; then cmd+=(--figures_dir "${FIGURES_DIR}"); fi
    "${cmd[@]}"
}

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r rank branch input_key <<< "$entry"
    run_one "$rank" "$branch" "$input_key"
done

echo Finished at: `date`
