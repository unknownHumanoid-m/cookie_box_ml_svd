#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=phase_single_eval_1d
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-02:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# Eval-only: runs the already-trained 1D-SVD single-pulse regressor twice —
# once against denoised data (Ypdf_denoised, denoised-trained ckpts) and
# once against noisy data (Ximg, noisy-trained ckpts). No retraining.
# ----------------------------------------------------------------------------

export EVAL_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/test"

# Denoised branch.
export DENOISED_MODEL_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/phase_single_svd1d_r8/phase_single_svd1d_r8.pth"
export DENOISED_HOW_MANY_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/how_many_denoised_svd1d_r8/how_many_denoised_svd1d_r8.pth"
export DENOISED_EVAL_IDENTIFIER="phase_single_svd1d_r8_eval_denoised"

# Noisy branch.
export NOISY_MODEL_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/phase_single_noisy_svd1d_r8/phase_single_noisy_svd1d_r8.pth"
export NOISY_HOW_MANY_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/how_many_noisy_svd1d_r8/how_many_noisy_svd1d_r8.pth"
export NOISY_EVAL_IDENTIFIER="phase_single_svd1d_r8_eval_noisy"

export BATCH_SIZE=64
export NUM_WORKERS=4

# ----------------------------------------------------------------------------

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions_1D_SVD
cd "$REGRESSION_DIR" || exit 1

run_eval () {
    local model_path=$1
    local how_many_path=$2
    local identifier=$3
    local branch=$4

    echo "============================================================"
    echo "[$branch] EVAL_DATA_DIRS=$EVAL_DATA_DIRS"
    echo "[$branch] MODEL_PATH=$model_path"
    echo "[$branch] HOW_MANY_PATH=$how_many_path"
    echo "[$branch] IDENTIFIER=$identifier"
    echo "============================================================"

    local cmd=(python3 eval_phase_single.py
        --data_dirs "${EVAL_DATA_DIRS}"
        --model_path "${model_path}"
        --how_many_path "${how_many_path}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --identifier "${identifier}"
    )
    if [ -n "${FIGURES_DIR:-}" ]; then cmd+=(--figures_dir "${FIGURES_DIR}"); fi
    "${cmd[@]}"
}

run_eval "$DENOISED_MODEL_PATH" "$DENOISED_HOW_MANY_PATH" "$DENOISED_EVAL_IDENTIFIER" "denoised"
run_eval "$NOISY_MODEL_PATH"    "$NOISY_HOW_MANY_PATH"    "$NOISY_EVAL_IDENTIFIER"    "noisy"

echo Finished at: `date`
