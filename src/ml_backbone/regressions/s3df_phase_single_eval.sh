#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=phase_single_eval
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-04:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# Eval-only: runs the already-trained 2D-SVD single-pulse regressor over every
# (rank, branch) pair — rank in {2, 4, 8}, branch in {denoised, noisy}.
# Each checkpoint's stored input_key drives which per-shot dataset is read
# (Ypdf_denoised for the denoised branch, Ximg for the noisy branch).
# Identifiers match the layout written by s3df_phase_single_train.sh /
# s3df_how_many_train.sh:
#     phase_single_svd2d_r<R>            (denoised)
#     phase_single_noisy_svd2d_r<R>      (noisy)
#     how_many_denoised_svd2d_r<R>       (denoised how_many gate)
#     how_many_noisy_svd2d_r<R>          (noisy how_many gate)
# ----------------------------------------------------------------------------

BASE_DATA_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data"
BASE_RUNS_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"

export BATCH_SIZE=64
export NUM_WORKERS=4

RANKS=(2 4 8)
BRANCHES=(denoised noisy)

# ----------------------------------------------------------------------------

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions
cd "$REGRESSION_DIR" || exit 1

run_eval () {
    local rank=$1
    local branch=$2

    local eval_data_dirs="${BASE_DATA_DIR}/svd2d_r${rank}_autoencoder/test"

    local phase_identifier how_many_identifier
    if [ "$branch" = "denoised" ]; then
        phase_identifier="phase_single_svd2d_r${rank}"
        how_many_identifier="how_many_denoised_svd2d_r${rank}"
    else
        phase_identifier="phase_single_noisy_svd2d_r${rank}"
        how_many_identifier="how_many_noisy_svd2d_r${rank}"
    fi

    local model_path="${BASE_RUNS_DIR}/${phase_identifier}/${phase_identifier}.pth"
    local how_many_path="${BASE_RUNS_DIR}/${how_many_identifier}/${how_many_identifier}.pth"
    local eval_identifier="${phase_identifier}_eval_${branch}"

    echo "============================================================"
    echo "[r${rank}/${branch}] EVAL_DATA_DIRS=${eval_data_dirs}"
    echo "[r${rank}/${branch}] MODEL_PATH=${model_path}"
    echo "[r${rank}/${branch}] HOW_MANY_PATH=${how_many_path}"
    echo "[r${rank}/${branch}] IDENTIFIER=${eval_identifier}"
    echo "============================================================"

    local cmd=(python3 eval_phase_single.py
        --data_dirs "${eval_data_dirs}"
        --model_path "${model_path}"
        --how_many_path "${how_many_path}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --identifier "${eval_identifier}"
    )
    if [ -n "${FIGURES_DIR:-}" ]; then cmd+=(--figures_dir "${FIGURES_DIR}"); fi
    "${cmd[@]}"
}

for rank in "${RANKS[@]}"; do
    for branch in "${BRANCHES[@]}"; do
        run_eval "$rank" "$branch"
    done
done

echo Finished at: `date`
