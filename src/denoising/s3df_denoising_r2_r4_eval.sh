#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=denoise_r2r4_eval
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 2

# Evaluate the denoising autoencoder on the 2D-SVD datasets for rank=2 and rank=4,
# sequentially, in a single SLURM job.

DATA_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"
MODEL_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"
COOKIE_DENOISING_DIR="/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Number of noisy-vs-denoised-vs-truth comparison figures to save.
export EVAL_NUM_COMPARISON_FIGURES=6
# Optional zero-pulse classifier. Leave empty to disable zero-masking.
export CLASSIFIER_PATH=""

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for RANK in 2 4; do
    echo "============================================================"
    echo "=== Evaluating 2D-SVD denoiser for rank=${RANK} at $(date) ==="
    echo "============================================================"

    export EVAL_DATA_DIRS="${DATA_ROOT}/2d_svd_r${RANK}_test/even-dist_Files.022_processed_svd2d_r${RANK}.h5":"${DATA_ROOT}/2d_svd_r${RANK}_test/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed_svd2d_r${RANK}.h5"

    export MODEL_SAVE_DIR="${MODEL_ROOT}/svd2d_r${RANK}_autoencoder/"
    export MODEL_IDENTIFIER="autoencoder_svd2d_r${RANK}"
    export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/${MODEL_IDENTIFIER}_best_model.pth"
    export EVAL_OUT_DIR="${MODEL_SAVE_DIR}/eval/"
    export EVAL_IDENTIFIER="${MODEL_IDENTIFIER}_eval"

    echo "EVAL_DATA_DIRS=$EVAL_DATA_DIRS"
    echo "EVAL_MODEL_PATH=$EVAL_MODEL_PATH"
    echo "EVAL_OUT_DIR=$EVAL_OUT_DIR"
    echo "EVAL_IDENTIFIER=$EVAL_IDENTIFIER"
    echo "EVAL_NUM_COMPARISON_FIGURES=$EVAL_NUM_COMPARISON_FIGURES"
    echo "CLASSIFIER_PATH=$CLASSIFIER_PATH"

    mkdir -p "${EVAL_OUT_DIR}"

    cd "$COOKIE_DENOISING_DIR" && python3 ximg_to_ypdf_autoencoder_straight_evaluation.py
done

echo "Finished at: $(date)"
