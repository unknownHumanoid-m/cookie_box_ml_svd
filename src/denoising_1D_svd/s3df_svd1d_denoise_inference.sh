#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=svd1d_denoise_inference
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-06:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml

# ----------------------------------------------------------------------------
# 1D-SVD autoencoder inference. Mirrors the input h5 tree adding
# Ypdf_denoised alongside Ximg / Ypdf. Also drops a handful of 2D
# reconstruction figures under FIGURES_DIR for diagnostic eyeballing.
# ----------------------------------------------------------------------------

# Split into two runs: train side (needed by downstream trainers) and
# test side (needed by downstream evaluators). Comment out either as needed.
export MODEL_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/svd1d_r8_autoencoder/svd1d_r8_autoencoder_best_model.pth"

export TRAIN_INPUT_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/1d_svd_r8_train"
export TRAIN_OUTPUT_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/train"

export TEST_INPUT_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/1d_svd_r8_test"
export TEST_OUTPUT_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/test"

export SVD_H=16
export SVD_W=512
export SVD_R=8
export BATCH_SIZE=256

# Diagnostics
export DIAG_EXAMPLES=6
export FIGURES_DIR="/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising_1D_svd/figures"

# ----------------------------------------------------------------------------

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

DENOISING_1D_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising_1D_svd

mkdir -p "${TRAIN_OUTPUT_DIR}" "${TEST_OUTPUT_DIR}" "${FIGURES_DIR}"

cd "${DENOISING_1D_DIR}"

echo "===== train split ====="
python3 denoise_inference_1d.py \
    --data_dirs "${TRAIN_INPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${TRAIN_OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --svd_h "${SVD_H}" --svd_w "${SVD_W}" --svd_r "${SVD_R}" \
    --diag_examples "${DIAG_EXAMPLES}" \
    --figures_dir "${FIGURES_DIR}" \
    --identifier "svd1d_r8_train_diag"

echo "===== test split ====="
python3 denoise_inference_1d.py \
    --data_dirs "${TEST_INPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${TEST_OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --svd_h "${SVD_H}" --svd_w "${SVD_W}" --svd_r "${SVD_R}" \
    --diag_examples "${DIAG_EXAMPLES}" \
    --figures_dir "${FIGURES_DIR}" \
    --identifier "svd1d_r8_test_diag"

echo Finished at: `date`
