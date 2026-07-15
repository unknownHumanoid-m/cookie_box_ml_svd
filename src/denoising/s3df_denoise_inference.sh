#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=denoise_infer
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16g
#SBATCH --time=0-06:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# Run the trained autoencoder over every SVD input file (train + test splits)
# and write mirrored denoised h5 files into a new scratch tree.
#
# Each output file preserves per-shot structure and adds Ypdf_denoised
# alongside Ypdf and Ximg. All group attrs (npulses, ...) are copied through
# so downstream classifiers / regressors can read the denoised output the
# same way they read the SVD files.
# ----------------------------------------------------------------------------

# --- Autoencoder weights to run inference with. ---
export MODEL_PATH="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/svd2d_r8_autoencoder/autoencoder_svd2d_r8_best_model.pth"

# --- Where the SVD inputs live. ---
export SVD_TRAIN_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_train"
export SVD_TEST_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_test"

# --- Where denoised outputs should go. ---
export DENOISED_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_denoised/svd2d_r8_autoencoder"
export DENOISED_TRAIN_DIR="${DENOISED_ROOT}/train"
export DENOISED_TEST_DIR="${DENOISED_ROOT}/test"

# Runtime knobs.
export BATCH_SIZE=128

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

COOKIE_DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising

mkdir -p "${DENOISED_TRAIN_DIR}" "${DENOISED_TEST_DIR}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "SVD_TRAIN_DIR=${SVD_TRAIN_DIR} -> ${DENOISED_TRAIN_DIR}"
echo "SVD_TEST_DIR=${SVD_TEST_DIR}   -> ${DENOISED_TEST_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"

cd "$COOKIE_DENOISING_DIR"

echo "=== test split ==="
python3 denoise_inference.py \
    --data_dirs "${SVD_TEST_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${DENOISED_TEST_DIR}" \
    --batch_size "${BATCH_SIZE}"

echo "=== train split ==="
python3 denoise_inference.py \
    --data_dirs "${SVD_TRAIN_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${DENOISED_TRAIN_DIR}" \
    --batch_size "${BATCH_SIZE}"

echo Finished at: `date`
