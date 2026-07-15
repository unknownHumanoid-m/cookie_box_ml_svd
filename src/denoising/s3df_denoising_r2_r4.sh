#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=denoise_r2r4
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 2

# Train the denoising autoencoder on the 2D-SVD datasets for rank=2 and rank=4,
# sequentially, in a single SLURM job.

DATA_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"
MODEL_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"
COOKIE_DENOISING_DIR="/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for RANK in 2 4; do
    echo "============================================================"
    echo "=== Training 2D-SVD denoiser for rank=${RANK} at $(date) ==="
    echo "============================================================"

    export TRAIN_DATA_DIRS="${DATA_ROOT}/2d_svd_r${RANK}_train/"
    export MODEL_SAVE_DIR="${MODEL_ROOT}/svd2d_r${RANK}_autoencoder/"
    export MODEL_IDENTIFIER="autoencoder_svd2d_r${RANK}"

    echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
    echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
    echo "MODEL_IDENTIFIER=$MODEL_IDENTIFIER"

    mkdir -p "${MODEL_SAVE_DIR}"

    cd "$COOKIE_DENOISING_DIR" && python3 ximg_to_ypdf_autoencoder_straight_training.py
done

echo "Finished at: $(date)"
