#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=svd1d_denoise_train
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=1-00:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml

# ----------------------------------------------------------------------------
# 1D-SVD autoencoder: train the segmented-MLP denoiser mapping the flat
# [U|S|V] Ximg vector to the corresponding flat Ypdf vector. This is the
# 1D counterpart to src/denoising/s3df_denoising.sh.
# ----------------------------------------------------------------------------

export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/1d_svd_r8_train"

# SVD layout of the input vectors. r8 means 8 retained components.
export SVD_H=16
export SVD_W=512
export SVD_R=8

export BATCH_SIZE=128
export LR=0.0005
export EPOCHS=200
export NUM_WORKERS=8

export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/svd1d_r8_autoencoder/"
export MODEL_IDENTIFIER="svd1d_r8_autoencoder"

# Optional override; defaults to <this dir>/figures/.
# export FIGURES_DIR="/some/other/figures/"

# ----------------------------------------------------------------------------

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

DENOISING_1D_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising_1D_svd

echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
echo "SVD_H=$SVD_H  SVD_W=$SVD_W  SVD_R=$SVD_R"
echo "BATCH_SIZE=$BATCH_SIZE  LR=$LR  EPOCHS=$EPOCHS"
echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "MODEL_IDENTIFIER=$MODEL_IDENTIFIER"

mkdir -p "${MODEL_SAVE_DIR}"

cd "${DENOISING_1D_DIR}" && python3 train_svd1d_autoencoder.py

echo Finished at: `date`
