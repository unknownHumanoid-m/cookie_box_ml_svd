#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=noisy_pipeline
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=1-12:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml

set -e

# ============================================================================
# Noisy-input baseline: train + eval how_many, phase_single, phase_two on
# Ximg (the noisy SVD input already in the same H5 files as Ypdf_denoised).
# Produces figures with '_noisy' identifiers and 'input_key=Ximg' titles so
# they can be compared side-by-side with the denoised-input results.
# ============================================================================

# ---- Data (shared across all six stages) -----------------------------------
export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd2d_r8_autoencoder/train"
export EVAL_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd2d_r8_autoencoder/test"
export INPUT_KEY="Ximg"

# ---- how_many classifier ---------------------------------------------------
export HM_MIN_PULSES=1
export HM_MAX_PULSES=4
export HM_BATCH_SIZE=128
export HM_LR=0.001
export HM_EPOCHS=50
export HM_PATIENCE=5
export HM_VAL_FRAC=0.2
export HM_NUM_WORKERS=4
export HM_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/how_many_noisy_svd2d_r8/"
export HM_IDENTIFIER="how_many_noisy_svd2d_r8"
export HM_MODEL_PATH="${HM_SAVE_DIR}/${HM_IDENTIFIER}.pth"
export HM_EVAL_IDENTIFIER="${HM_IDENTIFIER}_eval"

# ---- phase_single regressor ------------------------------------------------
export PS_NUM_CLASSES=2000
export PS_BATCH_SIZE=32
export PS_LR=0.001
export PS_EPOCHS=50
export PS_PATIENCE=5
export PS_VAL_FRAC=0.2
export PS_NUM_WORKERS=4
export PS_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/phase_single_noisy_svd2d_r8/"
export PS_IDENTIFIER="phase_single_noisy_svd2d_r8"
export PS_MODEL_PATH="${PS_SAVE_DIR}/${PS_IDENTIFIER}.pth"
export PS_EVAL_IDENTIFIER="${PS_IDENTIFIER}_eval"
export PS_EVAL_BATCH_SIZE=64

# ---- phase_two regressor ---------------------------------------------------
export PT_NUM_CLASSES=1000
export PT_BATCH_SIZE=32
export PT_LR=0.001
export PT_EPOCHS=50
export PT_PATIENCE=5
export PT_VAL_FRAC=0.2
export PT_NUM_WORKERS=4
export PT_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/phase_two_noisy_svd2d_r8/"
export PT_IDENTIFIER="phase_two_noisy_svd2d_r8"
export PT_MODEL_PATH="${PT_SAVE_DIR}/${PT_IDENTIFIER}.pth"
export PT_EVAL_IDENTIFIER="${PT_IDENTIFIER}_eval"
export PT_EVAL_BATCH_SIZE=64

# ============================================================================

echo starting run at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising
REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions

mkdir -p "${HM_SAVE_DIR}" "${PS_SAVE_DIR}" "${PT_SAVE_DIR}"

# ---------------------------------------------------------------------------
# 1) Train how_many on Ximg
# ---------------------------------------------------------------------------
echo "===== [1/6] train_how_many.py (input_key=${INPUT_KEY}) ====="
cd "${DENOISING_DIR}"
python3 train_how_many.py \
    --data_dirs "${TRAIN_DATA_DIRS}" \
    --input_key "${INPUT_KEY}" \
    --min_pulses "${HM_MIN_PULSES}" \
    --max_pulses "${HM_MAX_PULSES}" \
    --batch_size "${HM_BATCH_SIZE}" \
    --lr "${HM_LR}" \
    --epochs "${HM_EPOCHS}" \
    --patience "${HM_PATIENCE}" \
    --val_frac "${HM_VAL_FRAC}" \
    --num_workers "${HM_NUM_WORKERS}" \
    --save_dir "${HM_SAVE_DIR}" \
    --save_model "${HM_IDENTIFIER}.pth"

# ---------------------------------------------------------------------------
# 2) Evaluate how_many
# ---------------------------------------------------------------------------
echo "===== [2/6] evaluate_how_many.py ====="
python3 evaluate_how_many.py \
    --data_dirs "${EVAL_DATA_DIRS}" \
    --model_path "${HM_MODEL_PATH}" \
    --batch_size "${HM_BATCH_SIZE}" \
    --num_workers "${HM_NUM_WORKERS}" \
    --identifier "${HM_EVAL_IDENTIFIER}"

# ---------------------------------------------------------------------------
# 3) Train phase_single on Ximg
# ---------------------------------------------------------------------------
echo "===== [3/6] train_phase_single.py (input_key=${INPUT_KEY}) ====="
cd "${REGRESSION_DIR}"
python3 train_phase_single.py \
    --data_dirs "${TRAIN_DATA_DIRS}" \
    --input_key "${INPUT_KEY}" \
    --num_classes "${PS_NUM_CLASSES}" \
    --batch_size "${PS_BATCH_SIZE}" \
    --lr "${PS_LR}" \
    --epochs "${PS_EPOCHS}" \
    --patience "${PS_PATIENCE}" \
    --val_frac "${PS_VAL_FRAC}" \
    --num_workers "${PS_NUM_WORKERS}" \
    --save_dir "${PS_SAVE_DIR}" \
    --save_model "${PS_IDENTIFIER}.pth"

# ---------------------------------------------------------------------------
# 4) Evaluate phase_single (gated by how_many)
# ---------------------------------------------------------------------------
echo "===== [4/6] eval_phase_single.py ====="
python3 eval_phase_single.py \
    --data_dirs "${EVAL_DATA_DIRS}" \
    --model_path "${PS_MODEL_PATH}" \
    --how_many_path "${HM_MODEL_PATH}" \
    --batch_size "${PS_EVAL_BATCH_SIZE}" \
    --num_workers "${PS_NUM_WORKERS}" \
    --identifier "${PS_EVAL_IDENTIFIER}"

# ---------------------------------------------------------------------------
# 5) Train phase_two on Ximg
# ---------------------------------------------------------------------------
echo "===== [5/6] train_phase_two.py (input_key=${INPUT_KEY}) ====="
python3 train_phase_two.py \
    --data_dirs "${TRAIN_DATA_DIRS}" \
    --input_key "${INPUT_KEY}" \
    --num_classes "${PT_NUM_CLASSES}" \
    --batch_size "${PT_BATCH_SIZE}" \
    --lr "${PT_LR}" \
    --epochs "${PT_EPOCHS}" \
    --patience "${PT_PATIENCE}" \
    --val_frac "${PT_VAL_FRAC}" \
    --num_workers "${PT_NUM_WORKERS}" \
    --save_dir "${PT_SAVE_DIR}" \
    --save_model "${PT_IDENTIFIER}.pth"

# ---------------------------------------------------------------------------
# 6) Evaluate phase_two (gated by how_many)
# ---------------------------------------------------------------------------
echo "===== [6/6] eval_phase_two.py ====="
python3 eval_phase_two.py \
    --data_dirs "${EVAL_DATA_DIRS}" \
    --model_path "${PT_MODEL_PATH}" \
    --how_many_path "${HM_MODEL_PATH}" \
    --batch_size "${PT_EVAL_BATCH_SIZE}" \
    --num_workers "${PT_NUM_WORKERS}" \
    --identifier "${PT_EVAL_IDENTIFIER}"

echo Finished at: `date`
