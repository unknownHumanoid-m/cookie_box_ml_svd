#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=hm_phase2_r2r4
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-48:00:00
#SBATCH --gpus 1

# ----------------------------------------------------------------------------
# For RANK in {2, 4}:
#   0. Run the trained autoencoder over SVD train+test splits (denoise_inference.py).
#   1. Train the how_many classifier (train_how_many.py).
#   2. Train the two-pulse phase regressor (train_phase_two.py).
#   3. Evaluate the how_many classifier (evaluate_how_many.py).
#   4. Evaluate the two-pulse phase regressor (eval_phase_two.py).
# ----------------------------------------------------------------------------

# Roots (per-rank paths derived below).
DENOISED_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data"
RUNS_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"
SVD_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"

# Denoise-inference knobs.
DENOISE_BATCH_SIZE=128

# Which per-shot h5 dataset to feed the models.
#   Ypdf          -> clean truth (upper-bound run)
#   Ximg          -> raw SVD input (lower-bound run)
#   Ypdf_denoised -> autoencoder output (default; realistic end-to-end)
INPUT_KEY="Ypdf_denoised"

# how_many hyperparameters.
HM_MIN_PULSES=1
HM_MAX_PULSES=4
HM_BATCH_SIZE=128
HM_LR=0.001
HM_EPOCHS=50
HM_PATIENCE=5
HM_VAL_FRAC=0.2
HM_NUM_WORKERS=4

# phase_two hyperparameters.
P2_NUM_CLASSES=1000
P2_BATCH_SIZE=32
P2_LR=0.001
P2_EPOCHS=50
P2_PATIENCE=5
P2_VAL_FRAC=0.2
P2_NUM_WORKERS=4

# Eval knobs.
EVAL_BATCH_SIZE=64
EVAL_NUM_WORKERS=4

# ----------------------------------------------------------------------------

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising
REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions

for RANK in 2 4; do
    echo "============================================================"
    echo "=== RANK=${RANK} pipeline at $(date) ==="
    echo "============================================================"

    SVD_TRAIN_DIR="${SVD_ROOT}/2d_svd_r${RANK}_train"
    SVD_TEST_DIR="${SVD_ROOT}/2d_svd_r${RANK}_test"

    TRAIN_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/train"
    EVAL_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/test"

    AE_MODEL_PATH="${RUNS_ROOT}/svd2d_r${RANK}_autoencoder/autoencoder_svd2d_r${RANK}_best_model.pth"

    HM_SAVE_DIR="${RUNS_ROOT}/how_many_denoised_svd2d_r${RANK}/"
    HM_IDENTIFIER="how_many_denoised_svd2d_r${RANK}"
    HM_MODEL_PATH="${HM_SAVE_DIR}/${HM_IDENTIFIER}.pth"

    P2_SAVE_DIR="${RUNS_ROOT}/phase_two_svd2d_r${RANK}/"
    P2_IDENTIFIER="phase_two_svd2d_r${RANK}"
    P2_MODEL_PATH="${P2_SAVE_DIR}/${P2_IDENTIFIER}.pth"

    mkdir -p "${TRAIN_DATA_DIRS}" "${EVAL_DATA_DIRS}" "${HM_SAVE_DIR}" "${P2_SAVE_DIR}"

    echo "AE_MODEL_PATH=${AE_MODEL_PATH}"
    echo "SVD_TRAIN_DIR=${SVD_TRAIN_DIR} -> ${TRAIN_DATA_DIRS}"
    echo "SVD_TEST_DIR=${SVD_TEST_DIR}   -> ${EVAL_DATA_DIRS}"
    echo "HM_MODEL_PATH=${HM_MODEL_PATH}"
    echo "P2_MODEL_PATH=${P2_MODEL_PATH}"

    # ----- 0. Denoise inference over train + test SVD splits -----
    echo "--- [r${RANK}] Denoise inference (test) at $(date) ---"
    cd "$DENOISING_DIR" && python3 denoise_inference.py \
        --data_dirs "${SVD_TEST_DIR}" \
        --model_path "${AE_MODEL_PATH}" \
        --output_dir "${EVAL_DATA_DIRS}" \
        --batch_size "${DENOISE_BATCH_SIZE}"

    echo "--- [r${RANK}] Denoise inference (train) at $(date) ---"
    cd "$DENOISING_DIR" && python3 denoise_inference.py \
        --data_dirs "${SVD_TRAIN_DIR}" \
        --model_path "${AE_MODEL_PATH}" \
        --output_dir "${TRAIN_DATA_DIRS}" \
        --batch_size "${DENOISE_BATCH_SIZE}"

    # ----- 1. Train how_many -----
    echo "--- [r${RANK}] Training how_many at $(date) ---"
    cd "$DENOISING_DIR" && python3 train_how_many.py \
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

    # ----- 2. Train phase_two regressor -----
    echo "--- [r${RANK}] Training phase_two at $(date) ---"
    cd "$REGRESSION_DIR" && python3 train_phase_two.py \
        --data_dirs "${TRAIN_DATA_DIRS}" \
        --input_key "${INPUT_KEY}" \
        --num_classes "${P2_NUM_CLASSES}" \
        --batch_size "${P2_BATCH_SIZE}" \
        --lr "${P2_LR}" \
        --epochs "${P2_EPOCHS}" \
        --patience "${P2_PATIENCE}" \
        --val_frac "${P2_VAL_FRAC}" \
        --num_workers "${P2_NUM_WORKERS}" \
        --save_dir "${P2_SAVE_DIR}" \
        --save_model "${P2_IDENTIFIER}.pth"

    # ----- 3. Evaluate how_many -----
    echo "--- [r${RANK}] Evaluating how_many at $(date) ---"
    cd "$DENOISING_DIR" && python3 evaluate_how_many.py \
        --data_dirs "${EVAL_DATA_DIRS}" \
        --model_path "${HM_MODEL_PATH}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        --identifier "${HM_IDENTIFIER}_eval"

    # ----- 4. Evaluate phase_two regressor (uses how_many for gating) -----
    echo "--- [r${RANK}] Evaluating phase_two at $(date) ---"
    cd "$REGRESSION_DIR" && python3 eval_phase_two.py \
        --data_dirs "${EVAL_DATA_DIRS}" \
        --model_path "${P2_MODEL_PATH}" \
        --how_many_path "${HM_MODEL_PATH}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        --identifier "${P2_IDENTIFIER}_eval"

done

echo "Finished at: $(date)"
