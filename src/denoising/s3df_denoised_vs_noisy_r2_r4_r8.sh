#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=cmp_r2_r4_r8
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=3-00:00:00
#SBATCH --gpus 1

set -e

# ============================================================================
# For RANK in {2, 4, 8}, produce a denoised-vs-noisy comparison of the full
# classifier + phase-regression stack:
#   0. denoise_inference:   mirror SVD h5 -> +Ypdf_denoised
#   1. train_how_many       (per input_key)
#   2. evaluate_how_many
#   3. train_phase_single
#   4. eval_phase_single    (gated by how_many)
#   5. train_phase_two
#   6. eval_phase_two       (gated by how_many)
#
# Runs the whole stack twice per rank:
#   INPUT_KEY=Ypdf_denoised -> identifiers "..._denoised_svd2d_rR" / "..._svd2d_rR"
#   INPUT_KEY=Ximg          -> identifiers "..._noisy_svd2d_rR"
# so figures land side-by-side under each module's ./figures/ dir.
# ============================================================================

# ---- Roots (per-rank paths derived below) ---------------------------------
DENOISED_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data"
RUNS_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs"
SVD_ROOT="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"

# ---- Denoise-inference knob -----------------------------------------------
DENOISE_BATCH_SIZE=128
# Skip step 0 if the mirrored (denoised) h5 dirs already exist for this rank.
# Set SKIP_DENOISE_IF_EXISTS=0 to force re-mirroring.
SKIP_DENOISE_IF_EXISTS=1

# ---- how_many hyperparameters ---------------------------------------------
HM_MIN_PULSES=1
HM_MAX_PULSES=4
HM_BATCH_SIZE=128
HM_LR=0.001
HM_EPOCHS=50
HM_PATIENCE=5
HM_VAL_FRAC=0.2
HM_NUM_WORKERS=4

# ---- phase_single hyperparameters -----------------------------------------
PS_NUM_CLASSES=2000
PS_BATCH_SIZE=32
PS_LR=0.001
PS_EPOCHS=50
PS_PATIENCE=5
PS_VAL_FRAC=0.2
PS_NUM_WORKERS=4

# ---- phase_two hyperparameters --------------------------------------------
PT_NUM_CLASSES=1000
PT_BATCH_SIZE=32
PT_LR=0.001
PT_EPOCHS=50
PT_PATIENCE=5
PT_VAL_FRAC=0.2
PT_NUM_WORKERS=4

# ---- Eval knobs -----------------------------------------------------------
EVAL_BATCH_SIZE=64
EVAL_NUM_WORKERS=4

# ============================================================================

echo "starting run at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising
REGRESSION_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/regressions

# Runs one full (train + eval) stack for a given rank and input_key.
# Args: RANK INPUT_KEY TAG
#   TAG = "denoised" (identifiers .._svd2d_rR / how_many_denoised_svd2d_rR)
#   TAG = "noisy"    (identifiers .._noisy_svd2d_rR)
run_stack() {
    local RANK=$1
    local INPUT_KEY=$2
    local TAG=$3

    local TRAIN_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/train"
    local EVAL_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/test"

    local HM_ID PS_ID PT_ID
    if [[ "${TAG}" == "denoised" ]]; then
        HM_ID="how_many_denoised_svd2d_r${RANK}"
        PS_ID="phase_single_svd2d_r${RANK}"
        PT_ID="phase_two_svd2d_r${RANK}"
    else
        HM_ID="how_many_noisy_svd2d_r${RANK}"
        PS_ID="phase_single_noisy_svd2d_r${RANK}"
        PT_ID="phase_two_noisy_svd2d_r${RANK}"
    fi

    local HM_SAVE_DIR="${RUNS_ROOT}/${HM_ID}/"
    local HM_MODEL_PATH="${HM_SAVE_DIR}/${HM_ID}.pth"
    local PS_SAVE_DIR="${RUNS_ROOT}/${PS_ID}/"
    local PS_MODEL_PATH="${PS_SAVE_DIR}/${PS_ID}.pth"
    local PT_SAVE_DIR="${RUNS_ROOT}/${PT_ID}/"
    local PT_MODEL_PATH="${PT_SAVE_DIR}/${PT_ID}.pth"

    mkdir -p "${HM_SAVE_DIR}" "${PS_SAVE_DIR}" "${PT_SAVE_DIR}"

    echo "============================================================"
    echo "=== r${RANK} ${TAG} (input_key=${INPUT_KEY}) at $(date) ==="
    echo "============================================================"
    echo "TRAIN_DATA_DIRS=${TRAIN_DATA_DIRS}"
    echo "EVAL_DATA_DIRS =${EVAL_DATA_DIRS}"
    echo "HM_MODEL_PATH  =${HM_MODEL_PATH}"
    echo "PS_MODEL_PATH  =${PS_MODEL_PATH}"
    echo "PT_MODEL_PATH  =${PT_MODEL_PATH}"

    # ----- 1. Train how_many ----------------------------------------------
    echo "--- [r${RANK}/${TAG}] train_how_many at $(date) ---"
    cd "${DENOISING_DIR}" && python3 train_how_many.py \
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
        --save_model "${HM_ID}.pth"

    # ----- 2. Evaluate how_many -------------------------------------------
    echo "--- [r${RANK}/${TAG}] evaluate_how_many at $(date) ---"
    cd "${DENOISING_DIR}" && python3 evaluate_how_many.py \
        --data_dirs "${EVAL_DATA_DIRS}" \
        --model_path "${HM_MODEL_PATH}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        --identifier "${HM_ID}_eval"

    # ----- 3. Train phase_single ------------------------------------------
    echo "--- [r${RANK}/${TAG}] train_phase_single at $(date) ---"
    cd "${REGRESSION_DIR}" && python3 train_phase_single.py \
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
        --save_model "${PS_ID}.pth"

    # ----- 4. Evaluate phase_single (gated by how_many) -------------------
    echo "--- [r${RANK}/${TAG}] eval_phase_single at $(date) ---"
    cd "${REGRESSION_DIR}" && python3 eval_phase_single.py \
        --data_dirs "${EVAL_DATA_DIRS}" \
        --model_path "${PS_MODEL_PATH}" \
        --how_many_path "${HM_MODEL_PATH}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        --identifier "${PS_ID}_eval"

    # ----- 5. Train phase_two ---------------------------------------------
    echo "--- [r${RANK}/${TAG}] train_phase_two at $(date) ---"
    cd "${REGRESSION_DIR}" && python3 train_phase_two.py \
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
        --save_model "${PT_ID}.pth"

    # ----- 6. Evaluate phase_two (gated by how_many) ----------------------
    echo "--- [r${RANK}/${TAG}] eval_phase_two at $(date) ---"
    cd "${REGRESSION_DIR}" && python3 eval_phase_two.py \
        --data_dirs "${EVAL_DATA_DIRS}" \
        --model_path "${PT_MODEL_PATH}" \
        --how_many_path "${HM_MODEL_PATH}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        --identifier "${PT_ID}_eval"
}

for RANK in 2 4 8; do
    SVD_TRAIN_DIR="${SVD_ROOT}/2d_svd_r${RANK}_train"
    SVD_TEST_DIR="${SVD_ROOT}/2d_svd_r${RANK}_test"
    TRAIN_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/train"
    EVAL_DATA_DIRS="${DENOISED_ROOT}/svd2d_r${RANK}_autoencoder/test"
    AE_MODEL_PATH="${RUNS_ROOT}/svd2d_r${RANK}_autoencoder/autoencoder_svd2d_r${RANK}_best_model.pth"

    # ----- 0. Denoise inference over train + test SVD splits --------------
    if [[ "${SKIP_DENOISE_IF_EXISTS}" == "1" && -d "${TRAIN_DATA_DIRS}" && -d "${EVAL_DATA_DIRS}" ]]; then
        echo "--- [r${RANK}] skipping denoise_inference (${TRAIN_DATA_DIRS} and ${EVAL_DATA_DIRS} already exist) ---"
    else
        mkdir -p "${TRAIN_DATA_DIRS}" "${EVAL_DATA_DIRS}"
        echo "--- [r${RANK}] denoise_inference (test) at $(date) ---"
        cd "${DENOISING_DIR}" && python3 denoise_inference.py \
            --data_dirs "${SVD_TEST_DIR}" \
            --model_path "${AE_MODEL_PATH}" \
            --output_dir "${EVAL_DATA_DIRS}" \
            --batch_size "${DENOISE_BATCH_SIZE}"

        echo "--- [r${RANK}] denoise_inference (train) at $(date) ---"
        cd "${DENOISING_DIR}" && python3 denoise_inference.py \
            --data_dirs "${SVD_TRAIN_DIR}" \
            --model_path "${AE_MODEL_PATH}" \
            --output_dir "${TRAIN_DATA_DIRS}" \
            --batch_size "${DENOISE_BATCH_SIZE}"
    fi

    # ----- Run both stacks against the mirrored h5 tree -------------------
    run_stack "${RANK}" "Ypdf_denoised" "denoised"
    run_stack "${RANK}" "Ximg"          "noisy"
done

echo "Finished at: $(date)"
