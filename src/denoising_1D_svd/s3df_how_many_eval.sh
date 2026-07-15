#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=how_many_eval
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-02:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml


# ----------------------------------------------------------------------------
# ***** Edit these to point at the data / models you want *****
# The values below are passed to evaluate_how_many.py as CLI flags.
# ----------------------------------------------------------------------------

# ':'-separated list of h5 files (or directories) to evaluate against.
# Default: autoencoder-denoised test set (mirrors 2d_svd_test).
export EVAL_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/svd1d_r8_autoencoder/test"

# Trained-model checkpoint (produced by train_how_many.py).
export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/how_many_denoised_svd1d_r8/"
export MODEL_IDENTIFIER="how_many_denoised_svd1d_r8"
export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/${MODEL_IDENTIFIER}.pth"

# Optional overrides. Leave empty to reuse the values stored in the .pth.
export INPUT_KEY=""
export MIN_PULSES=""
export MAX_PULSES=""

# Runtime knobs.
export BATCH_SIZE=64
export NUM_WORKERS=4

# Output figure identifier (defaults to the .pth basename inside the script).
export EVAL_IDENTIFIER="${MODEL_IDENTIFIER}_eval"

# Optional override for figure output; defaults to <repo>/src/denoising/figures/.
# export FIGURES_DIR="/some/other/figures/"

# ----------------------------------------------------------------------------
# End of user config.
# ----------------------------------------------------------------------------

echo starting run 1 at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

COOKIE_DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising_1D_svd

echo "EVAL_DATA_DIRS=$EVAL_DATA_DIRS"
echo "EVAL_MODEL_PATH=$EVAL_MODEL_PATH"
echo "EVAL_IDENTIFIER=$EVAL_IDENTIFIER"
echo "INPUT_KEY(override)=${INPUT_KEY:-<from checkpoint>}"
echo "MIN_PULSES(override)=${MIN_PULSES:-<from checkpoint>}"
echo "MAX_PULSES(override)=${MAX_PULSES:-<from checkpoint>}"

CMD=(python3 evaluate_how_many.py
    --data_dirs "${EVAL_DATA_DIRS}"
    --model_path "${EVAL_MODEL_PATH}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --identifier "${EVAL_IDENTIFIER}"
)
if [ -n "${INPUT_KEY}" ];   then CMD+=(--input_key   "${INPUT_KEY}");   fi
if [ -n "${MIN_PULSES}" ];  then CMD+=(--min_pulses  "${MIN_PULSES}");  fi
if [ -n "${MAX_PULSES}" ];  then CMD+=(--max_pulses  "${MAX_PULSES}");  fi
if [ -n "${FIGURES_DIR:-}" ]; then CMD+=(--figures_dir "${FIGURES_DIR}"); fi

cd "$COOKIE_DENOISING_DIR" && "${CMD[@]}"

echo Finished at: `date`
