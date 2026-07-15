#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=lstm_cls
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 1
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml


# ----------------------------------------------------------------------------
# ***** Edit these to point at the data / models you want *****
# Values are passed to the LSTM training/eval scripts as CLI flags.
# Data locations mirror src/denoising/s3df_how_many_{train,eval}.sh so this
# LSTM run and the how_many run train/eval on the same shots.
# ----------------------------------------------------------------------------

# --- TRAINING config (used by lstm_pulseNum_classifier_straight_training.py) ---
export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_train/even-dist_Files.001_processed_svd2d_r8.h5":"/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_train/even-dist_Files.002_processed_svd2d_r8.h5":"/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_train/even-dist_Files.003_processed_svd2d_r8.h5"

# Per-shot h5 dataset the LSTM consumes. Ypdf = clean truth (matches how_many).
export INPUT_KEY="Ypdf"

# Pulse-count range. Matches the how_many default so class labels line up.
export MIN_PULSES=1
export MAX_PULSES=4

# Training hyperparameters.
export BATCH_SIZE=32
export LR=0.0001
export EPOCHS=50
export PATIENCE=5
export VAL_FRAC=0.2
export NUM_WORKERS=4

# LSTM architecture. Feature dim is derived from the data (16 for the current
# processed pipeline); only the recurrent width / depth are configurable here.
export HIDDEN_SIZE=128
export NUM_LSTM_LAYERS=3
export DROPOUT=0.2
export LSTM_DROPOUT=0.2

# Where to save the .pth (identifier + directory).
export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/lstm_pulseNum_svd2d_r8/"
export MODEL_IDENTIFIER="lstm_pulseNum_svd2d_r8"

# --- EVALUATION config (used by lstm_pulseNum_classifier_straight_eval.py) ---
export EVAL_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_test/even-dist_Files.022_processed_svd2d_r8.h5":"/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_test/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed_svd2d_r8.h5"

# Trained-model checkpoint to evaluate.
export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/${MODEL_IDENTIFIER}.pth"
export EVAL_IDENTIFIER="${MODEL_IDENTIFIER}_eval"
export EVAL_BATCH_SIZE=64

# Optional override for figure output; defaults to <this-dir>/figures/.
# export FIGURES_DIR="/some/other/figures/"

# ----------------------------------------------------------------------------
# End of user config.
# ----------------------------------------------------------------------------

if [ -z "$1" ]; then
    echo "No script specified. Usage: sbatch s3df_classifier.sh [training|evaluation]"
    exit 1
fi

echo starting run 1 at: `date`
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

COOKIE_LSTM_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/ml_backbone/classifiers

echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
echo "EVAL_DATA_DIRS=$EVAL_DATA_DIRS"
echo "INPUT_KEY=$INPUT_KEY  MIN_PULSES=$MIN_PULSES  MAX_PULSES=$MAX_PULSES"
echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR  MODEL_IDENTIFIER=$MODEL_IDENTIFIER"

mkdir -p "${MODEL_SAVE_DIR}"

case "$1" in
    training)
        CMD=(python3 lstm_pulseNum_classifier_straight_training.py
            --data_dirs        "${TRAIN_DATA_DIRS}"
            --input_key        "${INPUT_KEY}"
            --min_pulses       "${MIN_PULSES}"
            --max_pulses       "${MAX_PULSES}"
            --batch_size       "${BATCH_SIZE}"
            --lr               "${LR}"
            --epochs           "${EPOCHS}"
            --patience         "${PATIENCE}"
            --val_frac         "${VAL_FRAC}"
            --num_workers      "${NUM_WORKERS}"
            --hidden_size      "${HIDDEN_SIZE}"
            --num_lstm_layers  "${NUM_LSTM_LAYERS}"
            --dropout          "${DROPOUT}"
            --lstm_dropout     "${LSTM_DROPOUT}"
            --save_dir         "${MODEL_SAVE_DIR}"
            --save_model       "${MODEL_IDENTIFIER}.pth"
        )
        if [ -n "${FIGURES_DIR:-}" ]; then
            CMD+=(--figures_dir "${FIGURES_DIR}")
        fi
        cd "$COOKIE_LSTM_DIR" && "${CMD[@]}"
        ;;
    evaluation)
        CMD=(python3 lstm_pulseNum_classifier_straight_eval.py
            --data_dirs   "${EVAL_DATA_DIRS}"
            --model_path  "${EVAL_MODEL_PATH}"
            --batch_size  "${EVAL_BATCH_SIZE}"
            --num_workers "${NUM_WORKERS}"
            --identifier  "${EVAL_IDENTIFIER}"
        )
        if [ -n "${FIGURES_DIR:-}" ]; then
            CMD+=(--figures_dir "${FIGURES_DIR}")
        fi
        cd "$COOKIE_LSTM_DIR" && "${CMD[@]}"
        ;;
    *)
        echo "Invalid script specified. Usage: sbatch s3df_classifier.sh [training|evaluation]"
        exit 1
        ;;
esac

echo Finished at: `date`
