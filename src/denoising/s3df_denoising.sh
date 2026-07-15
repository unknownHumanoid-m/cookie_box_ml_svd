#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:tmox42619@ampere
#SBATCH --job-name=reg
#SBATCH --output=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --error=/sdf/home/m/miaed/slurm_logs/output-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32g
#SBATCH --time=0-24:00:00
#SBATCH --gpus 2
# source ~/.bashrc
# source /sdf/group/lcls/ds/tools/conda_envs/jackh_pytorch/bin/activate cookie_ml


# ----------------------------------------------------------------------------
# ***** Edit these to point at the data / models you want *****
# All paths are picked up by the Python scripts via os.environ.
# You can also override them on the sbatch command line, e.g.
#   sbatch --export=ALL,TRAIN_DATA_DIRS=/my/other/dir/ s3df_denoising.sh training
# ----------------------------------------------------------------------------

# --- TRAINING config (used by ximg_to_ypdf_autoencoder_straight_training.py) ---
# ':'-separated list of directories containing the training h5 files.
export TRAIN_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_train/"

# Where trained model + checkpoints go.
export MODEL_SAVE_DIR="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/svd2d_r8_autoencoder/"
# Identifier prefix for saved files.
export MODEL_IDENTIFIER="autoencoder_svd2d_r8"
# Optional override for figure output; defaults to <repo>/src/denoising/figures/.
# export FIGURES_DIR="/some/other/figures/"

# --- EVALUATION config (used by ximg_to_ypdf_autoencoder_straight_evaluation.py) ---
# ':'-separated list of directories containing the eval/test h5 files.
export EVAL_DATA_DIRS="/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_test/even-dist_Files.022_processed_svd2d_r8.h5":"/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/2d_svd_test/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed_svd2d_r8.h5"


# Path to the .pth checkpoint to evaluate.
export EVAL_MODEL_PATH="${MODEL_SAVE_DIR}/${MODEL_IDENTIFIER}_best_model.pth"
# Where evaluation artifacts (results.h5, results.txt) get written.
export EVAL_OUT_DIR="${MODEL_SAVE_DIR}/eval/"
export EVAL_IDENTIFIER="${MODEL_IDENTIFIER}_eval"
# Optional override for eval figure output.
# export EVAL_FIGURES_DIR="/some/other/figures/"
# Number of noisy-vs-denoised-vs-truth comparison figures to save.
export EVAL_NUM_COMPARISON_FIGURES=6
# Optional zero-pulse classifier. Leave empty to disable zero-masking.
export CLASSIFIER_PATH=""

# ----------------------------------------------------------------------------
# End of user config.
# ----------------------------------------------------------------------------

# Check if the script argument is provided
if [ -z "$1" ]; then
    echo "No script specified. Usage: sbatch s3df_denoising.sh [training|fineTuning|evaluation]"
    exit 1
fi

echo starting run 1 at: `date`
# Check which GPU is being used
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Run the Python script with the specified arguments
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

COOKIE_DENOISING_DIR=/sdf/home/m/miaed/jack_cookiebox_copy/COOKIE_ML/src/denoising

echo "TRAIN_DATA_DIRS=$TRAIN_DATA_DIRS"
echo "MODEL_SAVE_DIR=$MODEL_SAVE_DIR"
echo "MODEL_IDENTIFIER=$MODEL_IDENTIFIER"
echo "EVAL_DATA_DIRS=$EVAL_DATA_DIRS"
echo "EVAL_MODEL_PATH=$EVAL_MODEL_PATH"
echo "EVAL_OUT_DIR=$EVAL_OUT_DIR"
echo "EVAL_IDENTIFIER=$EVAL_IDENTIFIER"
echo "EVAL_NUM_COMPARISON_FIGURES=$EVAL_NUM_COMPARISON_FIGURES"
echo "CLASSIFIER_PATH=$CLASSIFIER_PATH"

case "$1" in
    training)
        cd "$COOKIE_DENOISING_DIR" && python3 ximg_to_ypdf_autoencoder_straight_training.py
        ;;
    fineTuning)
        cd "$COOKIE_DENOISING_DIR" && python3 ximg_to_ypdf_autoencoder_straight_training_fineTuning.py
        ;;
    evaluation)
        cd "$COOKIE_DENOISING_DIR" && python3 ximg_to_ypdf_autoencoder_straight_evaluation.py
        ;;
    *)
        echo "Invalid script specified. Usage: sbatch this_script.sh [training|fineTuning|evaluation]"
        exit 1
        ;;
esac

echo Finished at: `date`
