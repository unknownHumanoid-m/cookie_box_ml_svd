# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`COOKIE_ML` is the SLAC LCLS attosecond cookiebox ML pipeline. The trained
model chain is a deployment target for **FPGA inference** — architectural
decisions should favor low-latency inference over training throughput.

This tree is a personal copy of `jhirschm`'s `COOKIE_ML`; expect uncommitted
edits on `main`. Always `git status` / `git diff` before rewriting scripts,
since local overrides (paths, hyperparams) are common and often not yet
upstreamed.

## Pipeline architecture

The pipeline is stage-by-stage; understanding it requires reading multiple
files, so the flow is documented here:

1. **`src/data_processing/`** — turns raw cookiebox-sim HDF5 into processed
   npz. The SVD stage (`svd_dataset_generator.py`, `generate_svd_datasets.sh`,
   `generate_2d_svd_r8.sh`) writes one h5 file per split with per-shot groups:
   - `Ximg` — noisy SVD input (2D or 1D)
   - `Ypdf` — clean truth
   - `attrs["npulses"]`, `attrs["phases"]`
   Optionally `Ypdf_denoised` is added later by the autoencoder inference
   step.

2. **`src/denoising/`** — small conv autoencoder that maps `Ximg → Ypdf`.
   - Training: `ximg_to_ypdf_autoencoder_straight_training.py`, launched via
     `s3df_denoising.sh` / `s3df_denoising_r2_r4.sh`.
   - Inference: `denoise_inference.py` **mirrors** the input h5 tree into
     an output directory and adds a `Ypdf_denoised` dataset to every shot
     group while copying `Ximg`, `Ypdf`, and all attrs through. That mirrored
     tree is what every downstream script reads.

3. **`src/denoising/train_how_many.py`** — `SimpleMLP` classifier over pulse
   count in `[min_pulses, max_pulses]`. Labels are remapped to `0..K-1` for
   `CrossEntropyLoss`, and the top-class means "≥ `max_pulses`". Checkpoints
   pickle `state_dict + input_shape + num_classes + min_pulses + max_pulses +
   input_key`, so `evaluate_how_many.py` and downstream regressors can
   reconstruct the architecture and label semantics without extra flags.

4. **`src/ml_backbone/regressions/`** — per-pulse-count phase regressors on
   the 2D-SVD `Ypdf_denoised`:
   - **1-pulse**: `train_phase_single.py`. Target is `phases[0] ∈ [0, 2π)`.
   - **2-pulse**: `train_phase_two.py`. Target is `arccos(cos(φ₁-φ₂)) ∈ [0, π]`
     (arccos-of-cos folds the pulse-order ambiguity into a single scalar).
   Both use `resnets.resnet18` with a **classification head over N phase
   bins**; the scalar phase is recovered via a softmax-weighted mean of the
   bin indices (`logits_to_phase` / `logits_to_diff`). MSE is computed on that
   decoded scalar.

5. **Eval routes by the classifier, not by truth.** `eval_phase_single.py` /
   `eval_phase_two.py` run `how_many` on every shot and only feed the phase
   regressor the shots the classifier calls "1" (or "2"). This is the
   realistic on-FPGA gate — do not "fix" it to use ground-truth npulses.

### h5 convention (important, not obvious from any single file)

Every shot in a processed h5 is a group whose datasets include `Ximg`,
`Ypdf`, and optionally `Ypdf_denoised`, plus attrs `npulses` and `phases`.
`--input_key` selects which per-shot dataset to feed the model (`Ypdf`,
`Ximg`, or `Ypdf_denoised`).

### 1D vs 2D SVD

Two parallel copies of the classifier/regressor stack exist because SVD can
be stored either as `(H, W)` (default, rank R = number of retained
components, e.g. shape `(16, 512)`) or as a flat `(L,)` vector:

- **2D SVD**: `src/denoising/` + `src/ml_backbone/regressions/`.
  Regressors use ResNet18 and unsqueeze a channel dim to feed 2D convs.
- **1D SVD**: `src/denoising_1D_svd/` + `src/ml_backbone/regressions_1D_SVD/`.
  Regressors use a deep MLP (`PhaseMLP`) — the 2D convs are meaningless once
  the SVD is flat, and the channel-unsqueeze would break. Same softmax-bin
  head, same `logits_to_phase` / `logits_to_diff` decoder, so downstream code
  is unchanged.

When adding a new stage to either side of the split, mirror the change on
the other side unless you have a specific reason not to.

## Running training and eval

### SLURM (production)

Every `src/**/s3df_*.sh` is a `sbatch`-able launcher. The top of each file
is user-config (`TRAIN_DATA_DIRS`, `INPUT_KEY`, `MODEL_SAVE_DIR`,
`MODEL_IDENTIFIER`, hyperparams); the bottom builds the python command and
`cd`s into the right source dir. Defaults on this fork:

- Partition: `ampere`
- Account: `lcls:tmox42619@ampere`
- Logs: `/sdf/home/m/miaed/slurm_logs/output-%j.txt`

Typical commands:

```
sbatch src/denoising/s3df_denoising.sh
sbatch src/denoising/s3df_denoise_inference.sh
sbatch src/denoising/s3df_how_many_train.sh
sbatch src/ml_backbone/regressions/s3df_phase_single_train.sh
sbatch src/ml_backbone/regressions/s3df_phase_two_train.sh
sbatch src/ml_backbone/regressions/s3df_phase_single_eval.sh
sbatch src/ml_backbone/regressions/s3df_phase_two_eval.sh
```

1D-SVD counterparts live under `src/denoising_1D_svd/` and
`src/ml_backbone/regressions_1D_SVD/` with matching names.

### Interactive (debugging)

Every entrypoint takes CLI flags — no need to go through SLURM:

```
python3 train_how_many.py \
    --data_dirs /path/to/mirrored/denoised/train \
    --input_key Ypdf_denoised \
    --min_pulses 1 --max_pulses 4 \
    --save_dir /path/to/runs/ --save_model how_many.pth

python3 evaluate_how_many.py \
    --data_dirs /path/to/mirrored/denoised/test \
    --model_path /path/to/runs/how_many.pth

python3 eval_phase_single.py \
    --data_dirs /path/to/mirrored/denoised/test \
    --model_path /path/to/runs/phase_single.pth \
    --how_many_path /path/to/runs/how_many.pth
```

`--data_dirs` is `":"`-separated and accepts a mix of `.h5` files and
directories.

## Conventions to keep

- **Checkpoints are dicts, not raw state dicts.** They pack the training-time
  `input_shape`, `input_key`, `num_classes`, and (for `how_many`) the
  `min_pulses`/`max_pulses` range. Never save `torch.save(model.state_dict(),
  ...)` on its own — eval scripts read those extra keys.
- **`--input_key` is the switch between clean-truth / raw / denoised runs.**
  Don't hard-code it in training scripts; keep it as a CLI flag so a
  clean-vs-denoised comparison is one env-var change in the `.sh` file.
- **Figures land next to the code by default** (`./figures/` beside the
  entrypoint), unless `--figures_dir` overrides.
