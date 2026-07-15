# -*- coding: utf-8 -*-
"""
Straight-through training script for the 1D-SVD autoencoder.

Mirrors src/denoising/ximg_to_ypdf_autoencoder_straight_training.py but
uses the segmented-MLP model (see svd1d_autoencoder.py) instead of a 2D
conv autoencoder, and runs on the flat 1D-SVD Ximg/Ypdf vectors.

Config comes from env vars (set by the .sh launcher) with sensible
defaults for interactive debugging.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchinfo import summary

from svd1d_autoencoder import SVD1D_Autoencoder, SegmentNormalizedMSE, DEFAULT_H, DEFAULT_W, DEFAULT_R

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, "..", "ml_backbone"))
sys.path.insert(0, utils_dir)
from utils import DataMilking_HalfAndHalf, CustomScheduler


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available!")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    default_train_dirs = (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/"
        "mrco_h5_svd/1d_svd_r8_train"
    )
    train_dirs_env = os.environ.get("TRAIN_DATA_DIRS", default_train_dirs)
    datapaths = [d for d in train_dirs_env.split(":") if d]
    print(f"Training on data dirs: {datapaths}")

    H = int(os.environ.get("SVD_H", DEFAULT_H))
    W = int(os.environ.get("SVD_W", DEFAULT_W))
    R = int(os.environ.get("SVD_R", DEFAULT_R))
    expected_L = H * R + R + R * W
    print(f"SVD layout: H={H}, W={W}, r={R}, expected L={expected_L}")

    data = DataMilking_HalfAndHalf(
        root_dirs=datapaths,
        pulse_handler=None,
        input_name="Ximg",
        labels=["Ypdf"],
        transform=None,
    )
    print(f"Total shots: {len(data)}")

    train_size = int(0.8 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size
    print(f"Train: {train_size}  Val: {val_size}  Test: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        data, [train_size, val_size, test_size]
    )

    batch_size = int(os.environ.get("BATCH_SIZE", 128))
    num_workers = int(os.environ.get("NUM_WORKERS", 8))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SVD1D_Autoencoder(H=H, W=W, r=R)

    criterion = SegmentNormalizedMSE(H=H, W=W, r=R)
    lr = float(os.environ.get("LR", 5e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_epochs = int(os.environ.get("EPOCHS", 200))
    scheduler = CustomScheduler(
        optimizer,
        patience=5,
        early_stop_patience=9,
        cooldown=2,
        lr_reduction_factor=0.5,
        max_num_epochs=max_epochs,
        improvement_percentage=0.001,
    )

    default_model_save_dir = (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/"
        "svd1d_r8_autoencoder/"
    )
    model_save_dir = os.environ.get("MODEL_SAVE_DIR", default_model_save_dir)
    figures_dir = os.environ.get(
        "FIGURES_DIR", os.path.join(current_dir, "figures")
    )
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    identifier = os.environ.get("MODEL_IDENTIFIER", "svd1d_r8_autoencoder")

    print(f"Model save dir: {model_save_dir}")
    print(f"Figures dir:    {figures_dir}")
    print(f"Identifier:     {identifier}")

    model.to(device)
    print(summary(model, input_size=(1, expected_L)))
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")

    best_model, best_epoch, final_train, final_val, best_val = model.train_model(
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        model_save_dir,
        identifier,
        device,
        checkpoints_enabled=True,
        resume_from_checkpoint=False,
        max_epochs=max_epochs,
        figures_dir=figures_dir,
    )

    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    with open(results_file, "w") as f:
        f.write("1D-SVD Autoencoder Training Results\n")
        f.write("===================================\n")
        f.write(f"Data Path: {datapaths}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write(f"Figures Directory: {figures_dir}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Max Epochs: {max_epochs}\n")
        f.write(f"Best epoch: {best_epoch + 1}\n")
        f.write(f"Final train loss: {final_train:.10f}\n")
        f.write(f"Final val loss:   {final_val:.10f}\n")
        f.write(f"Best val loss:    {best_val:.10f}\n")
        f.write(f"SVD layout: H={H}, W={W}, r={R}\n")

    # --- Diagnostics: reconstruct 2D images from a few val shots ------------
    try:
        from diagnostics_svd1d import plot_denoised_reconstructions
        model.eval()
        plot_denoised_reconstructions(
            model=model,
            dataset=val_dataset,
            device=device,
            figures_dir=figures_dir,
            identifier=identifier,
            n_examples=6,
            H=H, W=W, r=R,
        )
    except Exception as e:
        print(f"Diagnostics failed (non-fatal): {e}")

    print(f"Training completed. Results at {results_file}")


if __name__ == "__main__":
    main()
