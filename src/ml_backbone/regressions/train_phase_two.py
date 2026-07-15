# -*- coding: utf-8 -*-
"""
Train the ResNet18 two-pulse phase-difference regressor on autoencoder-denoised
2D-SVD h5 files.

Target: wrapped phase difference in [0, pi], i.e. arccos(cos(phi1 - phi2)).
Using arccos(cos(.)) folds the sign/order ambiguity of the two pulses into a
single scalar, which is what the old eval scatter plots consumed.

Only shots with npulses == 2 are used for training.

Example
-------
    python3 train_phase_two.py \
        --data_dirs /path/to/svd2d_r8_autoencoder/train \
        --save_dir /path/to/runs/phase_two/ \
        --save_model phase_two_resnet18.pth
"""

import os
import sys
import copy
import time
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resnets import resnet18


def load_two_pulse_h5(paths, input_key="Ypdf_denoised"):
    """Preload every 2-pulse shot into a TensorDataset of (image, wrapped_diff)."""
    files = []
    for p in paths:
        if not p:
            continue
        if os.path.isfile(p) and p.endswith(".h5"):
            files.append(p)
        elif os.path.isdir(p):
            files.extend(
                os.path.join(p, n) for n in sorted(os.listdir(p)) if n.endswith(".h5")
            )
        else:
            raise FileNotFoundError(f"Not an .h5 file or directory: {p}")
    if not files:
        raise RuntimeError(f"No .h5 files found under {paths}")

    inputs, diffs = [], []
    dropped = {"npulses!=2": 0}
    for path in files:
        print(f"phase_two: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n != 2:
                    dropped["npulses!=2"] += 1
                    continue
                img = np.asarray(grp[input_key][()], dtype=np.float32)
                # phases attr is stored as fraction of a turn ([0, 1]) — convert to radians.
                phis = np.asarray(grp.attrs["phases"], dtype=np.float32) * 2 * np.pi
                diff_wrapped = float(np.arccos(np.cos(phis[0] - phis[1])))  # [0, pi]
                inputs.append(img)
                diffs.append(diff_wrapped)
    if not inputs:
        raise RuntimeError(f"No 2-pulse shots found in {files}")

    x = torch.from_numpy(np.stack(inputs)).unsqueeze(1)
    y = torch.tensor(diffs, dtype=torch.float32).unsqueeze(1)
    print(
        f"phase_two: loaded {len(y)} shots, input_key={input_key}, "
        f"shape={tuple(x.shape[1:])}, dropped={dropped}"
    )
    return TensorDataset(x, y), tuple(x.shape[1:])


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def logits_to_diff(logits, max_val=np.pi):
    """Softmax over N bins spanning [0, pi] -> weighted-mean wrapped diff."""
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    indices = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
    diff = (probs * indices).sum(dim=1) * (max_val / num_classes)
    return diff.unsqueeze(1)


def run_one_epoch(model, loader, opt, criterion, device, train=True):
    model.train() if train else model.eval()
    running = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, diffs in loader:
            imgs, diffs = imgs.to(device), diffs.to(device)
            if train:
                opt.zero_grad()
            logits = model(imgs)
            pred = logits_to_diff(logits)
            loss = criterion(pred, diffs)
            if train:
                loss.backward()
                opt.step()
            running += loss.item()
    return running / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True)
    parser.add_argument("--input_key", type=str, default="Ypdf_denoised")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of bins spanning [0, pi].")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42); np.random.seed(42)

    data_paths = [d for d in args.data_dirs.split(":") if d]
    dataset, input_shape = load_two_pulse_h5(data_paths, args.input_key)

    val_size = int(args.val_frac * len(dataset))
    train_size = len(dataset) - val_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = resnet18(num_classes=args.num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience)

    train_losses, val_losses = [], []
    actual_epochs = 0

    start = time.time()
    print("Training two-pulse phase-diff regressor...")
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl = run_one_epoch(model, val_loader, optimizer, criterion, device, train=False)
        train_losses.append(tl); val_losses.append(vl)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train MSE: {tl:.6f} | Val MSE: {vl:.6f}")
        early_stop(vl, model)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    if early_stop.best_weights is not None:
        model.load_state_dict(early_stop.best_weights)
        print(f"Rolled back to best weights (Best Val MSE: {early_stop.best_loss:.6f})")
    print(f"Training time: {time.time() - start:.1f}s")

    if args.save_model is not None:
        save_dir = args.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.save_model)
        torch.save({
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "num_classes": args.num_classes,
            "input_key": args.input_key,
            "npulses_target": 2,
            "target_kind": "arccos_cos_diff",
        }, save_path)
        print(f"Best model saved to {save_path}")

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, actual_epochs + 1)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
    plt.title(f"Two-pulse phase-diff regressor | input_key={args.input_key}")
    plt.tight_layout()
    fig_name = args.save_model or "phase_two"
    fig_path = os.path.join(figures_dir, f"training_curve_{os.path.splitext(fig_name)[0]}.png")
    plt.savefig(fig_path)
    print(f"Loss curve saved to {fig_path}")


if __name__ == "__main__":
    main()
