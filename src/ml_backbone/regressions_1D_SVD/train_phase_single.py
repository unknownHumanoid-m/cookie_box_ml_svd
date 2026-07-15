# -*- coding: utf-8 -*-
"""
Train the deep-MLP single-pulse phase regressor on autoencoder-denoised
1D-SVD h5 files.

Reads a flat `Ypdf_denoised` `(L,)` vector per shot and regresses the single
phase angle (0..2*pi). Only shots with `npulses == 1` are used for training.
Downstream, the how_many MLP decides which shots go through this regressor
at inference time.

The 2D-SVD sibling (src/ml_backbone/regressions/train_phase_single.py) uses
a ResNet18; that stops making sense once the SVD is flat — there's no 2D
structure for a conv stack to exploit. This variant uses a `PhaseMLP` (a
deep fully-connected classification head over `num_classes` phase bins),
keeping the same softmax-weighted-mean decoder so the training and eval
targets are identical to the 2D path.

Example
-------
    python3 train_phase_single.py \
        --data_dirs /path/to/svd1d_r8_autoencoder/train \
        --save_dir  /path/to/runs/phase_single/ \
        --save_model phase_single_mlp.pth
"""

import os
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


class PhaseMLP(nn.Module):
    """Deep MLP head shared by the 1D-SVD phase regressors.

    Flattens the input, then a stack of Linear+ReLU+Dropout blocks, then a
    final Linear that produces `num_classes` phase bins. The caller decodes
    those logits via softmax-weighted mean (see logits_to_phase /
    logits_to_diff).
    """
    def __init__(self, input_size, num_classes,
                 hidden=(1024, 512, 512, 256, 128), dropout=0.3):
        super().__init__()
        dims = [int(input_size), *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], int(num_classes))]
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def load_single_pulse_h5(paths, input_key="Ypdf_denoised"):
    """Preload every 1-pulse shot into a TensorDataset of (image, phase)."""
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

    inputs, phases = [], []
    dropped = {"npulses!=1": 0}
    for path in files:
        print(f"phase_single: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n != 1:
                    dropped["npulses!=1"] += 1
                    continue
                img = np.asarray(grp[input_key][()], dtype=np.float32)
                inputs.append(img)
                # phases attr is stored as fraction of a turn ([0, 1]) — convert to radians.
                phases.append(float(grp.attrs["phases"][0]) * 2 * np.pi)
    if not inputs:
        raise RuntimeError(f"No 1-pulse shots found in {files}")

    x = torch.from_numpy(np.stack(inputs))  # (N, L) — PhaseMLP flattens internally
    y = torch.tensor(phases, dtype=torch.float32).unsqueeze(1)
    print(
        f"phase_single: loaded {len(y)} shots, input_key={input_key}, "
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


def logits_to_phase(logits, max_val=2 * np.pi):
    """Softmax over N phase bins -> weighted-mean phase in [0, max_val)."""
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    indices = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
    phase = (probs * indices).sum(dim=1) * (max_val / num_classes)
    return phase.unsqueeze(1)


def train_one_epoch(model, loader, opt, criterion, device, num_classes):
    model.train()
    running = 0.0
    for imgs, phases in loader:
        imgs, phases = imgs.to(device), phases.to(device)
        opt.zero_grad()
        logits = model(imgs)
        pred = logits_to_phase(logits)
        loss = criterion(pred, phases)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for imgs, phases in loader:
            imgs, phases = imgs.to(device), phases.to(device)
            logits = model(imgs)
            pred = logits_to_phase(logits)
            loss = criterion(pred, phases)
            running += loss.item()
    return running / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--input_key", type=str, default="Ypdf_denoised")
    parser.add_argument("--num_classes", type=int, default=2000,
                        help="Number of phase bins the ResNet outputs.")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Directory for the loss-curve PNG. Defaults to "
                             "./figures/ next to this file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    data_paths = [d for d in args.data_dirs.split(":") if d]
    dataset, input_shape = load_single_pulse_h5(data_paths, args.input_key)

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

    input_size = int(np.prod(input_shape))
    model = PhaseMLP(input_size=input_size, num_classes=args.num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience)

    train_losses, val_losses = [], []
    actual_epochs = 0

    start = time.time()
    print("Training single-pulse phase regressor...")
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device, args.num_classes)
        vl = evaluate(model, val_loader, criterion, device, args.num_classes)
        train_losses.append(tl)
        val_losses.append(vl)
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
            "npulses_target": 1,
            "hidden": model.hidden,
        }, save_path)
        print(f"Best model saved to {save_path}")

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)
    epochs_range = range(1, actual_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True)
    plt.title(f"Single-pulse phase regressor | input_key={args.input_key}")
    plt.tight_layout()
    fig_name = args.save_model or "phase_single"
    fig_path = os.path.join(figures_dir, f"training_curve_{os.path.splitext(fig_name)[0]}.png")
    plt.savefig(fig_path)
    print(f"Loss curve saved to {fig_path}")


if __name__ == "__main__":
    main()
