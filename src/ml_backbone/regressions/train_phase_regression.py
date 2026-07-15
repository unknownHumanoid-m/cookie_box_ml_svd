# -*- coding: utf-8 -*-
"""
Phase regressor over SVD features, with an optional count-hint from a frozen
`SimpleMLP` "how-many-pulses" classifier concatenated to the input.

Input: SVD coefficients from an h5 file (same layout as `train_how_many.py`
consumes), read from `--input_key` per shot group.
Target: `grp.attrs["phases"]`. Single-pulse mode regresses the first phase;
double-pulse mode regresses |phi0 - phi1| / (2*pi).

Example
-------
    python3 train_phase_regression.py \
        --data_dirs /path/to/svd2d_r8_train/ \
        --input_key Ypdf \
        --pulse_mode single \
        --how_many_ckpt /path/to/how_many_svd2d_r8.pth \
        --save_model phase_regress_svd2d_r8.pth
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

# Reuse the "how-many" MLP definition instead of duplicating it.
current_dir = os.path.dirname(os.path.abspath(__file__))
denoise_dir = os.path.abspath(os.path.join(current_dir, "../..", "denoising"))
sys.path.append(denoise_dir)
from train_how_many import SimpleMLP, EarlyStopping  # noqa: E402


def load_phase_h5(paths, input_key, pulse_mode, min_pulses, max_pulses):
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

    inputs, targets, counts = [], [], []
    for path in files:
        print(f"phase_regress: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n < min_pulses or n > max_pulses:
                    continue
                phases = np.asarray(grp.attrs["phases"], dtype=np.float32)
                if pulse_mode == "single":
                    if n < 1:
                        continue
                    y = np.asarray([phases[0]], dtype=np.float32)
                elif pulse_mode == "double":
                    if n < 2:
                        continue
                    diff = np.abs(phases[0] - phases[1]) / (2.0 * np.pi)
                    y = np.asarray([diff], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown pulse_mode: {pulse_mode}")
                inputs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                targets.append(y)
                counts.append(n - min_pulses)

    if not inputs:
        raise RuntimeError(
            f"No shots matched pulse_mode={pulse_mode}, "
            f"npulses in [{min_pulses}, {max_pulses}] in {files}"
        )

    x = torch.from_numpy(np.stack(inputs))
    y = torch.from_numpy(np.stack(targets))
    c = torch.tensor(counts, dtype=torch.long)
    input_shape = tuple(x.shape[1:])
    num_classes = max_pulses - min_pulses + 1
    print(
        f"phase_regress: loaded {len(y)} shots, input_key={input_key}, "
        f"shape={input_shape}, pulse_mode={pulse_mode}"
    )
    return TensorDataset(x, y, c), input_shape, num_classes


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def load_how_many(ckpt_path, device):
    """Rehydrate a frozen SimpleMLP from the checkpoint train_how_many.py writes."""
    ckpt = torch.load(ckpt_path, map_location=device)
    input_shape = tuple(ckpt["input_shape"])
    num_classes = int(ckpt["num_classes"])
    model = SimpleMLP(input_size=int(np.prod(input_shape)), num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, num_classes


def build_regressor_input(svd, how_many_model, hint_kind):
    """Flatten SVD, optionally concat count-hint from the frozen classifier."""
    flat = svd.view(svd.size(0), -1)
    if how_many_model is None or hint_kind == "none":
        return flat
    with torch.no_grad():
        logits = how_many_model(svd)
    if hint_kind == "softmax":
        hint = F.softmax(logits, dim=1)
    elif hint_kind == "argmax":
        hint = F.one_hot(logits.argmax(1), num_classes=logits.size(1)).float()
    elif hint_kind == "logits":
        hint = logits
    else:
        raise ValueError(f"Unknown hint_kind: {hint_kind}")
    return torch.cat([flat, hint], dim=1)


def train_one_epoch(model, loader, how_many_model, hint_kind, optimizer, criterion, device):
    model.train()
    running = 0.0
    n_batches = 0
    for svd, y, _c in loader:
        svd, y = svd.to(device), y.to(device)
        x = build_regressor_input(svd, how_many_model, hint_kind)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
        n_batches += 1
    return running / max(1, n_batches)


def evaluate(model, loader, how_many_model, hint_kind, criterion, device):
    model.eval()
    running = 0.0
    n_batches = 0
    with torch.no_grad():
        for svd, y, _c in loader:
            svd, y = svd.to(device), y.to(device)
            x = build_regressor_input(svd, how_many_model, hint_kind)
            pred = model(x)
            running += criterion(pred, y).item()
            n_batches += 1
    return running / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--input_key", type=str, default="Ypdf")
    parser.add_argument("--pulse_mode", type=str, default="single",
                        choices=("single", "double"))
    parser.add_argument("--min_pulses", type=int, default=1)
    parser.add_argument("--max_pulses", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=str, default="256,128,64",
                        help="Comma-separated FC hidden sizes.")
    parser.add_argument("--how_many_ckpt", type=str, default=None,
                        help="Path to a train_how_many.py checkpoint. "
                             "If unset, no count hint is used.")
    parser.add_argument("--hint_kind", type=str, default="softmax",
                        choices=("softmax", "argmax", "logits", "none"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_paths = [d for d in args.data_dirs.split(":") if d]
    dataset, input_shape, num_classes = load_phase_h5(
        data_paths, args.input_key, args.pulse_mode, args.min_pulses, args.max_pulses,
    )

    val_size = int(args.val_frac * len(dataset))
    train_size = len(dataset) - val_size
    print(f"Train size: {train_size}, Val size: {val_size}")
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    how_many_model = None
    hint_dim = 0
    if args.how_many_ckpt and args.hint_kind != "none":
        how_many_model, hm_classes = load_how_many(args.how_many_ckpt, device)
        hint_dim = hm_classes
        print(f"Count-hint enabled: kind={args.hint_kind}, dim={hint_dim}")
    else:
        print("Count-hint disabled.")

    svd_dim = int(np.prod(input_shape))
    input_size = svd_dim + hint_dim
    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    model = MLPRegressor(input_size=input_size, hidden=hidden,
                         dropout=args.dropout).to(device)
    print(f"Regressor input size: {input_size} (svd={svd_dim}, hint={hint_dim})")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early = EarlyStopping(patience=args.patience)

    train_losses, val_losses = [], []
    actual_epochs = 0

    start_time = time.time()
    print("Training the phase regressor...")
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl = train_one_epoch(model, train_loader, how_many_model, args.hint_kind,
                             optimizer, criterion, device)
        vl = evaluate(model, val_loader, how_many_model, args.hint_kind,
                      criterion, device)
        train_losses.append(tl); val_losses.append(vl)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train MSE: {tl:.6f} | Val MSE: {vl:.6f}")

        early(vl, model)
        if early.early_stop:
            print("Early stopping triggered! Cutting training short.")
            break

    if early.best_weights is not None:
        model.load_state_dict(early.best_weights)
        print(f"Rolled back to best weights (Best Val Loss: {early.best_loss:.6f})")

    print(f"Training time: {time.time() - start_time:.2f}s")

    if args.save_model is not None:
        save_dir = args.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.save_model)
        torch.save({
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "svd_dim": svd_dim,
            "hint_dim": hint_dim,
            "hint_kind": args.hint_kind,
            "hidden": hidden,
            "pulse_mode": args.pulse_mode,
            "min_pulses": args.min_pulses,
            "max_pulses": args.max_pulses,
            "input_key": args.input_key,
            "how_many_ckpt": args.how_many_ckpt,
        }, save_path)
        print(f"Best model saved to {save_path}")

    epochs_range = range(1, actual_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.title(f"Phase Regressor MSE | mode={args.pulse_mode} | hint={args.hint_kind}")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    plt.tight_layout()

    if args.save_model is not None:
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures"
        )
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f"training_data_for_{args.save_model}.png")
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
