# -*- coding: utf-8 -*-
"""
Train the LSTM pulse-number classifier on the same 2D SVD h5 files as
train_how_many.py so the two models can be compared head-to-head.

Reads `Ypdf` (16x512) from each h5 group, transposes to (512, 16) so the LSTM
sees a 512-step time series of 16-dim vectors, and remaps npulses to
0..(max_pulses - min_pulses).

Emits the same 3-panel training-curve PNG (CE loss / MSE / accuracy) that
train_how_many.py produces, into --figures_dir.
"""

import os
import copy
import argparse
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from lstm_pulseNum_classifier import CustomLSTMClassifier


def load_h5_for_lstm(paths, input_key, min_pulses, max_pulses):
    """Preload every h5 group whose npulses is in [min_pulses, max_pulses].
    Returns (TensorDataset(x, y), seq_len, feat_dim, num_classes).

    Inputs are shaped (seq_len=16, feat_dim=512) — 16 detector-angle steps of
    512-bin spectra, matching the original repo's LSTM setup.
    Labels are integer classes 0..(max-min) (CrossEntropyLoss compatible).
    """
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

    inputs, labels = [], []
    for path in files:
        print(f"lstm: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n < min_pulses or n > max_pulses:
                    continue
                arr = np.asarray(grp[input_key][()], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(
                        f"{path}:{shot} {input_key} has shape {arr.shape}, expected 2D"
                    )
                # Ximg/Ypdf are (16, 512). Feed the LSTM as
                # (seq_len=16, feat_dim=512): 16 detector-angle steps of
                # 512-bin spectra, matching the original repo's LSTM setup.
                if arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                inputs.append(arr)
                labels.append(n - min_pulses)
    if not inputs:
        raise RuntimeError(
            f"No shots with npulses in [{min_pulses}, {max_pulses}] in {files}"
        )

    x = torch.from_numpy(np.stack(inputs))
    y = torch.tensor(labels, dtype=torch.long)
    seq_len, feat_dim = x.shape[1], x.shape[2]
    num_classes = max_pulses - min_pulses + 1
    print(
        f"lstm: loaded {len(y)} shots, input_key={input_key}, "
        f"shape=(seq_len={seq_len}, feat_dim={feat_dim}), "
        f"classes={num_classes} (npulses {min_pulses}..{max_pulses})"
    )
    return TensorDataset(x, y), seq_len, feat_dim, num_classes


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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def mse_metric(logits, labels):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    return F.mse_loss(probs, one_hot, reduction="mean")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, torch.float32), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mse += mse_metric(logits, y).item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / len(loader), running_mse / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, torch.float32), y.to(device)
            logits = model(x)
            running_loss += criterion(logits, y).item()
            running_mse += mse_metric(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / len(loader), running_mse / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--input_key", type=str, default="Ypdf")
    parser.add_argument("--min_pulses", type=int, default=1)
    parser.add_argument("--max_pulses", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None,
                        help="Filename (relative to --save_dir) for the .pth.")
    parser.add_argument("--figures_dir", type=str, default=None,
                        help="Directory for the training-curve PNG. "
                             "Defaults to ./figures/ next to this file.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_paths = [d for d in args.data_dirs.split(":") if d]
    dataset, seq_len, feat_dim, num_classes = load_h5_for_lstm(
        data_paths, args.input_key, args.min_pulses, args.max_pulses,
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

    model = CustomLSTMClassifier(
        input_size=feat_dim,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        num_classes=num_classes,
        bidirectional=True,
        fc_layers=[32, 64],
        dropout_p=args.dropout,
        lstm_dropout=args.lstm_dropout,
        layer_norm=False,
        ignore_output_layer=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience)

    train_losses, train_accs, train_mses = [], [], []
    val_losses, val_accs, val_mses = [], [], []
    actual_epochs = 0

    start_time = time.time()
    print("Training the LSTM pulse-number classifier...")
    for epoch in range(args.epochs):
        actual_epochs += 1
        tl, tm, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, vm, va = evaluate(model, val_loader, criterion, device)

        train_losses.append(tl); train_mses.append(tm); train_accs.append(ta)
        val_losses.append(vl);   val_mses.append(vm);   val_accs.append(va)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"| Train Loss: {tl:.4f}, MSE: {tm:.4f}, Acc: {ta:.2f}% "
              f"| Val Loss: {vl:.4f}, MSE: {vm:.4f}, Acc: {va:.2f}%")

        early_stopping(vl, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Cutting training short.")
            break

    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
        print(f"Rolled back to best weights (Best Val Loss: {early_stopping.best_loss:.4f})")

    print(f"Training time: {time.time() - start_time:.2f}s")

    if args.save_model is not None:
        save_dir = args.save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, args.save_model)
        torch.save({
            "state_dict": model.state_dict(),
            "seq_len": seq_len,
            "feat_dim": feat_dim,
            "num_classes": num_classes,
            "min_pulses": args.min_pulses,
            "max_pulses": args.max_pulses,
            "input_key": args.input_key,
            "hidden_size": args.hidden_size,
            "num_lstm_layers": args.num_lstm_layers,
            "dropout": args.dropout,
            "lstm_dropout": args.lstm_dropout,
        }, save_path)
        print(f"Best model saved to {save_path}")

    # --------------------------------------------------------------------
    # 3-panel training curve (same layout as train_how_many.py)
    # --------------------------------------------------------------------
    epochs_range = range(1, actual_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.suptitle(
        f"LSTM Pulse-Number Classifier | input_key={args.input_key} "
        f"| classes {args.min_pulses}..{args.max_pulses}",
        fontsize=14,
    )
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")
    plt.title("CE Loss"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_mses, label="Train")
    plt.plot(epochs_range, val_mses, label="Val")
    plt.title("MSE"); plt.xlabel("Epoch"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_accs, label="Train")
    plt.plot(epochs_range, val_accs, label="Val")
    plt.title("Accuracy (%)"); plt.xlabel("Epoch"); plt.legend()

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
