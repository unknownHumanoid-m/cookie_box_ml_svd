# -*- coding: utf-8 -*-
"""
Evaluate the LSTM pulse-number classifier on 2D SVD h5 files, matching the
figure format of evaluate_how_many.py so the two classifiers can be compared
head-to-head.

Loads the metadata-rich checkpoint written by
lstm_pulseNum_classifier_straight_training.py (state_dict + hyperparameters +
data spec) and emits `confusion_matrix_for_<id>.png` and `images_for_<id>.png`
in the same layout as how_many.
"""

import os
import argparse
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_pulseNum_classifier import CustomLSTMClassifier


def load_h5_for_lstm(paths, input_key, min_pulses, max_pulses):
    """Same loader used in training — returns TensorDataset with inputs
    shaped (seq_len=16, feat_dim=512) and integer labels 0..(max-min)."""
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
                # Match training: (seq_len=16, feat_dim=512).
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
    return TensorDataset(x, y), x.shape[1], x.shape[2]


def plot_confusion_matrix(cm, save_path, accuracy, elapsed, class_labels):
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 3))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted # Pulses")
    ax.set_ylabel("True # Pulses")
    ax.set_title(f"Confusion Matrix\nAcc: {accuracy:.2f}%  |  Time: {elapsed:.3f}s")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .pth (from the training script).")
    parser.add_argument("--input_key", type=str, default=None,
                        help="Override the input_key stored in the checkpoint.")
    parser.add_argument("--min_pulses", type=int, default=None)
    parser.add_argument("--max_pulses", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--identifier", type=str, default=None,
                        help="Prefix for output figure filenames. "
                             "Defaults to the .pth basename (no extension).")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            f"{args.model_path} is not a straight_training checkpoint "
            f"(expected dict with 'state_dict' + metadata)."
        )

    input_key = args.input_key or ckpt["input_key"]
    min_pulses = args.min_pulses if args.min_pulses is not None else ckpt["min_pulses"]
    max_pulses = args.max_pulses if args.max_pulses is not None else ckpt["max_pulses"]
    seq_len = int(ckpt["seq_len"])
    feat_dim = int(ckpt["feat_dim"])
    num_classes = int(ckpt["num_classes"])

    print(f"input_key={input_key}, npulses range=[{min_pulses}, {max_pulses}], "
          f"seq_len={seq_len}, feat_dim={feat_dim}, num_classes={num_classes}")

    model = CustomLSTMClassifier(
        input_size=feat_dim,
        hidden_size=int(ckpt["hidden_size"]),
        num_lstm_layers=int(ckpt["num_lstm_layers"]),
        num_classes=num_classes,
        bidirectional=True,
        fc_layers=[32, 64],
        dropout_p=float(ckpt["dropout"]),
        lstm_dropout=float(ckpt["lstm_dropout"]),
        layer_norm=False,
        ignore_output_layer=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_paths = [d for d in args.data_dirs.split(":") if d]
    test_dataset, ds_seq, ds_feat = load_h5_for_lstm(
        data_paths, input_key, min_pulses, max_pulses,
    )
    if (ds_seq, ds_feat) != (seq_len, feat_dim):
        raise RuntimeError(
            f"Test data shape (seq_len={ds_seq}, feat_dim={ds_feat}) "
            f"does not match the checkpoint (seq_len={seq_len}, feat_dim={feat_dim})."
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    start_time = time.time()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, torch.float32)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    elapsed = time.time() - start_time
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Time to evaluate test cases: {elapsed:.3f}s")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    identifier = args.identifier or os.path.splitext(os.path.basename(args.model_path))[0]
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)

    class_labels = [str(min_pulses + i) for i in range(num_classes)]
    cm_path = os.path.join(figures_dir, f"confusion_matrix_for_{identifier}.png")
    plot_confusion_matrix(cm, cm_path, accuracy, elapsed, class_labels)

    # --------------------------------------------------------------------
    # Sample visualization: images with true/pred pulse counts.
    # Stored as (16, 512); display as-is to match evaluate_how_many.py.
    # --------------------------------------------------------------------
    num_samples = min(12, len(test_dataset))
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = np.atleast_2d(axs).ravel()
    fig.suptitle(
        f"LSTM Pulse-Number Classifier | input_key={input_key} "
        f"| Test Acc: {accuracy:.2f}% | Time: {elapsed:.3f}s "
        f"| npulses {min_pulses}..{max_pulses}",
        fontsize=14,
    )
    for i, idx in enumerate(indices):
        x, label = test_dataset[idx]
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device, torch.float32))
        true_pulses = int(label.item()) + min_pulses
        pred_pulses = int(logits.argmax(dim=1).cpu().item()) + min_pulses
        # x is already (16, 512) — display as-is.
        axs[i].imshow(x.numpy(), aspect="auto", cmap="magma_r")
        axs[i].axis("off")
        axs[i].set_title(f"True: {true_pulses}\nPred: {pred_pulses}")
    for ax in axs[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"images_for_{identifier}.png")
    plt.savefig(fig_path)
    print(f"Evaluation figure saved to {fig_path}")


if __name__ == "__main__":
    main()
