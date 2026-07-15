# -*- coding: utf-8 -*-
"""
Evaluate the "how many pulses" MLP on 1D-SVD h5 files.

Loads the checkpoint written by train_how_many.py (which stores input_shape,
num_classes, and the min/max pulses range), so architecture and label bounds
are pulled from the file. Same MLP as the 2D version; the only difference
is the sample-visualization block, which switches from `imshow` to `plot`
when the per-shot input is 1D.

Example
-------
    python3 evaluate_how_many.py \
        --data_dirs /path/to/mrco_h5_svd/1d_svd_test/ \
        --model_path /path/to/how_many_svd1d_r8.pth
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_how_many_h5(paths, input_key, min_pulses, max_pulses):
    """Preload every h5 group whose npulses is in [min_pulses, max_pulses]
    into a single TensorDataset. Labels are remapped to 0..(max-min).
    Accepts a mix of directories and .h5 files.
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
        print(f"how_many: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n < min_pulses:
                    continue
                n = min(n, max_pulses)
                inputs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                labels.append(n - min_pulses)
    if not inputs:
        raise RuntimeError(
            f"No shots with npulses in [{min_pulses}, {max_pulses}] in {files}"
        )

    x = torch.from_numpy(np.stack(inputs))
    y = torch.tensor(labels, dtype=torch.long)
    input_shape = tuple(x.shape[1:])
    num_classes = max_pulses - min_pulses + 1
    print(
        f"how_many: loaded {len(y)} shots, input_key={input_key}, "
        f"shape={input_shape}, classes={num_classes} "
        f"(npulses {min_pulses}..{max_pulses})"
    )
    return TensorDataset(x, y), input_shape, num_classes


class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def plot_confusion_matrix(cm, save_path, accuracy, elapsed, class_labels, input_key):
    """Plot and save an NxN confusion matrix (rows = true, cols = pred)."""
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 3))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted # Pulses')
    ax.set_ylabel('True # Pulses')
    ax.set_title(
        f'Confusion Matrix | input_key={input_key}\n'
        f'Acc: {accuracy:.2f}%  |  Time: {elapsed:.3f}s'
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', type=str, required=True,
                        help="':'-separated list of h5 files or directories.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained .pth (from train_how_many-Copy1.py).")
    parser.add_argument('--input_key', type=str, default=None,
                        help="Override the input_key stored in the checkpoint.")
    parser.add_argument('--min_pulses', type=int, default=None,
                        help="Override the min_pulses stored in the checkpoint.")
    parser.add_argument('--max_pulses', type=int, default=None,
                        help="Override the max_pulses stored in the checkpoint.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--figures_dir', type=str, default=None,
                        help="Directory for output PNGs. Defaults to ./figures/.")
    parser.add_argument('--identifier', type=str, default=None,
                        help="Prefix for output figure filenames. "
                             "Defaults to the .pth basename (no extension).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.model_path, map_location=device)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise RuntimeError(
            f"{args.model_path} is not a train_how_many checkpoint "
            f"(expected dict with 'state_dict' + metadata)."
        )

    input_key = args.input_key or ckpt['input_key']
    min_pulses = args.min_pulses if args.min_pulses is not None else ckpt['min_pulses']
    max_pulses = args.max_pulses if args.max_pulses is not None else ckpt['max_pulses']
    input_shape = tuple(ckpt['input_shape'])
    num_classes = int(ckpt['num_classes'])
    input_size = int(np.prod(input_shape))

    print(f"input_key={input_key}, npulses range=[{min_pulses}, {max_pulses}], "
          f"input_shape={input_shape}, num_classes={num_classes}")

    model = SimpleMLP(input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data_paths = [d for d in args.data_dirs.split(":") if d]
    test_dataset, _, _ = load_how_many_h5(
        data_paths, input_key, min_pulses, max_pulses,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    start_time = time.time()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

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
    class_labels[-1] = f"≥{max_pulses}"
    cm_path = os.path.join(figures_dir, f"confusion_matrix_for_{identifier}.png")
    plot_confusion_matrix(cm, cm_path, accuracy, elapsed, class_labels, input_key)

    # --------------------------------------------------------------------
    # Sample visualization: show a few 2D images with true/pred pulse counts.
    # --------------------------------------------------------------------
    num_samples = min(12, len(test_dataset))
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    indices = random.sample(range(len(test_dataset)), num_samples)

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = np.atleast_2d(axs).ravel()

    fig.suptitle(
        f"How Many Pulses | input_key={input_key} "
        f"| Test Acc: {accuracy:.2f}% | Time: {elapsed:.3f}s "
        f"| npulses {min_pulses}..{max_pulses}",
        fontsize=14,
    )

    for i, idx in enumerate(indices):
        img, label = test_dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
        true_pulses = int(label.item()) + min_pulses
        pred_pulses = int(logits.argmax(dim=1).cpu().item()) + min_pulses
        true_str = f"≥{max_pulses}" if true_pulses == max_pulses else str(true_pulses)
        pred_str = f"≥{max_pulses}" if pred_pulses == max_pulses else str(pred_pulses)

        arr = img.numpy()
        if arr.ndim == 1:
            axs[i].plot(arr)
            axs[i].set_xticks([]); axs[i].set_yticks([])
        else:
            axs[i].imshow(arr, aspect='auto', cmap='magma_r')
            axs[i].axis('off')
        axs[i].set_title(f"True: {true_str}\nPred: {pred_str}")

    for ax in axs[num_samples:]:
        ax.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, f"images_for_{identifier}.png")
    plt.savefig(fig_path)
    print(f"Evaluation figure saved to {fig_path}")


if __name__ == "__main__":
    main()
