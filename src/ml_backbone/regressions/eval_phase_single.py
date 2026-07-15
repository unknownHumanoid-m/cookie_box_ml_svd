# -*- coding: utf-8 -*-
"""
Evaluate the ResNet18 single-pulse phase regressor on autoencoder-denoised
2D-SVD h5 files, using the how_many MLP as an upstream filter.

Pipeline
--------
    (Ypdf_denoised, npulses truth) -> how_many MLP -> {1, 2, 3, >=4}
    Only shots the classifier calls "1" are fed to this regressor.
    Shots called 3 or >=4 are counted and logged, since no regressor exists
    yet for those classes.

Example
-------
    python3 eval_phase_single.py \
        --data_dirs /path/to/svd2d_r8_autoencoder/test \
        --model_path /path/to/phase_single_resnet18.pth \
        --how_many_path /path/to/how_many_denoised_svd2d_r8.pth
"""

import os
import sys
import time
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resnets import resnet18


def load_all_h5(paths, input_key):
    """Load every shot (regardless of npulses) as (image, phases_padded, npulses)."""
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
        raise RuntimeError(f"No .h5 files under {paths}")

    imgs, first_phase, npulses = [], [], []
    for path in files:
        print(f"eval_phase_single: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n < 1:
                    continue
                imgs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                # phases attr is stored as fraction of a turn ([0, 1]) — convert to radians.
                first_phase.append(float(grp.attrs["phases"][0]) * 2 * np.pi)
                npulses.append(n)
    x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
    p = torch.tensor(first_phase, dtype=torch.float32)
    n = torch.tensor(npulses, dtype=torch.long)
    print(f"eval_phase_single: loaded {len(n)} shots, shape={tuple(x.shape[1:])}")
    return TensorDataset(x, p, n)


class SimpleMLP(nn.Module):
    """Mirrors train_how_many.SimpleMLP (dropout removed for eval)."""
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


def logits_to_phase(logits, max_val=2 * np.pi):
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    indices = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
    phase = (probs * indices).sum(dim=1) * (max_val / num_classes)
    return phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True,
                        help=".pth from train_phase_single.py")
    parser.add_argument("--how_many_path", type=str, required=True,
                        help=".pth from train_how_many.py")
    parser.add_argument("--input_key", type=str, default=None,
                        help="Override the input_key stored in the checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--identifier", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------ regressor
    ckpt = torch.load(args.model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"{args.model_path} is not a train_phase_single checkpoint")
    input_key = args.input_key or ckpt["input_key"]
    num_classes = int(ckpt["num_classes"])
    model = resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"phase-single regressor loaded: num_classes={num_classes}, input_key={input_key}")

    # ------------------------------------------------------------ how_many
    hm_ckpt = torch.load(args.how_many_path, map_location=device)
    if not isinstance(hm_ckpt, dict) or "state_dict" not in hm_ckpt:
        raise RuntimeError(f"{args.how_many_path} is not a train_how_many checkpoint")
    hm_input_shape = tuple(hm_ckpt["input_shape"])
    hm_num_classes = int(hm_ckpt["num_classes"])
    hm_min = int(hm_ckpt["min_pulses"])
    hm_max = int(hm_ckpt["max_pulses"])
    hm_input_key = hm_ckpt["input_key"]
    if hm_input_key != input_key:
        print(f"WARNING: how_many was trained on {hm_input_key}, regressor on {input_key}")
    hm_model = SimpleMLP(int(np.prod(hm_input_shape)), hm_num_classes).to(device)
    hm_model.load_state_dict(hm_ckpt["state_dict"])
    hm_model.eval()
    print(f"how_many MLP loaded: classes {hm_min}..>={hm_max} ({hm_num_classes})")

    # ------------------------------------------------------------ data
    data_paths = [d for d in args.data_dirs.split(":") if d]
    ds = load_all_h5(data_paths, input_key)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # ------------------------------------------------------------ run
    start = time.time()
    hm_pred_counts = torch.zeros(hm_num_classes, dtype=torch.long)
    hm_true_counts = torch.zeros(hm_num_classes, dtype=torch.long)
    true_phases_kept, pred_phases_kept = [], []
    n_total = 0

    with torch.no_grad():
        for imgs, phases_true, npulses in loader:
            imgs = imgs.to(device)
            n_total += imgs.size(0)

            # how_many uses the same input; ignores extra channel dim.
            hm_logits = hm_model(imgs.squeeze(1))
            hm_pred = hm_logits.argmax(dim=1).cpu()

            # bin the how_many predictions and truths (truth clipped to hm_max like training)
            hm_pred_counts.scatter_add_(0, hm_pred, torch.ones_like(hm_pred))
            npulses_clipped = torch.clamp(npulses, max=hm_max) - hm_min
            hm_true_counts.scatter_add_(0, npulses_clipped,
                                        torch.ones_like(npulses_clipped))

            # Filter: only shots the classifier calls "1" (which is class index 1-hm_min = 0).
            target_class = 1 - hm_min
            keep = hm_pred == target_class
            if keep.any():
                sub_imgs = imgs[keep]
                logits = model(sub_imgs)
                pred_phase = logits_to_phase(logits).cpu()
                true_phases_kept.append(phases_true[keep])
                pred_phases_kept.append(pred_phase)

    elapsed = time.time() - start
    true_arr = torch.cat(true_phases_kept).numpy() if true_phases_kept else np.array([])
    pred_arr = torch.cat(pred_phases_kept).numpy() if pred_phases_kept else np.array([])

    # ------------------------------------------------------------ report
    print(f"\nTotal shots read: {n_total}")
    print("how_many predicted distribution:")
    class_labels = [str(hm_min + i) for i in range(hm_num_classes)]
    class_labels[-1] = f">={hm_max}"
    for lbl, cnt, truth in zip(class_labels, hm_pred_counts.tolist(), hm_true_counts.tolist()):
        print(f"  class {lbl}: pred={cnt:>7d}   true={truth:>7d}")

    kept = len(true_arr)
    print(f"\nShots routed to single-pulse regressor: {kept} / {n_total}")
    if kept == 0:
        print("Nothing to plot; exiting.")
        return
    mse = float(np.mean((pred_arr - true_arr) ** 2))
    print(f"MSE(phase) = {mse:.6f}")
    print(f"Missing regressors: 3-pulse (pred={hm_pred_counts[3-hm_min].item()}), "
          f">={hm_max}-pulse (pred={hm_pred_counts[-1].item()})")
    print(f"Elapsed: {elapsed:.1f}s")

    # ------------------------------------------------------------ figures
    identifier = args.identifier or os.path.splitext(os.path.basename(args.model_path))[0]
    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(true_arr, pred_arr, s=4, alpha=0.4, label="Predicted vs True")
    plt.plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", label="Ideal")
    plt.xlabel("True phase [rad]"); plt.ylabel("Predicted phase [rad]")
    plt.title(
        f"Single-pulse phase | input_key={input_key} | "
        f"routed shots: {kept}/{n_total} | MSE={mse:.4f}"
    )
    plt.legend(); plt.grid(True); plt.tight_layout()
    p_out = os.path.join(figures_dir, f"true_vs_pred_{identifier}.png")
    plt.savefig(p_out); plt.close()
    print(f"True-vs-Pred saved to {p_out}")


if __name__ == "__main__":
    main()
