# -*- coding: utf-8 -*-
"""
Evaluate the ResNet18 two-pulse phase-diff regressor on autoencoder-denoised
2D-SVD h5 files, using the how_many MLP as an upstream filter.

Only shots the classifier calls "2" are fed to this regressor. Everything else
is counted and logged.

Example
-------
    python3 eval_phase_two.py \
        --data_dirs /path/to/svd2d_r8_autoencoder/test \
        --model_path /path/to/phase_two_resnet18.pth \
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
    """Load every shot (any npulses>=1) as (image, wrapped_diff, npulses).

    For npulses != 2 the wrapped_diff field is NaN and is ignored downstream.
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
        raise RuntimeError(f"No .h5 files under {paths}")

    imgs, diffs, npulses = [], [], []
    for path in files:
        print(f"eval_phase_two: reading {path}")
        with h5py.File(path, "r") as f:
            for shot in f.keys():
                grp = f[shot]
                n = int(grp.attrs["npulses"])
                if n < 1:
                    continue
                imgs.append(np.asarray(grp[input_key][()], dtype=np.float32))
                if n == 2:
                    # phases attr is stored as fraction of a turn ([0, 1]) — convert to radians.
                    phis = np.asarray(grp.attrs["phases"], dtype=np.float32) * 2 * np.pi
                    d = float(np.arccos(np.cos(phis[0] - phis[1])))
                else:
                    d = float("nan")
                diffs.append(d)
                npulses.append(n)
    x = torch.from_numpy(np.stack(imgs)).unsqueeze(1)
    d = torch.tensor(diffs, dtype=torch.float32)
    n = torch.tensor(npulses, dtype=torch.long)
    print(f"eval_phase_two: loaded {len(n)} shots, shape={tuple(x.shape[1:])}")
    return TensorDataset(x, d, n)


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


def logits_to_diff(logits, max_val=np.pi):
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    indices = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
    return (probs * indices).sum(dim=1) * (max_val / num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--how_many_path", type=str, required=True)
    parser.add_argument("--input_key", type=str, default=None)
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
        raise RuntimeError(f"{args.model_path} is not a train_phase_two checkpoint")
    input_key = args.input_key or ckpt["input_key"]
    num_classes = int(ckpt["num_classes"])
    model = resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"phase-two regressor loaded: num_classes={num_classes}, input_key={input_key}")

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
    true_diffs_kept, pred_diffs_kept = [], []
    misroute_no_label = 0
    n_total = 0

    with torch.no_grad():
        for imgs, diffs_true, npulses in loader:
            imgs = imgs.to(device)
            n_total += imgs.size(0)

            hm_logits = hm_model(imgs.squeeze(1))
            hm_pred = hm_logits.argmax(dim=1).cpu()
            hm_pred_counts.scatter_add_(0, hm_pred, torch.ones_like(hm_pred))
            npulses_clipped = torch.clamp(npulses, max=hm_max) - hm_min
            hm_true_counts.scatter_add_(0, npulses_clipped,
                                        torch.ones_like(npulses_clipped))

            target_class = 2 - hm_min
            keep = hm_pred == target_class
            if keep.any():
                sub_imgs = imgs[keep]
                logits = model(sub_imgs)
                pred = logits_to_diff(logits).cpu()
                sub_diffs = diffs_true[keep]
                labeled = ~torch.isnan(sub_diffs)
                misroute_no_label += int((~labeled).sum().item())
                if labeled.any():
                    true_diffs_kept.append(sub_diffs[labeled])
                    pred_diffs_kept.append(pred[labeled])

    elapsed = time.time() - start
    true_arr = torch.cat(true_diffs_kept).numpy() if true_diffs_kept else np.array([])
    pred_arr = torch.cat(pred_diffs_kept).numpy() if pred_diffs_kept else np.array([])

    # ------------------------------------------------------------ report
    print(f"\nTotal shots read: {n_total}")
    print("how_many predicted distribution:")
    class_labels = [str(hm_min + i) for i in range(hm_num_classes)]
    class_labels[-1] = f">={hm_max}"
    for lbl, cnt, truth in zip(class_labels, hm_pred_counts.tolist(), hm_true_counts.tolist()):
        print(f"  class {lbl}: pred={cnt:>7d}   true={truth:>7d}")

    kept = len(true_arr)
    print(f"\nShots routed to 2-pulse regressor (correctly-labeled): {kept} / {n_total}")
    print(f"  routed but not truly 2-pulse (no truth diff): {misroute_no_label}")
    if kept == 0:
        print("Nothing to plot; exiting.")
        return
    # true_arr is already arccos(cos(Δφ)) from load_all_h5, and pred_arr comes
    # from logits_to_diff (bins in [0, pi]); wrap explicitly so this stage is
    # symmetric with eval_phase_single.
    true_wrapped = np.arccos(np.cos(true_arr))
    pred_wrapped = np.arccos(np.cos(pred_arr))
    mse = float(np.mean((pred_wrapped - true_wrapped) ** 2))
    print(f"MSE(arccos(cos(Δφ))) = {mse:.6f}")
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
    plt.scatter(true_wrapped, pred_wrapped, s=4, alpha=0.4, label="Pred vs True")
    plt.plot([0, np.pi], [0, np.pi], "r--", label="Ideal")
    plt.xlabel("True arccos(cos(Δφ))"); plt.ylabel("Pred arccos(cos(Δφ))")
    plt.title(f"Two-pulse arccos(cos(Δφ)) | input_key={input_key} | "
              f"kept {kept}/{n_total} | MSE={mse:.4f}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    p_out = os.path.join(figures_dir, f"true_vs_pred_{identifier}.png")
    plt.savefig(p_out); plt.close()
    print(f"Saved {p_out}")


if __name__ == "__main__":
    main()
