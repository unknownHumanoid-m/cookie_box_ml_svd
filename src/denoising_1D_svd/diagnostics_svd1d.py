# -*- coding: utf-8 -*-
"""
Diagnostics for the 1D-SVD autoencoder.

Reconstruction path: given a shot's 1D-SVD vector
[U_flat (H*r), S (r), V_flat (r*W)], we can rebuild the 2D image as
    img = U @ diag(S) @ V           (shape (H, W))

This module provides two helpers:

* plot_denoised_reconstructions(model, dataset, ...)
    Pull N shots from a torch Dataset, run them through a trained
    SVD1D_Autoencoder, and save a side-by-side figure per shot:
        col 0 : recon(Ximg)         -- noisy input in image space
        col 1 : recon(Ypdf)         -- clean truth in image space
        col 2 : recon(model(Ximg))  -- denoised output in image space
    Also writes the raw 1D vectors (Ximg / Ypdf / denoised) with U|S|V
    boundaries marked, so you can eyeball whether the model actually
    moved S and the segment structure.

* plot_h5_reconstructions(h5_paths, ...)
    Same idea but reads shots straight from a mirrored denoised h5 tree
    (Ximg / Ypdf / Ypdf_denoised datasets already present per shot).
    Handy after running denoise_inference_1d.py.
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py

from svd1d_autoencoder import (
    DEFAULT_H, DEFAULT_W, DEFAULT_R,
    svd1d_split_sizes,
)


def _reconstruct_2d_np(vec, H, W, r):
    nU, nS, nV = svd1d_split_sizes(H, W, r)
    U = vec[:nU].reshape(H, r)
    S = vec[nU:nU + nS]
    V = vec[nU + nS:nU + nS + nV].reshape(r, W)
    return U @ np.diag(S) @ V


def _row_figure(recons, one_d_vecs, titles, out_path, H, W, r, suptitle=None):
    """recons: list of (H, W) arrays. one_d_vecs: list of (L,) arrays.
    titles: list[str] of matching length. Saves a 2-row figure:
        row 0 : imshow of each 2D recon (shared vmin/vmax)
        row 1 : line plot of each 1D SVD vector, with U|S|V boundaries
    """
    n = len(recons)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6),
                             gridspec_kw={"height_ratios": [1.3, 1.0]})
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    # Shared color scale across recons for fair comparison.
    vmin = min(r_.min() for r_ in recons)
    vmax = max(r_.max() for r_ in recons)

    for i, (rec, vec, title) in enumerate(zip(recons, one_d_vecs, titles)):
        ax = axes[0, i]
        im = ax.imshow(rec, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time bin")
        ax.set_ylabel("detector")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, i]
        ax.plot(vec, linewidth=0.5)
        nU, nS, _ = svd1d_split_sizes(H, W, r)
        ax.axvline(nU, color="red", linestyle="--", linewidth=0.8,
                   label="U|S boundary")
        ax.axvline(nU + nS, color="orange", linestyle="--", linewidth=0.8,
                   label="S|V boundary")
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.legend(fontsize=7, loc="upper right")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_denoised_reconstructions(
    model,
    dataset,
    device,
    figures_dir,
    identifier,
    n_examples=6,
    H=DEFAULT_H,
    W=DEFAULT_W,
    r=DEFAULT_R,
    seed=0,
):
    """Sample n_examples shots from a torch Dataset (returning
    (Ximg_vec, [Ypdf_vec]) pairs), run the model, and save one figure per
    example under figures_dir.
    """
    os.makedirs(figures_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    if n_total == 0:
        print("plot_denoised_reconstructions: empty dataset, skipping.")
        return
    idxs = rng.choice(n_total, size=min(n_examples, n_total), replace=False)

    model.eval()
    for k, idx in enumerate(idxs):
        item = dataset[int(idx)]
        # DataMilking_HalfAndHalf returns (input, [label, ...]) with
        # labels wrapped in a list; be defensive.
        x = item[0]
        y = item[1]
        if isinstance(y, (list, tuple)):
            y = y[0]
        x = torch.as_tensor(x, dtype=torch.float32).flatten()
        y = torch.as_tensor(y, dtype=torch.float32).flatten()

        with torch.no_grad():
            out = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()

        vecs = [x.numpy(), y.numpy(), out.numpy()]
        recons = [_reconstruct_2d_np(v, H, W, r) for v in vecs]
        titles = ["recon(Ximg) - noisy",
                  "recon(Ypdf) - truth",
                  "recon(model(Ximg)) - denoised"]

        out_path = os.path.join(
            figures_dir, f"{identifier}_diag_shot{k:02d}_idx{int(idx)}.pdf"
        )
        _row_figure(recons, vecs, titles, out_path, H, W, r,
                    suptitle=f"{identifier} val idx={int(idx)}")
        print(f"wrote {out_path}")


def plot_h5_reconstructions(
    h5_paths,
    figures_dir,
    identifier,
    n_examples=6,
    H=DEFAULT_H,
    W=DEFAULT_W,
    r=DEFAULT_R,
    seed=0,
):
    """Pull n_examples shots from mirrored denoised h5 files (must have
    per-shot Ximg / Ypdf / Ypdf_denoised) and write reconstruction
    figures. Used after denoise_inference_1d.py.
    """
    os.makedirs(figures_dir, exist_ok=True)
    files = []
    for p in h5_paths:
        if not p:
            continue
        if os.path.isfile(p):
            files.append(p)
        else:
            files.extend(sorted(glob.glob(os.path.join(p, "*.h5"))))
    if not files:
        print("plot_h5_reconstructions: no .h5 files found.")
        return

    rng = np.random.default_rng(seed)
    picked = 0
    for path in files:
        if picked >= n_examples:
            break
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            rng.shuffle(keys)
            for shot in keys:
                if picked >= n_examples:
                    break
                g = f[shot]
                if "Ypdf_denoised" not in g:
                    continue
                x = np.asarray(g["Ximg"][()], dtype=np.float32)
                y = np.asarray(g["Ypdf"][()], dtype=np.float32)
                d = np.asarray(g["Ypdf_denoised"][()], dtype=np.float32)
                vecs = [x, y, d]
                recons = [_reconstruct_2d_np(v, H, W, r) for v in vecs]
                titles = [
                    "recon(Ximg) - noisy",
                    "recon(Ypdf) - truth",
                    "recon(Ypdf_denoised) - model",
                ]
                out_path = os.path.join(
                    figures_dir,
                    f"{identifier}_h5_diag_shot{picked:02d}.pdf",
                )
                _row_figure(
                    recons, vecs, titles, out_path, H, W, r,
                    suptitle=f"{identifier} {os.path.basename(path)}/{shot[:12]}",
                )
                print(f"wrote {out_path}")
                picked += 1


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dirs", required=True,
                   help="':'-separated list of denoised h5 files or dirs.")
    p.add_argument("--figures_dir", required=True)
    p.add_argument("--identifier", default="svd1d_r8_h5_diag")
    p.add_argument("--n_examples", type=int, default=6)
    p.add_argument("--svd_h", type=int, default=DEFAULT_H)
    p.add_argument("--svd_w", type=int, default=DEFAULT_W)
    p.add_argument("--svd_r", type=int, default=DEFAULT_R)
    args = p.parse_args()
    plot_h5_reconstructions(
        h5_paths=[d for d in args.h5_dirs.split(":") if d],
        figures_dir=args.figures_dir,
        identifier=args.identifier,
        n_examples=args.n_examples,
        H=args.svd_h, W=args.svd_w, r=args.svd_r,
    )
