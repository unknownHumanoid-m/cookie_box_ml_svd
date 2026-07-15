# -*- coding: utf-8 -*-
"""
Run the trained 1D-SVD autoencoder over every shot in a set of 1D-SVD h5
files and write mirrored output files under a target directory.

For each input file `foo.h5`, this writes `foo.h5` (same basename) under
`--output_dir`, preserving the per-shot group structure:

    /{shot_key}
        Ximg               (copied from source, flat (L,))
        Ypdf               (copied from source, flat (L,))
        Ypdf_denoised      (new: autoencoder output on Ximg, flat (L,))
        <all group attrs copied through, incl. npulses / phases>

Optionally emits diagnostic 2D-reconstruction figures for a sample of
shots (--diag_examples) into --figures_dir.
"""

import argparse
import os

import h5py
import numpy as np
import torch

from svd1d_autoencoder import SVD1D_Autoencoder, DEFAULT_H, DEFAULT_W, DEFAULT_R
from diagnostics_svd1d import plot_h5_reconstructions


def collect_input_files(paths):
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
    return files


def denoise_file(src_path, dst_path, model, device, batch_size):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["denoiser_source"] = os.path.basename(src_path)

        shot_keys = list(src.keys())
        n_shots = len(shot_keys)
        print(f"  {os.path.basename(src_path)}: {n_shots} shots")

        for start in range(0, n_shots, batch_size):
            chunk_keys = shot_keys[start:start + batch_size]
            ximgs = np.stack(
                [src[k]["Ximg"][()] for k in chunk_keys]
            ).astype(np.float32)
            x = torch.from_numpy(ximgs).to(device)  # (B, L)
            with torch.no_grad():
                y = model(x).cpu().numpy()          # (B, L)

            for i, k in enumerate(chunk_keys):
                g_src = src[k]
                g_dst = dst.create_group(k)
                for attr_k, attr_v in g_src.attrs.items():
                    g_dst.attrs[attr_k] = attr_v
                g_dst.create_dataset("Ximg", data=g_src["Ximg"][()])
                g_dst.create_dataset("Ypdf", data=g_src["Ypdf"][()])
                g_dst.create_dataset("Ypdf_denoised", data=y[i])

            if (start // batch_size) % 10 == 0 or start + batch_size >= n_shots:
                done = min(start + batch_size, n_shots)
                print(f"    [{done}/{n_shots}]")

    print(f"  wrote {dst_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run 1D-SVD autoencoder inference.")
    p.add_argument("--data_dirs", required=True,
                   help="':'-separated list of h5 files or dirs to denoise.")
    p.add_argument("--model_path", required=True,
                   help="Path to the .pth state_dict for SVD1D_Autoencoder.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write mirrored denoised h5 files into.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Shots per forward pass.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: skip).")
    p.add_argument("--svd_h", type=int, default=DEFAULT_H)
    p.add_argument("--svd_w", type=int, default=DEFAULT_W)
    p.add_argument("--svd_r", type=int, default=DEFAULT_R)
    p.add_argument("--diag_examples", type=int, default=6,
                   help="How many post-inference reconstruction figures to "
                        "emit. 0 disables.")
    p.add_argument("--figures_dir", default=None,
                   help="Where to save diagnostic figures. Defaults to "
                        "<output_dir>/figures/.")
    p.add_argument("--identifier", default="svd1d_r8_inference")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_path}")
    print(f"Out:    {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = args.figures_dir or os.path.join(args.output_dir, "figures")
    if args.diag_examples > 0:
        os.makedirs(figures_dir, exist_ok=True)

    model = SVD1D_Autoencoder(H=args.svd_h, W=args.svd_w, r=args.svd_r).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    paths = [d for d in args.data_dirs.split(":") if d]
    files = collect_input_files(paths)
    print(f"Denoising {len(files)} file(s).")

    for src in files:
        dst = os.path.join(args.output_dir, os.path.basename(src))
        if os.path.exists(dst) and not args.overwrite:
            print(f"  skip (exists): {dst}")
            continue
        denoise_file(src, dst, model, device, args.batch_size)

    if args.diag_examples > 0:
        print(f"Writing {args.diag_examples} diagnostic figures to {figures_dir}")
        plot_h5_reconstructions(
            h5_paths=[args.output_dir],
            figures_dir=figures_dir,
            identifier=args.identifier,
            n_examples=args.diag_examples,
            H=args.svd_h, W=args.svd_w, r=args.svd_r,
        )


if __name__ == "__main__":
    main()
