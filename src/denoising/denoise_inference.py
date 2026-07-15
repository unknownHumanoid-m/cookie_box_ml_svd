# -*- coding: utf-8 -*-
"""
Run the trained Ximg->Ypdf autoencoder over every shot in a set of SVD h5
files and write mirrored output files under a target directory.

For each input file `foo.h5`, this writes `foo.h5` (same basename) under
`--output_dir`, preserving the per-shot group structure:

    /{shot_key}
        Ximg               (copied from source)
        Ypdf               (copied from source; the clean truth)
        Ypdf_denoised      (new: autoencoder output on Ximg)
        <all group attrs copied through, incl. npulses>

Root-level attributes are copied too and augmented with `denoiser_source`
(the .pth weights) and `denoiser_input`.

Downstream scripts (train_how_many.py, evaluate_how_many.py, sinusoid
regressor, etc.) can then read `Ypdf_denoised` while still having the
noisy input and clean truth in the same file.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder


def build_autoencoder():
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()],
    ])
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()],
    ])
    return Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)


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
            x = torch.from_numpy(ximgs).unsqueeze(1).to(device)  # (B, 1, H, W)
            with torch.no_grad():
                y = model(x).squeeze(1).cpu().numpy()  # (B, H, W)

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
    p = argparse.ArgumentParser(description="Run autoencoder inference over SVD h5 files.")
    p.add_argument("--data_dirs", required=True,
                   help="':'-separated list of h5 files or directories to denoise.")
    p.add_argument("--model_path", required=True,
                   help="Path to the .pth state_dict for Ximg_to_Ypdf_Autoencoder.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write mirrored denoised h5 files into.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Shots per forward pass.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files (default: skip).")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_path}")
    print(f"Out:    {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_autoencoder().to(device)
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


if __name__ == "__main__":
    main()
