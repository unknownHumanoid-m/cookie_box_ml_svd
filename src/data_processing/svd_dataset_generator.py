"""
Generate a rank-r SVD version of the processed h5 dataset and save it for later use.

Given an input h5 whose groups each contain a 2D `Ximg` and a 2D `Ypdf` dataset
(plus scalar/vector attributes), this writes a new h5 alongside it in which
Ximg and Ypdf are replaced with either:

    --mode 2d : rank-r reconstruction, same 2D shape as the original
                (U[:, :r] @ diag(S[:r]) @ V[:r, :])

    --mode 1d : flat concatenation [U[:, :r].flatten(), S[:r], V[:r, :].flatten()]
                (mirrors the MNIST toy path in overlapped_dataset.py)

All group attributes are copied through unchanged. The rank r and mode are
recorded as attributes on the output file's root group.

Example
-------
    python3 svd_dataset_generator.py \
        --input  /sdf/.../scratch/miaed_mnis_data/mrco_h5/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed.h5 \
        --mode 2d --rank 15
"""
from dp_utils import *


DEFAULT_INPUT = (
    "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/"
    "mrco_h5/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed.h5"
)
DEFAULT_OUTPUT_DIR = (
    "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd"
)


def svd_2d(matrix, r):
    """Rank-r reconstruction of `matrix`. Returns array of the same shape."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    r_eff = min(r, S.shape[0])
    return (U[:, :r_eff] * S[:r_eff]) @ Vt[:r_eff, :]


def svd_1d(matrix, r):
    """Flat [U[:, :r].flatten(), S[:r], Vt[:r, :].flatten()] vector."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    r_eff = min(r, S.shape[0])
    return np.concatenate([U[:, :r_eff].flatten(), S[:r_eff], Vt[:r_eff, :].flatten()])


def build_output_path(input_path, output_dir, mode, rank):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}_svd{mode}_r{rank}.h5")


def generate_svd_h5(input_path, output_path, mode, rank):
    transform = svd_2d if mode == "2d" else svd_1d
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        dst.attrs["svd_mode"] = mode
        dst.attrs["svd_rank"] = rank
        dst.attrs["svd_source"] = os.path.basename(input_path)

        keys = list(src.keys())
        print(f"Processing {len(keys)} groups: {input_path} -> {output_path}")
        for i, key in enumerate(keys):
            g_src = src[key]
            g_dst = dst.create_group(key)
            for name, val in g_src.attrs.items():
                g_dst.attrs[name] = val

            ximg = g_src["Ximg"][:]
            ypdf = g_src["Ypdf"][:]
            g_dst.create_dataset("Ximg", data=transform(ximg, rank))
            g_dst.create_dataset("Ypdf", data=transform(ypdf, rank))
            g_dst.attrs["Ximg_orig_shape"] = ximg.shape
            g_dst.attrs["Ypdf_orig_shape"] = ypdf.shape

            if (i + 1) % 200 == 0 or i + 1 == len(keys):
                print(f"  [{i + 1}/{len(keys)}] done")
    print(f"Wrote {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate SVD-compressed h5 dataset.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Path to source processed h5.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write the SVD h5 into.")
    p.add_argument("--output", default=None,
                   help="Explicit output path. Overrides --output-dir naming.")
    p.add_argument("--mode", choices=["1d", "2d"], required=True,
                   help="'2d' for rank-r reconstruction, '1d' for flat U/S/V vector.")
    p.add_argument("--rank", type=int, default=15, help="Truncation rank r.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or build_output_path(
        args.input, args.output_dir, args.mode, args.rank
    )
    generate_svd_h5(args.input, output_path, args.mode, args.rank)
