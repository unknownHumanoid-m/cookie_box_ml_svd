from dp_utils import *

def plot_ximg_ypdf(h5_file_path, num_examples=3, out_prefix="demo_visual"):
    """
    Plot the Ximg and Ypdf datasets and print the attributes for a few example image keys.

    Args:
    - h5_file_path (str): Path to the HDF5 file.
    - num_examples (int): Number of example image keys to plot and print.
    - out_prefix (str): Filename prefix used for saved figures.
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        image_keys = list(h5f.keys())[:num_examples]

        for image_key in image_keys:
            print(f"Image Key: {image_key}")
            ximg = h5f[image_key]["Ximg"][:]
            ypdf = h5f[image_key]["Ypdf"][:]

            attributes = dict(h5f[image_key].attrs)
            print("Attributes:", attributes)

            # Plot Ximg
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"{image_key} - Ximg")
            plt.imshow(ximg, aspect='auto', cmap='magma_r')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title(f"{image_key} -Ypdf")
            plt.imshow(ypdf, aspect='auto', cmap='magma_r')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(f'figures/{out_prefix}_{image_key}.png')
            plt.close()


def _reshape_1d_svd(vec, orig_shape, rank):
    """Invert the 1D flat [U, S, V] encoding back to a rank-r reconstruction."""
    m, n = orig_shape
    r = min(rank, m, n)
    u_end = m * r
    s_end = u_end + r
    U = vec[:u_end].reshape(m, r)
    S = vec[u_end:s_end]
    Vt = vec[s_end:s_end + r * n].reshape(r, n)
    return (U * S) @ Vt


def plot_svd_h5(h5_file_path, num_examples=3, out_prefix="svd_visual"):
    """
    Plot Ximg/Ypdf from an SVD-processed h5 (produced by svd_dataset_generator.py).
    Handles both mode='2d' (data stored as reconstructed matrix) and mode='1d'
    (data stored as flat U/S/V vector; reshape back for display).
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        mode = h5f.attrs.get("svd_mode", "2d")
        rank = int(h5f.attrs.get("svd_rank", 0))
        source = h5f.attrs.get("svd_source", "<unknown>")
        print(f"SVD file: mode={mode} rank={rank} source={source}")

        image_keys = list(h5f.keys())[:num_examples]
        for image_key in image_keys:
            g = h5f[image_key]
            ximg = g["Ximg"][:]
            ypdf = g["Ypdf"][:]
            attributes = dict(g.attrs)
            print(f"Image Key: {image_key}  Attributes: {attributes}")

            if mode == "1d":
                ximg = _reshape_1d_svd(ximg, tuple(g.attrs["Ximg_orig_shape"]), rank)
                ypdf = _reshape_1d_svd(ypdf, tuple(g.attrs["Ypdf_orig_shape"]), rank)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title(f"{image_key} - Ximg (svd-{mode} r={rank})")
            plt.imshow(ximg, aspect='auto', cmap='magma_r')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title(f"{image_key} - Ypdf (svd-{mode} r={rank})")
            plt.imshow(ypdf, aspect='auto', cmap='magma_r')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(f'figures/{out_prefix}_{mode}_r{rank}_{image_key}.png')
            plt.close()


if __name__ == "__main__":
    #processed_h5_file_path = "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed.h5"
    svd_path = "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/Even_Distribution_Pulses_Processed_test_even-dist_Files.003_processed_svd2d_r8.h5"

    #plot_ximg_ypdf(processed_h5_file_path)

    os.makedirs("figures", exist_ok=True)
    if os.path.isdir(svd_path):
        for name in os.listdir(svd_path):
            if name.endswith(".h5"):
                plot_svd_h5(os.path.join(svd_path, name))
    elif os.path.isfile(svd_path):
        plot_svd_h5(svd_path)
    else:
        raise FileNotFoundError(svd_path)
