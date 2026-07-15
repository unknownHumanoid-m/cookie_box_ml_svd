from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from ximg_to_ypdf_autoencoder import Zero_PulseClassifier
from denoising_util import *


def save_comparison_figures(results_h5_path, figures_dir, identifier, num_examples=6):
    """Read `num_examples` groups from the eval-results h5 and write a single
    combined PNG (rows = examples, columns = noisy input / denoised output /
    truth) into `figures_dir`. Uses aspect='auto' and cmap='magma_r' to match
    src/data_processing/processed_data_viewer.py.
    """
    os.makedirs(figures_dir, exist_ok=True)
    with h5py.File(results_h5_path, "r") as h5file:
        group_keys = sorted(h5file.keys())
        n = min(num_examples, len(group_keys))
        if n == 0:
            print(f"No groups found in {results_h5_path}; skipping figure.")
            return None

        col_titles = ("Noisy Input (Ximg)", "Denoised Output", "Truth (Ypdf)")

        examples = []
        for key in group_keys[:n]:
            grp = h5file[key]
            examples.append((
                key,
                np.array(grp["input"]),
                np.array(grp["output"]),
                np.array(grp["target"]),
                grp.attrs.get("loss", float("nan")),
            ))

        all_vals = np.concatenate(
            [arr.ravel() for _, inp, out, tgt, _ in examples
             for arr in (inp, out, tgt)]
        )
        vmin, vmax = float(all_vals.min()), float(all_vals.max())

        fig, axes = plt.subplots(
            n, 3, figsize=(15, 2.5 * n), squeeze=False,
        )

        im = None
        for row, (key, inp, out, tgt, loss) in enumerate(examples):
            for col, (img, title) in enumerate(
                zip((inp, out, tgt), col_titles)
            ):
                ax = axes[row, col]
                im = ax.imshow(img, aspect="auto", cmap="magma_r",
                               vmin=vmin, vmax=vmax)
                if row == 0:
                    ax.set_title(title)
            axes[row, 0].set_ylabel(f"{key}\nMSE={loss:.2e}")

        fig.suptitle(f"{identifier} — input vs. denoised vs. truth ({n} examples)")
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.84])
        fig.colorbar(im, cax=cbar_ax)

        out_path = os.path.join(figures_dir, f"{identifier}_comparison.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote combined comparison figure: {out_path}")
        return out_path

# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))

# Add the utils directory to the Python path
sys.path.insert(0, utils_dir)
from utils import DataMilking_Nonfat, DataMilking, DataMilking_SemiSkimmed, DataMilking_HalfAndHalf
from utils import CustomScheduler

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------------------------------------------------------
    # Config (overridable from s3df_denoising.sh via env vars).
    # ----------------------------------------------------------------------
    # Test data dir(s). ':'-separated list allowed.
    default_eval_dirs = "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/"
    eval_dirs_env = os.environ.get("EVAL_DATA_DIRS", default_eval_dirs)
    datapaths = [d for d in eval_dirs_env.split(":") if d]

    # Autoencoder weights to evaluate.
    default_model_path = (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/"
        "svd2d_r8_autoencoder/autoencoder_svd2d_r8_best_model.pth"
    )
    best_model_path = os.environ.get("EVAL_MODEL_PATH", default_model_path)

    # Where to write eval results (h5, txt).
    default_eval_out = (
        "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/"
        "svd2d_r8_autoencoder/eval/"
    )
    model_save_dir = os.environ.get("EVAL_OUT_DIR", default_eval_out)

    identifier = os.environ.get("EVAL_IDENTIFIER", "autoencoder_svd2d_r8_eval")

    default_figures_dir = os.path.join(current_dir, "figures")
    figures_dir = os.environ.get("EVAL_FIGURES_DIR", default_figures_dir)

    num_comparison_figures = int(os.environ.get("EVAL_NUM_COMPARISON_FIGURES", "6"))

    # Optional zero-pulse classifier. If unset -> zero_masking disabled.
    classifier_path = os.environ.get("CLASSIFIER_PATH", "").strip()
    use_zero_masking = bool(classifier_path)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Eval data dirs:     {datapaths}")
    print(f"Model weights:      {best_model_path}")
    print(f"Eval output dir:    {model_save_dir}")
    print(f"Eval identifier:    {identifier}")
    print(f"Figures dir:        {figures_dir}")
    print(f"Zero-masking:       {use_zero_masking} ({classifier_path or 'disabled'})")

    # ----------------------------------------------------------------------
    # Dataset -> test-only DataLoader.
    # ----------------------------------------------------------------------
    data = DataMilking_HalfAndHalf(
        root_dirs=datapaths, pulse_handler=None, test_batch=1,
        input_name="Ximg", labels=["Ypdf"], transform=None,
    )

    train_size = int(0 * len(data))
    val_size = int(0 * len(data))
    test_size = int(len(data) - train_size - val_size)

    train_dataset, val_dataset, test_dataset = random_split(
        data, [train_size, val_size, test_size]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # ----------------------------------------------------------------------
    # Model (must match the architecture used at training time).
    # ----------------------------------------------------------------------
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])

    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]
    ])

    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)
    autoencoder.to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)

    # Optional zero-pulse classifier for masking.
    classifier = None
    if use_zero_masking:
        conv_layers = [
            [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
            [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
            [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
            [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
        ]

        def get_conv_output_size(input_size, conv_layers):
            x = torch.randn(input_size)
            model = nn.Sequential(*[layer for layer_pair in conv_layers
                                    for layer in layer_pair if layer is not None])
            x = model(x)
            return x.shape

        output_size = get_conv_output_size((1, 1, 16, 512), conv_layers)
        print(f"Output size after conv layers: {output_size}")

        fc_layers = [
            [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
            [nn.Linear(4, 1), None]
        ]

        classifier = Zero_PulseClassifier(conv_layers, fc_layers)
        classifier.to(device)
        cls_state = torch.load(classifier_path, map_location=device)
        # Drop side_network keys if present (legacy fine-tuned classifier state).
        keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
        cls_state = {k: v for k, v in cls_state.items()
                     if not any(key in k for key in keys_to_remove)}
        classifier.load_state_dict(cls_state)

    criterion = nn.MSELoss()
    print(summary(autoencoder, input_size=(1, 1, 16, 512)))

    autoencoder.evaluate_model(
        test_dataloader, criterion, device,
        save_results=True,
        results_dir=model_save_dir,
        results_filename=f"{identifier}_results.h5",
        zero_masking=use_zero_masking,
        zero_masking_model=classifier,
    )

    results_h5_path = os.path.join(model_save_dir, f"{identifier}_results.h5")
    save_comparison_figures(
        results_h5_path, figures_dir, identifier,
        num_examples=num_comparison_figures,
    )

    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n")
        f.write(f"Data Paths: {datapaths}\n")
        f.write(f"Model Weights: {best_model_path}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write(f"Figures Directory: {figures_dir}\n")
        f.write(f"Zero-masking: {use_zero_masking} ({classifier_path or 'disabled'})\n")
        f.write("\nModel Architecture\n")
        f.write("------------------\n")
        f.write(f"Encoder Layers: {encoder_layers}\n")
        f.write(f"Decoder Layers: {decoder_layers}\n")
        f.write("\nAdditional Notes\n")
        f.write("----------------\n")
        f.write("Evaluation of 2D SVD (rank-r reconstruction) Ximg -> Ypdf autoencoder.\n")
        f.write(str(summary(autoencoder, input_size=(1, 1, 16, 512))))


if __name__ == "__main__":
    main()
