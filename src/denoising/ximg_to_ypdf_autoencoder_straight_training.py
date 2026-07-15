from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder


from denoising_util import *

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
    # Input Data Paths and Output Save Paths

    # 2D SVD dataset (rank-r reconstruction, same 2D shape as original).
    # Training data directory (or ':'-separated list of dirs). Override via env
    # var TRAIN_DATA_DIRS so the user only edits s3df_denoising.sh.
    default_train_dirs = "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/miaed_mnis_data/mrco_h5_svd/"
    train_dirs_env = os.environ.get("TRAIN_DATA_DIRS", default_train_dirs)
    datapaths = [d for d in train_dirs_env.split(":") if d]
    print(f"Training on data dirs: {datapaths}")

    data = DataMilking_HalfAndHalf(root_dirs=datapaths, pulse_handler=None, input_name="Ximg", labels=["Ypdf"], transform=None)

    print(len(data))
    # Calculate the lengths for each split
    train_size = int(0.8 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


    # Example usage
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
    # decoder_layers = np.array([
    #     [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Tanh()]  # Example with Tanh activation
    #     # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    # ])
    #Changing back to sigmoid since normalized outputs to be 0 to 1
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
        # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    ])

    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)
    max_epochs = 200
    scheduler = CustomScheduler(optimizer, patience=5, early_stop_patience = 9, cooldown=2, lr_reduction_factor=0.5, max_num_epochs = max_epochs, improvement_percentage=0.001)
    # Scratch dir for heavy artifacts (checkpoints, .pth model files, dataset caches, run summaries).
    # Figures stay next to this .py file under ./figures/ (see below).
    # Both are overridable via env vars set in s3df_denoising.sh.
    default_model_save_dir = "/sdf/home/m/miaed/tmo_exp/tmo101347625/scratch/denoising_runs/svd2d_r8_autoencoder/"
    model_save_dir = os.environ.get("MODEL_SAVE_DIR", default_model_save_dir)

    default_figures_dir = os.path.join(current_dir, "figures")
    figures_dir = os.environ.get("FIGURES_DIR", default_figures_dir)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    identifier = os.environ.get("MODEL_IDENTIFIER", "autoencoder_svd2d_r8")
    print(f"Model save dir: {model_save_dir}")
    print(f"Figures dir:    {figures_dir}")
    print(f"Identifier:     {identifier}")
    autoencoder.to(device)
    # Get detailed GPU information if using CUDA
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        num_gpus = torch.cuda.device_count()
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert to GB
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
        device_info = f"{gpu_name} (Total GPUs: {num_gpus}, Total Memory: {gpu_memory_total:.2f} GB, " \
                    f"Reserved Memory: {gpu_memory_reserved:.2f} GB, Allocated Memory: {gpu_memory_allocated:.2f} GB)"
    elif device.type == 'mps':
        device_info = 'MPS (Apple Silicon GPU)'
    else:
        device_info = 'CPU'
    print(summary(autoencoder, input_size=(1, 1, 16, 512)))

    print(f"Using device: {device_info}")
    autoencoder.train_model(train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=max_epochs, figures_dir=figures_dir)

    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    # with open(results_file, 'w') as f:
    #     f.write("Model Training Results\n")
    #     f.write("======================\n")
    #     f.write(f"Data Path: {datapaths}\n")
    #     f.write(f"Model Save Directory: {model_save_dir}\n")
    #     f.write("\nModel Parameters and Hyperparameters\n")
    #     f.write("-----------------------------------\n")
    #     f.write(f"Patience: {scheduler.patience}\n")
    #     f.write(f"Cooldown: {scheduler.cooldown}\n")
    #     f.write(f"Learning Rate Reduction Factor: {scheduler.lr_reduction_factor}\n")
    #     f.write(f"Improvement Percentage: {scheduler.improvement_percentage}\n")
    #     f.write(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}\n")
    #     f.write("\nModel Architecture\n")
    #     f.write("------------------\n")
    #     f.write(f"Encoder Layers: {encoder_layers}\n")
    #     f.write(f"Decoder Layers: {decoder_layers}\n")
    #     f.write("\nAdditional Notes\n")
    #     f.write("----------------\n")
    #     f.write("Training on single pulse.\n")
    with open(results_file, 'w') as f:
        f.write("Model Training Results\n")
        f.write("======================\n")
        f.write(f"Data Path: {datapaths}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write(f"Device Used: {device_info}\n")
        f.write("\nModel Parameters and Hyperparameters\n")
        f.write("-----------------------------------\n")
        f.write(f"Patience: {scheduler.patience}\n")
        f.write(f"Cooldown: {scheduler.cooldown}\n")
        f.write(f"Learning Rate Reduction Factor: {scheduler.lr_reduction_factor}\n")
        f.write(f"Improvement Percentage: {scheduler.improvement_percentage}\n")
        f.write(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write("\nModel Architecture\n")
        f.write("------------------\n")
        f.write("Encoder Layers:\n")
        for layer in encoder_layers:
            f.write(f"{layer}\n")
        f.write("Decoder Layers:\n")
        for layer in decoder_layers:
            f.write(f"{layer}\n")
        f.write("\nAdditional Notes\n")
        f.write("----------------\n")
        f.write("Training on 2D SVD (rank-r reconstruction) Ximg -> Ypdf dataset.\n")
        f.write(f"Total Training Epochs: {max_epochs}\n")
        f.write(f"Data handled using DataMilking_HalfAndHalf with no pulse handler.\n")
        f.write(f"Figures Directory: {figures_dir}\n")
        f.write(f"Batch Size: {train_dataloader.batch_size}\n")
        f.write(f"Train Size: {train_size}, Validation Size: {val_size}, Test Size: {test_size}\n")
        f.write(str(summary(autoencoder, input_size=(1, 1, 16, 512))))

    print(f"Training completed. Results saved to {results_file}")


if __name__ == "__main__":

    main()