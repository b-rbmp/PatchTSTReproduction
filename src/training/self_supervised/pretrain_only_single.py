import os
import sys

# Add parent directory to path
sys.path.append(".")

import random
from config import PreTrainingConfig, parse_args
import numpy as np
import torch
from tqdm import tqdm
from src.training.self_supervised.utils import find_learning_rate, pre_train_model
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    config: PreTrainingConfig = parse_args(mode="pretrain")

    print("Configurations:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

    # Fix seed
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Enable CUDA
    use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Checkpoint folder
    checkpoint_folder = os.path.join(config.checkpoint_dir, "self_supervised", "pretrain", config.model_identifier)


    print(f"Device: {device}")
    learning_rate = find_learning_rate(
        config=config, device=device, head_type="pretrain", checkpoint_folder=checkpoint_folder, mode="pretrain"
    )

    print("Suggested Learning Rate:", learning_rate)
    pre_train_model(
        config=config,
        device=device,
        suggested_lr=learning_rate,
        checkpoint_folder=checkpoint_folder,
    )
