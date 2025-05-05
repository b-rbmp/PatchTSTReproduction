import os
import sys

# Add parent directory to path
sys.path.append(".")

import random
from config import FinetuneConfig, parse_args
import numpy as np
import torch
from tqdm import tqdm
from src.training.self_supervised.utils import (
    find_learning_rate,
    finetune_model,
    pre_train_model,
    test_model,
)
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config: FinetuneConfig = parse_args(mode="finetune")

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

    print(f"Device: {device}")
    if config.finetune_mode and config.linear_probe_mode:
        raise ValueError(
            "Invalid mode. Choose either finetune_mode or linear_probe_mode"
        )

    if not config.finetune_mode and not config.linear_probe_mode:
        raise ValueError(
            "Invalid mode. Choose either finetune_mode or linear_probe_mode"
        )

    # Checkpoint folder
    checkpoint_folder = os.path.join(config.checkpoint_dir, "self_supervised", "finetune", config.model_identifier)

    learning_rate = find_learning_rate(
        config=config, device=device, head_type="prediction", mode="finetuning", checkpoint_folder=checkpoint_folder
    )
    print("Suggested Finetuning Learning Rate:", learning_rate)

    finetune_model(
        config,
        device,
        learning_rate,
        checkpoint_folder,
        linear_probe_only=config.linear_probe_mode,
    )
    print("Finetuning Completed")
    # Test
    metrics = test_model(config, device, checkpoint_folder)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
