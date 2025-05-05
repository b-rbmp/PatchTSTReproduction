
import os
import sys

# Add parent directory to path
sys.path.append(".")

import random
from config import PretrainFinetuneConfig, parse_args
import numpy as np
import torch
from src.training.self_supervised.utils import find_learning_rate, finetune_model, pre_train_model, test_model
from src.utils.bootstrapping import calculate_bootstrap_statistics
import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    config: PretrainFinetuneConfig = parse_args(mode="pretrain_finetune")

    if config.finetune_mode and config.linear_probe_mode:
        raise ValueError("Invalid mode. Choose either finetune_mode or linear_probe_mode")
    
    if not config.finetune_mode and not config.linear_probe_mode:
        raise ValueError("Invalid mode. Choose either finetune_mode or linear_probe_mode")
    
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

    addon_str = "linear_probe" if config.linear_probe_mode else "end_to_end"
    # Checkpoint folder
    checkpoint_folder = os.path.join(config.checkpoint_dir, "self_supervised", addon_str, config.model_identifier)

    # Bootstrap:
    metrics_dict = {}
    for i in range(config.bootstrap_iterations):
        print(f"Bootstrap iteration {i + 1}/{config.bootstrap_iterations}")
        config.model_identifier = f"{config.model_identifier}_bootstrap_{i}"
        print(f"Device: {device}")
        learning_rate = find_learning_rate(config=config, device=device, head_type='pretrain', mode="pretrain", checkpoint_folder=checkpoint_folder)
        torch.cuda.empty_cache()
        print("Suggested Learning Rate:", learning_rate)
        pre_train_model(config=config, device=device, suggested_lr=learning_rate, checkpoint_folder=checkpoint_folder)
        torch.cuda.empty_cache()
        learning_rate = find_learning_rate(config=config, device=device, head_type='prediction', mode="finetuning", checkpoint_folder=checkpoint_folder)
        print("Suggested Finetuning Learning Rate:", learning_rate)

        torch.cuda.empty_cache()

        finetune_model(config, device, learning_rate, checkpoint_folder, linear_probe_only=config.linear_probe_mode)
        print("Finetuning Completed")
        # Test
        metrics = test_model(config, device, checkpoint_folder)
        torch.cuda.empty_cache()
        metrics_dict[config.model_identifier] = metrics

        # Remove the checkpoint model after testing
        checkpoint_model = os.path.join(checkpoint_folder, f"checkpoint.pt")
        os.remove(checkpoint_model)


    # Calculate bootstrap statistics with confidence intervals
    final_metrics = calculate_bootstrap_statistics(metrics_dict)
    print("Final metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

