import os
import sys

# Add parent directory to path
sys.path.append(".")

import random
from config import TrainingConfig, parse_args
import numpy as np
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataloader_manager import DataLoaderManager, DataParams, get_dataloaders
from src.models.patchTST.encoders.supervised_patchTST import PatchTST
from src.utils.early_stopping import EarlyStopping
from src.utils.scheduler import adjust_lr
from src.utils.metrics import calculate_metrics
from src.utils.bootstrapping import calculate_bootstrap_statistics


def visualize_results(input_truth, input_pred, path):

    plt.figure(figsize=(10, 5))
    plt.plot(input_truth, label="Ground Truth", color="blue")
    plt.plot(input_pred, label="Prediction", color="red")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def get_model(config: TrainingConfig):

    # Get model
    if config.model == "PatchTST":
        model = PatchTST(configs=config)

    return model


def get_data_loader_manager(config: TrainingConfig) -> DataLoaderManager:
    # Get data loader parameters
    data_loader_params = DataParams(
        dset=config.dataset,
        context_points=config.input_length,
        target_points=config.prediction_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        features=config.features,
        use_time_features=True if config.embed == "timeFeatures" else False,
    )
    # Get data loaders
    data_loader_manager = get_dataloaders(data_loader_params)

    return data_loader_manager


def train_model(config: TrainingConfig, train_identifier: str, device: torch.device):
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    train_loader = data_loader_manager.train_loader
    val_loader = data_loader_manager.val_loader

    # Total number of training steps per epoch
    train_steps = len(train_loader)

    # Create checkpoint folder
    checkpoint_folder = os.path.join(config.checkpoint_dir, train_identifier)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Get model and move to device
    model = get_model(config).to(device)

    # Get optimizer, criterion, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.learning_rate,
        total_steps=None,
        steps_per_epoch=train_steps,
        pct_start=config.lr_pct_start,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
        last_epoch=-1,
        epochs=config.epochs,
    )

    # If fp16 is enabled, create GradScaler
    if config.fp16:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    best_val_loss = float("inf")

    early_stopping = EarlyStopping(
        patience=config.patience,
        verbose=True,
        path=os.path.join(checkpoint_folder, "checkpoint.pt"),
        best_metric=best_val_loss,
    )

    for epoch in range(config.epochs):
        train_loss = 0
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            optimizer.zero_grad()

            if config.fp16:
                with torch.amp.autocast(device_type=device.type):
                    # Forward pass
                    if config.model == "PatchTST" or "Linear" in config.model:
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                    else:
                        # Decoder input
                        decoder_input = torch.zeros_like(
                            y[:, -config.prediction_length :, :]
                        ).float()
                        decoder_input = (
                            torch.cat(
                                [y[:, : config.prediction_length, :], decoder_input],
                                dim=1,
                            )
                            .float()
                            .to(device)
                        )

                        x, y, x_mark, y_mark = batch
                        x, y, x_mark, y_mark = (
                            x.to(device),
                            y.to(device),
                            x_mark.to(device),
                            y_mark.to(device),
                        )
                        if config.output_attention:
                            y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                        else:
                            y_hat = model(x, x_mark, decoder_input, y_mark)

                    feature_dim = -1 if config.features == "MS" else 0
                    y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                    y = y[:, -config.prediction_length :, feature_dim:]

                    loss = criterion(y_hat, y)

                # Backward pass with GradScaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal FP32 training
                if config.model == "PatchTST" or "Linear" in config.model:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                else:
                    # Decoder input
                    decoder_input = torch.zeros_like(
                        y[:, -config.prediction_length :, :]
                    ).float()
                    decoder_input = (
                        torch.cat(
                            [y[:, : config.prediction_length, :], decoder_input], dim=1
                        )
                        .float()
                        .to(device)
                    )
                    x, y, x_mark, y_mark = batch
                    x, y, x_mark, y_mark = (
                        x.to(device),
                        y.to(device),
                        x_mark.to(device),
                        y_mark.to(device),
                    )
                    if config.output_attention:
                        y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                    else:
                        y_hat = model(x, x_mark, decoder_input, y_mark)

                feature_dim = -1 if config.features == "MS" else 0
                y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                y = y[:, -config.prediction_length :, feature_dim:]
                loss = criterion(y_hat, y)

                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            # Adjust learning rate (OneCycleLR) if needed
            if config.learning_rate_adjustment == "TST":
                adjust_lr(optimizer, scheduler, epoch + 1, config)
                scheduler.step()

        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            if config.fp16:
                # Use autocast for validation as well
                with torch.amp.autocast(device_type=device.type):
                    for batch in tqdm(
                        val_loader,
                        desc=f"Validation - Epoch {epoch + 1}/{config.epochs}",
                    ):
                        if config.model == "PatchTST" or "Linear" in config.model:
                            x, y = batch
                            x, y = x.to(device), y.to(device)
                            y_hat = model(x)
                        else:
                            # Decoder input
                            decoder_input = torch.zeros_like(
                                y[:, -config.prediction_length :, :]
                            ).float()
                            decoder_input = (
                                torch.cat(
                                    [
                                        y[:, : config.prediction_length, :],
                                        decoder_input,
                                    ],
                                    dim=1,
                                )
                                .float()
                                .to(device)
                            )
                            x, y, x_mark, y_mark = batch
                            x, y, x_mark, y_mark = (
                                x.to(device),
                                y.to(device),
                                x_mark.to(device),
                                y_mark.to(device),
                            )
                            if config.output_attention:
                                y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                            else:
                                y_hat = model(x, x_mark, decoder_input, y_mark)

                        feature_dim = -1 if config.features == "MS" else 0
                        y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                        y = y[:, -config.prediction_length :, feature_dim:]

                        loss = criterion(y_hat, y)
                        val_loss += loss.item()
            else:
                for batch in tqdm(
                    val_loader, desc=f"Validation - Epoch {epoch + 1}/{config.epochs}"
                ):
                    if config.model == "PatchTST" or "Linear" in config.model:
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                    else:
                        # Decoder input
                        decoder_input = torch.zeros_like(
                            y[:, -config.prediction_length :, :]
                        ).float()
                        decoder_input = (
                            torch.cat(
                                [y[:, : config.prediction_length, :], decoder_input],
                                dim=1,
                            )
                            .float()
                            .to(device)
                        )
                        x, y, x_mark, y_mark = batch
                        x, y, x_mark, y_mark = (
                            x.to(device),
                            y.to(device),
                            x_mark.to(device),
                            y_mark.to(device),
                        )
                        if config.output_attention:
                            y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                        else:
                            y_hat = model(x, x_mark, decoder_input, y_mark)

                    feature_dim = -1 if config.features == "MS" else 0
                    y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                    y = y[:, -config.prediction_length :, feature_dim:]

                    loss = criterion(y_hat, y)
                    val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        early_stopping(val_loss, model, optimizer, epoch)
        best_val_loss = early_stopping.best_metric

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if config.learning_rate_adjustment != "TST":
            adjust_lr(optimizer, scheduler, epoch + 1, config)
            scheduler.step()

    print(f"Training completed. Best validation loss: {best_val_loss}")
    model = get_model(config).to(device)
    best_state_dict = torch.load(early_stopping.path)["model_state_dict"]
    # Delete model.revin_layer.mean and model.revin_layer.stdev from the state dict
    if config.model == "PatchTST":
        del best_state_dict["model.revin_layer.mean"]
        del best_state_dict["model.revin_layer.stdev"]
    model.load_state_dict(best_state_dict)
    return model


def test_model(
    model, config: TrainingConfig, train_identifier: str, device: torch.device
):
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    test_loader = data_loader_manager.test_loader

    # Get model and move to device
    model = model.to(device)

    results_path = "./test_results/" + train_identifier + "/"
    os.makedirs(results_path, exist_ok=True)

    model.eval()
    predictions = []
    ground_truth = []
    inputs = []

    use_autocast = config.fp16

    with torch.no_grad():
        count = 0
        for batch in tqdm(test_loader, desc="Testing"):
            if use_autocast:
                with torch.amp.autocast(device_type=device.type):
                    if config.model == "PatchTST" or "Linear" in config.model:
                        x, y = batch
                        x, y = x.to(device), y.to(device)
                        y_hat = model(x)
                    else:
                        # Decoder input
                        decoder_input = torch.zeros_like(
                            y[:, -config.prediction_length :, :]
                        ).float()
                        decoder_input = (
                            torch.cat(
                                [y[:, : config.prediction_length, :], decoder_input],
                                dim=1,
                            )
                            .float()
                            .to(device)
                        )
                        x, y, x_mark, y_mark = batch
                        x, y, x_mark, y_mark = (
                            x.to(device),
                            y.to(device),
                            x_mark.to(device),
                            y_mark.to(device),
                        )
                        if config.output_attention:
                            y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                        else:
                            y_hat = model(x, x_mark, decoder_input, y_mark)

                    feature_dim = -1 if config.features == "MS" else 0
                    y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                    y = y[:, -config.prediction_length :, feature_dim:]
            else:
                # Normal inference
                if config.model == "PatchTST" or "Linear" in config.model:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                else:
                    # Decoder input
                    decoder_input = torch.zeros_like(
                        y[:, -config.prediction_length :, :]
                    ).float()
                    decoder_input = (
                        torch.cat(
                            [y[:, : config.prediction_length, :], decoder_input], dim=1
                        )
                        .float()
                        .to(device)
                    )
                    x, y, x_mark, y_mark = batch
                    x, y, x_mark, y_mark = (
                        x.to(device),
                        y.to(device),
                        x_mark.to(device),
                        y_mark.to(device),
                    )
                    if config.output_attention:
                        y_hat = model(x, x_mark, decoder_input, y_mark)[0]
                    else:
                        y_hat = model(x, x_mark, decoder_input, y_mark)

                feature_dim = -1 if config.features == "MS" else 0
                y_hat = y_hat[:, -config.prediction_length :, feature_dim:]
                y = y[:, -config.prediction_length :, feature_dim:]

            y_hat_detach = y_hat.detach().cpu().numpy()
            y_detach = y.detach().cpu().numpy()
            x_detach = x.detach().cpu().numpy()
            predictions.append(y_hat_detach)
            ground_truth.append(y_detach)
            inputs.append(x_detach)

            if count % 20 == 0:
                input_truth = np.concatenate(
                    (x_detach[0, :, -1], y_detach[0, :, -1]), axis=0
                )
                input_pred = np.concatenate(
                    (x_detach[0, :, -1], y_hat_detach[0, :, -1]), axis=0
                )
                visualize_results(
                    input_truth,
                    input_pred,
                    os.path.join(results_path, f"sample_{count}.png"),
                )

            count += 1

        # Remove elements from list that are not homogeneous in last dimension
        # Example: Filter out any element if its second dimension doesn't match the max length in predictions.
        max_len = max(arr.shape[0] for arr in predictions)

        filtered_predictions, filtered_ground_truth, filtered_inputs = [], [], []
        for pred, gt, inp in zip(predictions, ground_truth, inputs):
            if pred.shape[0] == max_len:
                filtered_predictions.append(pred)
                filtered_ground_truth.append(gt)
                filtered_inputs.append(inp)

        filtered_predictions = np.array(filtered_predictions)
        filtered_ground_truth = np.array(filtered_ground_truth)
        filtered_inputs = np.array(filtered_inputs)

        # Then reshape as needed
        filtered_predictions = filtered_predictions.reshape(
            -1, filtered_predictions.shape[-2], filtered_predictions.shape[-1]
        )
        filtered_ground_truth = filtered_ground_truth.reshape(
            -1, filtered_ground_truth.shape[-2], filtered_ground_truth.shape[-1]
        )
        filtered_inputs = filtered_inputs.reshape(
            -1, filtered_inputs.shape[-2], filtered_inputs.shape[-1]
        )

        metrics = calculate_metrics(filtered_predictions, filtered_ground_truth)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Write results to a text file
        with open(os.path.join(results_path, "metrics.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    return metrics


if __name__ == "__main__":
    config = parse_args()

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

    # Train model if in train mode
    if config.train_mode:
        metrics_dict = {}
        for i in range(config.bootstrap_iterations):
            print(f"Bootstrap iteration {i + 1}/{config.bootstrap_iterations}")
            train_identifier = f"{config.model_identifier}_bootstrap_{i}"
            # Train model from scratch
            best_model = train_model(config, train_identifier, device)

            # Test model on the best checkpoint
            metrics = test_model(best_model, config, train_identifier, device)
            torch.cuda.empty_cache()
            metrics_dict[train_identifier] = metrics

        # Calculate bootstrap statistics with confidence intervals
        final_metrics = calculate_bootstrap_statistics(metrics_dict)
        print("Final metrics:")
        for key, value in final_metrics.items():
            print(f"{key}: {value}")
    else:
        raise ValueError("Not Implemented.")
