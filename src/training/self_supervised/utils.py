import os

import numpy as np
from src.models.patchTST.revin.revin import RevIN
from src.data.dataloader_manager import DataLoaderManager, DataParams, get_dataloaders
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from src.training.self_supervised.config import FinetuneConfig, PreTrainingConfig, SupervisedTrainingConfig, TransferLearningConfig
from src.models.patchTST.encoders.unsupervised_patchTST import PatchTST
from src.utils.scheduler import ExponentialLR
from src.utils.early_stopping import EarlyStopping
from torch.optim import lr_scheduler


def load_weights(weights_path, model, exclude_head=True, device="cpu") -> nn.Module:
    """
    Transfers weights from a saved state dictionary into the given model.

    Args:
        weights_path (str): Path to the saved weights file.
        model (torch.nn.Module): The target model to which weights will be transferred.
        exclude_head (bool): If True, layers with "head" in their name will be skipped.
        device (str): Device on which to load the weights (e.g., "cpu" or "cuda").

    Returns:
        torch.nn.Module: The model with the transferred weights.
    """
    # Load the saved weights, ensuring they are mapped to the specified device.
    loaded_state_dict = torch.load(weights_path, map_location=device)[
        "model_state_dict"
    ]

    matched_layers = 0  # Count of layers that were successfully matched
    unmatched_layers = []  # List to record layers that did not match or were missing

    # Iterate over the model's state dictionary to transfer weights.
    for layer_name, target_param in model.state_dict().items():
        # Optionally skip layers that are part of the model head.
        if exclude_head and "head" in layer_name:
            continue

        # Check if the layer exists in the loaded weights.
        if layer_name in loaded_state_dict:
            matched_layers += 1
            loaded_param = loaded_state_dict[layer_name]
            # Transfer the weight if the shape matches.
            if loaded_param.shape == target_param.shape:
                target_param.copy_(loaded_param)
            else:
                # Record layer if the shapes do not match.
                unmatched_layers.append(layer_name)
        else:
            # Record layer if it is missing from the loaded weights.
            unmatched_layers.append(layer_name)

    # Raise an error if no layers matched.
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")

    # Report any unmatched layers.
    if unmatched_layers:
        print(f"Unmatched layers: {unmatched_layers}")
    else:
        print(f"Weights from {weights_path} successfully transferred!")

    # Move the model to the specified device.
    model = model.to(device)
    return model


def create_patch(input_tensor, patch_length, stride):
    """
    Splits the input tensor into overlapping patches.

    The function extracts patches from the end of the sequence such that
    the patches cover a contiguous block of the sequence, with a specified
    patch length and stride between patches.

    Args:
        input_tensor (torch.Tensor): Tensor of shape [batch_size, sequence_length, num_vars].
        patch_length (int): The length of each patch.
        stride (int): The step size between consecutive patches.

    Returns:
        patches (torch.Tensor): Tensor of shape [batch_size, num_patches, num_vars, patch_length],
                                where each patch is an extracted segment from the sequence.
        num_patches (int): The number of patches created.
    """
    batch_size, sequence_length, num_vars = input_tensor.shape

    # Calculate the number of patches that can be extracted.
    # Using max(sequence_length, patch_length) ensures at least one patch is created.
    num_patches = (max(sequence_length, patch_length) - patch_length) // stride + 1

    # Determine the total length of the sequence block that will be covered by the patches.
    # The block starts from the position 'start_index' so that the last patch aligns with the end.
    target_length = patch_length + stride * (num_patches - 1)
    start_index = sequence_length - target_length

    # Extract the contiguous block from the end of the sequence.
    # The resulting tensor has shape [batch_size, target_length, num_vars].
    trimmed_sequence = input_tensor[:, start_index:, :]

    # Use unfold to create patches along the sequence dimension.
    # The resulting tensor has shape [batch_size, num_patches, num_vars, patch_length].
    patches = trimmed_sequence.unfold(dimension=1, size=patch_length, step=stride)

    return patches, num_patches


def random_masking(input_tensor, mask_ratio):
    """
    Applies random masking to a tensor of patches.

    The function randomly selects a subset of patches to keep based on the mask ratio,
    sets the remaining patches to zeros, and returns the masked tensor along with the
    kept patches, a binary mask, and the indices required to restore the original order.

    Args:
        input_tensor (torch.Tensor): Tensor of shape
            [batch_size, num_patches, num_vars, patch_length].
        mask_ratio (float): Fraction of patches to mask (set to zero). Must be in [0, 1].

    Returns:
        masked_tensor (torch.Tensor): Tensor with masked patches restored to the original order,
            shape [batch_size, num_patches, num_vars, patch_length].
        kept_tensor (torch.Tensor): Tensor containing only the kept patches,
            shape [batch_size, num_keep, num_vars, patch_length] where num_keep = num_patches * (1 - mask_ratio).
        mask (torch.Tensor): Binary mask of shape [batch_size, num_patches, num_vars] where 0 indicates a kept patch and 1 a masked patch.
        restore_indices (torch.Tensor): Indices used to restore the original order,
            shape [batch_size, num_patches, num_vars].
    """
    # Unpack the input tensor dimensions.
    batch_size, num_patches, num_vars, patch_length = input_tensor.shape

    # Clone the input tensor to avoid modifying the original.
    x = input_tensor.clone()

    # Determine the number of patches to keep.
    num_keep = int(num_patches * (1 - mask_ratio))

    # Generate random noise for each patch (for each sample and variable).
    # Shape: [batch_size, num_patches, num_vars]
    noise = torch.rand(batch_size, num_patches, num_vars, device=input_tensor.device)

    # For each sample and variable, sort the noise values.
    # Patches with lower noise values will be kept.
    # ids_shuffle holds the indices that would sort the noise in ascending order.
    ids_shuffle = torch.argsort(
        noise, dim=1
    )  # Shape: [batch_size, num_patches, num_vars]

    # Compute indices to restore the original order after masking.
    restore_indices = torch.argsort(
        ids_shuffle, dim=1
    )  # Shape: [batch_size, num_patches, num_vars]

    # Select indices for the patches to keep.
    ids_keep = ids_shuffle[:, :num_keep, :]  # Shape: [batch_size, num_keep, num_vars]

    # Gather the kept patches along the num_patches dimension.
    # Expand ids_keep to include the patch_length dimension.
    kept_tensor = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, patch_length)
    )
    # kept_tensor shape: [batch_size, num_keep, num_vars, patch_length]

    # Create a tensor of zeros for the patches that will be removed.
    removed_tensor = torch.zeros(
        batch_size,
        num_patches - num_keep,
        num_vars,
        patch_length,
        device=input_tensor.device,
    )

    # Concatenate the kept patches with the removed (zero) patches along the patch dimension.
    x_combined = torch.cat(
        [kept_tensor, removed_tensor], dim=1
    )  # Shape: [batch_size, num_patches, num_vars, patch_length]

    # Restore the original order of patches using the restore indices.
    masked_tensor = torch.gather(
        x_combined,
        dim=1,
        index=restore_indices.unsqueeze(-1).repeat(1, 1, 1, patch_length),
    )
    # masked_tensor shape: [batch_size, num_patches, num_vars, patch_length]

    # Generate a binary mask indicating which patches were kept (0) and which were masked (1).
    mask = torch.ones(batch_size, num_patches, num_vars, device=input_tensor.device)
    mask[:, :num_keep, :] = 0

    # Reorder the mask to match the original order of patches.
    mask = torch.gather(mask, dim=1, index=restore_indices)

    return masked_tensor, kept_tensor, mask, restore_indices


def patch_mse_loss(predictions, targets, mask):
    """
    Computes the mean squared error (MSE) loss over patches, weighted by a mask.

    This function calculates the MSE loss between the predicted and target patches,
    averages the loss over the patch_length dimension, applies a mask to focus the loss
    on valid entries, and then normalizes by the sum of the mask values.

    Args:
        predictions (torch.Tensor): Predicted tensor of shape
            [batch_size, num_patches, num_vars, patch_length].
        targets (torch.Tensor): Ground truth tensor of shape
            [batch_size, num_patches, num_vars, patch_length].
        mask (torch.Tensor): Mask tensor of shape
            [batch_size, num_patches, num_vars] that indicates which entries should contribute to the loss.

    Returns:
        torch.Tensor: A scalar tensor representing the masked average MSE loss.
    """
    # Compute the squared error between predictions and targets.
    squared_error = (predictions - targets) ** 2

    # Average the squared error over the patch_length dimension.
    # Resulting shape: [batch_size, num_patches, num_vars].
    mse_per_patch = squared_error.mean(dim=-1)

    # Apply the mask to focus on valid patches, then sum the masked losses.
    total_loss = (mse_per_patch * mask).sum()

    # Normalize by the sum of the mask values to get the average loss.
    average_loss = total_loss / mask.sum()

    return average_loss


def get_data_loader_manager(config: PreTrainingConfig) -> DataLoaderManager:
    # Get data loader parameters
    data_loader_params = DataParams(
        dset=config.dataset,
        context_points=config.input_length,
        target_points=config.prediction_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        features=config.features,
        use_time_features=(
            True
            if hasattr(config, "embed") and config.embed == "timeFeatures"
            else False
        ),
    )
    # Get data loaders
    data_loader_manager = get_dataloaders(data_loader_params)

    return data_loader_manager


# Find learning rate using the valley method
def find_learning_rate(
    config: PreTrainingConfig | FinetuneConfig | TransferLearningConfig,
    device,
    starting_lr=1e-7,
    end_lr=10,
    num_iter=100,
    beta=0.98,
    head_type="pretrain",
    mode="pretrain",
    checkpoint_folder: str = None,
):

    num_patches = (
        max(config.input_length, config.patch_length) - config.patch_length
    ) // config.stride + 1
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    train_loader = data_loader_manager.train_loader



    # Get model and move to device
    model = PatchTST(
        input_channels=data_loader_manager.vars,
        target_dim=config.prediction_length,
        patch_length=config.patch_length,
        stride=config.stride,
        num_patches=num_patches,
        num_layers=config.num_encoder_layers,
        model_dim=config.d_model,
        num_heads=config.n_heads,
        feedforward_dim=config.d_fcn,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        activation="relu",
        head_type=head_type,
        residual_attention=False,
        shared_projection=True,
    ).to(device)

    if mode == "finetuning":
        config: FinetuneConfig = config
        if hasattr(config, "pretrained_model"):
            pretrained_model = config.pretrained_model
        else:
            pretrained_model = os.path.join(checkpoint_folder, "checkpoint.pt")
        model = load_weights(
            pretrained_model, model, exclude_head=True, device=device
        )
        loss_func = torch.nn.MSELoss(reduction="mean")
    elif mode == "pretrain":
        loss_func = patch_mse_loss
    elif mode == "supervised":
        loss_func = torch.nn.MSELoss(reduction="mean")
    # -- Before Fit --

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    # Metrics
    losses, lrs = [], []
    best_loss, avg_loss = np.inf, 0
    train_iter = 0

    # Set the learning rate
    lrs = [starting_lr] * len(optimizer.param_groups)

    # Initialize the proper learning rate policy
    scheduler = ExponentialLR(optimizer, end_lr, num_iter)

    # -- END --
    num_epochs = train_iter // len(train_loader) + 1
    for epoch in range(num_epochs):
        train_loss = 0
        # -- Before Epoch Train --

        # -- END --
        model.train()
        for batch in tqdm(
            train_loader, desc=f"[LR FINDER] Epoch {epoch + 1}/{num_epochs}"
        ):
            optimizer.zero_grad()
            # -- Before Batch Train --

            # -- END --
            x, y = batch
            x, y = x.to(device), y.to(device)
            # -- Before Forward --

            # Revin
            if config.revin:
                revin = RevIN(
                    num_features=data_loader_manager.vars,
                    eps=1e-5,
                    affine=False,
                )
                x_revin = revin(x, "norm")
                x = x_revin

            if mode == "finetuning" or mode == "supervised":
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                x = input_patch
            else:  # Pretraining
                # Patching and masking
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                input_mask, _, mask, _ = random_masking(input_patch, config.mask_ratio)
                mask = mask.bool()
                x = input_mask  # Masked input
                y = input_patch  # Unmasked input

            # -- END --

            y_hat = model(x)

            # -- After Forward --
            # DON'T DENORM HERE
            # if config.revin:
            #     y_hat = revin(y_hat, "denorm")
            # -- END --
            if mode == "finetuning" or mode == "supervised":
                loss = loss_func(y_hat, y)
            else:
                loss = loss_func(y_hat, y, mask)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            # -- After Batch Train --
            train_iter += 1
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

            # Smooth the loss if beta is specified
            avg_loss = beta * avg_loss + (1 - beta) * loss.detach().item()
            smoothed_loss = avg_loss / (1 - beta**train_iter)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            # Stop if exploding losses
            if smoothed_loss > 4 * best_loss:
                break
            if train_iter > num_iter:
                break

            # -- END --

        # -- After Epoch Train --

        # -- END --
        train_loss /= len(train_loader)

    # -- After Fit --
    # Reset Gradient
    optimizer.zero_grad()

    # Determine the suggested learning rate based on the longest "valley" in the loss curve.
    num_points = len(losses)
    # These variables will mark the start and end indices of the longest decreasing sequence (valley)
    longest_valley_start, longest_valley_end = 0, 0

    # Compute the length of the longest decreasing sequence (LDS) ending at each index.
    # Each element in `lds` represents the length of the valley ending at that loss point.
    lds = [1] * num_points
    for i in range(1, num_points):
        for j in range(i):
            # If the current loss is lower than a previous loss and extending the valley
            # increases the LDS length at index i, then update it.
            if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
        # Update the overall longest valley if the current valley ending at i is longer.
        if lds[i] > lds[longest_valley_end]:
            longest_valley_end = i
            longest_valley_start = longest_valley_end - lds[i]

    # Divide the valley into three equal sections and pick an index in the middle section.
    valley_length = longest_valley_end - longest_valley_start
    section_length = valley_length / 3
    # Choose an index roughly in the center of the middle section:
    chosen_index = longest_valley_start + int(section_length) + int(section_length / 2)

    # Use the chosen index to select the corresponding learning rate.
    lr_suggestion = float(lrs[chosen_index])
    return lr_suggestion
    # -- END --


def pre_train_model(
    config: PreTrainingConfig,
    device: torch.device,
    suggested_lr: float,
    checkpoint_folder: str,
):

    num_patches = (
        max(config.input_length, config.patch_length) - config.patch_length
    ) // config.stride + 1
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    train_loader = data_loader_manager.train_loader
    val_loader = data_loader_manager.val_loader

    # Total number of training steps per epoch
    train_steps = len(train_loader)

    # Create checkpoint folder
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Get model and move to device
    model = PatchTST(
        input_channels=data_loader_manager.vars,
        target_dim=config.prediction_length,
        patch_length=config.patch_length,
        stride=config.stride,
        num_patches=num_patches,
        num_layers=config.num_encoder_layers,
        model_dim=config.d_model,
        num_heads=config.n_heads,
        feedforward_dim=config.d_fcn,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        activation="relu",
        head_type="pretrain",
        residual_attention=False,
        shared_projection=True,
    ).to(device)

    # -- Before Fit --

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    max_lr = suggested_lr if suggested_lr else config.learning_rate
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=None,
        steps_per_epoch=train_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
        epochs=config.epochs,
    )
    lrs = []

    # -- END --
    best_val_loss = float("inf")

    early_stopping = EarlyStopping(
        patience=config.patience,
        verbose=True,
        path=os.path.join(checkpoint_folder, "checkpoint.pt"),
        best_metric=best_val_loss,
    )
    for epoch in range(config.epochs):
        train_loss = 0
        # -- Before Epoch Train --

        # -- END --
        model.train()
        for batch in tqdm(
            train_loader, desc=f"[Pretrain] Epoch {epoch + 1}/{config.epochs}"
        ):
            optimizer.zero_grad()
            # -- Before Batch Train --

            # -- END --
            x, y = batch
            x, y = x.to(device), y.to(device)
            # -- Before Forward --

            # Revin
            if config.revin:
                revin = RevIN(
                    num_features=data_loader_manager.vars,
                    eps=1e-5,
                    affine=False,
                )
                x_revin = revin(x, "norm")
                x = x_revin

            # Patching and masking
            input_patch, num_patches = create_patch(
                x, config.patch_length, config.stride
            )
            input_mask, _, mask, _ = random_masking(input_patch, config.mask_ratio)
            mask = mask.bool()
            x = input_mask  # Masked input
            y = input_patch  # Unmasked input

            # -- END --

            y_hat = model(x)

            # -- After Forward --
            # DON'T DENORM HERE
            # if config.revin:
            #     y_hat = revin(y_hat, "denorm")
            # -- END --
            loss = patch_mse_loss(y_hat, y, mask)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

            # -- After Batch Train --
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
            # -- END --

        # -- After Epoch Train --

        # -- END --
        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation - Epoch {epoch + 1}/{config.epochs}"
            ):
                x, y = batch
                x, y = x.to(device), y.to(device)

                # -- Before Forward --

                # Revin
                if config.revin:
                    revin = RevIN(
                        num_features=data_loader_manager.vars,
                        eps=1e-5,
                        affine=False,
                    )
                    x_revin = revin(x, "norm")
                    x = x_revin

                # Patching and masking
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                input_mask, _, mask, _ = random_masking(input_patch, config.mask_ratio)
                mask = mask.bool()
                x = input_mask  # Masked input
                y = input_patch  # Unmasked input

                # -- END --

                y_hat = model(x)

                # -- After Forward --
                # DON'T DENORM HERE
                # if config.revin:
                #     y_hat = revin(y_hat, "denorm")
                # -- END --
                loss = patch_mse_loss(y_hat, y, mask)

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
    # -- After Fit --
    # Reset Gradient
    optimizer.zero_grad()


def finetune_model(
    config: FinetuneConfig,
    device: torch.device,
    suggested_lr: float,
    checkpoint_folder: str,
    linear_probe_only: bool,
):

    num_patches = (
        max(config.input_length, config.patch_length) - config.patch_length
    ) // config.stride + 1
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    train_loader = data_loader_manager.train_loader
    val_loader = data_loader_manager.val_loader

    # Total number of training steps per epoch
    train_steps = len(train_loader)

    # Create checkpoint folder
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Finetune the head of freeze_epochs > 0:
    if config.freeze_epochs > 0:
        print("(Linear Probing) Finetuning the Head, while Freezing backbone")

        # Get model and move to device
        model = PatchTST(
            input_channels=data_loader_manager.vars,
            target_dim=config.prediction_length,
            patch_length=config.patch_length,
            stride=config.stride,
            num_patches=num_patches,
            num_layers=config.num_encoder_layers,
            model_dim=config.d_model,
            num_heads=config.n_heads,
            feedforward_dim=config.d_fcn,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            activation="relu",
            head_type="prediction",
            residual_attention=False,
            shared_projection=True,
        ).to(device)

        # Load weights
        if hasattr(config, "pretrained_model"):
            pretrained_model = config.pretrained_model
        else:
            pretrained_model = os.path.join(checkpoint_folder, "checkpoint.pt")
        model = load_weights(
            pretrained_model, model, exclude_head=True, device=device
        )
        if hasattr(model, "head"):
            print("Found model Head - Freezing Backbone")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            raise Exception("Model does not have a head attribute")

        # -- Before Fit --

        # Optimizer
        loss_func = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

        max_lr = suggested_lr if suggested_lr else config.learning_rate
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=None,
            steps_per_epoch=train_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
            last_epoch=-1,
            verbose=False,
            epochs=config.freeze_epochs,
        )
        lrs = []

        # -- END --
        best_val_loss = float("inf")

        early_stopping = EarlyStopping(
            patience=config.patience,
            verbose=True,
            path=os.path.join(checkpoint_folder, "checkpoint.pt"),
            best_metric=best_val_loss,
        )
        for epoch in range(config.freeze_epochs):
            train_loss = 0
            # -- Before Epoch Train --

            # -- END --
            model.train()
            for batch in tqdm(
                train_loader,
                desc=f"[Head Finetuning] Epoch {epoch + 1}/{config.freeze_epochs}",
            ):
                optimizer.zero_grad()
                # -- Before Batch Train --

                # -- END --
                x, y = batch
                x, y = x.to(device), y.to(device)
                # -- Before Forward --

                # Revin
                if config.revin:
                    revin = RevIN(
                        num_features=data_loader_manager.vars,
                        eps=1e-5,
                        affine=False,
                    )
                    x_revin = revin(x, "norm")
                    x = x_revin

                # Patching and masking
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                x = input_patch

                # -- END --

                y_hat = model(x)

                # -- After Forward --
                if config.revin:
                    y_hat = revin(y_hat, "denorm")
                # -- END --
                loss = loss_func(y_hat, y)
                loss.backward()
                train_loss += loss.item()

                optimizer.step()

                # -- After Batch Train --
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
                # -- END --

            # -- After Epoch Train --

            # -- END --
            train_loss /= len(train_loader)

            val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"[Head Finetuning] Validation - Epoch {epoch + 1}/{config.freeze_epochs}",
                ):
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                    # -- Before Forward --

                    # Revin
                    if config.revin:
                        revin = RevIN(
                            num_features=data_loader_manager.vars,
                            eps=1e-5,
                            affine=False,
                        )
                        x_revin = revin(x, "norm")
                        x = x_revin

                    # Patching and masking
                    input_patch, num_patches = create_patch(
                        x, config.patch_length, config.stride
                    )
                    x = input_patch

                    # -- END --

                    y_hat = model(x)

                    # -- After Forward --
                    if config.revin:
                        y_hat = revin(y_hat, "denorm")
                    # -- END --
                    loss = loss_func(y_hat, y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{config.freeze_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            early_stopping(val_loss, model, optimizer, epoch)

            best_val_loss = early_stopping.best_metric

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # -- After Fit --
        # Reset Gradient
        optimizer.zero_grad()

    # Unallocate model
    del model
    torch.cuda.empty_cache()

    # Finetune the entire network if n_epochs > 0
    if config.finetune_epochs > 0 and not linear_probe_only:
        # Get model and move to device
        model = PatchTST(
            input_channels=data_loader_manager.vars,
            target_dim=config.prediction_length,
            patch_length=config.patch_length,
            stride=config.stride,
            num_patches=num_patches,
            num_layers=config.num_encoder_layers,
            model_dim=config.d_model,
            num_heads=config.n_heads,
            feedforward_dim=config.d_fcn,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            activation="relu",
            head_type="prediction",
            residual_attention=False,
            shared_projection=True,
        )

        # Load weights after head finetuning
        model = load_weights(
            os.path.join(checkpoint_folder, "checkpoint.pt"),
            model,
            exclude_head=False,
        )

        model = model.to(device)

        print("Finetuning the entire network, unfreezing all params")
        for param in model.parameters():
            param.requires_grad = True

        # -- Before Fit --

        # Optimizer
        loss_func = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(model.parameters(), config.learning_rate // 2)

        max_lr = suggested_lr if suggested_lr else config.learning_rate
        max_lr = max_lr // 2
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=None,
            steps_per_epoch=train_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
            last_epoch=-1,
            verbose=False,
            epochs=config.finetune_epochs,
        )
        lrs = []

        # -- END --
        best_val_loss = float("inf")

        early_stopping = EarlyStopping(
            patience=config.patience,
            verbose=True,
            path=os.path.join(checkpoint_folder, "checkpoint.pt"),
            best_metric=best_val_loss,
        )
        for epoch in range(config.finetune_epochs):
            train_loss = 0
            # -- Before Epoch Train --

            # -- END --
            model.train()
            for batch in tqdm(
                train_loader,
                desc=f"[Full Finetuning] Epoch {epoch + 1}/{config.finetune_epochs}",
            ):
                optimizer.zero_grad()
                # -- Before Batch Train --

                # -- END --
                x, y = batch
                x, y = x.to(device), y.to(device)
                # -- Before Forward --

                # Revin
                if config.revin:
                    revin = RevIN(
                        num_features=data_loader_manager.vars,
                        eps=1e-5,
                        affine=False,
                    )
                    x_revin = revin(x, "norm")
                    x = x_revin

                # Patching and masking
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                x = input_patch

                # -- END --

                y_hat = model(x)

                # -- After Forward --
                if config.revin:
                    y_hat = revin(y_hat, "denorm")
                # -- END --
                loss = loss_func(y_hat, y)
                loss.backward()
                train_loss += loss.item()

                optimizer.step()

                # -- After Batch Train --
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
                # -- END --

            # -- After Epoch Train --

            # -- END --
            train_loss /= len(train_loader)

            val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"[Full Finetuning] Validation - Epoch {epoch + 1}/{config.finetune_epochs}",
                ):
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                    # -- Before Forward --

                    # Revin
                    if config.revin:
                        revin = RevIN(
                            num_features=data_loader_manager.vars,
                            eps=1e-5,
                            affine=False,
                        )
                        x_revin = revin(x, "norm")
                        x = x_revin

                    # Patching and masking
                    input_patch, num_patches = create_patch(
                        x, config.patch_length, config.stride
                    )
                    x = input_patch

                    # -- END --

                    y_hat = model(x)

                    # -- After Forward --
                    if config.revin:
                        y_hat = revin(y_hat, "denorm")
                    # -- END --
                    loss = loss_func(y_hat, y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}/{config.finetune_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            early_stopping(val_loss, model, optimizer, epoch)

            best_val_loss = early_stopping.best_metric

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # -- After Fit --
        # Reset Gradient
        optimizer.zero_grad()

def supervised_train_model(
    config: SupervisedTrainingConfig,
    device: torch.device,
    suggested_lr: float,
    checkpoint_folder: str,
):

    num_patches = (
        max(config.input_length, config.patch_length) - config.patch_length
    ) // config.stride + 1
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    train_loader = data_loader_manager.train_loader
    val_loader = data_loader_manager.val_loader

    # Total number of training steps per epoch
    train_steps = len(train_loader)

    # Create checkpoint folder
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Get model and move to device
    model = PatchTST(
        input_channels=data_loader_manager.vars,
        target_dim=config.prediction_length,
        patch_length=config.patch_length,
        stride=config.stride,
        num_patches=num_patches,
        num_layers=config.num_encoder_layers,
        model_dim=config.d_model,
        num_heads=config.n_heads,
        feedforward_dim=config.d_fcn,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        activation="relu",
        head_type="prediction",
        residual_attention=False,
        shared_projection=True,
    ).to(device)

    # -- Before Fit --

    # Optimizer
    loss_func = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    max_lr = suggested_lr if suggested_lr else config.learning_rate
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=None,
        steps_per_epoch=train_steps,
        pct_start=0.2,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
        epochs=config.epochs,
    )
    lrs = []

    # -- END --
    best_val_loss = float("inf")

    early_stopping = EarlyStopping(
        patience=config.patience,
        verbose=True,
        path=os.path.join(checkpoint_folder, "checkpoint.pt"),
        best_metric=best_val_loss,
    )
    for epoch in range(config.epochs):
        train_loss = 0
        # -- Before Epoch Train --

        # -- END --
        model.train()
        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
        ):
            optimizer.zero_grad()
            # -- Before Batch Train --

            # -- END --
            x, y = batch
            x, y = x.to(device), y.to(device)
            # -- Before Forward --

            # Revin
            if config.revin:
                revin = RevIN(
                    num_features=data_loader_manager.vars,
                    eps=1e-5,
                    affine=False,
                )
                x_revin = revin(x, "norm")
                x = x_revin

            # Patching and masking
            input_patch, num_patches = create_patch(
                x, config.patch_length, config.stride
            )
            x = input_patch

            # -- END --

            y_hat = model(x)

            # -- After Forward --
            if config.revin:
                y_hat = revin(y_hat, "denorm")
            # -- END --
            loss = loss_func(y_hat, y)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

            # -- After Batch Train --
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
            # -- END --

        # -- After Epoch Train --

        # -- END --
        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Validation - Epoch {epoch + 1}/{config.epochs}",
            ):
                x, y = batch
                x, y = x.to(device), y.to(device)

                # -- Before Forward --

                # Revin
                if config.revin:
                    revin = RevIN(
                        num_features=data_loader_manager.vars,
                        eps=1e-5,
                        affine=False,
                    )
                    x_revin = revin(x, "norm")
                    x = x_revin

                # Patching and masking
                input_patch, num_patches = create_patch(
                    x, config.patch_length, config.stride
                )
                x = input_patch

                # -- END --

                y_hat = model(x)

                # -- After Forward --
                if config.revin:
                    y_hat = revin(y_hat, "denorm")
                # -- END --
                loss = loss_func(y_hat, y)

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
    # -- After Fit --
    # Reset Gradient
    optimizer.zero_grad()



def test_model(
    config: FinetuneConfig,
    device: torch.device,
    checkpoint_folder: str,
):

    num_patches = (
        max(config.input_length, config.patch_length) - config.patch_length
    ) // config.stride + 1
    # Get data loaders
    data_loader_manager = get_data_loader_manager(config)
    test_loader = data_loader_manager.test_loader

    # Get model and move to device
    model = PatchTST(
        input_channels=data_loader_manager.vars,
        target_dim=config.prediction_length,
        patch_length=config.patch_length,
        stride=config.stride,
        num_patches=num_patches,
        num_layers=config.num_encoder_layers,
        model_dim=config.d_model,
        num_heads=config.n_heads,
        feedforward_dim=config.d_fcn,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        activation="relu",
        head_type="prediction",
        residual_attention=False,
        shared_projection=True,
    ).to(device)

    # Load weights after head finetuning
    model = load_weights(
        os.path.join(checkpoint_folder, "checkpoint.pt"),
        model,
        exclude_head=False,
        device=device,
    )

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            # -- Before Forward --

            # Revin
            if config.revin:
                revin = RevIN(
                    num_features=data_loader_manager.vars,
                    eps=1e-5,
                    affine=False,
                )
                x_revin = revin(x, "norm")
                x = x_revin

            # Patching and masking
            input_patch, num_patches = create_patch(
                x, config.patch_length, config.stride
            )
            x = input_patch

            # -- END --

            y_hat = model(x)

            # -- After Forward --
            if config.revin:
                y_hat = revin(y_hat, "denorm")

            y_true.append(y)
            y_pred.append(y_hat)
            # -- END --

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Calculate MSE and MAE
    mse = F.mse_loss(y_true, y_pred).item()
    mae = F.l1_loss(y_true, y_pred).item()

    metrics = {
        "mse": mse,
        "mae": mae,
    }

    return metrics
