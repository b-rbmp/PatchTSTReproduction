import argparse
from dataclasses import dataclass
from typing import Union

@dataclass
class SupervisedTrainingConfig:
    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool

    # Bootstrap parameters
    bootstrap_iterations: int



@dataclass
class PreTrainingConfig:
    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool


    # Pretraining parameters
    mask_ratio: float

@dataclass
class FinetuneConfig:
    # Params
    finetune_mode: bool
    linear_probe_mode: bool
    pretrained_model: str
    freeze_epochs: int
    finetune_epochs: int

    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool


    # Pretraining parameters
    mask_ratio: float

@dataclass
class PretrainFinetuneConfig:
    # Params
    finetune_mode: bool
    linear_probe_mode: bool
    freeze_epochs: int
    finetune_epochs: int
    epochs: int

    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool


    # Pretraining parameters
    mask_ratio: float

    # Bootstrap parameters
    bootstrap_iterations: int

@dataclass
class TransferLearningConfig:
    # Params
    finetune_mode: bool
    linear_probe_mode: bool
    freeze_epochs: int
    finetune_epochs: int
    epochs: int

    # General parameters
    model_identifier: str
    seed: int
    model: str
    input_length: int
    prediction_length: int
    dataset: str
    dataset_origin: str
    checkpoint_dir: str
    features: str

    # Formers
    d_model: int
    n_heads: int
    num_encoder_layers: int
    d_fcn: int
    dropout: float



    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    head_dropout: float
    revin: bool

    # Training parameters
    batch_size: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool


    # Pretraining parameters
    mask_ratio: float

    # Bootstrap parameters
    bootstrap_iterations: int

def get_parser_supervised() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST supervised training")

    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=256,
        help="Fully connected layer dimension (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )

    # Bootstrap parameters
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5,
        help="Number of bootstrap iterations (default)"
    )
    
    return parser

def get_parser_pretrain() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST pre-training learning")

    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=256,
        help="Fully connected layer dimension (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )
    
    # Pretraining parameters
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Masking ratio for the input (default)"
    )
    return parser

def get_parser_finetune() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST finetune learning")

    # Parameters
    parser.add_argument(
        "--finetune_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable finetuning (and disable Linear Probe Mode)",
    )
    parser.add_argument(
        "--linear_probe_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear probe (Don't forget to disable Finetune Mode)",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="checkpoints/pretrain/PatchTST/checkpoint.pt",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Number of finetuning (whole network) epochs (default: 20)",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of finetuning head epochs (default: 20)",
    )
    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=256,
        help="Fully connected layer dimension (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )
    

    # Pretraining parameters
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Masking ratio for the input (default)"
    )
    return parser

def get_parser_pretrain_finetune() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST pretrain finetune learning")

    # Parameters
    parser.add_argument(
        "--finetune_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable finetuning (and disable Linear Probe Mode)",
    )
    parser.add_argument(
        "--linear_probe_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear probe (Don't forget to disable Finetune Mode)",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Number of finetuning (whole network) epochs (default: 20)",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of finetuning head epochs (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of pretraining epochs (default: 100)",
    )
    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=256,
        help="Fully connected layer dimension (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )
    

    # Pretraining parameters
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Masking ratio for the input (default)"
    )

    # Bootstrap parameters
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5,
        help="Number of bootstrap iterations (default)"
    )
    return parser

def get_parser_transfer_learning() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST pretrain finetune learning")

    # Parameters
    parser.add_argument(
        "--finetune_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable finetuning (and disable Linear Probe Mode)",
    )
    parser.add_argument(
        "--linear_probe_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable linear probe (Don't forget to disable Finetune Mode)",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=20,
        help="Number of finetuning (whole network) epochs (default: 20)",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10,
        help="Number of finetuning head epochs (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of pretraining epochs (default: 100)",
    )
    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=512,
        help="Input sequence length (default: 512)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=96,
        help="Prediction length (default: 96)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="etth1",
        help="Dataset name (default: etth1)",
    )
    parser.add_argument(
        "--dataset_origin",
        type=str,
        default="electricity",
        help="Dataset origin name (default: electricity)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )


    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model dimension (default: 128)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of heads (default: 16)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=256,
        help="Fully connected layer dimension (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )


    # PatchTST model parameters
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
        help="Kernel size (default: 25)",
    )
    parser.add_argument(
        "--patch_length",
        type=int,
        default=12,
        help="Patch length (default: 12)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (default: 12)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="Head dropout (default: 0.2)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--use_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA",
    )
    

    # Pretraining parameters
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="Masking ratio for the input (default)"
    )

    # Bootstrap parameters
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=5,
        help="Number of bootstrap iterations (default)"
    )
    return parser


def parse_args(mode: str = "pretrain") -> PreTrainingConfig | FinetuneConfig | PretrainFinetuneConfig | SupervisedTrainingConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    if mode == "pretrain":
        parser = get_parser_pretrain()
        args = parser.parse_args()
        
        return PreTrainingConfig(
            model_identifier=args.model_identifier,
            seed=args.seed,
            model=args.model,
            input_length=args.input_length,
            prediction_length=args.prediction_length,
            dataset=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            features=args.features,
            kernel_size=args.kernel_size,
            patch_length=args.patch_length,
            stride=args.stride,
            patch_padding=args.patch_padding,
            head_dropout=args.head_dropout,
            revin=args.revin,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,

            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_encoder_layers,
            d_fcn=args.d_fcn,
            dropout=args.dropout,

            mask_ratio=args.mask_ratio,

        )
    elif mode == "finetune":
        parser = get_parser_finetune()
        args = parser.parse_args()
        
        return FinetuneConfig(
            finetune_mode=args.finetune_mode,
            linear_probe_mode=args.linear_probe_mode,
            pretrained_model=args.pretrained_model,
            freeze_epochs=args.freeze_epochs,
            finetune_epochs=args.finetune_epochs,

            model_identifier=args.model_identifier,
            seed=args.seed,
            model=args.model,
            input_length=args.input_length,
            prediction_length=args.prediction_length,
            dataset=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            features=args.features,
            kernel_size=args.kernel_size,
            patch_length=args.patch_length,
            stride=args.stride,
            patch_padding=args.patch_padding,
            head_dropout=args.head_dropout,
            revin=args.revin,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,

            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_encoder_layers,
            d_fcn=args.d_fcn,
            dropout=args.dropout,

            mask_ratio=args.mask_ratio,

        )
    elif mode == "pretrain_finetune":
        parser = get_parser_pretrain_finetune()
        args = parser.parse_args()
        
        return PretrainFinetuneConfig(
            finetune_mode=args.finetune_mode,
            linear_probe_mode=args.linear_probe_mode,
            freeze_epochs=args.freeze_epochs,
            finetune_epochs=args.finetune_epochs,
            epochs=args.epochs,

            model_identifier=args.model_identifier,
            seed=args.seed,
            model=args.model,
            input_length=args.input_length,
            prediction_length=args.prediction_length,
            dataset=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            features=args.features,
            kernel_size=args.kernel_size,
            patch_length=args.patch_length,
            stride=args.stride,
            patch_padding=args.patch_padding,
            head_dropout=args.head_dropout,
            revin=args.revin,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,

            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_encoder_layers,
            d_fcn=args.d_fcn,
            dropout=args.dropout,

            mask_ratio=args.mask_ratio,

            bootstrap_iterations=args.bootstrap_iterations,
        )

    elif mode == "supervised":
        parser = get_parser_supervised()
        args = parser.parse_args()
        return SupervisedTrainingConfig(
            model_identifier=args.model_identifier,
            seed=args.seed,
            model=args.model,
            input_length=args.input_length,
            prediction_length=args.prediction_length,
            dataset=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            features=args.features,
            kernel_size=args.kernel_size,
            patch_length=args.patch_length,
            stride=args.stride,
            patch_padding=args.patch_padding,
            head_dropout=args.head_dropout,
            revin=args.revin,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,

            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_encoder_layers,
            d_fcn=args.d_fcn,
            dropout=args.dropout,

            bootstrap_iterations=args.bootstrap_iterations,
        )
    elif mode == "transfer_learning":
        parser = get_parser_transfer_learning()
        args = parser.parse_args()
        return TransferLearningConfig(
            finetune_mode=args.finetune_mode,
            linear_probe_mode=args.linear_probe_mode,
            freeze_epochs=args.freeze_epochs,
            finetune_epochs=args.finetune_epochs,
            epochs=args.epochs,
            

            model_identifier=args.model_identifier,
            seed=args.seed,
            model=args.model,
            input_length=args.input_length,
            prediction_length=args.prediction_length,
            dataset=args.dataset,
            dataset_origin=args.dataset_origin,
            checkpoint_dir=args.checkpoint_dir,
            features=args.features,
            kernel_size=args.kernel_size,
            patch_length=args.patch_length,
            stride=args.stride,
            patch_padding=args.patch_padding,
            head_dropout=args.head_dropout,
            revin=args.revin,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            use_cuda=args.use_cuda,

            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.num_encoder_layers,
            d_fcn=args.d_fcn,
            dropout=args.dropout,

            mask_ratio=args.mask_ratio,

            bootstrap_iterations=args.bootstrap_iterations,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")