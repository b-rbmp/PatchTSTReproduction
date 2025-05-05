import argparse
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # General parameters
    model_identifier: str
    seed: int
    train_mode: bool
    model: str
    input_length: int
    start_token_length: int
    prediction_length: int
    dataset: str
    checkpoint_dir: str
    features: str
    target_feature: str
    encoding_freq: str

    # Formers parameters
    embed: str
    output_attention: bool
    encoder_input_size: int
    decoder_input_size: int
    output_size: int
    d_model: int
    n_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    d_fcn: int
    moving_avg_window_length: int
    attention_factor: int
    encoder_distil: bool
    dropout: float
    activation: str
    output_activation: bool
    do_predict: bool


    # PatchTST model parameters
    kernel_size: int
    patch_length: int
    stride: int
    patch_padding: str
    fc_dropout: float
    head_dropout: float
    revin: bool
    revin_affine: bool
    subtract_last: bool
    decomposition: bool
    individual_head: bool

    # Training parameters
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    num_workers: int
    use_cuda: bool
    learning_rate_adjustment: str
    lr_pct_start: float

    # Bootstrap parameters
    bootstrap_iterations: int

    # Ablation
    only_patching: bool

    # FP16
    fp16: bool


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PatchTST training.
    """
    parser = argparse.ArgumentParser(description="PatchTST few-shot learning")

    # General parameters
    parser.add_argument(
        "--model_identifier",
        type=str,
        default="PatchTST",
        help="Model identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable training mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PatchTST",
        help="Model architecture (default: PatchTST)",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=96,
        help="Input sequence length (default: 96)",
    )
    parser.add_argument(
        "--start_token_length",
        type=int,
        default=48,
        help="Start token length (default: 48)",
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
        default="weather",
        help="Dataset name (default: weather)",
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
        default="T",
        help="Forecasting Task - [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )
    parser.add_argument(
        "--target_feature",
        type=str,
        default="OT",
        help="Target feature",
    )
    parser.add_argument(
        "--encoding_freq",
        type=str,
        default="h",
        help="Frequency for encoding - [s: Secondly, t: Minutely, h: Hourly, d: Daily, b: Business days, w: Weekly, m: Monthly]",
    )

    # Formers parameters
    parser.add_argument(
        "--embed",
        type=str,
        default="fixed",
        help="Embedding type (default: fixed) - [timeFeatures, fixed, learned]",
    )
    parser.add_argument(
        "--output_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable output attention",
    )
    parser.add_argument(
        "--encoder_input_size",
        type=int,
        default=7,
        help="Encoder input size (default: 7)",
    )
    parser.add_argument(
        "--decoder_input_size",
        type=int,
        default=7,
        help="Decoder input size (default: 7)",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=7,
        help="Output size (default: 7)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension (default: 512)",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of heads (default: 8)",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=2,
        help="Number of encoder layers (default: 2)",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=1,
        help="Number of decoder layers (default: 1)",
    )
    parser.add_argument(
        "--d_fcn",
        type=int,
        default=2048,
        help="Fully connected layer dimension (default: 2048)",
    )
    parser.add_argument(
        "--moving_avg_window_length",
        type=int,
        default=25,
        help="Moving average window length (default: 25)",
    )
    parser.add_argument(
        "--attention_factor",
        type=int,
        default=1,
        help="Attention factor (default: 1)",
    )
    parser.add_argument(
        "--encoder_distil",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable encoder distillation",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="Activation function (default: gelu)",
    )
    parser.add_argument(
        "--output_activation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable output activation",
    )
    parser.add_argument(
        "--do_predict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable prediction",
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
        default=16,
        help="Patch length (default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride (default: 8)",
    )
    parser.add_argument(
        "--patch_padding",
        type=str,
        default="end",
        help="'None: None; end: Padding on the end",
    )
    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.05,
        help="Fully connected layer dropout (default: 0.05)",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.0,
        help="Head dropout (default: 0.0)",
    )
    parser.add_argument(
        "--revin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RevIn",
    )
    parser.add_argument(
        "--revin_affine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RevIn affine",
    )
    parser.add_argument(
        "--subtract_last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable subtract last",
    )
    parser.add_argument(
        "--decomposition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable decomposition",
    )
    parser.add_argument(
        "--individual_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable individual head",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
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
    parser.add_argument(
        "--learning_rate_adjustment",
        type=str,
        default="type3",
        help="Learning rate adjustment (default: type3)",
    )
    parser.add_argument(
        "--lr_pct_start",
        type=float,
        default=0.3,
        help="Learning rate percentage start (default: 0.3)",
    )
    
    # Bootstrap parameters
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1,
        help="Number of bootstrap iterations",
    )

    # Ablations
    parser.add_argument(
        "--only_patching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable only patching",
    )

    # FP16
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FP16 training",
    )

    return parser


def parse_args() -> TrainingConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()

    return TrainingConfig(
        model_identifier=args.model_identifier,
        seed=args.seed,
        train_mode=args.train_mode,
        model=args.model,
        input_length=args.input_length,
        start_token_length=args.start_token_length,
        prediction_length=args.prediction_length,
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        features=args.features,
        target_feature=args.target_feature,
        encoding_freq=args.encoding_freq,
        kernel_size=args.kernel_size,
        patch_length=args.patch_length,
        stride=args.stride,
        patch_padding=args.patch_padding,
        fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout,
        revin=args.revin,
        revin_affine=args.revin_affine,
        subtract_last=args.subtract_last,
        decomposition=args.decomposition,
        individual_head=args.individual_head,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_workers=args.num_workers,
        use_cuda=args.use_cuda,
        learning_rate_adjustment=args.learning_rate_adjustment,
        lr_pct_start=args.lr_pct_start,
        bootstrap_iterations=args.bootstrap_iterations,

        embed=args.embed,
        output_attention=args.output_attention,
        encoder_input_size=args.encoder_input_size,
        decoder_input_size=args.decoder_input_size,
        output_size=args.output_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_fcn=args.d_fcn,
        moving_avg_window_length=args.moving_avg_window_length,
        attention_factor=args.attention_factor,
        encoder_distil=args.encoder_distil,
        dropout=args.dropout,
        activation=args.activation,
        output_activation=args.output_activation,
        do_predict=args.do_predict,
        only_patching=args.only_patching,
        fp16=args.fp16,
    )