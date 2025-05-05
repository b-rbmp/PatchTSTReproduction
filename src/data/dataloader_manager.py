import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd

# Ensure the src directory is in the Python path
sys.path.append(__file__.replace("src/data/dataloader_manager.py", ""))

from src.data.datasets import (
    ETTDataset,
    CustomDataset,
)

# Define supported datasets
SUPPORTED_DATASETS = [
    "ettm1",
    "ettm2",
    "etth1",
    "etth2",
    "electricity",
    "traffic",
    "illness",
    "weather",
    "exchange",
]

@dataclass
class DataParams:
    dset: str
    context_points: int
    target_points: int
    batch_size: int
    num_workers: int
    features: str
    use_time_features: bool

@dataclass
class DatasetInfo:
    """
    Dataclass to store information about each dataset.
    """

    dataset_cls: Any  # The Dataset class
    root_path: Path
    data_path: str


# Mapping dataset names to DatasetInfo entries with updated class references
DATASETS_INFO: Dict[str, DatasetInfo] = {
    "ettm1": DatasetInfo(
        dataset_cls=ETTDataset,
        root_path=Path("datasets/ETT-small/"),
        data_path="ETTm1.csv",
    ),
    "ettm2": DatasetInfo(
        dataset_cls=ETTDataset,
        root_path=Path("datasets/ETT-small/"),
        data_path="ETTm2.csv",
    ),
    "etth1": DatasetInfo(
        dataset_cls=ETTDataset,
        root_path=Path("datasets/ETT-small/"),
        data_path="ETTh1.csv",
    ),
    "etth2": DatasetInfo(
        dataset_cls=ETTDataset,
        root_path=Path("datasets/ETT-small/"),
        data_path="ETTh2.csv",
    ),
    "electricity": DatasetInfo(
        dataset_cls=CustomDataset,
        root_path=Path("datasets/electricity/"),
        data_path="electricity.csv",
    ),
    "traffic": DatasetInfo(
        dataset_cls=CustomDataset,
        root_path=Path("datasets/traffic/"),
        data_path="traffic.csv",
    ),
    "weather": DatasetInfo(
        dataset_cls=CustomDataset,
        root_path=Path("datasets/weather/"),
        data_path="weather.csv",
    ),
    "illness": DatasetInfo(
        dataset_cls=CustomDataset,
        root_path=Path("datasets/illness/"),
        data_path="national_illness.csv",
    ),
    "exchange": DatasetInfo(
        dataset_cls=CustomDataset,
        root_path=Path("datasets/exchange_rate/"),
        data_path="exchange_rate.csv",
    ),
}


class DataLoaderManager:
    """
    Manages train/validation/test DataLoaders for a given dataset class.
    Can also create additional DataLoaders if needed.
    """

    def __init__(
        self,
        dataset_cls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
    ):
        """
        Args:
            dataset_cls: A dataset class (e.g., ETTDatasetMinute) with an interface like ETTDatasetMinute(..., partition="train").
            dataset_kwargs (dict): Keyword arguments passed to the dataset_cls except 'partition'.
            batch_size (int): Batch size for the DataLoader.
            workers (int): Number of CPU workers for data loading.
            collate_fn (callable, optional): Collate function to merge a list of samples into a mini-batch.
            shuffle_train (bool): Whether to shuffle the training dataset.
            shuffle_val (bool): Whether to shuffle the validation dataset.
        """
        super().__init__()
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        # Remove any pre-existing 'partition' key to avoid conflicts
        dataset_kwargs.pop("partition", None)
        self.dataset_kwargs = dataset_kwargs

        # Instantiate DataLoaders
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
        self.test_loader = self._create_test_loader()

    def _create_train_loader(self):
        return self._create_loader(partition="train", shuffle=self.shuffle_train)

    def _create_val_loader(self):
        return self._create_loader(partition="val", shuffle=self.shuffle_val)

    def _create_test_loader(self):
        return self._create_loader(partition="test", shuffle=False)

    def _create_loader(self, partition: str, shuffle: bool = False):
        """
        Internal helper to instantiate a DataLoader for the specified partition.
        """
        dataset = self.dataset_cls(**self.dataset_kwargs, partition=partition)
        if len(dataset) == 0:
            return None  # Return None if the dataset for this split is empty
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(cls, parser):
        """
        Utility for command-line interface integration.
        Example:
            parser = argparse.ArgumentParser()
            DataLoaderManager.add_cli(parser)
            args = parser.parse_args()
        """
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="Number of CPU workers for PyTorch DataLoader.",
        )

    def add_loader(self, data, batch_size=None, shuffle=False, **kwargs):
        """
        Create or return an existing DataLoader for arbitrary data.

        Args:
            data: Can be a PyTorch DataLoader or Dataset.
                  If it's already a DataLoader, simply return it.
                  If it's a Dataset, a new DataLoader is created.
            batch_size (int, optional): Override the default batch size.
            shuffle (bool, optional): Whether to shuffle the data.
            **kwargs: Additional parameters for the DataLoader.

        Returns:
            A PyTorch DataLoader.
        """
        if isinstance(data, DataLoader):
            # Already a DataLoader, just return as is
            return data

        if batch_size is None:
            batch_size = self.batch_size

        if not isinstance(data, Dataset):
            raise ValueError("Argument 'data' must be a PyTorch Dataset or DataLoader.")

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )


def get_dataloaders(params: DataParams) -> DataLoaderManager:
    """
    Create and configure DataLoaderManager based on the provided parameters.

    Args:
        params: An object containing dataset and DataLoader configurations. Expected attributes:
            - dset (str): Name of the dataset.
            - context_points (int): Number of context points.
            - target_points (int): Number of target points.
            - batch_size (int): Batch size for DataLoaders.
            - num_workers (int): Number of worker processes for DataLoaders.
            - features (str): Type of features to use.
            - use_time_features (bool, optional): Whether to use time features.

    Returns:
        DataLoaderManager: An instance managing train, validation, and test DataLoaders.

    Raises:
        ValueError: If an unsupported dataset is specified or if the train split is empty.
    """
    # Validate dataset
    if params.dset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unrecognized dataset '{params.dset}'. Supported datasets: {SUPPORTED_DATASETS}"
        )

    # Retrieve dataset information
    dataset_info = DATASETS_INFO[params.dset]

    # Prepare dataset arguments
    # The improved code uses window_sizes, data_dir, file_name, feature_mode, apply_scaling, etc.
    size = [params.context_points, 0, params.target_points]
    dataset_kwargs = {
        "data_dir": str(dataset_info.root_path),
        "file_name": dataset_info.data_path,
        "feature_mode": params.features,
        "apply_scaling": True,
        "window_sizes": size,
        "add_time_features": getattr(params, "use_time_features", False),
    }

    # Initialize DataLoaderManager
    dataloader_manager = DataLoaderManager(
        dataset_cls=dataset_info.dataset_cls,
        dataset_kwargs=dataset_kwargs,
        batch_size=params.batch_size,
        workers=params.num_workers,
        shuffle_train=True,
        shuffle_val=False,
    )

    # Extract and assign additional attributes
    try:
        first_train_sample = next(iter(dataloader_manager.train_loader))
        if len(first_train_sample) == 4:
            inputs, targets, _, _ = first_train_sample
        else:
            inputs, targets = first_train_sample
        # Number of variables/features
        dataloader_manager.vars = inputs.shape[2]
        # Store context length
        dataloader_manager.len = params.context_points
        # Example usage: store c based on shape of targets
        dataloader_manager.c = targets.shape[0]
    except StopIteration:
        raise ValueError(f"The train split for dataset '{params.dset}' is empty.")

    return dataloader_manager


if __name__ == "__main__":

    def test_dataloader(params):
        """
        Create and iterate over train/val/test DataLoaders, then print shape details.
        """
        print(f"Testing with use_time_features={params.use_time_features}")
        dataloaders = get_dataloaders(params)

        for split_name, loader in [
            ("train", dataloaders.train_loader),
            ("val", dataloaders.val_loader),
            ("test", dataloaders.test_loader),
        ]:
            if loader is None:
                print(f"{split_name} loader is None (split might be empty).")
                continue

            print(f"\n=== {split_name.upper()} LOADER ===")
            for batch_idx, batch in enumerate(loader):
                # Some datasets return (seq_x, seq_y), others return (seq_x, seq_y, seq_x_mark, seq_y_mark)
                if len(batch) == 4:
                    seq_x, seq_y, seq_x_mark, seq_y_mark = batch
                    print(
                        f"  Batch {batch_idx}: "
                        f"seq_x={tuple(seq_x.shape)}, "
                        f"seq_y={tuple(seq_y.shape)}, "
                        f"seq_x_mark={tuple(seq_x_mark.shape)}, "
                        f"seq_y_mark={tuple(seq_y_mark.shape)}"
                    )
                else:
                    seq_x, seq_y = batch
                    print(
                        f"  Batch {batch_idx}: "
                        f"seq_x={tuple(seq_x.shape)}, "
                        f"seq_y={tuple(seq_y.shape)}"
                    )
                # Stop after a few batches for brevity
                if batch_idx >= 2:
                    break

    # Test 1: no time features
    class ParamsNoTime:
        dset = "etth2"
        context_points = 384
        target_points = 96
        batch_size = 64
        num_workers = 2
        features = "M"
        use_time_features = False

    test_dataloader(ParamsNoTime())

    # Test 2: with time features
    class ParamsWithTime:
        dset = "etth2"
        context_points = 384
        target_points = 96
        batch_size = 64
        num_workers = 2
        features = "M"
        use_time_features = True

    test_dataloader(ParamsWithTime())
