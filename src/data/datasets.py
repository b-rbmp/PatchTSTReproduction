import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Union
from src.data.time_features import time_features


def _to_tensor(*arrays: np.ndarray) -> Tuple[torch.Tensor, ...]:
    """Converts each NumPy array to a float32 torch.Tensor."""
    return tuple(torch.from_numpy(arr).float() for arr in arrays)


class BaseTimeSeriesDataset(Dataset):
    """
    Base class for time-series datasets, handling:
      - File I/O
      - Partition slicing (train/val/test)
      - Scaling and inverse transform
      - Time feature extraction
      - Sequence/window slicing
    """

    def __init__(
        self,
        data_dir: str,
        file_name: str,
        partition: str,
        window_sizes: Optional[List[int]] = None,
        feature_mode: str = "S",
        target_col: str = "OT",
        apply_scaling: bool = True,
        time_encoding: int = 0,
        frequency: str = "h",
        add_time_features: bool = False,
    ):
        assert partition in [
            "train",
            "val",
            "test",
            "pred",
        ], "Partition must be 'train', 'val', 'test', or 'pred'."

        # Default window sizes if none are provided
        if window_sizes is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = window_sizes

        self.data_dir = data_dir
        self.file_name = file_name
        self.partition = partition
        self.feature_mode = feature_mode
        self.target_col = target_col
        self.apply_scaling = apply_scaling
        self.time_encoding = time_encoding
        self.frequency = frequency
        self.add_time_features = add_time_features

        self.scaler = StandardScaler()
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None

    def _read_data(self) -> None:
        """
        Reads and processes data from CSV, applies scaling, and sets up time features.
        Should be overridden or extended by subclasses for custom partitioning logic.
        """
        raise NotImplementedError("Subclasses must implement _read_data().")

    def __getitem__(self, idx: int):
        """
        Retrieves a single time-series slice:
        seq_x (input), seq_y (labels), optional time features.
        """
        seq_start = idx
        seq_end = seq_start + self.seq_len
        label_start = seq_end - self.label_len
        label_end = label_start + self.label_len + self.pred_len

        seq_x = self.data_x[seq_start:seq_end]
        seq_y = self.data_y[label_start:label_end]

        seq_x_mark = self.data_stamp[seq_start:seq_end]
        seq_y_mark = self.data_stamp[label_start:label_end]

        if self.add_time_features:
            return _to_tensor(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else:
            return _to_tensor(seq_x, seq_y)

    def __len__(self) -> int:
        """
        Returns the number of possible sequences in the current partition.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Reverses the scaling transformation.
        """
        return self.scaler.inverse_transform(arr)


class ETTDataset(BaseTimeSeriesDataset):
    """
    Single class that can handle both Hourly (ETTh) and Minute (ETTm) ETT data.
    Partitioning is determined by the partition argument and the frequency chosen.
    """

    def __init__(
        self,
        data_dir: str,
        partition: str = "train",
        window_sizes: Optional[List[int]] = None,
        feature_mode: str = "S",
        file_name: str = "ETTh1.csv",
        target_col: str = "OT",
        apply_scaling: bool = True,
        time_encoding: int = 0,
        frequency: str = "h",
        add_time_features: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            file_name=file_name,
            partition=partition,
            window_sizes=window_sizes,
            feature_mode=feature_mode,
            target_col=target_col,
            apply_scaling=apply_scaling,
            time_encoding=time_encoding,
            frequency=frequency,
            add_time_features=add_time_features,
        )
        # Maps partition strings to an index for start/end slicing
        self.partition_map = {"train": 0, "val": 1, "test": 2, "pred": -1}
        self._read_data()

    def _read_data(self) -> None:
        """
        Reads CSV, partitions data, applies scaling, and sets up time stamps.
        Assumes ETT data has a 'date' column and uses fixed-size splits:
          - For Hourly dataset (ETTh), the total length is typically 12+4+4 months.
          - For Minute dataset (ETTm), typically 12+4+4 months * 24 * 4 steps/day.
        """
        df_raw = pd.read_csv(os.path.join(self.data_dir, self.file_name))
        # Example: partitioning for ETTh1
        if "m" in self.file_name.lower():
            # Minute dataset
            total_steps_per_day = 24 * 4
        else:
            # Hourly dataset
            total_steps_per_day = 24

        # Example fixed-split for ETTh/ETTm: 12 months train, 4 months val, 4 months test
        train_steps = 12 * 30 * total_steps_per_day
        val_steps = 4 * 30 * total_steps_per_day
        test_steps = 4 * 30 * total_steps_per_day

        # Partition slicing
        start_indices = [
            0,
            train_steps - self.seq_len,
            train_steps + val_steps - self.seq_len,
        ]
        end_indices = [
            train_steps,
            train_steps + val_steps,
            train_steps + val_steps + test_steps,
        ]

        if self.partition == "pred":
            # Prediction slice can be the last seq_len chunk
            start_idx = len(df_raw) - self.seq_len
            end_idx = len(df_raw)
        else:
            partition_idx = self.partition_map[self.partition]
            start_idx = start_indices[partition_idx]
            end_idx = end_indices[partition_idx]

        # Feature selection
        if self.feature_mode in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]  # skip first column, usually 'date'
        else:  # Single-target
            df_data = df_raw[[self.target_col]]

        # Fit scaler on training portion only
        if self.apply_scaling:
            train_data = df_data.iloc[start_indices[0] : end_indices[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Time features
        df_stamp = df_raw[["date"]].iloc[start_idx:end_idx].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        if self.time_encoding == 0:
            # Basic cyclical/datetime splits
            df_stamp["month"] = df_stamp["date"].apply(lambda x: x.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda x: x.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda x: x.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda x: x.hour)
            # For minute data, optionally store minute or quarter-of-hour
            if "m" in self.file_name.lower():
                df_stamp["minute"] = df_stamp["date"].apply(lambda x: x.minute // 15)
            time_array = df_stamp.drop(["date"], axis=1).values
        else:
            # Fourier-based time features
            time_array = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.frequency
            )
            time_array = time_array.transpose(1, 0)

        # Final data slices
        self.data_x = data[start_idx:end_idx]
        self.data_y = data[start_idx:end_idx]
        self.data_stamp = time_array


class CustomDataset(BaseTimeSeriesDataset):
    """
    General-purpose dataset with configurable train/val/test ratios and a customizable time column.
    """

    def __init__(
        self,
        data_dir: str,
        partition: str = "train",
        window_sizes: Optional[List[int]] = None,
        feature_mode: str = "S",
        file_name: str = "ETTh1.csv",
        target_col: str = "OT",
        apply_scaling: bool = True,
        time_encoding: int = 0,
        frequency: str = "h",
        time_col: str = "date",
        add_time_features: bool = False,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
    ):
        super().__init__(
            data_dir=data_dir,
            file_name=file_name,
            partition=partition,
            window_sizes=window_sizes,
            feature_mode=feature_mode,
            target_col=target_col,
            apply_scaling=apply_scaling,
            time_encoding=time_encoding,
            frequency=frequency,
            add_time_features=add_time_features,
        )
        self.time_col = time_col
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self._read_data()

    def _read_data(self) -> None:
        df_raw = pd.read_csv(os.path.join(self.data_dir, self.file_name))
        total_len = len(df_raw)
        num_train = int(total_len * self.train_ratio)
        num_test = int(total_len * self.test_ratio)
        num_val = total_len - num_train - num_test

        start_indices = [
            0,
            num_train - self.seq_len,
            total_len - num_test - self.seq_len,
        ]
        end_indices = [num_train, num_train + num_val, total_len]

        partition_map = {"train": 0, "val": 1, "test": 2}
        start_idx = start_indices[partition_map[self.partition]]
        end_idx = end_indices[partition_map[self.partition]]

        if self.feature_mode in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]  # all except first col (often date)
        else:
            df_data = df_raw[[self.target_col]]

        if self.apply_scaling:
            train_data = df_data.iloc[start_indices[0] : end_indices[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col]].iloc[start_idx:end_idx].copy()
        df_stamp[self.time_col] = pd.to_datetime(df_stamp[self.time_col])

        if self.time_encoding == 0:
            df_stamp["month"] = df_stamp[self.time_col].apply(lambda x: x.month)
            df_stamp["day"] = df_stamp[self.time_col].apply(lambda x: x.day)
            df_stamp["weekday"] = df_stamp[self.time_col].apply(lambda x: x.weekday())
            df_stamp["hour"] = df_stamp[self.time_col].apply(lambda x: x.hour)
            time_array = df_stamp.drop([self.time_col], axis=1).values
        else:
            time_array = time_features(
                pd.to_datetime(df_stamp[self.time_col].values), freq=self.frequency
            )
            time_array = time_array.transpose(1, 0)

        self.data_x = data[start_idx:end_idx]
        self.data_y = data[start_idx:end_idx]
        self.data_stamp = time_array


class PredictionDataset(BaseTimeSeriesDataset):
    """
    One-shot prediction dataset: only last seq_len samples are used to predict next pred_len steps.
    Useful for final inference or "production" forecasting scenarios.
    """

    def __init__(
        self,
        data_dir: str,
        file_name: str = "ETTh1.csv",
        target_col: str = "OT",
        window_sizes: Optional[List[int]] = None,
        feature_mode: str = "S",
        apply_scaling: bool = True,
        inverse: bool = False,
        time_encoding: int = 0,
        frequency: str = "15min",
        cols: Optional[List[str]] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            file_name=file_name,
            partition="pred",
            window_sizes=window_sizes,
            feature_mode=feature_mode,
            target_col=target_col,
            apply_scaling=apply_scaling,
            time_encoding=time_encoding,
            frequency=frequency,
            add_time_features=True,  # often helpful for inference
        )
        self.inverse = inverse
        self.cols = cols
        self._read_data()

    def _read_data(self) -> None:
        df_raw = pd.read_csv(os.path.join(self.data_dir, self.file_name))
        if self.cols:
            cols = (
                ["date"]
                + [c for c in self.cols if c != self.target_col]
                + [self.target_col]
            )
            df_raw = df_raw[cols]
        else:
            # Use all columns except that 'date' is assumed first, 'target_col' last
            tmp_cols = list(df_raw.columns)
            tmp_cols.remove(self.target_col)
            tmp_cols.remove("date")
            df_raw = df_raw[["date"] + tmp_cols + [self.target_col]]

        start_idx = len(df_raw) - self.seq_len
        end_idx = len(df_raw)

        if self.feature_mode in ["M", "MS"]:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target_col]]

        if self.apply_scaling:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Future timestamps for the prediction window
        tmp_stamp = df_raw[["date"]].iloc[start_idx:end_idx].copy()
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp["date"])
        pred_dates = pd.date_range(
            start=tmp_stamp["date"].values[-1],
            periods=self.pred_len + 1,
            freq=self.frequency,
        )
        # Combine historical dates + predicted future dates
        all_dates = list(tmp_stamp["date"].values) + list(pred_dates[1:])
        df_stamp = pd.DataFrame({"date": all_dates})

        if self.time_encoding == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda x: x.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda x: x.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda x: x.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda x: x.hour)
            df_stamp["minute"] = df_stamp["date"].apply(lambda x: x.minute // 15)
            time_array = df_stamp.drop(["date"], axis=1).values
        else:
            time_array = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.frequency
            )
            time_array = time_array.transpose(1, 0)

        self.data_x = data[start_idx:end_idx]
        if self.inverse:
            self.data_y = df_data.values[start_idx:end_idx]
        else:
            self.data_y = self.data_x
        self.data_stamp = time_array

    def __getitem__(self, idx: int):
        seq_start = idx
        seq_end = seq_start + self.seq_len
        label_start = seq_end - self.label_len
        label_end = label_start + self.label_len + self.pred_len

        seq_x = self.data_x[seq_start:seq_end]
        # For inference, only the history is typically "inverse-transformed"
        seq_y = self.data_y[label_start : label_start + self.label_len]

        seq_x_mark = self.data_stamp[seq_start:seq_end]
        seq_y_mark = self.data_stamp[label_start:label_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len + 1
