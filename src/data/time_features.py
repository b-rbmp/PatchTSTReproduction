from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class AbstractTimeFeature(ABC):
    """Base class for deriving numerical time features in [-0.5, 0.5]."""

    @abstractmethod
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray: ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SecondOfMinute(AbstractTimeFeature):
    """Encodes the second (0-59) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(AbstractTimeFeature):
    """Encodes the minute (0-59) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(AbstractTimeFeature):
    """Encodes the hour (0-23) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(AbstractTimeFeature):
    """Encodes the weekday (Mon=0-Sun=6) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(AbstractTimeFeature):
    """Encodes the day of month (1-31) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(AbstractTimeFeature):
    """Encodes the day of year (1–365/366) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(AbstractTimeFeature):
    """Encodes the month (1-12) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(AbstractTimeFeature):
    """Encodes the ISO week (1-52/53) as a value in [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # isocalendar().week returns an Int64Index; it can be 52 or 53 depending on the year
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def select_time_features(freq_str: str) -> List[AbstractTimeFeature]:
    """
    Return a list of time features suited to the specified frequency string.
    e.g., "12H", "5min", "1D" etc.
    """
    offset_map = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)
    for off_type, feature_classes in offset_map.items():
        if isinstance(offset, off_type):
            return [cls() for cls in feature_classes]

    # If frequency is not recognized, provide a guide
    msg = (
        f"Unsupported frequency {freq_str}\n"
        "Supported offsets include:\n"
        "  - Y or A (yearly)\n"
        "  - M (monthly)\n"
        "  - W (weekly)\n"
        "  - D (daily)\n"
        "  - B (business days)\n"
        "  - H (hourly)\n"
        "  - T/min (minutely)\n"
        "  - S (secondly)\n"
    )
    raise ValueError(msg)


def time_features(dates: pd.DatetimeIndex, freq: str = "H") -> np.ndarray:
    """Stack the appropriate time features for a given datetime index and frequency."""
    features = select_time_features(freq)
    return np.vstack([f(dates) for f in features])
