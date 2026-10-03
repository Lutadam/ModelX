"""Feature building. PLACEHOLDER until Shazia replaces `build_features`.

Interface agreed with the team:
    build_features(readings: pd.DataFrame) -> pd.DataFrame
Input has a DatetimeIndex and a `segment` column. Output is indexed by
timestamp at FEATURE_STEP, holds feature columns only, and may contain NaN in
warm-up rows.
"""
import pandas as pd

from config import (
    EXPECTED_SAMPLING_SECONDS,
    FEATURE_STEP,
    MIN_WINDOW_COVERAGE,
    ROLLING_WINDOW,
    SENSOR_COLUMNS,
)


def rolling_features(readings: pd.DataFrame, columns: list, window: str = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling mean and standard deviation, computed within segments only.

    Uses a time-based window, so it is correct at any sampling rate. A window
    needs MIN_WINDOW_COVERAGE of the expected rows, so the first minutes of
    each segment (and any thin stretch) give NaN instead of a misleading value.

    Args:
        readings: Data with a DatetimeIndex and a `segment` column.
        columns: Sensor columns to summarise.
        window: Rolling window length, as a pandas offset string.

    Returns:
        One row per input row, with `<column>_mean_<window>` and
        `<column>_std_<window>` columns.
    """
    window_seconds = pd.Timedelta(window).total_seconds()
    minimum_rows_in_window = int(window_seconds / EXPECTED_SAMPLING_SECONDS * MIN_WINDOW_COVERAGE)
    rows_by_segment = readings.groupby("segment")[columns]
    rolling_mean = rows_by_segment.rolling(window, min_periods=minimum_rows_in_window).mean().droplevel(0)
    rolling_std = rows_by_segment.rolling(window, min_periods=minimum_rows_in_window).std().droplevel(0)
    return rolling_mean.add_suffix(f"_mean_{window}").join(rolling_std.add_suffix(f"_std_{window}"))


def build_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Placeholder feature builder: rolling mean and std, one row per minute.

    Args:
        readings: Data with a DatetimeIndex and a `segment` column.

    Returns:
        Feature columns only, indexed by timestamp at FEATURE_STEP. Minutes
        with no data, and warm-up rows, are NaN.
    """
    rolling_summary = rolling_features(readings, SENSOR_COLUMNS)
    return rolling_summary.resample(FEATURE_STEP).last()
