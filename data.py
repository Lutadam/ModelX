"""Loading the MetroPT-3 data and building the risk labels."""
from pathlib import Path

import pandas as pd

from config import DATA_FILE, FAILURES, GAP_THRESHOLD, WARNING_HORIZON


def load_dataset(csv_path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the CSV with a sorted DatetimeIndex and a contiguous-segment id.

    Args:
        csv_path: Location of the MetroPT-3 CSV file.

    Returns:
        One row per reading, indexed by timestamp, with all sensor columns and
        an extra `segment` column. A new segment starts after every gap longer
        than GAP_THRESHOLD, so windows can be kept from spanning an outage.
    """
    readings = pd.read_csv(csv_path, parse_dates=["timestamp"])
    readings = readings.drop(columns=[c for c in readings.columns if c.startswith("Unnamed")])
    readings = readings.sort_values("timestamp").set_index("timestamp")
    time_since_previous_row = readings.index.to_series().diff()
    readings["segment"] = (time_since_previous_row > GAP_THRESHOLD).cumsum().to_numpy()
    return readings


def make_labels(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Label each timestamp as risk (1), normal (0) or excluded (<NA>).

    Risk is [failure start - WARNING_HORIZON, failure start). Rows from a
    failure's start until its `excluded_until` time are excluded, so failure
    rows are never labelled positive. Exclusion wins over risk if they overlap.

    Args:
        timestamps: The timestamps to label.

    Returns:
        Nullable integer series aligned to `timestamps`.
    """
    labels = pd.Series(0, index=timestamps, dtype="Int8")
    for failure in FAILURES:
        in_risk_window = (timestamps >= failure.start - WARNING_HORIZON) & (timestamps < failure.start)
        labels[in_risk_window] = 1
    for failure in FAILURES:
        in_failure_or_maintenance = (timestamps >= failure.start) & (timestamps <= failure.excluded_until)
        labels[in_failure_or_maintenance] = pd.NA
    return labels
