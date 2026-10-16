"""Data loading, segment creation, risk labeling, and false-alarm budget calculation.

Owner: Shazia Baboorally (Data & Features Specialist)
Part of Team ModelX (RailGuard AI)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DATA_FILE,
    FAILURES,
    FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
    FEATURE_STEP,
    GAP_THRESHOLD,
    MOTOR_CURRENT_COLUMN,
    OPERATING_MOTOR_CURRENT_AMPS,
    WARNING_HORIZON,
)


def load_dataset(csv_path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the MetroPT-3 CSV dataset with sorted DatetimeIndex and contiguous segments.

    Args:
        csv_path: Location of the MetroPT-3 CSV file.

    Returns:
        pd.DataFrame indexed by timestamp containing all sensor channels and a
        `segment` integer column that increments after every outage gap > GAP_THRESHOLD.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # 1. Load CSV and parse timestamps
    try:
        readings = pd.read_csv(csv_path, parse_dates=["timestamp"])
    except ValueError as error:  # e.g. no "timestamp" column: the wrong file
        raise ValueError(f"{csv_path} does not look like the MetroPT-3 CSV: {error}") from error

    # 2. Drop unnamed index columns if present
    readings = readings.drop(columns=[c for c in readings.columns if c.startswith("Unnamed")])

    # 3. Sort chronologically and set DatetimeIndex
    readings = readings.sort_values("timestamp").set_index("timestamp")

    # 4. Calculate sampling interval deltas to detect outage gaps
    time_deltas = readings.index.to_series().diff()

    # 5. Create contiguous segment IDs (increment ID whenever a gap exceeds GAP_THRESHOLD)
    is_new_segment = time_deltas > GAP_THRESHOLD
    readings["segment"] = is_new_segment.cumsum()

    return readings


def operating_per_minute(readings: pd.DataFrame) -> pd.Series:
    """Flag the minutes in which the compressor motor is running.

    The project's one definition of an operating minute, used for the budget and
    by the evaluation.

    Args:
        readings: Data with a DatetimeIndex and a motor current column.

    Returns:
        Float series per minute: 1.0 when the mean motor current is above
        OPERATING_MOTOR_CURRENT_AMPS, 0.0 otherwise. Minutes with no readings
        are left out.
    """
    mean_motor_current = readings[MOTOR_CURRENT_COLUMN].resample(FEATURE_STEP).mean().dropna()
    return (mean_motor_current > OPERATING_MOTOR_CURRENT_AMPS).astype(float)


def make_labels(timestamps: pd.DatetimeIndex) -> pd.Series:
    """Create binary risk labels for predictive maintenance.

    Risk labeling logic:
    - y = 1 (Risk Window): Rows in [failure_start - 3 hours, failure_start)
    - y = np.nan (Excluded): Rows inside [failure_start, excluded_until] (maintenance/breakdown)
    - y = 0 (Normal): All other normal operating rows

    Args:
        timestamps: The timestamps to label (agreed interface: timestamps in,
            so it works for raw readings and for the 1-minute feature grid).

    Returns:
        pd.Series containing 0.0, 1.0, or NaN values indexed by timestamp.
    """
    labels = pd.Series(0.0, index=timestamps, name="target")

    for failure in FAILURES:
        # Define 3-hour warning window prior to failure start
        warning_start = failure.start - WARNING_HORIZON

        # Label 3-hour advance warning window as Risk (y = 1.0)
        risk_mask = (timestamps >= warning_start) & (timestamps < failure.start)
        labels.loc[risk_mask] = 1.0

        # Exclude rows during failure and repair periods (y = NaN)
        exclusion_mask = (timestamps >= failure.start) & (timestamps <= failure.excluded_until)
        labels.loc[exclusion_mask] = np.nan

    return labels


def operating_hours_and_budget(operating_per_minute: pd.Series) -> dict:
    """Operating hours and the false-alarm budget (Condition 4) from per-minute flags.

    One definition of an operating hour for the whole project: a minute counts
    as operating when its mean motor current is above OPERATING_MOTOR_CURRENT_AMPS
    (operating_per_minute above, the same flag the evaluation uses).
    The budget is 1 false alert per 100 operating hours. The motor is on about
    9 h per day on average (evaluation/horizon_figure.py), so that is about 1 false alert
    per 11 days.

    Args:
        operating_per_minute: 1.0 for minutes with the motor on, 0.0 otherwise.

    Returns:
        dict with total_minutes, operating_minutes, total_operating_hours and
        max_false_alarms_budget.
    """
    operating_minutes = int(operating_per_minute.sum())
    total_operating_hours = operating_minutes / 60
    return {
        "total_minutes": len(operating_per_minute),
        "operating_minutes": operating_minutes,
        "total_operating_hours": round(total_operating_hours, 2),
        "max_false_alarms_budget": int(total_operating_hours * FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET),
    }


def calculate_operating_hours_and_budget(readings: pd.DataFrame, verbose: bool = True) -> dict:
    """Operating hours and the false-alarm budget of the raw readings (Condition 4).

    Args:
        readings: Output of `load_dataset`.
        verbose: Print the summary box.

    Returns:
        The dict of `operating_hours_and_budget`, plus total_rows.
    """
    stats = {"total_rows": len(readings), **operating_hours_and_budget(operating_per_minute(readings))}
    if verbose:
        print("==================================================")
        print(" SHAZIA'S DATA & CONDITION 4 BUDGET SUMMARY")
        print("==================================================")
        print(f" Total Rows Loaded:           {stats['total_rows']:,}")
        print(f" Operating Minutes:           {stats['operating_minutes']:,}")
        print(f" Total Active Operating Time: {stats['total_operating_hours']} hours")
        print(f" Condition 4 Budget:          Max {stats['max_false_alarms_budget']} False Alerts")
        print("==================================================")
    return stats
