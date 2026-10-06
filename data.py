"""Data loading, segment creation, risk labeling, and false-alarm budget calculation.

Owner: Shazia Baboorally (Data & Features Specialist)
Part of Team ModelX (RailGuard AI)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DATA_FILE,
    EXPECTED_SAMPLING_SECONDS,
    FAILURES,
    GAP_THRESHOLD,
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
    readings = pd.read_csv(csv_path, parse_dates=["timestamp"])

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


def make_labels(readings: pd.DataFrame) -> pd.Series:
    """Create binary risk labels for predictive maintenance.

    Risk labeling logic:
    - y = 1 (Risk Window): Rows in [failure_start - 3 hours, failure_start)
    - y = np.nan (Excluded): Rows inside [failure_start, excluded_until] (maintenance/breakdown)
    - y = 0 (Normal): All other normal operating rows

    Args:
        readings: pd.DataFrame indexed by timestamp

    Returns:
        pd.Series containing 0.0, 1.0, or NaN values indexed by timestamp.
    """
    labels = pd.Series(0.0, index=readings.index, name="target")

    for failure in FAILURES:
        # Define 3-hour warning window prior to failure start
        warning_start = failure.start - WARNING_HORIZON

        # Label 3-hour advance warning window as Risk (y = 1.0)
        risk_mask = (readings.index >= warning_start) & (readings.index < failure.start)
        labels.loc[risk_mask] = 1.0

        # Exclude rows during failure and repair periods (y = NaN)
        exclusion_mask = (readings.index >= failure.start) & (readings.index <= failure.excluded_until)
        labels.loc[exclusion_mask] = np.nan

    return labels


def calculate_operating_hours_and_budget(readings: pd.DataFrame) -> dict:
    """Calculate total active operating hours and the False-Alarm Budget (Condition 4).

    Condition 4 Requirement:
    - Operating assumption: Motor Current > 1.0 A defines active train operation.
    - False-alarm budget: Maximum 1 false alert per 100 operating hours (~1 per week at 14h/day).

    Args:
        readings: pd.DataFrame containing sensor columns

    Returns:
        dict containing total operating hours, active rows, and max false-alarm count.
    """
    # Active operation mask (Motor current > 1.0 A or TP2 > 1.0 bar)
    if "Motor_current" in readings.columns:
        active_mask = readings["Motor_current"] > 1.0
    elif "TP2" in readings.columns:
        active_mask = readings["TP2"] > 1.0
    else:
        active_mask = pd.Series(True, index=readings.index)

    active_rows = int(active_mask.sum())

    # Calculate operating hours (each row is nominally 10 seconds = 0.1 Hz)
    total_operating_hours = (active_rows * EXPECTED_SAMPLING_SECONDS) / 3600.0

    # Enforce budget: 1 false alert per 100 operating hours
    max_false_alarms = int(total_operating_hours // 100)

    stats = {
        "total_rows": len(readings),
        "active_rows": active_rows,
        "total_operating_hours": round(total_operating_hours, 2),
        "max_false_alarms_budget": max_false_alarms,
    }

    print("==================================================")
    print(" SHAZIA'S DATA & CONDITION 4 BUDGET SUMMARY")
    print("==================================================")
    print(f" Total Rows Loaded:           {stats['total_rows']:,}")
    print(f" Active Operating Rows:       {stats['active_rows']:,}")
    print(f" Total Active Operating Time: {stats['total_operating_hours']} hours")
    print(f" Condition 4 Budget:          Max {stats['max_false_alarms_budget']} False Alerts")
    print("==================================================")

    return stats