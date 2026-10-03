"""Week 1 first result: rule baselines and Random Forest on the time-respecting fold.

Train on F1-F3, test on F4 (cut at config.TEST_START). Every method is judged
at event level: was the failure caught, with how many hours of warning, and how
many false alerts per 100 operating hours against the budget. All methods use
the same minutes: labelled, with complete features and sensor data.
"""
import random

import numpy as np
import pandas as pd

from alerts import (
    count_false_alerts,
    detect_failure,
    operating_hours_of_normal_minutes,
    operating_per_minute,
)
from config import (
    FAILURES,
    FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
    RANDOM_SEED,
    TEST_START,
)
from data import load_dataset, make_labels
from features import build_features
from models import predict_alarms, train_random_forest
from rules import LowPressureRule, lowest_pressure_per_minute, lps_alarm_per_minute

NON_FEATURE_COLUMNS = ["label", "lps_alarm", "lowest_pressure", "operating"]


def build_minute_table(readings: pd.DataFrame) -> pd.DataFrame:
    """Put labels, rule inputs and features on one per-minute table.

    Args:
        readings: Output of `load_dataset`.

    Returns:
        One row per minute that has a label, complete features and sensor
        data. Columns: `label`, `lps_alarm`, `lowest_pressure`, `operating`,
        then the features.
    """
    features = build_features(readings)
    minutes = features.index
    table = pd.DataFrame(
        {
            "label": make_labels(minutes),
            "lps_alarm": lps_alarm_per_minute(readings).reindex(minutes).astype(float),
            "lowest_pressure": lowest_pressure_per_minute(readings).reindex(minutes),
            "operating": operating_per_minute(readings).reindex(minutes),
        },
        index=minutes,
    ).join(features)
    return table.dropna()


def judge_alarms(alarm: pd.Series, test: pd.DataFrame, test_failure_start: pd.Timestamp) -> dict:
    """Event-level result of one method on the test fold.

    Args:
        alarm: Boolean alarm series aligned to `test`.
        test: The test rows from `build_minute_table`.
        test_failure_start: Start time of the held-out failure.

    Returns:
        Dict with caught, warning_hours, false_alerts, normal_operating_hours,
        false_alerts_per_100_operating_hours and within_budget.
    """
    detection = detect_failure(alarm, test_failure_start)
    false_alerts = count_false_alerts(alarm, test["label"])
    normal_operating_hours = operating_hours_of_normal_minutes(test["label"], test["operating"])
    false_alerts_per_operating_hour = false_alerts / normal_operating_hours if normal_operating_hours else float("nan")
    return {
        "caught": detection["caught"],
        "warning_hours": round(detection["warning_hours"], 2),
        "false_alerts": false_alerts,
        "normal_operating_hours": round(normal_operating_hours, 1),
        "false_alerts_per_100_operating_hours": round(100 * false_alerts_per_operating_hour, 2),
        "within_budget": false_alerts_per_operating_hour <= FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
    }


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    readings = load_dataset()
    print(f"{len(readings)} readings loaded")
    minute_table = build_minute_table(readings)
    train = minute_table[minute_table.index < TEST_START]
    test = minute_table[minute_table.index >= TEST_START]
    feature_names = [c for c in minute_table.columns if c not in NON_FEATURE_COLUMNS]
    print(f"train: {len(train)} minutes, {int(train['label'].sum())} risk | "
          f"test: {len(test)} minutes, {int(test['label'].sum())} risk")

    low_pressure_rule = LowPressureRule(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(
        train["lowest_pressure"], train["label"], train["operating"])
    print(f"Pressure cut-off set on training data: {low_pressure_rule.pressure_cutoff:.3f}")
    forest = train_random_forest(train[feature_names], train["label"])

    held_out_failure = FAILURES[-1]
    results = pd.DataFrame(
        {
            "LPS active (current practice)": judge_alarms(
                test["lps_alarm"].astype(bool), test, held_out_failure.start),
            "Low pressure (TP3/Reservoirs)": judge_alarms(
                low_pressure_rule.predict(test["lowest_pressure"]), test, held_out_failure.start),
            "Random Forest": judge_alarms(
                predict_alarms(forest, test[feature_names]), test, held_out_failure.start),
        }
    )
    print(f"\nHeld-out failure {held_out_failure.name}, budget "
          f"{100 * FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET:.0f} false alert per 100 operating hours")
    print(results.to_string())


if __name__ == "__main__":
    main()
