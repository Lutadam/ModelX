"""Turning per-minute alarms into alerts, and counting them against the budget.

An alert is one burst of alarm minutes: it costs one inspection however long
it lasts. The definition of an alert lives in ONE place, evaluation.alert_starts
(cooldown, plus a new alert when an alarm stays on for REALERT_AFTER), so the
thresholds set here are judged by the same rule in the evaluation.
"""
import pandas as pd

from config import (
    FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
    FEATURE_STEP,
    MOTOR_CURRENT_COLUMN,
    OPERATING_MOTOR_CURRENT_AMPS,
    WARNING_HORIZON,
)
from evaluation import EvalConfig, alert_starts


def operating_per_minute(readings: pd.DataFrame) -> pd.Series:
    """Flag the minutes in which the compressor motor is running.

    Args:
        readings: Data with a DatetimeIndex and a motor current column.

    Returns:
        Float series per minute: 1.0 when the mean motor current is above
        OPERATING_MOTOR_CURRENT_AMPS, 0.0 otherwise. Minutes with no readings
        are left out.
    """
    mean_motor_current = readings[MOTOR_CURRENT_COLUMN].resample(FEATURE_STEP).mean().dropna()
    return (mean_motor_current > OPERATING_MOTOR_CURRENT_AMPS).astype(float)


def alert_start_times(alarm: pd.Series) -> pd.DatetimeIndex:
    """Return the time each alert begins (same rule as the evaluation).

    A new alert starts after more than ALERT_COOLDOWN without an alarm, or when
    an alarm is still on REALERT_AFTER after the current alert began.

    Args:
        alarm: Boolean alarm series on a DatetimeIndex, sorted by time.

    Returns:
        Timestamps of the alert starts.
    """
    return alert_starts(alarm, EvalConfig())


def count_false_alerts(alarm: pd.Series, labels: pd.Series) -> int:
    """Count alerts that begin on a normal (label 0) minute.

    Args:
        alarm: Boolean alarm series on a DatetimeIndex.
        labels: Labels from `make_labels`, indexed by the same minutes.

    Returns:
        The number of false alerts.
    """
    label_at_alert_start = labels.reindex(alert_start_times(alarm))
    return int((label_at_alert_start == 0).fillna(False).sum())


def operating_hours_of_normal_minutes(labels: pd.Series, operating: pd.Series) -> float:
    """Hours of operation in the normal (label 0) minutes. One row is one minute.

    Args:
        labels: Labels from `make_labels`.
        operating: Output of `operating_per_minute`, aligned to the labels.

    Returns:
        Operating hours inside the normal minutes.
    """
    is_normal = (labels == 0).fillna(False).to_numpy(dtype=bool)
    return float(operating[is_normal].sum()) / 60


def detect_failure(alarm: pd.Series, failure_start: pd.Timestamp) -> dict:
    """Say whether a failure was caught, and with how many hours of warning.

    Caught means at least one alarm inside the risk window, the WARNING_HORIZON
    before the failure start. Warning is measured from the first such alarm.

    Args:
        alarm: Boolean alarm series on a DatetimeIndex.
        failure_start: Start time of the failure.

    Returns:
        Dict with `caught` (bool) and `warning_hours` (0.0 when not caught).
    """
    in_risk_window = (alarm.index >= failure_start - WARNING_HORIZON) & (alarm.index < failure_start)
    alarm_times_in_window = alarm.index[in_risk_window & alarm.to_numpy(dtype=bool)]
    if len(alarm_times_in_window) == 0:
        return {"caught": False, "warning_hours": 0.0}
    return {"caught": True, "warning_hours": (failure_start - alarm_times_in_window[0]).total_seconds() / 3600}


def judge_alarms(alarm: pd.Series, test: pd.DataFrame, test_failure_start: pd.Timestamp) -> dict:
    """Event-level result of one method on one test block.

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
