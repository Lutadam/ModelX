"""Shared helpers for Atul's two test scripts (test_random_forest.py and test_isolation_forest.py).

Not run on its own. Owner: Mundram Baboo Khoomeshwarsingh (Atul), ModelX (RailGuard AI).
"""
import sys
from pathlib import Path

# Put the project root (where config.py and main.py live) on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from alerts import (
    alert_start_times,
    count_false_alerts,
    detect_failure,
    judge_alarms,
    operating_hours_of_normal_minutes,
)
from config import (
    ALERT_COOLDOWN,
    FAILURES,
    FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
    REALERT_AFTER,
    TEST_START,
    THRESHOLD_CANDIDATES,
    WARNING_HORIZON,
)
from minute_table import get_feature_names, load_minute_table
from rules import LowPressureRule

failed_checks = []


def check(condition: bool, message: str) -> None:
    """Print a tick or a cross for one check and remember the failures."""
    if condition:
        print(f"  ✓ {message}")
    else:
        print(f"  ❌ {message}")
        failed_checks.append(message)


def print_header(title: str) -> None:
    print("==================================================")
    print(f" {title}")
    print("==================================================")


def reference_alert_starts(alarm: pd.Series) -> list:
    """Alert starts worked out one alarm minute at a time (the slow, obvious way).

    New alert after a quiet spell longer than ALERT_COOLDOWN, or when the alarm
    is still on REALERT_AFTER after the current alert began.
    """
    starts, previous_alarm, current_alert_start = [], None, None
    for time in alarm.index[alarm.to_numpy(dtype=bool)]:
        quiet_too_long = previous_alarm is None or time - previous_alarm > ALERT_COOLDOWN
        on_too_long = current_alert_start is not None and time >= current_alert_start + REALERT_AFTER
        if quiet_too_long or on_too_long:
            starts.append(time)
            current_alert_start = time
        previous_alarm = time
    return starts


def test_alerts(test: pd.DataFrame) -> None:
    """Alert counting and caught / warning hours, on the real F4 test block."""
    print("\n[Alerts] Grouping alarms, false alerts, caught and warning hours (real data, F4 test block)...")
    real_alarms = {
        "LPS active": test["lps_alarm"].astype(bool),
        "motor running": test["operating"] == 1,
    }
    for alarm_name, alarm in real_alarms.items():
        reference_starts = reference_alert_starts(alarm)
        check(list(alert_start_times(alarm)) == reference_starts,
              f"{alarm_name}: {len(reference_starts):,} alerts, same as the minute-by-minute count")
        reference_false = sum(1 for time in reference_starts if test["label"][time] == 0)
        check(count_false_alerts(alarm, test["label"]) == reference_false,
              f"{alarm_name}: {reference_false} false alerts (alerts starting on a normal minute)")

    failure_start = FAILURES[-1].start
    window_start = failure_start - WARNING_HORIZON
    motor_running = test["operating"] == 1
    alarms_in_window = test.index[(test.index >= window_start) & (test.index < failure_start) & motor_running.to_numpy()]
    detection = detect_failure(motor_running, failure_start)
    check(detection["caught"] == (len(alarms_in_window) > 0),
          f"'Motor running' has {len(alarms_in_window)} alarm minutes in the 3 h before F4, so caught = {detection['caught']}")
    if detection["caught"]:
        expected_hours = (failure_start - alarms_in_window[0]).total_seconds() / 3600
        check(abs(detection["warning_hours"] - expected_hours) < 1e-9,
              f"Warning measured from the first alarm in the window = {expected_hours:.2f} h")
    before_window = motor_running & (test.index < window_start)
    check(not detect_failure(before_window, failure_start)["caught"],
          "Alarms only before the 3 h window are not a catch")
    check(not detect_failure(motor_running & False, failure_start)["caught"], "No alarm at all is not caught")


def test_low_pressure_rule_budget(train: pd.DataFrame) -> None:
    """The cut-off is the most sensitive one that keeps the real training minutes within the budget."""
    print("\n[Low-pressure rule] Cut-off must respect the false-alert budget (real training data, F1-F3)...")
    rule = LowPressureRule(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(
        train["lowest_pressure"], train["label"], train["operating"])
    normal_hours = operating_hours_of_normal_minutes(train["label"], train["operating"])
    rate = count_false_alerts(rule.predict(train["lowest_pressure"]), train["label"]) / normal_hours
    check(rate <= FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
          f"Cut-off {rule.pressure_cutoff:.3f} bar gives {100 * rate:.2f} false alerts per 100 operating hours (budget 1.00)")

    normal_pressure = train.loc[train["label"] == 0, "lowest_pressure"]
    candidates = np.unique(normal_pressure.quantile(np.linspace(0, 0.5, THRESHOLD_CANDIDATES)))
    check(rule.pressure_cutoff in candidates, "The cut-off is one of the candidates from normal minutes only")
    higher = candidates[candidates > rule.pressure_cutoff]
    if len(higher):
        higher_rate = count_false_alerts(train["lowest_pressure"] < higher[0], train["label"]) / normal_hours
        check(higher_rate > FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
              f"The next, more sensitive cut-off ({higher[0]:.3f} bar) would break the budget")


def load_train_test() -> tuple:
    """Load the real data and split it at TEST_START, checking for leakage.

    Returns:
        Tuple of (train, test, feature_names).
    """
    print("\n[Real data] Train/test split and leakage checks...")
    minute_table = load_minute_table()
    train = minute_table[minute_table.index < TEST_START]
    test = minute_table[minute_table.index >= TEST_START]
    check(train.index.max() < TEST_START <= test.index.min(), f"All training minutes are before {TEST_START}, all test minutes after")
    check(not minute_table["label"].isna().any(), "No excluded (failure or maintenance) minute is in the table")
    held_out = FAILURES[-1]
    check((test.index >= held_out.start).any() and (train.index >= held_out.start).sum() == 0,
          f"{held_out.name} appears only in the test set")
    print(f"  train: {len(train)} minutes, {int(train['label'].sum())} risk | test: {len(test)} minutes, {int(test['label'].sum())} risk")
    check(int(train["label"].sum()) > 0 and int(test["label"].sum()) > 0, "Both sets contain risk minutes")
    return train, test, get_feature_names(minute_table)


def print_result(model_name: str, model_alarm: pd.Series, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print the event-level result of one model next to the two rule baselines on F4."""
    print(f"\n[Result] {model_name} against the rules on the time-respecting fold (train F1-F3, test F4)...")
    low_pressure_rule = LowPressureRule(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(
        train["lowest_pressure"], train["label"], train["operating"])
    held_out_failure = FAILURES[-1]
    results = pd.DataFrame(
        {
            "LPS active (current practice)": judge_alarms(test["lps_alarm"].astype(bool), test, held_out_failure.start),
            "Low pressure (TP3/Reservoirs)": judge_alarms(
                low_pressure_rule.predict(test["lowest_pressure"]), test, held_out_failure.start),
            model_name: judge_alarms(model_alarm, test, held_out_failure.start),
        }
    )
    print(f"  Held-out failure {held_out_failure.name}, budget {100 * FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET:.0f} false alert per 100 operating hours\n")
    print(results.to_string())


def finish() -> None:
    """Print the summary and exit with a non-zero code if any check failed."""
    print("\n==================================================")
    if failed_checks:
        print(f" ❌ {len(failed_checks)} CHECK(S) FAILED:")
        for message in failed_checks:
            print(f"    - {message}")
    else:
        print(" 🎉 ALL CHECKS PASSED")
    print("==================================================")
    sys.exit(1 if failed_checks else 0)
