"""Shared helpers for Atul's two test scripts (test_random_forest.py and test_isolation_forest.py).

Not run on its own. Owner: Mundram Baboo Khoomeshwarsingh (Atul), ModelX (RailGuard AI).
"""
import sys
from pathlib import Path

# Put the project root (where config.py and main.py live) on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from alerts import alert_start_times, count_false_alerts, detect_failure
from config import FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET, FAILURES, TEST_START
from data import load_dataset
from main import NON_FEATURE_COLUMNS, build_minute_table, judge_alarms
from rules import LowPressureRule

failed_checks = []


def check(condition: bool, message: str) -> None:
    """Print a tick or a cross for one check and remember the failures."""
    if condition:
        print(f"  ✓ {message}")
    else:
        print(f"  ❌ {message}")
        failed_checks.append(message)


def minutes_from(start: str, count: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="1min")


def print_header(title: str) -> None:
    print("==================================================")
    print(f" {title}")
    print("==================================================")


def test_alerts() -> None:
    print("\n[Alerts] Grouping alarms, false alerts, caught and warning hours (made-up data)...")
    minutes = minutes_from("2020-01-01", 24 * 60)

    # Alarms at minutes 0-2 and again 4 hours later: one burst, then another
    alarm = pd.Series(False, index=minutes)
    alarm.iloc[[0, 1, 2, 240, 241]] = True
    starts = alert_start_times(alarm)
    check(len(starts) == 2, f"Two bursts of alarms 4 h apart give 2 alerts (got {len(starts)})")

    close_together = pd.Series(False, index=minutes)
    close_together.iloc[[0, 100]] = True
    check(len(alert_start_times(close_together)) == 1, "Two alarms 100 min apart are one alert (inside the 3 h cooldown)")

    # An alert starting inside a risk window is not a false alert
    labels = pd.Series(0.0, index=minutes)
    labels.iloc[238:] = 1.0
    check(count_false_alerts(alarm, labels) == 1, "Only the alert on a normal minute counts as false (the one in the risk window does not)")

    failure_start = minutes[600]
    in_window = pd.Series(False, index=minutes)
    in_window[failure_start - pd.Timedelta(hours=2)] = True
    caught = detect_failure(in_window, failure_start)
    check(caught["caught"] and abs(caught["warning_hours"] - 2.0) < 1e-9, "An alarm 2 h before the failure gives caught with 2.0 h of warning")
    too_early = pd.Series(False, index=minutes)
    too_early[failure_start - pd.Timedelta(hours=5)] = True
    check(not detect_failure(too_early, failure_start)["caught"], "An alarm 5 h before the failure is outside the 3 h window, so not caught")
    check(not detect_failure(pd.Series(False, index=minutes), failure_start)["caught"], "No alarm at all is not caught")


def test_low_pressure_rule_budget() -> None:
    print("\n[Low-pressure rule] Cut-off must respect the false-alert budget (made-up data)...")
    minutes = minutes_from("2020-01-01", 20000)  # about 333 operating hours
    pressure = pd.Series(9.0, index=minutes)
    for dip_start in (1000, 5000, 9000, 13000, 17000):  # 5 dips, each far apart
        pressure.iloc[dip_start:dip_start + 2] = 3.0
    labels = pd.Series(0.0, index=minutes)
    operating = pd.Series(1.0, index=minutes)

    rule = LowPressureRule(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(pressure, labels, operating)
    false_alerts = count_false_alerts(rule.predict(pressure), labels)
    rate = false_alerts / (len(minutes) / 60)
    check(rate <= FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
          f"5 dips would be 1.5 per 100 h, so the rule backs off: {100 * rate:.2f} per 100 h (budget 1.00)")
    check(rule.pressure_cutoff == 3.0, f"Cut-off stays at the lowest pressure seen, 3.0 (got {rule.pressure_cutoff})")

    few_dips = pressure.copy()
    few_dips.iloc[13000:13002] = 9.0
    few_dips.iloc[17000:17002] = 9.0  # only 3 dips left: 0.9 per 100 h, inside the budget
    relaxed_rule = LowPressureRule(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(few_dips, labels, operating)
    check(relaxed_rule.pressure_cutoff > 3.0, f"With only 3 dips the budget allows a higher cut-off (got {relaxed_rule.pressure_cutoff})")


def load_train_test() -> tuple:
    """Load the real data and split it at TEST_START, checking for leakage.

    Returns:
        Tuple of (train, test, feature_names).
    """
    print("\n[Real data] Train/test split and leakage checks...")
    minute_table = build_minute_table(load_dataset())
    train = minute_table[minute_table.index < TEST_START]
    test = minute_table[minute_table.index >= TEST_START]
    check(train.index.max() < TEST_START <= test.index.min(), f"All training minutes are before {TEST_START}, all test minutes after")
    check(not minute_table["label"].isna().any(), "No excluded (failure or maintenance) minute is in the table")
    held_out = FAILURES[-1]
    check((test.index >= held_out.start).any() and (train.index >= held_out.start).sum() == 0,
          f"{held_out.name} appears only in the test set")
    print(f"  train: {len(train)} minutes, {int(train['label'].sum())} risk | test: {len(test)} minutes, {int(test['label'].sum())} risk")
    check(int(train["label"].sum()) > 0 and int(test["label"].sum()) > 0, "Both sets contain risk minutes")
    feature_names = [c for c in minute_table.columns if c not in NON_FEATURE_COLUMNS]
    return train, test, feature_names


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
