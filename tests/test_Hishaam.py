"""Tests for Hishaam's evaluation module (evaluation/core.py), on the real MetroPT-3 data.

Owner: Lauthan Muhammad Hishaam Ibn Afzal (Evaluator & Ethics), Team ModelX.

Every check runs on the real per-minute table. The alarms come from real
signals (the LPS alarm, low pressure, the motor running). Where a check needs
the right answer, it is worked out a second, simpler way (a plain loop) on the
same real data, and the two must agree.

Run from anywhere:  python tests/test_Hishaam.py
Uses the cached per-minute table (built from the full dataset the first time).
"""
import sys
from itertools import product
from pathlib import Path

# Put the project root (where config.py lives) on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from evaluation.core import (
    EvalConfig,
    alert_starts,
    chance_caught_by_luck,
    count_false_alerts,
    cross_fitted_scores,
    evaluate_fold,
    false_alert_rate,
    make_folds,
    normal_operating_hours,
    probability_at_least,
    split,
    summarise,
    threshold_for_budget,
)
from data.minute_table import load_minute_table

SETTINGS = EvalConfig()
LOW_PRESSURE_TEST_CUTOFF_BAR = 8.0  # only to make a real alarm series for the checks
failed_checks = []


def check(condition: bool, message: str) -> None:
    """Print a tick or a cross for one check and remember the failures."""
    if condition:
        print(f"  ✓ {message}")
    else:
        print(f"  ❌ {message}")
        failed_checks.append(message)


# ------------------------------------------------------------------ the slow, obvious way

def reference_alert_starts(alarm: pd.Series, settings: EvalConfig) -> list:
    """Alert starts worked out one alarm minute at a time.

    A new alert starts after a quiet spell longer than the cooldown, or when
    the alarm is still on `realert_after` after the current alert began.
    """
    starts, previous_alarm, current_alert_start = [], None, None
    for time in alarm.index[alarm.to_numpy(dtype=bool)]:
        quiet_too_long = previous_alarm is None or time - previous_alarm > settings.alert_cooldown
        on_too_long = current_alert_start is not None and time >= current_alert_start + settings.realert_after
        if quiet_too_long or on_too_long:
            starts.append(time)
            current_alert_start = time
        previous_alarm = time
    return starts


def reference_false_alerts(alarm: pd.Series, labels: pd.Series, settings: EvalConfig) -> int:
    return sum(1 for time in reference_alert_starts(alarm, settings) if labels[time] == 0)


def real_alarms(minute_table: pd.DataFrame) -> dict:
    return {
        "LPS active": minute_table["lps_alarm"].astype(bool),
        f"pressure below {LOW_PRESSURE_TEST_CUTOFF_BAR} bar": minute_table["lowest_pressure"] < LOW_PRESSURE_TEST_CUTOFF_BAR,
        "motor running": minute_table["operating"] == 1,
    }


# ------------------------------------------------------------------ alerts

def test_alerts(minute_table: pd.DataFrame) -> None:
    print("\n[1] Alarm minutes become alerts (one alert = one inspection)")
    labels = minute_table["label"]
    for alarm_name, alarm in real_alarms(minute_table).items():
        starts = list(alert_starts(alarm, SETTINGS))
        check(starts == reference_alert_starts(alarm, SETTINGS),
              f"{alarm_name}: {len(starts):,} alerts, same as the minute-by-minute count")
        check(count_false_alerts(alarm, labels, SETTINGS) == reference_false_alerts(alarm, labels, SETTINGS),
              f"{alarm_name}: false alerts = alerts starting on a normal minute")

    no_alarm = minute_table["lps_alarm"].astype(bool) & False
    check(len(alert_starts(no_alarm, SETTINGS)) == 0, "No alarm minutes = 0 alerts")

    motor_running = minute_table["operating"] == 1
    alarm_times = motor_running.index[motor_running.to_numpy()]
    bursts = 1 + int((np.diff(alarm_times.to_numpy()) > SETTINGS.alert_cooldown.to_timedelta64()).sum())
    check(len(alert_starts(motor_running, SETTINGS)) > bursts,
          f"An alarm that stays on counts again every {SETTINGS.realert_after} "
          f"({len(alert_starts(motor_running, SETTINGS)):,} alerts from {bursts:,} bursts)")


def test_operating_hours(minute_table: pd.DataFrame) -> None:
    print("\n[2] Operating hours and the false-alert rate")
    labels, operating = minute_table["label"], minute_table["operating"]
    expected_hours = int(((labels == 0) & (operating == 1)).sum()) / 60
    hours = normal_operating_hours(labels, operating)
    check(abs(hours - expected_hours) < 1e-9, f"{hours:,.1f} operating hours in normal minutes (counted minute by minute)")

    lps = minute_table["lps_alarm"].astype(bool)
    expected_rate = reference_false_alerts(lps, labels, SETTINGS) / expected_hours
    check(abs(false_alert_rate(lps, labels, operating, SETTINGS) - expected_rate) < 1e-12,
          f"LPS false-alert rate = {100 * expected_rate:.2f} per 100 operating hours")


# ------------------------------------------------------------------ folds

def test_folds(minute_table: pd.DataFrame) -> list:
    print("\n[3] Leave-one-failure-out folds")
    index = minute_table.index
    folds = make_folds(index.min(), index.max(), config.FAILURES)
    check(len(folds) == len(config.FAILURES), f"One fold per failure ({len(folds)})")

    for fold in folds:
        check(fold.test_start <= fold.held_out.start <= fold.test_end,
              f"{fold.name}: its failure lies inside its own test block")
    check(all(earlier.test_end < later.test_start for earlier, later in zip(folds, folds[1:])),
          "Test blocks never overlap")
    times_covered = sum(int(((index >= fold.test_start) & (index <= fold.test_end)).sum()) for fold in folds)
    check(times_covered == len(index), f"Every one of the {len(index):,} real minutes is in exactly one test block")

    for fold in folds:
        train, test = split(minute_table, fold, SETTINGS)
        too_close = ((train.index >= fold.test_start - SETTINGS.purge_gap)
                     & (train.index <= fold.test_end + SETTINGS.purge_gap)).any()
        check(not too_close and not train.index.isin(test.index).any(),
              f"{fold.name}: no training minute within {SETTINGS.purge_gap} of the test block")

    check(folds[-1].test_start == config.FAILURES[2].excluded_until + pd.Timedelta(seconds=1),
          "The last fold starts right after F3's maintenance (same as the original split)")
    return folds


# ------------------------------------------------------------------ threshold

def test_threshold(minute_table: pd.DataFrame, folds: list) -> None:
    print("\n[4] Threshold from the false-alarm budget (real training minutes of the last fold)")
    train, _ = split(minute_table, folds[-1], SETTINGS)
    labels, operating = train["label"], train["operating"]
    real_scores = {
        "low pressure (-lowest pressure)": -train["lowest_pressure"],
        "motor current variability (Motor_current_std_60m)": train["Motor_current_std_60m"],
    }
    for score_name, scores in real_scores.items():
        threshold = threshold_for_budget(scores, labels, operating, SETTINGS)
        rate = false_alert_rate(scores >= threshold, labels, operating, SETTINGS)
        check(rate <= SETTINGS.budget_per_operating_hour,
              f"{score_name}: threshold {threshold:.4g} stays within budget ({100 * rate:.2f} per 100 h)")

        normal_scores = scores[(labels == 0).to_numpy()]
        candidates = np.unique(normal_scores.quantile(np.linspace(0.5, 1.0, config.THRESHOLD_CANDIDATES)))
        check(np.isinf(threshold) or threshold in candidates,
              f"{score_name}: the threshold is one of the normal-minute candidates (risk minutes are not used)")
        if np.isfinite(threshold) and (candidates < threshold).any():
            looser = candidates[candidates < threshold].max()
            check(false_alert_rate(scores >= looser, labels, operating, SETTINGS) > SETTINGS.budget_per_operating_hour,
                  f"{score_name}: the next, more sensitive threshold would break the budget")

    check(np.isinf(threshold_for_budget(operating, labels, operating, SETTINGS)),
          "Using 'motor running' as the score breaks the budget at every threshold, so it never alarms (inf)")


def test_cross_fitting(minute_table: pd.DataFrame, folds: list) -> None:
    print("\n[5] Cross-fitted scores never come from a model trained on the same block")
    problems = []

    def fit_and_score(fit_rows: pd.DataFrame, score_rows: pd.DataFrame) -> np.ndarray:
        # Stands in for a model: it only checks what it was given
        near_block = ((fit_rows.index >= score_rows.index.min() - SETTINGS.purge_gap)
                      & (fit_rows.index <= score_rows.index.max() + SETTINGS.purge_gap))
        if near_block.any():
            problems.append(score_rows.index.min())
        return score_rows["lowest_pressure"].to_numpy()

    scores = cross_fitted_scores(minute_table, folds, SETTINGS, fit_and_score)
    check(not problems, f"No model saw a training minute within {SETTINGS.purge_gap} of the minutes it scored")
    check(scores.notna().mean() > 0.9, f"{scores.notna().mean():.1%} of the real minutes get an out-of-sample score")


# ------------------------------------------------------------------ one fold

def test_evaluate_fold(minute_table: pd.DataFrame, folds: list) -> None:
    fold = folds[-1]
    failure = fold.held_out
    print(f"\n[6] Event-level metrics on the real {fold.name} fold")
    _, test = split(minute_table, fold, SETTINGS)
    window_start = failure.start - SETTINGS.warning_horizon
    in_window = (test.index >= window_start) & (test.index < failure.start)

    motor_running = test["operating"] == 1
    alarms_in_window = test.index[in_window & motor_running.to_numpy()]
    row = evaluate_fold("motor running", fold, test, motor_running, SETTINGS)
    expected_caught = len(alarms_in_window) > 0
    check(row["caught"] == expected_caught,
          f"'Motor running' {'caught' if expected_caught else 'missed'} {failure.name}: "
          f"{len(alarms_in_window)} alarm minutes in the risk window")
    if expected_caught:
        expected_hours = round((failure.start - alarms_in_window[0]).total_seconds() / 3600, 2)
        check(row["warning_hours"] == expected_hours, f"Warning measured from the first alarm = {expected_hours} h")
    check(row["false_alerts"] == reference_false_alerts(motor_running, test["label"], SETTINGS),
          f"False alerts = {row['false_alerts']}, same as the minute-by-minute count")

    early_only = motor_running & (test.index < window_start)
    row_early = evaluate_fold("motor running, before the window only", fold, test, early_only, SETTINGS)
    check(not row_early["caught"] and np.isnan(row_early["warning_hours"]),
          "Alarms only before the 3 h window are not a catch; warning left blank")

    unknown_onset = EvalConfig(unknown_onset={failure.name})
    row_unknown = evaluate_fold("motor running", fold, test, motor_running, unknown_onset)
    check(row_unknown["caught"] == expected_caught and np.isnan(row_unknown["warning_hours"]),
          "Unknown onset (like F1): caught is still reported, warning hours are not")

    true_labels = test["label"].astype(float)
    row_perfect = evaluate_fold("the labels themselves", fold, test, true_labels >= 1, SETTINGS,
                                scores=true_labels, threshold=1)
    check(row_perfect["pr_auc"] == 1.0 and row_perfect["recall"] == 1.0,
          "Scoring with the real labels gives PR-AUC 1.0 and recall 1.0")


# ------------------------------------------------------------------ luck and summary

def test_luck_and_summary(minute_table: pd.DataFrame, folds: list) -> None:
    print("\n[7] Chance of catching by luck, and the summary across folds")
    test_blocks = [split(minute_table, fold, SETTINGS)[1] for fold in folds]

    last_test = test_blocks[-1]
    risk_operating_hours = last_test.loc[last_test["label"] == 1, "operating"].sum() / 60
    expected = 1 - np.exp(-SETTINGS.budget_per_operating_hour * risk_operating_hours)
    check(abs(chance_caught_by_luck(last_test, SETTINGS) - expected) < 1e-12,
          f"A blind detector at the budget catches {folds[-1].held_out.name} by luck with chance {expected:.4f} "
          f"({risk_operating_hours:.2f} operating hours in its risk window)")

    chances = [chance_caught_by_luck(test, SETTINGS) for test in test_blocks]
    for catches in range(len(chances) + 1):
        brute_force = sum(np.prod([chance if hit else 1 - chance for chance, hit in zip(chances, outcome)])
                          for outcome in product([False, True], repeat=len(chances)) if sum(outcome) >= catches)
        check(abs(probability_at_least(catches, chances) - brute_force) < 1e-12,
              f"P(at least {catches} of {len(chances)} caught by luck) = {brute_force:.2e}, same as listing every outcome")

    per_fold = pd.DataFrame([evaluate_fold("LPS active", fold, test, test["lps_alarm"].astype(bool), SETTINGS)
                             for fold, test in zip(folds, test_blocks)])
    summary = summarise(per_fold).loc["LPS active"]
    check(summary["failures caught"] == f"{int(per_fold['caught'].sum())} of {len(folds)}",
          f"Summary counts {summary['failures caught']} caught")
    pooled = round(100 * per_fold["false_alerts"].sum() / per_fold["normal_operating_hours"].sum(), 2)
    check(summary["false alerts per 100 operating hours"] == pooled,
          f"False-alert rate is pooled over the folds ({pooled} per 100 h), not averaged")


def main() -> None:
    print("==================================================")
    print(" HISHAAM'S EVALUATION TESTS (real data)")
    print("==================================================")
    minute_table = load_minute_table()
    test_alerts(minute_table)
    test_operating_hours(minute_table)
    folds = test_folds(minute_table)
    test_threshold(minute_table, folds)
    test_cross_fitting(minute_table, folds)
    test_evaluate_fold(minute_table, folds)
    test_luck_and_summary(minute_table, folds)
    print("\n==================================================")
    if failed_checks:
        print(f" {len(failed_checks)} CHECK(S) FAILED")
        for message in failed_checks:
            print(f"  - {message}")
        sys.exit(1)
    print(" ALL CHECKS PASSED")
    print("==================================================")


if __name__ == "__main__":
    main()
