"""Tests for Hishaam's evaluation module (evaluation_Hishaam.py).

Owner: Lauthan Muhammad Hishaam Ibn Afzal (Evaluator & Ethics), Team ModelX.

Every check uses small hand-made data whose right answer is known in advance,
so a wrong count or a leaky split shows up as a cross. No dataset needed.

Run from anywhere:  python tests/test_Hishaam.py
"""
import sys
from pathlib import Path

# Put the project root (where evaluation_Hishaam.py lives) on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from evaluation_Hishaam import (
    METROPT_FAILURES,
    EvalConfig,
    Failure,
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

CONFIG = EvalConfig()
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


# ------------------------------------------------------------------ alerts

def test_alerts() -> None:
    print("\n[1] Alarm minutes become alerts (one alert = one inspection)")
    minutes = minutes_from("2020-01-01", 24 * 60)

    far_apart = pd.Series(False, index=minutes)
    far_apart.iloc[[0, 1, 2, 240, 241]] = True  # burst, then another 4 h later
    check(len(alert_starts(far_apart, CONFIG)) == 2, "Two bursts 4 h apart = 2 alerts")

    close = pd.Series(False, index=minutes)
    close.iloc[[0, 100]] = True  # 100 min apart, inside the 3 h cooldown
    check(len(alert_starts(close, CONFIG)) == 1, "Two alarms 100 min apart = 1 alert")

    check(len(alert_starts(pd.Series(False, index=minutes), CONFIG)) == 0, "No alarms = 0 alerts")

    three_days = minutes_from("2020-01-01", 3 * 24 * 60)
    stuck_on = pd.Series(True, index=three_days)
    check(len(alert_starts(stuck_on, CONFIG)) == 3,
          "An alarm stuck on for 3 days = 3 alerts (re-alert every 24 h), not 1")

    labels = pd.Series(0.0, index=minutes)
    labels.iloc[238:420] = 1.0  # the second burst starts inside a risk window
    check(count_false_alerts(far_apart, labels, CONFIG) == 1,
          "Only the alert starting on a normal minute is false (the one in the risk window is a hit)")


def test_operating_hours() -> None:
    print("\n[2] Operating hours and the false-alert rate")
    minutes = minutes_from("2020-01-01", 600)
    labels = pd.Series(0.0, index=minutes)
    labels.iloc[540:] = 1.0  # last 60 minutes are risk
    operating = pd.Series(0.0, index=minutes)
    operating.iloc[:300] = 1.0  # motor ran for the first 300 minutes (all normal)
    check(abs(normal_operating_hours(labels, operating) - 5.0) < 1e-9,
          "300 operating minutes inside normal time = 5.0 hours")

    alarm = pd.Series(False, index=minutes)
    alarm.iloc[10] = True
    check(abs(false_alert_rate(alarm, labels, operating, CONFIG) - 0.2) < 1e-9,
          "1 false alert in 5 operating hours = 0.2 per hour")


# ------------------------------------------------------------------ folds

def test_folds() -> None:
    print("\n[3] Leave-one-failure-out folds")
    index = pd.date_range("2020-02-01", "2020-09-01", freq="1h")
    folds = make_folds(index.min(), index.max(), METROPT_FAILURES)
    check(len(folds) == 4, "One fold per failure (4)")

    for fold in folds:
        check(fold.test_start <= fold.held_out.start <= fold.test_end,
              f"{fold.name}: its failure lies inside its own test block")

    blocks_disjoint = all(a.test_end < b.test_start for a, b in zip(folds, folds[1:]))
    check(blocks_disjoint, "Test blocks never overlap")
    covered = sum(((index >= f.test_start) & (index <= f.test_end)).sum() for f in folds)
    check(covered == len(index), "Every hour of the data is in exactly one test block")

    table = pd.DataFrame({"label": 0.0}, index=index)
    for fold in folds:
        train, test = split(table, fold, CONFIG)
        too_close = ((train.index >= fold.test_start - CONFIG.purge_gap)
                     & (train.index <= fold.test_end + CONFIG.purge_gap)).any()
        check(not too_close and len(set(train.index) & set(test.index)) == 0,
              f"{fold.name}: no training row within 24 h of the test block")

    check(folds[-1].test_start == METROPT_FAILURES[2].excluded_until + pd.Timedelta(seconds=1),
          "Fold 4 starts right after F3's maintenance (same as the original split)")


# ------------------------------------------------------------------ threshold

def test_threshold() -> None:
    print("\n[4] Threshold from the false-alarm budget")
    rng = np.random.default_rng(0)
    minutes = minutes_from("2020-01-01", 30 * 24 * 60)  # 30 days
    labels = pd.Series(0.0, index=minutes)
    operating = pd.Series(1.0, index=minutes)
    scores = pd.Series(rng.random(len(minutes)), index=minutes)

    threshold = threshold_for_budget(scores, labels, operating, CONFIG)
    rate = false_alert_rate(scores >= threshold, labels, operating, CONFIG)
    check(rate <= CONFIG.budget_per_operating_hour,
          f"Chosen threshold stays within budget on training data ({100 * rate:.2f} per 100 h)")

    # The search tries a grid of 501 quantiles of the normal scores; the next
    # grid step below the chosen threshold must break the budget.
    grid = np.unique(scores.quantile(np.linspace(0.5, 1.0, 501)))
    looser = grid[grid < threshold].max()
    rate_looser = false_alert_rate(scores >= looser, labels, operating, CONFIG)
    check(rate_looser > CONFIG.budget_per_operating_hour,
          "The next, more sensitive threshold would break the budget (so it is the most sensitive allowed)")

    risk_only = labels.copy()
    risk_only.iloc[:100] = 1.0
    scores_with_risk = scores.copy()
    scores_with_risk.iloc[:100] = 5.0  # risk minutes score high
    t2 = threshold_for_budget(scores_with_risk, risk_only, operating, CONFIG)
    check(abs(t2 - threshold) < 0.05,
          "Risk minutes do not move the threshold (it is set on normal minutes only)")

    always_alarming = pd.Series(1.0, index=minutes)  # every minute identical
    check(np.isinf(threshold_for_budget(always_alarming, labels, operating, CONFIG)),
          "If no threshold fits the budget, the method never alarms (inf)")


def test_cross_fitting() -> None:
    print("\n[5] Cross-fitted scores never come from a model trained on the same block")
    index = pd.date_range("2020-02-01", "2020-09-01", freq="1h")
    folds = make_folds(index.min(), index.max(), METROPT_FAILURES)
    table = pd.DataFrame({"label": 0.0}, index=index)
    for failure in METROPT_FAILURES:
        table.loc[(index >= failure.start - CONFIG.warning_horizon) & (index < failure.start), "label"] = 1.0

    leaks = []

    def fit_and_score(fit_rows, score_rows):
        # A "model" that remembers its training timestamps and reports any overlap
        if len(set(fit_rows.index) & set(score_rows.index)) > 0:
            leaks.append(True)
        return np.zeros(len(score_rows))

    scores = cross_fitted_scores(table, folds, CONFIG, fit_and_score)
    check(not leaks, "No row is scored by a model that trained on it")
    check(scores.notna().mean() > 0.9, "Almost every row receives an out-of-sample score")


# ------------------------------------------------------------------ one fold

def toy_fold():
    """A 5-day test block with one failure on day 4, motor always running."""
    failure = Failure("FX", pd.Timestamp("2020-01-04 12:00"), pd.Timestamp("2020-01-04 18:00"),
                      pd.Timestamp("2020-01-04 18:00"))
    index = minutes_from("2020-01-01", 5 * 24 * 60)
    keep = ~((index >= failure.start) & (index <= failure.excluded_until))
    index = index[keep]
    label = ((index >= failure.start - CONFIG.warning_horizon) & (index < failure.start)).astype(float)
    test = pd.DataFrame({"label": label, "operating": 1.0}, index=index)
    fold = make_folds(index.min(), index.max(), [failure])[0]
    return fold, test, failure


def test_evaluate_fold() -> None:
    print("\n[6] Event-level metrics on one fold")
    fold, test, failure = toy_fold()

    alarm = pd.Series(False, index=test.index)
    alarm[failure.start - pd.Timedelta(hours=2)] = True  # 2 h before the failure
    alarm[pd.Timestamp("2020-01-02 08:00")] = True       # one false alarm
    row = evaluate_fold("toy", fold, test, alarm, CONFIG)
    check(row["caught"], "Alarm inside the risk window = caught")
    check(row["warning_hours"] == 2.0, f"Warning measured from the first alarm = 2.0 h (got {row['warning_hours']})")
    check(row["false_alerts"] == 1, "The alarm on day 2 is the only false alert")

    late = pd.Series(False, index=test.index)
    late[failure.start - pd.Timedelta(hours=4)] = True  # before the 3 h window
    row_late = evaluate_fold("toy", fold, test, late, CONFIG)
    check(not row_late["caught"] and np.isnan(row_late["warning_hours"]),
          "An alarm 4 h early (outside the 3 h window) is not a catch; warning left blank")

    unknown = CONFIG.__class__(unknown_onset={"FX"})
    row_unknown = evaluate_fold("toy", fold, test, alarm, unknown)
    check(row_unknown["caught"] and np.isnan(row_unknown["warning_hours"]),
          "Unknown onset (like F1): caught is reported, warning hours are not")

    scores = pd.Series(test["label"].to_numpy(), index=test.index)  # a perfect score
    row_perfect = evaluate_fold("toy", fold, test, scores >= 1, CONFIG, scores=scores, threshold=1)
    check(row_perfect["pr_auc"] == 1.0 and row_perfect["recall"] == 1.0,
          "A perfect score gets PR-AUC 1.0 and recall 1.0")


# ------------------------------------------------------------------ luck and summary

def test_luck_and_summary() -> None:
    print("\n[7] Chance of catching by luck, and the summary across folds")
    fold, test, _ = toy_fold()
    expected = 1 - np.exp(-CONFIG.budget_per_operating_hour * 3)  # 3 operating hours in the window
    check(abs(chance_caught_by_luck(test, CONFIG) - expected) < 1e-12,
          f"Blind detector at budget catches a 3 h window with chance {expected:.4f}")

    check(abs(probability_at_least(2, [0.5, 0.5]) - 0.25) < 1e-12, "P(2 of 2 coin flips) = 0.25")
    check(abs(probability_at_least(1, [0.5, 0.5]) - 0.75) < 1e-12, "P(at least 1 of 2 coin flips) = 0.75")
    check(probability_at_least(0, [0.1, 0.2]) == 1.0, "P(at least 0) = 1")

    per_fold = pd.DataFrame([
        {"method": "M", "failure": "A", "caught": True, "warning_hours": 2.0, "onset_known": True,
         "false_alerts": 1, "normal_operating_hours": 100.0, "normal_weeks": 1.0,
         "chance_caught_by_luck": 0.01, "pr_auc": 0.5, "f2": 0.4},
        {"method": "M", "failure": "B", "caught": False, "warning_hours": float("nan"), "onset_known": True,
         "false_alerts": 0, "normal_operating_hours": 300.0, "normal_weeks": 3.0,
         "chance_caught_by_luck": 0.01, "pr_auc": 0.1, "f2": 0.0},
    ])
    summary = summarise(per_fold).loc["M"]
    check(summary["failures caught"] == "1 of 2", "Summary counts 1 of 2 caught")
    check(summary["false alerts per 100 operating hours"] == 0.25,
          "False-alert rate is pooled: 1 alert / 400 h = 0.25 per 100 h (not the mean of 1.0 and 0)")


def main() -> None:
    print("==================================================")
    print(" HISHAAM'S EVALUATION TESTS")
    print("==================================================")
    test_alerts()
    test_operating_hours()
    test_folds()
    test_threshold()
    test_cross_fitting()
    test_evaluate_fold()
    test_luck_and_summary()
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
