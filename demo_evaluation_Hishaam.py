"""Demo: runs the whole evaluation on FAKE per-minute data, with no project code.

    python demo_evaluation_Hishaam.py

The fake data has the real MetroPT-3 dates and failures, so the folds look
like the real ones, and the same columns as main.build_minute_table():
label, lps_alarm, lowest_pressure, operating, then the features.
Three methods, named as in main.py, are evaluated:
  * "LPS active (current practice)": a fixed alarm, no threshold;
  * "Low pressure (TP3/Reservoirs)": score -lowest_pressure, thresholded from the budget;
  * "Random Forest": a learned model, thresholded on cross-fitted scores.
Swap the fake table for build_minute_table(load_dataset()) later; nothing else changes.
The numbers this prints are MEANINGLESS. Only the mechanics are being tested.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation_Hishaam import (
    METROPT_FAILURES,
    EvalConfig,
    cross_fitted_scores,
    describe_folds,
    evaluate_fold,
    make_folds,
    plot_fold,
    save_results,
    split,
    summarise,
    threshold_for_budget,
)

RANDOM_SEED = 42  # same as config.RANDOM_SEED
NON_FEATURE_COLUMNS = ["label", "lps_alarm", "lowest_pressure", "operating"]  # same as main.py


def fake_minute_table(failures: list, config: EvalConfig) -> pd.DataFrame:
    """Per-minute table with the same columns as main.build_minute_table(), fake values."""
    rng = np.random.default_rng(RANDOM_SEED)
    index = pd.date_range("2020-02-01", "2020-09-01", freq="1min")
    pressure = 9.0 + rng.normal(0, 0.15, len(index))
    vibration = rng.normal(0, 1, len(index))
    label = np.zeros(len(index))
    keep = np.ones(len(index), bool)
    for number, failure in enumerate(failures):
        window = (index >= failure.start - config.warning_horizon) & (index < failure.start)
        label[window] = 1
        drift = 0.2 if failure.name == "F3" else 1.0  # F3 made hard on purpose
        pressure[window] -= drift * np.linspace(0, 1, window.sum())
        vibration[window] += 2 * drift
        keep &= ~((index >= failure.start) & (index <= failure.excluded_until))  # remove failure/maintenance
    for start in rng.integers(0, len(index) - 30, 40):  # random dips: false-alarm material
        pressure[start:start + 30] -= rng.uniform(0.3, 0.9)
    table = pd.DataFrame({
        "label": label,
        "lps_alarm": (rng.random(len(index)) < 0.0005).astype(float),
        "lowest_pressure": pressure,
        "operating": (rng.random(len(index)) < 0.6).astype(float),
        "TP3_mean": pressure + rng.normal(0, 0.05, len(index)),
        "Motor_current_std": vibration,
    }, index=index)
    return table[keep]


def main() -> None:
    config = EvalConfig()
    minutes = fake_minute_table(METROPT_FAILURES, config)
    features = [c for c in minutes.columns if c not in NON_FEATURE_COLUMNS]

    folds = make_folds(minutes.index.min(), minutes.index.max(), METROPT_FAILURES)
    print("Leave-one-failure-out folds:")
    print(describe_folds(minutes, folds, config).to_string())

    def forest_fit_and_score(fit_rows: pd.DataFrame, score_rows: pd.DataFrame) -> np.ndarray:
        forest = RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                        random_state=RANDOM_SEED, n_jobs=-1)
        forest.fit(fit_rows[features], fit_rows["label"].astype(int))
        return forest.predict_proba(score_rows[features])[:, 1]

    rows = []
    for fold in folds:
        train, test = split(minutes, fold, config)

        # Method 1: current practice, a fixed alarm with no score.
        lps_alarm = test["lps_alarm"].astype(bool)
        rows.append(evaluate_fold("LPS active (current practice)", fold, test, lps_alarm, config))

        # Method 2: low-pressure rule. Lower pressure = more risk, so the score is -lowest_pressure.
        rule_threshold = threshold_for_budget(-train["lowest_pressure"], train["label"], train["operating"], config)
        rule_scores = -test["lowest_pressure"]
        rule_alarm = rule_scores >= rule_threshold
        rows.append(evaluate_fold("Low pressure (TP3/Reservoirs)", fold, test, rule_alarm, config,
                                  rule_scores, rule_threshold))
        plot_fold("Low pressure (TP3/Reservoirs)", fold, test, rule_alarm, config, rule_scores, rule_threshold)

        # Method 3: a learned model, thresholded on out-of-sample training scores.
        oof = cross_fitted_scores(train, folds, config, forest_fit_and_score)
        scored = oof.notna()
        forest_threshold = threshold_for_budget(oof[scored], train.loc[scored, "label"],
                                                train.loc[scored, "operating"], config)
        forest_scores = pd.Series(forest_fit_and_score(train, test), index=test.index)
        forest_alarm = forest_scores >= forest_threshold
        rows.append(evaluate_fold("Random Forest", fold, test, forest_alarm, config, forest_scores, forest_threshold))
        plot_fold("Random Forest", fold, test, forest_alarm, config, forest_scores, forest_threshold)

    per_fold = pd.DataFrame(rows)
    summary = summarise(per_fold)
    save_results(per_fold, summary)

    print("\nPer fold:")
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(per_fold[["method", "failure", "caught", "warning_hours", "false_alerts",
                        "false_alerts_per_100_operating_hours", "within_budget",
                        "chance_caught_by_luck", "pr_auc", "f2"]].to_string(index=False))
    print("\nAcross folds:")
    print(summary.to_string())
    print("\nWritten: results/ and figures/  (FAKE data: numbers are meaningless)")


if __name__ == "__main__":
    main()
