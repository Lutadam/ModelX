"""Run every method on the four folds and judge them all the same way.

Methods:
  * LPS active (current practice): a fixed alarm, nothing to tune.
  * Low pressure rule: its cut-off is set from the false-alarm budget on the training minutes.
  * Random Forest, Isolation Forest, CNN+GRU: each gives a risk score per minute.
    The alarm threshold is the most sensitive one that keeps false alerts within
    the budget on out-of-sample training scores (`evaluation.cross_fitted_scores`),
    never on the test block.

A scored method is a function `fit_and_score(fit_rows, score_rows)` that trains
a FRESH model on `fit_rows` and returns one score per row of `score_rows`.
"""
import pandas as pd

import config
from console import run_with_progress
from evaluation import (
    EvalConfig,
    cross_fitted_scores,
    evaluate_fold,
    plot_fold,
    split,
    summarise,
    threshold_for_budget,
)
from models import IsolationForestDetector, train_random_forest
from models.deep_learning_model import make_fit_and_score as make_deep_learning_fit_and_score
from rules import LowPressureRule

LPS_METHOD = "LPS active (current practice)"
LOW_PRESSURE_METHOD = "Low pressure (TP3/Reservoirs)"
RANDOM_FOREST_METHOD = "Random Forest"
ISOLATION_FOREST_METHOD = "Isolation Forest"
DEEP_LEARNING_METHOD = "CNN+GRU (deep learning)"


# ============================================================ the scored methods

def make_random_forest_fit_and_score(feature_names: list):
    def fit_and_score(fit_rows: pd.DataFrame, score_rows: pd.DataFrame):
        forest = train_random_forest(fit_rows[feature_names], fit_rows["label"])
        risk_class_position = list(forest.classes_).index(1)
        return forest.predict_proba(score_rows[feature_names])[:, risk_class_position]

    return fit_and_score


def make_isolation_forest_fit_and_score(feature_names: list, settings: EvalConfig):
    def fit_and_score(fit_rows: pd.DataFrame, score_rows: pd.DataFrame):
        # Train only: the cut-off is set from out-of-sample scores in judge_scored_method
        detector = IsolationForestDetector(settings.budget_per_operating_hour).fit_forest(
            fit_rows[feature_names], fit_rows["label"])
        return detector.anomaly_scores(score_rows[feature_names]).to_numpy()

    return fit_and_score


def build_scored_methods(feature_names: list, settings: EvalConfig, device) -> dict:
    """Name -> fit_and_score for every method that gives a score per minute.

    Args:
        feature_names: Feature columns of the minute table.
        settings: Evaluation settings.
        device: From `models.compute.get_device`, for the CNN+GRU.
    """
    return {
        RANDOM_FOREST_METHOD: make_random_forest_fit_and_score(feature_names),
        ISOLATION_FOREST_METHOD: make_isolation_forest_fit_and_score(feature_names, settings),
        DEEP_LEARNING_METHOD: make_deep_learning_fit_and_score(feature_names, device),
    }


# ============================================================ judging one method on one fold

def judge_method(method_name: str, fold, test: pd.DataFrame, alarm: pd.Series, settings: EvalConfig,
                 scores: pd.Series = None, threshold: float = None) -> dict:
    """Judge one method on one fold and save its timeline figure. Returns one results row."""
    result = evaluate_fold(method_name, fold, test, alarm, settings, scores, threshold)
    plot_fold(method_name, fold, test, alarm, settings, scores, threshold, folder=config.FIGURES_DIR)
    return result


def judge_low_pressure_rule(fold, train: pd.DataFrame, test: pd.DataFrame, settings: EvalConfig) -> dict:
    rule = LowPressureRule(settings.budget_per_operating_hour).fit(
        train["lowest_pressure"], train["label"], train["operating"])
    alarm = rule.predict(test["lowest_pressure"])
    # A lower pressure means more risk, so the score is the pressure with its sign flipped.
    return judge_method(LOW_PRESSURE_METHOD, fold, test, alarm, settings,
                        -test["lowest_pressure"], -rule.pressure_cutoff)


def judge_scored_method(method_name: str, fit_and_score, fold, folds: list, train: pd.DataFrame,
                        test: pd.DataFrame, settings: EvalConfig) -> dict:
    out_of_sample_scores = cross_fitted_scores(train, folds, settings, fit_and_score)
    has_score = out_of_sample_scores.notna()
    threshold = threshold_for_budget(
        out_of_sample_scores[has_score], train.loc[has_score, "label"], train.loc[has_score, "operating"], settings)

    # Minutes without a full window (CNN+GRU only) have no evidence of risk: score 0, never an alarm.
    test_scores = pd.Series(fit_and_score(train, test), index=test.index).fillna(0.0)
    return judge_method(method_name, fold, test, test_scores >= threshold, settings, test_scores, threshold)


# ============================================================ all folds

def run_all_folds(minute_table: pd.DataFrame, folds: list, scored_methods: dict,
                  settings: EvalConfig) -> tuple:
    """Judge every method on every fold.

    Args:
        minute_table: Output of `build_minute_table`.
        folds: From `folds.get_loeo_folds`.
        scored_methods: From `build_scored_methods`.
        settings: Evaluation settings.

    Returns:
        (per_fold, summary): one row per method and fold, and one row per method.
    """
    result_rows = []
    for fold_number, fold in enumerate(folds, start=1):
        print(f"\n  Fold {fold_number}/{len(folds)}: {fold.name}", flush=True)
        train, test = split(minute_table, fold, settings)

        result_rows.append(run_with_progress(LPS_METHOD, lambda: judge_method(
            LPS_METHOD, fold, test, test["lps_alarm"].astype(bool), settings)))
        result_rows.append(run_with_progress(LOW_PRESSURE_METHOD, lambda: judge_low_pressure_rule(
            fold, train, test, settings)))
        for method_name, fit_and_score in scored_methods.items():
            result_rows.append(run_with_progress(method_name, lambda: judge_scored_method(
                method_name, fit_and_score, fold, folds, train, test, settings)))

    per_fold = pd.DataFrame(result_rows)
    return per_fold, summarise(per_fold)
