"""Test script for Atul's Random Forest baseline.

Owner: Mundram Baboo Khoomeshwarsingh (Atul), ModelX (RailGuard AI)

Run from anywhere: python tests/test_random_forest.py
Uses the cached per-minute table. Takes a few minutes: the threshold needs three
extra trainings (out-of-sample scores on the training failures).
"""
import _atul_common as common
from config import FAILURES
from evaluation import EvalConfig, cross_fitted_scores, false_alert_rate, make_folds, threshold_for_budget
from models import predict_random_forest_alarms, train_random_forest
from run_folds import make_random_forest_fit_and_score


def budget_threshold(train, test, feature_names) -> float:
    """The same threshold main.py uses: the budget applied to out-of-sample training scores."""
    settings = EvalConfig()
    folds = make_folds(train.index.min(), test.index.max(), FAILURES)
    out_of_sample_scores = cross_fitted_scores(train, folds, settings, make_random_forest_fit_and_score(feature_names))
    has_score = out_of_sample_scores.notna()
    threshold = threshold_for_budget(out_of_sample_scores[has_score], train.loc[has_score, "label"],
                                     train.loc[has_score, "operating"], settings)
    rate = false_alert_rate(out_of_sample_scores[has_score] >= threshold, train.loc[has_score, "label"],
                            train.loc[has_score, "operating"], settings)
    common.check(rate <= settings.budget_per_operating_hour,
                 f"Threshold {threshold:.3f} keeps the out-of-sample training scores within budget "
                 f"({100 * rate:.2f} per 100 h)")
    return threshold


def test_random_forest(train, test, feature_names) -> object:
    print("\n[Random Forest] Trains, predicts with the budget threshold, and gives the same answer twice...")
    threshold = budget_threshold(train, test, feature_names)
    first_forest = train_random_forest(train[feature_names], train["label"])
    second_forest = train_random_forest(train[feature_names], train["label"])
    first_alarms = predict_random_forest_alarms(first_forest, test[feature_names], threshold)
    second_alarms = predict_random_forest_alarms(second_forest, test[feature_names], threshold)
    common.check(len(first_alarms) == len(test), "One alarm decision per test minute")
    common.check(bool((first_alarms == second_alarms).all()), "Two trainings with the fixed seed give identical alarms")
    common.check(sorted(first_forest.classes_) == [0, 1], "The forest learned both classes, normal (0) and risk (1)")
    return first_alarms


def main() -> None:
    common.print_header("RUNNING ATUL'S RANDOM FOREST TESTS")
    train, test, feature_names = common.load_train_test()
    common.test_alerts(test)
    common.test_low_pressure_rule_budget(train)
    alarms = test_random_forest(train, test, feature_names)
    common.print_result("Random Forest", alarms, train, test)
    common.finish()


if __name__ == "__main__":
    main()
