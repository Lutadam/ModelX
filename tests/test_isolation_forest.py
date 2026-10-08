"""Test script for Atul's Isolation Forest baseline.

Owner: Mundram Baboo Khoomeshwarsingh (Atul), ModelX (RailGuard AI)

Run from anywhere: python tests/test_isolation_forest.py
Takes a minute or two because it loads the full dataset.
"""
import _atul_common as common
from alerts import count_false_alerts, operating_hours_of_normal_minutes
from config import FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET
from models import IsolationForestDetector


def test_isolation_forest(train, test, feature_names) -> object:
    print("\n[Isolation Forest] Trains on normal minutes, sets its cut-off within the budget, is repeatable...")
    detector = IsolationForestDetector(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(
        train[feature_names], train["label"], train["operating"])
    repeat_detector = IsolationForestDetector(FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET).fit(
        train[feature_names], train["label"], train["operating"])

    alarms = detector.predict(test[feature_names])
    common.check(len(alarms) == len(test), "One alarm decision per test minute")
    common.check(bool((alarms == repeat_detector.predict(test[feature_names])).all()),
                 "Two trainings with the fixed seed give identical alarms")
    common.check(detector.score_cutoff == repeat_detector.score_cutoff, "The score cut-off is the same both times")

    training_alarms = detector.predict(train[feature_names])
    false_alerts = count_false_alerts(training_alarms, train["label"])
    normal_hours = operating_hours_of_normal_minutes(train["label"], train["operating"])
    rate = false_alerts / normal_hours
    common.check(rate <= FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
                 f"On the training normal minutes the cut-off gives {100 * rate:.2f} false alerts per 100 operating hours (budget 1.00)")

    scores = detector.anomaly_scores(train[feature_names])
    is_normal = (train["label"] == 0).to_numpy()
    common.check(scores[is_normal].quantile(0.5) < scores[is_normal].max(), "Anomaly scores vary: the most unusual normal minute scores above the typical one")
    print(f"  score cut-off set on training data: {detector.score_cutoff:.4f}")
    return alarms


def main() -> None:
    common.print_header("RUNNING ATUL'S ISOLATION FOREST TESTS")
    common.test_alerts()
    common.test_low_pressure_rule_budget()
    train, test, feature_names = common.load_train_test()
    alarms = test_isolation_forest(train, test, feature_names)
    common.print_result("Isolation Forest", alarms, train, test)
    common.finish()


if __name__ == "__main__":
    main()
