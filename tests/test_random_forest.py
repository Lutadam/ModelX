"""Test script for Atul's Random Forest baseline.

Owner: Mundram Baboo Khoomeshwarsingh (Atul), ModelX (RailGuard AI)

Run from anywhere: python tests/test_random_forest.py
Takes a minute or two because it loads the full dataset.
"""
import _atul_common as common
from models import predict_random_forest_alarms, train_random_forest


def test_random_forest(train, test, feature_names) -> object:
    print("\n[Random Forest] Trains, predicts, and gives the same answer twice...")
    first_forest = train_random_forest(train[feature_names], train["label"])
    second_forest = train_random_forest(train[feature_names], train["label"])
    first_alarms = predict_random_forest_alarms(first_forest, test[feature_names])
    second_alarms = predict_random_forest_alarms(second_forest, test[feature_names])
    common.check(len(first_alarms) == len(test), "One alarm decision per test minute")
    common.check(bool((first_alarms == second_alarms).all()), "Two trainings with the fixed seed give identical alarms")
    common.check(sorted(first_forest.classes_) == [0, 1], "The forest learned both classes, normal (0) and risk (1)")
    return first_alarms


def main() -> None:
    common.print_header("RUNNING ATUL'S RANDOM FOREST TESTS")
    common.test_alerts()
    common.test_low_pressure_rule_budget()
    train, test, feature_names = common.load_train_test()
    alarms = test_random_forest(train, test, feature_names)
    common.print_result("Random Forest", alarms, train, test)
    common.finish()


if __name__ == "__main__":
    main()
