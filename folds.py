"""Leave-one-failure-out folds with the purge-gap check (Hitesh).

Condition 2 (leak-free folds) and refinement 3 (purge check). The folds are
the team's (`evaluation.make_folds`), so the CNN+GRU is judged on the same test
blocks as every other method. This module adds the checks a sequence model needs:

* the purge gap must cover the input sequence plus the longest rolling feature
  (`evaluation.minimum_purge_gap`), and
* `assert_purge_check` proves no training row is closer to a test block than that.
"""
import pandas as pd

import config
from evaluation import EvalConfig, check_purge_gap, make_folds, masks, minimum_purge_gap

# Same failures as config.FAILURES, as plain dicts for display (`pd.DataFrame(METROPT_FAILURES)`).
METROPT_FAILURES = [failure._asdict() for failure in config.FAILURES]


def get_loeo_folds(minute_table: pd.DataFrame, settings: EvalConfig,
                   sequence_minutes: int = config.DEEP_LEARNING_SEQUENCE_MINUTES) -> list:
    """One fold per failure: each failure is held out in turn.

    Args:
        minute_table: Output of `build_minute_table`.
        settings: Evaluation settings (they hold the purge gap).
        sequence_minutes: Length of the CNN+GRU window, in minutes.

    Returns:
        The folds, in time order. Raises ValueError if the purge gap is shorter
        than the sequence plus the rolling window.
    """
    check_purge_gap(settings, pd.Timedelta(minutes=sequence_minutes), pd.Timedelta(config.ROLLING_WINDOW))
    return make_folds(minute_table.index.min(), minute_table.index.max(), config.FAILURES)


def assert_purge_check(folds: list, minute_table: pd.DataFrame, settings: EvalConfig,
                       sequence_minutes: int = config.DEEP_LEARNING_SEQUENCE_MINUTES,
                       verbose: bool = True) -> bool:
    """Check every fold: no training row inside a test block, and a big enough gap around it.

    A training window reaches `sequence_minutes` back and a rolling feature
    reaches further back by the rolling window, so the nearest training row
    must be at least that far from the test block on both sides.

    Returns:
        True when every fold passes. Raises AssertionError otherwise.
    """
    required_gap = minimum_purge_gap(pd.Timedelta(minutes=sequence_minutes), pd.Timedelta(config.ROLLING_WINDOW))
    for fold in folds:
        train_mask, test_mask = masks(minute_table.index, fold, settings)
        train_times = minute_table.index[train_mask]
        assert not (train_mask & test_mask).any(), f"{fold.name}: a test minute is in the training set"

        training_before_test = train_times[train_times < fold.test_start]
        training_after_test = train_times[train_times > fold.test_end]
        gap_before = fold.test_start - training_before_test.max() if len(training_before_test) else None
        gap_after = training_after_test.min() - fold.test_end if len(training_after_test) else None
        for side, gap in (("before", gap_before), ("after", gap_after)):
            assert gap is None or gap >= required_gap, \
                f"{fold.name}: training data is {gap} {side} the test block, need at least {required_gap}"
        if verbose:
            print(f"[PASS] {fold.name:<14} train {int(train_mask.sum()):>7,} | test {int(test_mask.sum()):>7,} | "
                  f"gap before {gap_before}, after {gap_after} (need >= {required_gap})")
    return True
