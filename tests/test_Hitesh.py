"""Tests for Hitesh's CNN+GRU (models/) and folds.py, on the real MetroPT-3 data.

Run from anywhere: python tests/test_Hitesh.py
Needs PyTorch and the dataset (config.DATA_FILE). Takes a few minutes: it loads
the full dataset and trains the network briefly on one real fold. It runs on a
GPU if there is one, otherwise on the CPU.
"""
import sys
from pathlib import Path

# Put the project root on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

import config
from evaluation import EvalConfig, make_folds, split
from folds import METROPT_FAILURES, assert_purge_check, get_loeo_folds
from minute_table import get_feature_names, load_minute_table
from models.cnn_gru_network import RailGuardHybridNet
from models.compute import describe_device, get_device
from models.deep_learning_model import make_fit_and_score
from models.sequences import find_window_end_rows, gather_windows
from sklearn.preprocessing import StandardScaler

SEQUENCE_MINUTES = config.DEEP_LEARNING_SEQUENCE_MINUTES
failed_checks = []


def check(condition: bool, message: str) -> None:
    """Print a tick or a cross for one check and remember the failures."""
    if condition:
        print(f"  ✓ {message}")
    else:
        print(f"  ❌ {message}")
        failed_checks.append(message)


def test_network_shapes(minute_table: pd.DataFrame) -> None:
    print("\n[1] Network shape, on 16 real windows")
    feature_names = get_feature_names(minute_table)
    scaled_values = torch.as_tensor(
        StandardScaler().fit_transform(minute_table[feature_names]), dtype=torch.float32)
    end_rows = torch.as_tensor(find_window_end_rows(minute_table.index, SEQUENCE_MINUTES)[:16], dtype=torch.long)
    real_windows = gather_windows(scaled_values, end_rows, SEQUENCE_MINUTES)
    check(real_windows.shape == (16, SEQUENCE_MINUTES, len(feature_names)),
          f"16 real windows of {SEQUENCE_MINUTES} minutes x {len(feature_names)} features")
    network = RailGuardHybridNet(num_features=len(feature_names)).eval()
    with torch.no_grad():
        check(network(real_windows).shape == (16, 1), "16 real windows give 16 risk logits")
    notebook_network = RailGuardHybridNet(num_features=26)  # the notebook used 26 features
    check(sum(parameter.numel() for parameter in notebook_network.parameters() if parameter.requires_grad) == 53_313,
          "With the notebook's 26 features: 53,313 trainable parameters (same architecture as the notebook)")


def test_windows_on_real_minutes(minute_table: pd.DataFrame) -> None:
    print("\n[2] Windows never cross a gap in the real data")
    held_out = config.FAILURES[2]  # F3: the table has a long removed stretch here
    around_gap = minute_table[(minute_table.index >= held_out.start - pd.Timedelta(days=2))
                              & (minute_table.index <= held_out.excluded_until + pd.Timedelta(days=2))]
    gap_sizes = around_gap.index.to_series().diff().dropna()
    check(gap_sizes.max() > pd.Timedelta(hours=12), f"the slice contains a real gap ({gap_sizes.max()})")

    window_ends = find_window_end_rows(around_gap.index, SEQUENCE_MINUTES)
    for row in window_ends:
        window_times = around_gap.index[row - SEQUENCE_MINUTES + 1: row + 1]
        if window_times[-1] - window_times[0] != pd.Timedelta(minutes=SEQUENCE_MINUTES - 1):
            check(False, f"window ending at {around_gap.index[row]} spans a gap")
            return
    check(True, f"all {len(window_ends):,} windows are {SEQUENCE_MINUTES} unbroken minutes")

    check(len(window_ends) <= len(around_gap) - (SEQUENCE_MINUTES - 1),
          "never more windows than rows minus the first sequence_minutes - 1")
    check(len(window_ends) < len(around_gap) - (SEQUENCE_MINUTES - 1),
          "the gap removes some windows: the minutes right after it have no full window")


def test_folds_on_real_minutes(minute_table: pd.DataFrame) -> EvalConfig:
    print("\n[3] Folds and the purge check")
    settings = EvalConfig()
    folds = get_loeo_folds(minute_table, settings)
    check([fold.held_out.name for fold in folds] == ["F1", "F2", "F3", "F4"], "four folds, one per failure, in time order")
    check(folds == make_folds(minute_table.index.min(), minute_table.index.max(), config.FAILURES),
          "same folds as the team's evaluation, so the methods are comparable")
    check(assert_purge_check(folds, minute_table, settings, verbose=False),
          f"purge audit passes with the {config.PURGE_GAP} gap")
    check([failure["name"] for failure in METROPT_FAILURES] == ["F1", "F2", "F3", "F4"], "METROPT_FAILURES lists F1 to F4")

    too_short = EvalConfig(purge_gap=pd.Timedelta(minutes=30))
    try:
        get_loeo_folds(minute_table, too_short)
        check(False, "a 30-minute gap is refused (shorter than sequence plus rolling window)")
    except ValueError:
        check(True, "a 30-minute gap is refused (shorter than sequence plus rolling window)")
    try:
        assert_purge_check(folds, minute_table, too_short, verbose=False)
        check(False, "a 30-minute gap fails the purge audit")
    except AssertionError:
        check(True, "a 30-minute gap fails the purge audit")
    return settings


def test_fit_and_score_on_real_fold(minute_table: pd.DataFrame, settings: EvalConfig) -> None:
    print("\n[4] fit_and_score on the real hold-out-F4 fold (1 epoch, 5,000 normal windows)")
    device = get_device(require_gpu=False)
    print(f"  {describe_device(device)}")
    check(device.type in ("cuda", "mps", "cpu"), "a device was chosen")

    feature_names = get_feature_names(minute_table)
    fold = get_loeo_folds(minute_table, settings)[-1]
    train, test = split(minute_table, fold, settings)
    test = test.iloc[:5 * 24 * 60]  # first 5 days of the test block keeps the test short
    fit_and_score = make_fit_and_score(feature_names, device, epochs=1, max_normal_windows=5_000)

    scores = fit_and_score(train, test)
    check(len(scores) == len(test), "one score per test minute")
    check(np.isnan(scores[:SEQUENCE_MINUTES - 1]).all(), "the first sequence_minutes - 1 minutes have no window (NaN)")
    has_window = ~np.isnan(scores)
    check(has_window.sum() > 0 and ((scores[has_window] >= 0) & (scores[has_window] <= 1)).all(),
          "every other minute has a probability between 0 and 1")
    check(np.allclose(scores, fit_and_score(train, test), atol=1e-5, equal_nan=True),
          "a second run with the same seed gives the same scores")
    check(not train.index.isin(test.index).any(), "no test minute is in the training rows")


def main() -> None:
    print("=" * 50)
    print(" HITESH'S CNN+GRU TESTS (real data)")
    print("=" * 50)
    minute_table = load_minute_table()
    test_network_shapes(minute_table)
    test_windows_on_real_minutes(minute_table)
    settings = test_folds_on_real_minutes(minute_table)
    test_fit_and_score_on_real_fold(minute_table, settings)

    print("\n" + "=" * 50)
    if failed_checks:
        print(f" {len(failed_checks)} CHECK(S) FAILED:")
        for message in failed_checks:
            print(f"   - {message}")
        print("=" * 50)
        sys.exit(1)
    print(" ALL CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
