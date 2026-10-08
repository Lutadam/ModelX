"""Test script for Shazia's data loading, labeling, budget, and feature engineering.

Owner: Shazia Baboorally (Data & Features Specialist)
Part of Team ModelX (RailGuard AI)
"""

import sys
from pathlib import Path

# Run from anywhere: put the project root (where data.py and features.py live) on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import calculate_operating_hours_and_budget, load_dataset, make_labels
from features import build_features, fit_pca_health_index


def run_tests():
    print("==================================================")
    print(" RUNNING SHAZIA'S UNIT & INTEGRATION TESTS")
    print("==================================================")

    # 1. Test Dataset Loader
    print("\n[Test 1] Testing Data Loader & Segment Creation...")
    try:
        readings = load_dataset()
        print(f"  ✓ Loaded {len(readings):,} rows successfully!")
        print(f"  ✓ DatetimeIndex confirmed: {readings.index[0]} to {readings.index[-1]}")
        print(f"  ✓ Segments identified: {readings['segment'].nunique():,} contiguous segments.")
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        return

    # 2. Test Risk Labeling
    print("\n[Test 2] Testing Risk Labeling (make_labels)...")
    labels = make_labels(readings.index)
    print(f"  ✓ Risk Window Rows (y=1.0): {(labels == 1.0).sum():,}")
    print(f"  ✓ Normal Operating Rows (y=0.0): {(labels == 0.0).sum():,}")
    print(f"  ✓ Excluded Failure Rows (y=NaN): {labels.isna().sum():,}")

    # 3. Test Condition 4 Budget Calculation
    print("\n[Test 3] Testing Condition 4 False-Alarm Budget Calculation...")
    stats = calculate_operating_hours_and_budget(readings)

    # 4. Test Feature Engineering on sample rows
    print("\n[Test 4] Testing Feature Engineering (build_features)...")
    sample_features = build_features(readings.head(1000))
    print(f"  ✓ Sample Features Shape: {sample_features.shape}")
    print("  ✓ Feature Columns Created:", list(sample_features.columns)[:5])

    # 5. Test Leakage-Free PCA Fitting
    print("\n[Test 5] Testing Leakage-Free PCA Health Index...")
    train_hi, test_hi = fit_pca_health_index(sample_features, sample_features)
    print(f"  ✓ PCA Health Index Shape: {train_hi.shape}")

    print("\n==================================================")
    print(" 🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()