"""RailGuard AI: run every method on the four leave-one-failure-out folds.

    python main.py

Steps: check the GPU, load the per-minute table of the real MetroPT-3 data
(cached after the first run), work out the false-alarm budget, make the folds
and check the purge gap, judge each method the same way on each fold, save
results/ and figures/.
"""
import random
import time

import numpy as np

import config
from console import print_results, print_step, print_table, print_title, print_values
from data import operating_hours_and_budget
from evaluation import EvalConfig, describe_folds, save_results
from folds import assert_purge_check, get_loeo_folds
from minute_table import get_feature_names, load_minute_table
from models.compute import describe_device, get_device
from run_folds import build_scored_methods, run_all_folds

STEP_COUNT = 5


def main() -> None:
    run_start = time.perf_counter()
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print_title("RAILGUARD AI: four-fold evaluation on MetroPT-3")

    print_step(1, STEP_COUNT, "Compute device")
    device = get_device()  # first, so a missing GPU stops the run before the slow steps
    print(f"  {describe_device(device)}")

    print_step(2, STEP_COUNT, "Per-minute table (real MetroPT-3 data)")
    minute_table = load_minute_table(verbose=True)
    feature_names = get_feature_names(minute_table)
    print_values({
        "Minutes": f"{len(minute_table):,}",
        "Features": len(feature_names),
        "Risk minutes": f"{int(minute_table['label'].sum()):,}",
    })

    print_step(3, STEP_COUNT, "False-alarm budget")
    budget = operating_hours_and_budget(minute_table["operating"])
    print_values({
        "Operating minutes": f"{budget['operating_minutes']:,}",
        "Operating hours": f"{budget['total_operating_hours']:,.1f}",
        "Budget over these minutes": f"at most {budget['max_false_alarms_budget']} false alerts",
    })

    print_step(4, STEP_COUNT, "Folds and purge check")
    settings = EvalConfig()  # every setting from config.py
    folds = get_loeo_folds(minute_table, settings)
    assert_purge_check(folds, minute_table, settings, verbose=False)
    print(f"  Purge check passed for all {len(folds)} folds (gap {config.PURGE_GAP}).")
    print_table(describe_folds(minute_table, folds, settings))

    print_step(5, STEP_COUNT, "Training and judging every method on every fold")
    scored_methods = build_scored_methods(feature_names, settings, device)
    per_fold, summary = run_all_folds(minute_table, folds, scored_methods, settings)
    save_results(per_fold, summary, config.RESULTS_DIR)

    print_results(per_fold, summary, settings.budget_per_operating_hour)
    print(f"\n  Files written: {config.RESULTS_DIR}/ (per_fold.csv, summary.csv, results.json), "
          f"{config.FIGURES_DIR}/ (timelines)")
    print(f"  Total time: {(time.perf_counter() - run_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
