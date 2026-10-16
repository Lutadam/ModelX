"""RailGuard AI: run every method on the four leave-one-failure-out folds.

    python main.py

Steps: check the GPU, load the per-minute table of the real MetroPT-3 data
(cached after the first run), the data checks (horizon figure, gap analysis),
work out the false-alarm budget, make the folds and check the purge gap, judge
each method the same way on each fold, save results/ and figures/
(report_figures.py for the summary figures). One command makes everything the
report and the demo need.
"""
import random
import sys
import time
import traceback
from datetime import datetime
from typing import NoReturn

import numpy as np

import config
from data.dataset import operating_hours_and_budget
from data.minute_table import get_feature_names, load_minute_table
from evaluation.console import print_results, print_step, print_table, print_title, print_values
from evaluation.core import EvalConfig, describe_folds, save_results
from evaluation.folds import assert_purge_check, get_loeo_folds
from evaluation.gap_analysis import make_gap_report
from evaluation.horizon_figure import make_horizon_report
from evaluation.report_figures import make_report_figures
from evaluation.run_folds import build_scored_methods, run_all_folds
from models.compute import describe_device, get_device

STEP_COUNT = 6


def describe_run(device, run_start_time: datetime) -> dict:
    """When, on what and with which settings the results were made (top of results.json)."""
    return {
        "started": f"{run_start_time:%Y-%m-%d %H:%M}",
        "device": describe_device(device),
        "settings": {
            "random_seed": config.RANDOM_SEED,
            "warning_horizon": str(config.WARNING_HORIZON),
            "false_alerts_per_operating_hour_budget": config.FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
            "alert_cooldown": str(config.ALERT_COOLDOWN),
            "realert_after": str(config.REALERT_AFTER),
            "purge_gap": str(config.PURGE_GAP),
            "operating_motor_current_amps": config.OPERATING_MOTOR_CURRENT_AMPS,
            "random_forest_trees": config.RANDOM_FOREST_TREES,
            "isolation_forest_trees": config.ISOLATION_FOREST_TREES,
            "deep_learning_epochs": config.DEEP_LEARNING_EPOCHS,
            "deep_learning_sequence_minutes": config.DEEP_LEARNING_SEQUENCE_MINUTES,
            "deep_learning_risk_weight": config.DEEP_LEARNING_RISK_WEIGHT,
            "deep_learning_use_cudnn": config.DEEP_LEARNING_USE_CUDNN,
        },
    }


def main() -> None:
    run_start_time = datetime.now()
    run_start = time.perf_counter()
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print_title("RAILGUARD AI: four-fold evaluation on MetroPT-3")

    print_step(1, STEP_COUNT, "Compute device")
    try:
        device = get_device()  # first, so a missing GPU stops the run before the slow steps
    except RuntimeError as error:
        stop(str(error))
    print(f"  {describe_device(device)}")

    print_step(2, STEP_COUNT, "Per-minute table (real MetroPT-3 data)")
    try:
        minute_table = load_minute_table(verbose=True)
    except FileNotFoundError:
        stop(f"The MetroPT-3 CSV is not at {config.DATA_FILE}. Download it from UCI (see README, Setup) "
             "or change DATA_FILE in config.py.")
    feature_names = get_feature_names(minute_table)
    print_values({
        "Minutes": f"{len(minute_table):,}",
        "Features": len(feature_names),
        "Risk minutes": f"{int(minute_table['label'].sum()):,}",
    })

    print_step(3, STEP_COUNT, "Data checks: the 48 h before each failure, and the gaps in the data")
    # Figures and tables for the report only; no method depends on them, so a problem here warns and the run goes on.
    try:
        make_horizon_report(config.DATA_FILE)
    except Exception:
        print("\n  WARNING: the horizon figure could not be made. The evaluation continues. Error:")
        traceback.print_exc()
    try:
        make_gap_report(minute_table)
    except Exception:
        print("\n  WARNING: the gap analysis could not be made. The evaluation continues. Error:")
        traceback.print_exc()

    print_step(4, STEP_COUNT, "False-alarm budget")
    budget = operating_hours_and_budget(minute_table["operating"])
    print_values({
        "Operating minutes": f"{budget['operating_minutes']:,}",
        "Operating hours": f"{budget['total_operating_hours']:,.1f}",
        "Budget over these minutes": f"at most {budget['max_false_alarms_budget']} false alerts",
    })

    print_step(5, STEP_COUNT, "Folds and purge check")
    settings = EvalConfig()  # every setting from config.py
    folds = get_loeo_folds(minute_table, settings)
    assert_purge_check(folds, minute_table, settings, verbose=False)
    print(f"  Purge check passed for all {len(folds)} folds (gap {config.PURGE_GAP}).")
    print_table(describe_folds(minute_table, folds, settings))

    print_step(6, STEP_COUNT, "Training and judging every method on every fold")
    scored_methods = build_scored_methods(feature_names, settings, device)
    per_fold, summary, traces = run_all_folds(minute_table, folds, scored_methods, settings)
    results_dir = config.RESULTS_DIR
    try:
        save_results(per_fold, summary, results_dir, describe_run(device, run_start_time))
    except PermissionError as error:
        # Usually a results file is open in Excel. Keep the run: write next to it instead.
        results_dir = config.RESULTS_DIR.with_name(f"{config.RESULTS_DIR.name}_{run_start_time:%Y%m%d_%H%M}")
        print(f"\n  WARNING: could not write {error.filename} (open in another program?). "
              f"Writing the results to {results_dir}/ instead.")
        save_results(per_fold, summary, results_dir, describe_run(device, run_start_time))
    try:
        make_report_figures(per_fold, traces, minute_table, settings)
    except Exception:  # the results are already saved: a figure problem must not lose the run
        print("\n  WARNING: the summary figures could not be drawn. The results files are complete. Error:")
        traceback.print_exc()

    print_results(per_fold, summary, settings.budget_per_operating_hour)
    print(f"\n  Files written: {results_dir}/ (per_fold.csv, summary.csv, results.json), "
          f"{config.FIGURES_DIR}/ (data/, summary/, alerts/, scores/, timelines/)")
    print(f"  Total time: {(time.perf_counter() - run_start) / 60:.1f} min")


def stop(message: str) -> NoReturn:
    """End the run with a clear message instead of a traceback, for problems outside the code."""
    print(f"\n  STOPPED: {message}")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop("interrupted (Ctrl+C).")
