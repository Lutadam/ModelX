"""Shared constants for the ModelX pipeline.

Every other module imports its settings from here, so a value such as the
warning horizon or the random seed is defined in exactly one place.
"""
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------- reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------- data
DATA_FILE = Path.home() / "Desktop" / "metropt+3+dataset" / "MetroPT3(AirCompressor).csv"
FIGURES_DIR = Path("figures")
FIGURES_DATA_DIR = FIGURES_DIR / "data"            # horizon figure, gaps: what the data looks like
FIGURES_SUMMARY_DIR = FIGURES_DIR / "summary"      # every method side by side
FIGURES_ALERTS_DIR = FIGURES_DIR / "alerts"        # every alert in each test block
FIGURES_SCORES_DIR = FIGURES_DIR / "scores"        # score vs threshold, one file per method
FIGURES_TIMELINES_DIR = FIGURES_DIR / "timelines"  # 48 h before each failure, one subfolder per method

EXPECTED_SAMPLING_SECONDS = 10  # file is 0.1 Hz (confirmed from timestamps)

# 5 min is 30x the nominal interval; observed jitter is only 9-13 s, so
# anything longer is a real outage, not timing noise.
GAP_THRESHOLD = pd.Timedelta(minutes=5)

# ---------------------------------------------------------------- labels
WARNING_HORIZON = pd.Timedelta(hours=3)  # risk = [failure start - 3 h, failure start)


class Failure(NamedTuple):
    """One compressor failure.

    Rows from `start` up to and including `excluded_until` are neither risk
    nor normal and are dropped from training and evaluation. `excluded_until`
    is the maintenance time where known, otherwise the failure end.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    excluded_until: pd.Timestamp


FAILURES = [  # renumbered: the UCI table lists #1 twice
    Failure("F1", pd.Timestamp("2020-04-18 00:00"), pd.Timestamp("2020-04-18 23:59"),
            pd.Timestamp("2020-04-18 23:59")),
    Failure("F2", pd.Timestamp("2020-05-29 23:30"), pd.Timestamp("2020-05-30 06:00"),
            pd.Timestamp("2020-05-30 06:00")),
    Failure("F3", pd.Timestamp("2020-06-05 10:00"), pd.Timestamp("2020-06-07 14:30"),
            pd.Timestamp("2020-06-08 16:00")),  # maintenance finished 8 Jun 16:00
    Failure("F4", pd.Timestamp("2020-07-15 14:30"), pd.Timestamp("2020-07-15 19:00"),
            pd.Timestamp("2020-07-16 00:00")),  # maintenance finished 16 Jul 00:00
]

# ---------------------------------------------------------------- time-respecting fold
# Train on everything before the end of F3's maintenance (so F1-F3), test on
# everything after it (so F4). Assumption: change here if the team agrees a
# different cut.
TEST_START = pd.Timestamp("2020-06-08 16:00")

# ---------------------------------------------------------------- features
FEATURE_STEP = "1min"       # feature rows, labels and alarms all live on this grid
ROLLING_WINDOW = "60min"
MIN_WINDOW_COVERAGE = 0.7   # share of expected rows a window needs (tolerates 12 s stretches)

SENSOR_COLUMNS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
]

# ---------------------------------------------------------------- baselines
LPS_COLUMN = "LPS"
PRESSURE_COLUMNS = ("TP3", "Reservoirs")

# Condition 4 of the proposal feedback: at most 1 false alert per train per
# 100 operating hours. Thresholds are set on the training folds only.
FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET = 1 / 100

# Alarm minutes less than this apart belong to one alert (one inspection).
# Assumption: reuse the warning horizon, since a warning is acted on within it.
ALERT_COOLDOWN = WARNING_HORIZON

# An alarm still on this long after its alert began counts as a NEW alert (the
# depot would inspect again the next day). Used by evaluation/core.py only.
REALERT_AFTER = pd.Timedelta(hours=24)

# Every budget search (rules, Isolation Forest, evaluation) tries this many
# candidate thresholds, spread evenly over the score quantiles.
THRESHOLD_CANDIDATES = 501

# PLACEHOLDER: Shazia owns the operating-hour definition (condition 4). A
# minute counts as operating when the mean motor current is above this.
MOTOR_CURRENT_COLUMN = "Motor_current"
OPERATING_MOTOR_CURRENT_AMPS = 1.0

ISOLATION_FOREST_TREES = 200
RANDOM_FOREST_TREES = 200

# ---------------------------------------------------------------- evaluation
RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")  # saved per-minute table (minute_table.load_minute_table); safe to delete
F_BETA = 2.0                          # F-score weighting: a missed failure costs more than a false alarm
UNKNOWN_ONSET_FAILURES = {"F1"}       # start time is a placeholder, so hours of warning are not reported
TIME_ORDERED_FAILURE = FAILURES[-1].name  # the only fold that trains on earlier failures only

# Timeline figures (evaluation/core.py and evaluation/horizon_figure.py) show this stretch around each failure.
FIGURE_HOURS_BEFORE_FAILURE = 48
FIGURE_HOURS_AFTER_FAILURE = 2
# Training rows this close to a test block are dropped, so no training row
# overlaps a test row. Must be at least the CNN+GRU sequence plus ROLLING_WINDOW.
PURGE_GAP = pd.Timedelta(hours=24)

# ---------------------------------------------------------------- deep learning (CNN + GRU)
# True: stop with an error when PyTorch finds no GPU. False: fall back to the CPU (slow).
REQUIRE_GPU = True
# cuDNN is NVIDIA's fast library for the GRU. Off because, with the RTX 3060's
# driver 555 (CUDA 12.5) and this PyTorch (CUDA 12.6, cuDNN 9.10), training a GRU
# with cuDNN makes Python crash on exit. Off = PyTorch's own GPU code: slower,
# same model. Turn it back on after a driver update to CUDA 12.6 or newer.
DEEP_LEARNING_USE_CUDNN = False
DEEP_LEARNING_SEQUENCE_MINUTES = 120    # minutes of history the network sees per prediction
DEEP_LEARNING_EPOCHS = 3
DEEP_LEARNING_BATCH_SIZE = 256
DEEP_LEARNING_LEARNING_RATE = 1e-3
DEEP_LEARNING_RISK_WEIGHT = 15.0        # a risk window counts this many times more than a normal one in the loss
DEEP_LEARNING_MAX_NORMAL_WINDOWS = 60_000  # normal training windows kept per training run (all risk windows are kept)
DEEP_LEARNING_CNN_CHANNELS = 32
DEEP_LEARNING_GRU_HIDDEN_SIZE = 64
DEEP_LEARNING_GRU_LAYERS = 2
DEEP_LEARNING_DROPOUT = 0.2
