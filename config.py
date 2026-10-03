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

# PLACEHOLDER: Shazia owns the operating-hour definition (condition 4). A
# minute counts as operating when the mean motor current is above this.
MOTOR_CURRENT_COLUMN = "Motor_current"
OPERATING_MOTOR_CURRENT_AMPS = 1.0

RANDOM_FOREST_TREES = 200
RANDOM_FOREST_ALARM_PROBABILITY = 0.5
