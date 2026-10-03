"""Threshold-rule baselines.

Both rules produce one alarm decision per FEATURE_STEP minute, so they can be
compared with the Random Forest on the same rows.
"""
import numpy as np
import pandas as pd

from alerts import count_false_alerts, operating_hours_of_normal_minutes
from config import FEATURE_STEP, LPS_COLUMN, PRESSURE_COLUMNS


def lps_alarm_per_minute(readings: pd.DataFrame) -> pd.Series:
    """Rule (a): current practice. The alarm is on when LPS is active. No tuning.

    Args:
        readings: Data with a DatetimeIndex and an LPS column.

    Returns:
        Boolean series, True for minutes in which LPS was active at least once.
        Minutes with no readings are left out.
    """
    lps_active = (readings[LPS_COLUMN] == 1).astype(float)
    return lps_active.resample(FEATURE_STEP).max().dropna().astype(bool)


def lowest_pressure_per_minute(readings: pd.DataFrame) -> pd.Series:
    """Lowest value of TP3 and Reservoirs seen in each minute.

    Args:
        readings: Data with a DatetimeIndex and the PRESSURE_COLUMNS.

    Returns:
        Float series per minute. Minutes with no readings are left out.
    """
    lowest_of_both_sensors = readings[list(PRESSURE_COLUMNS)].min(axis=1)
    return lowest_of_both_sensors.resample(FEATURE_STEP).min().dropna()


class LowPressureRule:
    """Rule (b): alarm when TP3 or Reservoirs falls below a cut-off.

    The cut-off is set on the normal periods of the training folds only: the
    highest (most sensitive) cut-off whose false alerts per operating hour
    stay within the budget. It is never set on the held-out fold.
    """

    def __init__(self, false_alerts_per_operating_hour_budget: float):
        """Create an unfitted rule.

        Args:
            false_alerts_per_operating_hour_budget: Most false alerts allowed
                per operating hour, for example 1 / 100.
        """
        self.false_alerts_per_operating_hour_budget = false_alerts_per_operating_hour_budget
        self.pressure_cutoff = None

    def fit(self, lowest_pressure_training: pd.Series, labels_training: pd.Series,
            operating_training: pd.Series) -> "LowPressureRule":
        """Set the cut-off from the normal training minutes.

        Candidate cut-offs are tried from the lowest upwards. The search stops
        at the first one that breaks the budget, and the one before it is kept.

        Args:
            lowest_pressure_training: Output of `lowest_pressure_per_minute`
                restricted to the training period.
            labels_training: Labels from `make_labels`, aligned to the above.
            operating_training: Output of `operating_per_minute`, aligned too.

        Returns:
            The fitted rule (self).
        """
        is_normal = (labels_training == 0).fillna(False).to_numpy(dtype=bool)
        normal_operating_hours = operating_hours_of_normal_minutes(labels_training, operating_training)
        if normal_operating_hours == 0:
            raise ValueError("No operating hours in the normal training minutes.")

        normal_pressure = lowest_pressure_training[is_normal]
        candidate_cutoffs = np.unique(normal_pressure.quantile(np.linspace(0, 0.5, 501)))
        # The lowest candidate equals the minimum pressure, so nothing is below it: no alarms.
        self.pressure_cutoff = candidate_cutoffs[0]
        for candidate_cutoff in candidate_cutoffs[1:]:
            alarm = lowest_pressure_training < candidate_cutoff
            false_alerts = count_false_alerts(alarm, labels_training)
            if false_alerts / normal_operating_hours > self.false_alerts_per_operating_hour_budget:
                break
            self.pressure_cutoff = candidate_cutoff
        return self

    def predict(self, lowest_pressure: pd.Series) -> pd.Series:
        """Return True for minutes whose lowest pressure is below the cut-off.

        Args:
            lowest_pressure: Output of `lowest_pressure_per_minute`.

        Returns:
            Boolean alarm series aligned to the input.
        """
        if self.pressure_cutoff is None:
            raise RuntimeError("Call fit() before predict().")
        return lowest_pressure < self.pressure_cutoff
