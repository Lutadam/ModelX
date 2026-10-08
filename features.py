"""Feature engineering pipeline for RailGuard AI.

Owner: Shazia Baboorally (Data & Features Specialist)
Part of Team ModelX (RailGuard AI)
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    EXPECTED_SAMPLING_SECONDS,
    FEATURE_STEP,
    MIN_WINDOW_COVERAGE,
    MOTOR_CURRENT_COLUMN,
    OPERATING_MOTOR_CURRENT_AMPS,
    RANDOM_SEED,
    ROLLING_WINDOW,
    SENSOR_COLUMNS,
)


def rolling_features(
    readings: pd.DataFrame,
    columns: list = None,
    window: str = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Compute segment-grouped rolling mean and standard deviation.

    Uses a time-based window ('60min') strictly grouped by `segment` so that
    rolling calculations never cross an outage gap.

    Args:
        readings: pd.DataFrame indexed by timestamp with a `segment` column.
        columns: List of sensor column names to compute rolling stats for.
        window: Rolling window duration string (default: '60min').

    Returns:
        pd.DataFrame of rolling features indexed by timestamp.
    """
    if columns is None:
        columns = [c for c in SENSOR_COLUMNS if c in readings.columns]

    # Calculate minimum required rows (70% coverage of 60 mins at 0.1 Hz = 252 rows)
    nominal_rows_per_window = int(pd.Timedelta(window).total_seconds() / EXPECTED_SAMPLING_SECONDS)
    min_periods = int(MIN_WINDOW_COVERAGE * nominal_rows_per_window)

    roll_df = pd.DataFrame(index=readings.index)
    grouped = readings.groupby("segment")

    for col in columns:
        # 1. Rolling Mean per segment
        roll_df[f"{col}_mean_60m"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        # 2. Rolling Standard Deviation per segment
        roll_df[f"{col}_std_60m"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std()
        )

    return roll_df


def compute_duty_cycle(readings: pd.DataFrame, window: str = ROLLING_WINDOW) -> pd.Series:
    """Compute the motor current duty cycle (% of time motor is running).

    Args:
        readings: pd.DataFrame with the motor current column and 'segment'.
        window: Rolling window duration string (default: '60min').

    Returns:
        pd.Series of duty cycle fractions indexed by timestamp. A reading
        counts as running above OPERATING_MOTOR_CURRENT_AMPS (config.py).
    """
    nominal_rows = int(pd.Timedelta(window).total_seconds() / EXPECTED_SAMPLING_SECONDS)
    min_periods = int(MIN_WINDOW_COVERAGE * nominal_rows)

    motor_active = (readings[MOTOR_CURRENT_COLUMN] > OPERATING_MOTOR_CURRENT_AMPS).astype(float)

    grouped = motor_active.groupby(readings["segment"])
    duty_cycle = grouped.transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
    duty_cycle.name = "duty_cycle_60m"

    return duty_cycle


def compute_duty_cycle_frequency_features(readings: pd.DataFrame, window: str = ROLLING_WINDOW) -> pd.DataFrame:
    """Describe the rhythm of the compressor's on/off cycling over a trailing window.

    The motor on/off signal is averaged per minute and, for each trailing
    window, its spectrum is taken (constant part removed). Windows never span
    two segments, and any window with a missing minute gives NaN. Unlike a
    plain spectral energy, which is only the variance again, these describe
    how fast the compressor cycles.

    Args:
        readings: pd.DataFrame with the motor current column and `segment`.
        window: Window length as a pandas offset string (default: '60min').

    Returns:
        pd.DataFrame at FEATURE_STEP with two columns:
        `duty_cycle_dominant_cycles_per_hour` (frequency of the strongest
        cycling component, 0 when the motor does not cycle) and
        `duty_cycle_dominant_power_share` (its share of the cycling power,
        close to 1 for a regular rhythm and low for an irregular one).
    """
    window_minutes = int(pd.Timedelta(window) / pd.Timedelta(FEATURE_STEP))
    motor_on = (readings[MOTOR_CURRENT_COLUMN] > OPERATING_MOTOR_CURRENT_AMPS).astype(float)
    motor_on_per_minute = motor_on.resample(FEATURE_STEP).mean()  # NaN for minutes with no readings
    # Forward-fill so empty minutes stay inside their segment and the minute grid stays unbroken.
    segment_per_minute = readings["segment"].resample(FEATURE_STEP).max().ffill()

    frequency_names = ["duty_cycle_dominant_cycles_per_hour", "duty_cycle_dominant_power_share"]
    frequency_features = pd.DataFrame(np.nan, index=motor_on_per_minute.index, columns=frequency_names)
    cycles_per_hour = np.fft.rfftfreq(window_minutes, d=1.0)[1:] * 60  # skip the constant component

    for _, segment_minutes in motor_on_per_minute.groupby(segment_per_minute):
        segment_values = segment_minutes.to_numpy()
        if len(segment_values) < window_minutes:
            continue
        windows = sliding_window_view(segment_values, window_minutes)
        is_complete = ~np.isnan(windows).any(axis=1)
        complete_windows = windows[is_complete]
        centred_windows = complete_windows - complete_windows.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(centred_windows, axis=1))[:, 1:] ** 2
        total_power = power.sum(axis=1)
        has_cycling = total_power > 0
        strongest_bin = power.argmax(axis=1)
        dominant_frequency = np.where(has_cycling, cycles_per_hour[strongest_bin], 0.0)
        dominant_power_share = np.divide(
            power.max(axis=1), total_power, out=np.zeros_like(total_power), where=has_cycling
        )
        window_end_minutes = segment_minutes.index[window_minutes - 1 :][is_complete]
        frequency_features.loc[window_end_minutes, frequency_names[0]] = dominant_frequency
        frequency_features.loc[window_end_minutes, frequency_names[1]] = dominant_power_share

    return frequency_features


def build_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Master feature builder function matching the team interface contract.

    Interface agreed with team:
        build_features(readings: pd.DataFrame) -> pd.DataFrame

    Args:
        readings: pd.DataFrame indexed by timestamp with a `segment` column.

    Returns:
        pd.DataFrame containing all engineered features, one row per
        FEATURE_STEP (1 minute). Minutes with no data and warm-up rows are NaN.
    """
    # 1. Compute rolling statistics and motor duty cycle (one row per reading)
    roll_stats = rolling_features(readings)
    duty_cycle = compute_duty_cycle(readings)

    # Interface contract: one row per FEATURE_STEP (1 minute), not per 10 s reading
    features_per_minute = pd.concat([roll_stats, duty_cycle], axis=1).resample(FEATURE_STEP).last()

    # 2. Frequency features of the duty cycle, already one row per minute
    frequency_features = compute_duty_cycle_frequency_features(readings)

    return pd.concat([features_per_minute, frequency_features], axis=1)


def fit_pca_health_index(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Fit StandardScaler and PCA on TRAIN features only to produce a 1D Health Index.

    CRUCIAL LEAKAGE RULE:
    Fit scalers and PCA strictly on training fold data, then transform both train and test.

    Rows with any NaN (warm-up rows, empty minutes) are dropped from both sets,
    so no made-up values are fed to the scaler.

    Args:
        train_features: pd.DataFrame of features for training folds. Pass only
            labelled rows, without the excluded failure and maintenance rows.
        test_features: pd.DataFrame of features for test fold.

    Returns:
        Tuple of (train_health_index, test_health_index) pd.Series, each
        indexed by the rows that had complete features.
    """
    clean_train = train_features.dropna()
    clean_test = test_features.dropna()

    scaler = StandardScaler()
    pca = PCA(n_components=1, random_state=RANDOM_SEED)

    # Fit ON TRAIN ONLY
    scaled_train = scaler.fit_transform(clean_train)
    pca_train_vals = pca.fit_transform(scaled_train).ravel()

    # Transform test features using fitted scaler and PCA
    scaled_test = scaler.transform(clean_test)
    pca_test_vals = pca.transform(scaled_test).ravel()

    train_health_index = pd.Series(pca_train_vals, index=clean_train.index, name="pca_health_index")
    test_health_index = pd.Series(pca_test_vals, index=clean_test.index, name="pca_health_index")

    return train_health_index, test_health_index