"""Feature engineering pipeline for RailGuard AI.

Owner: Shazia Baboorally (Data & Features Specialist)
Part of Team ModelX (RailGuard AI)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    EXPECTED_SAMPLING_SECONDS,
    MIN_WINDOW_COVERAGE,
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
        readings: pd.DataFrame with 'Motor_current' or 'TP2' and 'segment'.
        window: Rolling window duration string (default: '60min').

    Returns:
        pd.Series of duty cycle percentages indexed by timestamp.
    """
    nominal_rows = int(pd.Timedelta(window).total_seconds() / EXPECTED_SAMPLING_SECONDS)
    min_periods = int(MIN_WINDOW_COVERAGE * nominal_rows)

    col = "Motor_current" if "Motor_current" in readings.columns else "TP2"
    motor_active = (readings[col] > 1.0).astype(float)

    grouped = motor_active.groupby(readings["segment"])
    duty_cycle = grouped.transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
    duty_cycle.name = "duty_cycle_60m"

    return duty_cycle


def compute_fft_spectral_energy(readings: pd.DataFrame, window: int = 360) -> pd.DataFrame:
    """Compute Fast Fourier Transform (FFT) spectral energy over sliding windows.

    At 0.1 Hz sampling rate, a 360-row window corresponds to 60 minutes of data.
    FFT measures low-frequency duty cycle dynamics and pressure fluctuation energy.

    Args:
        readings: pd.DataFrame with sensor columns.
        window: Window length in rows (360 rows = 60 mins at 0.1 Hz).

    Returns:
        pd.DataFrame containing FFT spectral energy features.
    """
    fft_df = pd.DataFrame(index=readings.index)
    target_cols = [c for c in ["TP2", "TP3", "Motor_current"] if c in readings.columns]

    for col in target_cols:
        series = readings[col].values
        fft_energy = np.full(len(series), np.nan)

        # Compute rolling FFT over sliding row windows
        for i in range(window, len(series)):
            chunk = series[i - window : i]
            if not np.isnan(chunk).any():
                fft_vals = np.abs(np.fft.rfft(chunk - np.mean(chunk)))
                fft_energy[i] = np.sum(fft_vals**2) / len(chunk)

        fft_df[f"{col}_fft_energy"] = fft_energy

    return fft_df


def build_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Master feature builder function matching the team interface contract.

    Interface agreed with team:
        build_features(readings: pd.DataFrame) -> pd.DataFrame

    Args:
        readings: pd.DataFrame indexed by timestamp with a `segment` column.

    Returns:
        pd.DataFrame containing all engineered features indexed by timestamp.
    """
    # 1. Compute rolling statistics
    roll_stats = rolling_features(readings)

    # 2. Compute motor duty cycle
    duty_cycle = compute_duty_cycle(readings)

    # 3. Compute FFT spectral energy
    fft_stats = compute_fft_spectral_energy(readings)

    # Combine all feature DataFrames
    features = pd.concat([roll_stats, duty_cycle, fft_stats], axis=1)

    return features


def fit_pca_health_index(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Fit StandardScaler and PCA on TRAIN features only to produce a 1D Health Index.

    CRUCIAL LEAKAGE RULE:
    Fit scalers and PCA strictly on training fold data, then transform both train and test.

    Args:
        train_features: pd.DataFrame of features for training folds.
        test_features: pd.DataFrame of features for test fold.

    Returns:
        Tuple of (train_health_index, test_health_index) pd.Series.
    """
    clean_train = train_features.dropna()
    clean_test = test_features.fillna(0)

    scaler = StandardScaler()
    pca = PCA(n_components=1, random_state=42)

    # Fit ON TRAIN ONLY
    scaled_train = scaler.fit_transform(clean_train)
    pca_train_vals = pca.fit_transform(scaled_train).ravel()

    # Transform test features using fitted scaler and PCA
    scaled_test = scaler.transform(clean_test)
    pca_test_vals = pca.transform(scaled_test).ravel()

    train_health_index = pd.Series(pca_train_vals, index=clean_train.index, name="pca_health_index")
    test_health_index = pd.Series(pca_test_vals, index=clean_test.index, name="pca_health_index")

    return train_health_index, test_health_index