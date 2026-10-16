"""The per-minute table every method reads: labels, rule inputs and features on one grid.

Building it from the 1.5 million raw readings takes minutes, so `load_minute_table`
keeps a copy in config.CACHE_DIR. The copy is rebuilt automatically when the CSV
or any code that shapes the table (listed in TABLE_SOURCE_FILES) changes.
"""
import hashlib
from pathlib import Path

import pandas as pd

import config
from data.dataset import load_dataset, make_labels, operating_per_minute
from data.features import build_features
from models.rules import lowest_pressure_per_minute, lps_alarm_per_minute

NON_FEATURE_COLUMNS = ["label", "lps_alarm", "lowest_pressure", "operating"]
TABLE_SOURCE_FILES = ["config.py", "data/dataset.py", "data/features.py", "models/rules.py", "evaluation/alerts.py",
                      "data/minute_table.py"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_minute_table(readings: pd.DataFrame) -> pd.DataFrame:
    """Put labels, rule inputs and features on one per-minute table.

    Args:
        readings: Output of `load_dataset`.

    Returns:
        One row per minute that has a label, complete features and sensor
        data. Columns: `label`, `lps_alarm`, `lowest_pressure`, `operating`,
        then the features.
    """
    features = build_features(readings)
    minutes = features.index
    table = pd.DataFrame(
        {
            "label": make_labels(minutes),
            "lps_alarm": lps_alarm_per_minute(readings).reindex(minutes).astype(float),
            "lowest_pressure": lowest_pressure_per_minute(readings).reindex(minutes),
            "operating": operating_per_minute(readings).reindex(minutes),
        },
        index=minutes,
    ).join(features)
    return table.dropna()


def cache_key(csv_path: Path) -> str:
    """Changes whenever the CSV or the code that builds the table changes."""
    hasher = hashlib.sha256()
    csv_stat = csv_path.stat()
    hasher.update(f"{csv_stat.st_size}-{csv_stat.st_mtime_ns}".encode())
    for file_name in TABLE_SOURCE_FILES:
        hasher.update((PROJECT_ROOT / file_name).read_bytes())
    return hasher.hexdigest()[:16]


def load_minute_table(csv_path: Path = config.DATA_FILE, verbose: bool = False) -> pd.DataFrame:
    """The per-minute table: from the cache when nothing changed, otherwise built and cached.

    Args:
        csv_path: The MetroPT-3 CSV.
        verbose: Say whether the cache was used.

    Returns:
        Same as `build_minute_table(load_dataset(csv_path))`.
    """
    cache_dir = PROJECT_ROOT / config.CACHE_DIR
    cache_file = cache_dir / f"minute_table_{cache_key(csv_path)}.pkl"
    if cache_file.exists():
        try:
            minute_table = pd.read_pickle(cache_file)
            if verbose:
                print(f"  Per-minute table loaded from the cache ({cache_file.name})")
            return minute_table
        except Exception as error:  # damaged file or saved by another pandas version: rebuilding is always safe
            print(f"  WARNING: the cached table could not be read ({type(error).__name__}: {error}). Rebuilding it.")

    if verbose:
        print("  Building the per-minute table from the raw readings (cached for next time)...")
    minute_table = build_minute_table(load_dataset(csv_path))
    try:
        cache_dir.mkdir(exist_ok=True)
        for old_cache_file in cache_dir.glob("minute_table_*.pkl"):
            old_cache_file.unlink()
        minute_table.to_pickle(cache_file)
    except OSError as error:  # the table is still correct, only the next run is slower
        print(f"  WARNING: could not save the cache ({error}). The next run will rebuild the table.")
    return minute_table


def get_feature_names(minute_table: pd.DataFrame) -> list:
    """The feature columns: everything that is not a label, a rule input or the operating flag."""
    return [column for column in minute_table.columns if column not in NON_FEATURE_COLUMNS]
