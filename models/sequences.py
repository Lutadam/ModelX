"""Cutting the per-minute table into the fixed-length windows the CNN+GRU reads.

A window is the last `sequence_minutes` minutes up to and including one minute.
Windows that would cross a gap in the data (a removed failure, maintenance or
outage) are not allowed: they would join two stretches that never ran together.
"""
import numpy as np
import pandas as pd
import torch


def find_window_end_rows(minute_index: pd.DatetimeIndex, sequence_minutes: int) -> np.ndarray:
    """Row numbers whose previous `sequence_minutes` minutes are all present.

    Args:
        minute_index: Sorted one-minute timestamps of a table (may have gaps).
        sequence_minutes: Window length in minutes.

    Returns:
        Row numbers that can end a window. The first `sequence_minutes - 1` rows
        and every row too close after a gap are not included.
    """
    if len(minute_index) < sequence_minutes:
        return np.array([], dtype=int)
    minutes_since_start = ((minute_index - minute_index[0]) / pd.Timedelta(minutes=1)).to_numpy()
    candidate_rows = np.arange(sequence_minutes - 1, len(minute_index))
    minutes_spanned = minutes_since_start[candidate_rows] - minutes_since_start[candidate_rows - (sequence_minutes - 1)]
    return candidate_rows[np.isclose(minutes_spanned, sequence_minutes - 1)]


def gather_windows(scaled_values: torch.Tensor, end_rows: torch.Tensor, sequence_minutes: int) -> torch.Tensor:
    """Build a batch of windows from the whole table, on the table's own device.

    Args:
        scaled_values: (rows, features) tensor of the scaled feature table.
        end_rows: 1-D long tensor, the row that ends each window.
        sequence_minutes: Window length in minutes.

    Returns:
        (len(end_rows), sequence_minutes, features) tensor.
    """
    offsets = torch.arange(-(sequence_minutes - 1), 1, device=scaled_values.device)
    return scaled_values[end_rows.unsqueeze(1) + offsets]
