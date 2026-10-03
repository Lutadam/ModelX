# ModelX

Early warning of compressor failures on the MetroPT-3 data.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The CSV is expected at `Desktop/metropt+3+dataset/MetroPT3(AirCompressor).csv`
(change `DATA_FILE` in `config.py`).

## Layout

| File | Purpose |
|---|---|
| `config.py` | All constants: paths, seed, failure windows, thresholds |
| `data.py` | `load_dataset()` and `make_labels()` |
| `features.py` | `build_features(readings)`, a placeholder until Shazia's version replaces it |
| `rules.py` | LPS rule and low-pressure rule |
| `models.py` | Random Forest |
| `alerts.py` | Alarm minutes to alerts, false alerts per operating hour, caught / hours of warning. Hishaam's `evaluate.py` will hold the full evaluation |
| `main.py` | Train on F1-F3, test on F4, print the event-level comparison |

## Interfaces

- `load_dataset() -> DataFrame`: DatetimeIndex, sensor columns, `segment`.
- `build_features(readings) -> DataFrame`: indexed by timestamp at 1-minute steps, feature columns only, NaN allowed in warm-up rows.
- `make_labels(timestamps) -> Series`: 1 risk, 0 normal, `<NA>` excluded.
- `LowPressureRule(budget).fit(lowest_pressure, labels, operating).predict(lowest_pressure) -> bool Series`.
- `alert_start_times(alarm)`, `count_false_alerts(alarm, labels)`, `detect_failure(alarm, failure_start) -> dict`.

## Labels

Risk is the 3 hours before a failure starts: `[start - 3h, start)`. Rows from the failure start until the end of the failure, or until maintenance finished where known (F3 until 8 Jun 16:00, F4 until 16 Jul 00:00), are excluded. Failure rows are never positive.
