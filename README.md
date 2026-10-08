# ModelX (RailGuard AI)

Early warning of air-compressor failures on the MetroPT-3 data (Metro do Porto). A warning means taking the train out of service at the end of its current run, so the warning horizon is fixed at 3 hours and never tuned.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The CSV is expected at `Desktop/metropt+3+dataset/MetroPT3(AirCompressor).csv`
(change `DATA_FILE` in `config.py`). The data is not committed (CC BY 4.0, Davari et al. 2021, doi:10.24432/C5VW3R). Download it from UCI.

Seeds are fixed from `RANDOM_SEED` in `config.py`, so two runs print the same numbers.

## Layout

| File | Owner | Purpose |
|---|---|---|
| `config.py` | Atul | All constants: paths, seed, failure windows, budget, operating-hour threshold |
| `main.py` | Atul | Train on F1-F3, test on F4, print the event-level comparison |
| `rules.py` | Atul | LPS rule and low-pressure rule |
| `models/` | Atul, Hitesh | One module per model: `random_forest.py` and `isolation_forest.py` (Atul). Hitesh adds the deep learning module here |
| `alerts.py` | Atul | Alarm minutes to alerts, false alerts per operating hour, caught / hours of warning |
| `data.py` | Shazia | `load_dataset()`, `make_labels()`, operating-hour budget summary |
| `features.py` | Shazia | `build_features()`, rolling, duty-cycle and frequency features, PCA health index |
| `tests/test_shazia.py` | Shazia | Runs her data and feature code end to end: `python tests/test_shazia.py` |
| `tests/test_random_forest.py`, `tests/test_isolation_forest.py` | Atul | One script per model: alert and rule checks, leakage checks, the model's own checks, and its F4 result next to the rules. `tests/_atul_common.py` holds the shared helpers |

Still to come: `folds.py` (Hitesh, four purged leave-one-event-out folds), `evaluate.py` (Hishaam, full event-level metrics, PR-AUC, F2) and the deep learning model (Hitesh). `alerts.py` holds only what the baselines need.

## Interfaces

- `load_dataset() -> DataFrame`: DatetimeIndex, sensor columns, `segment` (new segment after every gap over 5 minutes).
- `build_features(readings) -> DataFrame`: indexed by timestamp at 1-minute steps, feature columns only, NaN in warm-up rows and minutes with no data.
- `make_labels(timestamps) -> Series`: 1.0 risk, 0.0 normal, NaN excluded. Takes timestamps, not the readings table.
- `fit_pca_health_index(train_features, test_features) -> (Series, Series)`: fitted on training rows only, NaN rows dropped.
- `LowPressureRule(budget).fit(lowest_pressure, labels, operating).predict(lowest_pressure) -> bool Series`.
- `train_random_forest(features, labels)` and `predict_random_forest_alarms(forest, features) -> bool Series` (from `models`).
- `IsolationForestDetector(budget).fit(features, labels, operating).predict(features) -> bool Series`: trained on normal training minutes only, score cut-off set on the training folds to meet the budget.
- A new model in `models/` should return a boolean alarm series indexed by minute, so `judge_alarms` in `main.py` can score it.
- `alert_start_times(alarm)`, `count_false_alerts(alarm, labels)`, `detect_failure(alarm, failure_start) -> dict`.

## Labels

Risk is the 3 hours before a failure starts: `[start - 3h, start)`. Rows from the failure start until the end of the failure, or until maintenance finished where known (F3 until 8 Jun 16:00, F4 until 16 Jul 00:00), are excluded. Failure rows are never positive and are never counted as false alerts.

## How results are judged

- **Operating hour:** a minute where the motor current is above `OPERATING_MOTOR_CURRENT_AMPS` (1.0 A, placeholder owned by Shazia).
- **Budget:** at most 1 false alert per train per 100 operating hours (`FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET`).
- **Alert:** alarm minutes less than 3 hours apart (`ALERT_COOLDOWN`, an assumption) count as one alert. A false alert is one that starts on a normal minute.
- **Caught:** at least one alarm inside the 3-hour risk window. Warning hours are measured from the first such alarm.
- **Thresholds** are set on the training folds only. The low-pressure cut-off is the highest value whose false alerts stay within the budget.
- **Time-respecting fold:** train on everything before 8 Jun 16:00 (F1-F3), test on everything after (F4). The cut is `TEST_START` in `config.py`.

## Current status

First result on F4 (re-run `python main.py` for the current numbers): none of the LPS rule, the low-pressure rule, the Random Forest or the Isolation Forest caught the failure. In the 3 hours before F4 the lowest pressure stays near 8 bar, the same level as normal running, so no pressure cut-off that meets the budget fires. The Random Forest uses a fixed 0.5 probability cut-off that is not yet calibrated to the budget.

Not yet in `main.py`: the PCA health index, the budget summary from `data.py`, all four folds, the deep learning model.
