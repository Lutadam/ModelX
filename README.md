# ModelX (RailGuard AI)

Early warning of air-compressor failures on the MetroPT-3 data (Metro do Porto). A warning means taking the train out of service at the end of its current run, so the warning horizon is fixed at 3 hours and never tuned.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The CSV is expected at `Desktop/metropt+3+dataset/MetroPT3(AirCompressor).csv` (change `DATA_FILE` in `config.py`). The data is not committed (CC BY 4.0, Davari et al. 2021, doi:10.24432/C5VW3R). Download it from UCI.

Seeds are fixed from `RANDOM_SEED` in `config.py`, so two runs print the same numbers.

The per-minute table takes a few minutes to build from the 1.5 million readings, so it is saved in `cache/` after the first run and reused. It is rebuilt automatically when the CSV or any code that shapes it changes. Deleting `cache/` is always safe.

### GPU (needed for the CNN+GRU)

`requirements.txt` installs the CUDA 12.6 build of PyTorch on Windows and Linux (needs an NVIDIA driver that supports CUDA 12.x) and the normal build on a Mac, where Apple Silicon uses its GPU through MPS. Torch 2.14.1 only has CUDA 12.6 and 13.0 builds: with a driver that supports CUDA 13, use `cu130` instead of `cu126` in `requirements.txt`.

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

`DEEP_LEARNING_USE_CUDNN = False` in `config.py`: with driver 555 (CUDA 12.5), training the GRU with cuDNN makes Python crash on exit (Windows code 0xC0000409) after the results are written. With cuDNN off the model still trains on the GPU, with PyTorch's own GPU code (slower, same model). Set it to `True` on a machine whose driver supports CUDA 12.6 or newer.

`python main.py` checks for a GPU first and stops with a clear message if there is none. To run on the CPU anyway (slow), set `REQUIRE_GPU = False` in `config.py`.

## What `main.py` does

1. Checks the GPU.
2. Loads the per-minute table of the real MetroPT-3 data (labels, rule inputs, features), from the cache when possible.
3. Works out the operating hours and the false-alarm budget.
4. Makes four leave-one-failure-out folds and checks the purge gap.
5. Judges every method the same way on every fold: LPS rule, low-pressure rule, Random Forest, Isolation Forest, CNN+GRU. Writes `results/per_fold.csv`, `results/summary.csv`, `results/results.json` and `figures/timeline_<method>_<failure>.png`. Numbers in the report are copied from `results/`, never retyped from the console.

## Layout

| File | Owner | Purpose |
|---|---|---|
| `main.py` | Atul | Entry point: wires every step below together |
| `config.py` | Atul | Every setting shared by more than one file or that changes the results: paths, seed, failures, budget, alert rules, threshold search, purge gap, figure window, GPU and deep learning settings. Values used in one place only for display or speed (colours, figure sizes, scoring batch) stay in their own file |
| `data.py` | Shazia | `load_dataset()`, `make_labels()`, operating hours and budget |
| `features.py` | Shazia | `build_features()`: rolling, duty-cycle, frequency and motor-run features; PCA health index |
| `minute_table.py` | Atul | `load_minute_table()` / `build_minute_table()`: labels, rule inputs and features on one per-minute table, cached |
| `rules.py` | Atul | LPS rule and low-pressure rule |
| `alerts.py` | Atul | Operating minutes, alarm minutes to alerts (same rule as `evaluation.py`), caught / warning hours, `judge_alarms` for one test block |
| `models/random_forest.py`, `models/isolation_forest.py` | Atul | The two baselines |
| `models/deep_learning_model.py` | Hitesh | CNN+GRU training and scoring (`make_fit_and_score`) |
| `models/cnn_gru_network.py`, `models/sequences.py`, `models/compute.py` | Hitesh | The network, the window cutting, GPU choice and seeds |
| `folds.py` | Hitesh | Four folds and the purge-gap check (condition 2, refinement 3) |
| `evaluation.py` | Hishaam | Folds, alerts, budget thresholds, metrics, summary, results files, timeline figures |
| `run_folds.py` | Atul | Runs every method on every fold through `evaluation.py` |
| `console.py` | Atul | Terminal output of `main.py` |
| `horizon_figure.py` | Hishaam | Standalone, not part of `main.py`: `python horizon_figure.py`. Sensors in the 48 h before each failure, checks the 3 h horizon (refinement 1) |

Tests, all on the real data. Where a check needs the right answer, it is worked out a second, simpler way (a plain loop) on the same real data and the two must agree:

| Test | Run |
|---|---|
| `tests/test_shazia.py` | `python tests/test_shazia.py` |
| `tests/test_random_forest.py`, `tests/test_isolation_forest.py` (helpers in `tests/_atul_common.py`) | `python tests/test_random_forest.py` |
| `tests/test_Hishaam.py` | `python tests/test_Hishaam.py` |
| `tests/test_Hitesh.py` | `python tests/test_Hitesh.py` |

On Windows, run with `PYTHONUTF8=1` if a test crashes on the tick characters.

## Interfaces

- `load_dataset() -> DataFrame`: DatetimeIndex, sensor columns, `segment` (new segment after every gap over 5 minutes).
- `build_features(readings) -> DataFrame`: indexed by timestamp at 1-minute steps, feature columns only, NaN in warm-up rows and minutes with no data.
- `make_labels(timestamps) -> Series`: 1.0 risk, 0.0 normal, NaN excluded. Takes timestamps, not the readings table.
- `load_minute_table() -> DataFrame`: `label`, `lps_alarm`, `lowest_pressure`, `operating`, then the features; no NaN. Cached version of `build_minute_table(load_dataset())`.
- `operating_hours_and_budget(operating_per_minute) -> dict`: operating hours and the most false alerts the budget allows.
- A **new scored model** provides `fit_and_score(fit_rows, score_rows) -> one risk score per row of score_rows` (higher = riskier). It trains a FRESH model on `fit_rows` each call. Add it to `build_scored_methods` in `run_folds.py`. The evaluation sets its threshold from the budget.
- `LowPressureRule(budget).fit(lowest_pressure, labels, operating).predict(lowest_pressure) -> bool Series`.
- `IsolationForestDetector(budget).fit(...)` trains and sets its own cut-off; `.fit_forest(features, labels)` only trains (the four-fold run sets the cut-off from out-of-sample scores).
- `predict_random_forest_alarms(forest, features, threshold) -> bool Series`: the threshold comes from the budget, never a fixed 0.5.
- `fit_pca_health_index(train_features, test_features) -> (Series, Series)`: fitted on training rows only, NaN rows dropped.

## Labels

Risk is the 3 hours before a failure starts: `[start - 3h, start)`. Rows from the failure start until the end of the failure, or until maintenance finished where known (F3 until 8 Jun 16:00, F4 until 16 Jul 00:00), are excluded. Failure rows are never positive and are never counted as false alerts.

## How results are judged

- **Folds:** each of the four failures is held out in turn. Training rows within `PURGE_GAP` (24 h) of the test block are dropped. The gap must be at least the CNN+GRU window (120 min) plus the longest rolling feature (60 min); `folds.py` checks this.
- **Operating hour:** one definition everywhere: a minute whose mean motor current is above `OPERATING_MOTOR_CURRENT_AMPS` (1.0 A, placeholder owned by Shazia). The motor is on about 9 h a day, so the budget is about 1 false alert per 11 days.
- **Budget:** at most 1 false alert per train per 100 operating hours (`FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET`).
- **Thresholds** are the most sensitive ones that stay within the budget on the normal training minutes only. Learned models use out-of-sample (cross-fitted) training scores, because a model scored on its own training data looks perfect and gets a threshold that is too low.
- **Alert:** one definition everywhere (`evaluation.alert_starts`): alarm minutes less than 3 hours apart (`ALERT_COOLDOWN`) are one alert, and an alarm still on 24 hours after its alert began counts as a new alert. A false alert is one that starts on a normal minute.
- **Caught:** at least one alarm inside the 3-hour risk window. Warning hours are measured from the first such alarm (not reported for F1, whose start time is a midnight placeholder).
- **Reported per method:** failures caught out of 4 and out of the 3 with a known onset (F1 excluded), false alerts per 100 operating hours (pooled over the folds), PR-AUC and F2 per fold, the time-ordered F4 fold on its own, and the chance of catching that many failures by luck. Accuracy is not reported: always answering "normal" scores about 99.8%.

## Findings so far (from the data, before any tuning)

- **F1 cannot be caught:** its start is a midnight placeholder and the compressor was off for the whole 3 hours before it. It is reported separately ("caught, known onset").
- **Only F4 shows a warning sign** in the run length and TP3 pressure (`python horizon_figure.py`): the compressor ran 14.6 h without a stop before F4 (normal maximum 1.8 h), starting well before the 3 h window. F2 and F3 show nothing before the window in these two signals.
- **Low pressure happens in normal running.** There are 33 episodes below 3 bar. In every one both pressure sensors agree, and in 31 the LPS switch trips too, so they are real pressure losses, not sensor faults, and they stay in the data. They are why the low-pressure rule's budget cut-off drops to about 0.9 bar in the F4 fold.
- **Thresholds that meet the budget on the training folds often go over it on the test folds.** This is a result to report, not something to tune away.

## Limitations to state in the report

- Folds 1 to 3 train on failures that happened later. This assumes each air leak is a separate, repaired event.
- Four results are a small set, not a statistically powered estimate.
- The F3 test block has only about 6 days of normal running, so its false-alarm rate is uncertain. This is why rates are pooled.
- F1's start time is a placeholder.
- The `motor_run_minutes` feature was added after the horizon figure showed F4's long run, so any F4 catch that relies on it is not an independent test.
- The CNN+GRU is untuned (3 epochs, risk weight 15, from Hitesh's notebook). Tuning it properly needs a validation step inside the training folds; with only 2 or 3 training failures per fold that means nested leave-one-failure-out runs, several times the current training time.

## Team status (checked against the code and the tests)

| Who | Done | Open |
|---|---|---|
| Shazia | `data.py`, `features.py`, labels, budget (one operating-hour definition), motor-run feature, PCA health index, her test | Operating threshold (1.0 A) is still a placeholder. The PCA health index is not used by any method: the sign of the first component is arbitrary, so it needs a decision on which direction means risk |
| Atul | `config.py`, rules, alerts (now the same alert rule as the evaluation), Random Forest (budget threshold, no fixed 0.5), Isolation Forest, their tests | |
| Hishaam | `evaluation.py` (now also counts catches with a known onset) and its tests, `horizon_figure.py` | |
| Hitesh | CNN+GRU design and notebook | `folds.py` and `models/` were written from his notebook with Claude, not by Hitesh. He needs to review them, and to decide whether the model should be tuned (see Limitations) |
| Whole team | | Commit `results/` and `figures/` once the numbers for the report are final. Both are in `.gitignore` (every run recreates them), so add them with `git add -f results/ figures/` |

The proposal feedback text was not available when this table was written, so conditions are checked only against what the code says: condition 2 (folds) and refinement 3 (purge check) are implemented, condition 3 and refinement 1 (horizon) have `horizon_figure.py`, condition 4 (budget) is in the thresholds, refinement 2 (operating hours) rests on the 1.0 A placeholder.
