# Evaluation module: Hishaam (Evaluator & Ethics)

Team ModelX, RailGuard AI. Owner: Lauthan Muhammad Hishaam Ibn Afzal.

This module judges every method (rules, Random Forest, Isolation Forest, CNN+GRU) the same way, on all four MetroPT-3 failures. It is the `evaluate.py` listed as "still to come" in the team README (github.com/Lutadam/ModelX). It is **standalone**: it imports nothing from the rest of the project, so it can be plugged in once everyone's code is merged.

## Files

| File | Purpose |
|---|---|
| `evaluation_Hishaam.py` | The module: folds, budget thresholds, metrics, summary, results files, figures |
| `demo_evaluation_Hishaam.py` | Runs the full evaluation on **fake** data to show it works (its numbers mean nothing) |
| `requirements.txt` | Identical to the team's pinned `requirements.txt` |
| `README_Hishaam.md` | This file |

## Try it on its own

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo_evaluation_Hishaam.py
```

This prints the fold table and the results, and writes `results/` and `figures/`.

## What it does

1. **Leave-one-failure-out folds** (feedback condition 2). Each of the four failures is held out in turn, so there are four test results instead of one. Training rows within 24 h of the test block are dropped (purge gap) so near-identical neighbouring readings don't leak into training. The F4 fold is the same as the original proposal split.
2. **Thresholds from the false-alarm budget** (condition 4). Every method gets the most sensitive threshold that keeps false alerts at or below 1 per 100 operating hours on the **normal training minutes only**. No failure is spent on tuning. Learned models are thresholded on cross-fitted (out-of-sample) scores, because a model scored on its own training data looks perfect and gets a threshold that is too low.
3. **Metrics per fold:**
   - Event level (main): caught or not, hours of warning, false alerts per 100 operating hours and per week, within budget, chance of catching the failure by luck.
   - Window level (secondary): PR-AUC (compare with the risk base rate), F2, precision, recall.
   - Accuracy is deliberately not reported: always answering "normal" scores about 99.8%.
4. **Summary across folds:** one row per method, with failures caught out of 4, pooled false-alert rates, and the probability of that many catches by luck.
5. **Outputs:** `results/per_fold.csv`, `results/summary.csv`, `results/results.json`, and `figures/timeline_<method>_<failure>.png` (the 48 h before each failure: score, threshold, alarms), used for error analysis.

## What the rest of the project must hand over

### 1. A per-minute table

A pandas DataFrame **indexed by timestamp, one row per minute**, with at least:

| Column | Meaning | From |
|---|---|---|
| `label` | 1 = risk (the 3 h before a failure), 0 = normal. Failure and maintenance minutes **removed** (no NaN left) | `data.make_labels` (Shazia) |
| `operating` | 1 if the compressor ran in that minute, else 0 | `alerts.operating_per_minute` (motor current > 1.0 A) |
| `lps_alarm`, `lowest_pressure` | Inputs of the two rules | `rules.py` (Atul) |
| feature columns | anything the models need | `features.build_features` (Shazia) |

The team's `build_minute_table()` in `main.py` already produces exactly this, after `dropna()`. The feature columns are everything not in `main.NON_FEATURE_COLUMNS`.

### 2. The failures

Pass `config.FAILURES`. `Failure` here has the same fields as `config.Failure` (`name`, `start`, `end`, `excluded_until`), and `METROPT_FAILURES` is a copy of `config.FAILURES` used by the demo. Correct dates in `config.py` only.

### 3. From each method, per fold

- A **risk score per test minute** (higher = riskier) and a way to compute it on training rows, **or**
- For a fixed rule with no score (e.g. LPS active), a **True/False alarm per test minute**.

For learned models, wrap training + scoring in one function:

```python
def fit_and_score(fit_rows, score_rows):
    model = ...                     # a FRESH model each call
    model.fit(fit_rows[features], fit_rows["label"])
    return model.predict_proba(score_rows[features])[:, 1]
```

## How to plug it into the project

Copy `evaluation_Hishaam.py` into the project root as `evaluate.py` (next to `main.py`), then add a script such as `run_folds.py`. It uses the team's own rules and models, unchanged:

```python
import pandas as pd

import config
from data import load_dataset
from main import NON_FEATURE_COLUMNS, build_minute_table
from models import IsolationForestDetector, train_random_forest
from rules import LowPressureRule
from evaluate import (
    EvalConfig, cross_fitted_scores, describe_folds, evaluate_fold,
    make_folds, plot_fold, save_results, split, summarise, threshold_for_budget,
)

settings = EvalConfig.from_project_config(config)    # horizon, cooldown, budget from config.py
minutes = build_minute_table(load_dataset())
features = [c for c in minutes.columns if c not in NON_FEATURE_COLUMNS]
folds = make_folds(minutes.index.min(), minutes.index.max(), config.FAILURES)
print(describe_folds(minutes, folds, settings))


def forest_fit_and_score(fit_rows, score_rows):
    forest = train_random_forest(fit_rows[features], fit_rows["label"])   # a FRESH model each call
    return forest.predict_proba(score_rows[features])[:, list(forest.classes_).index(1)]


rows = []
for fold in folds:
    train, test = split(minutes, fold, settings)

    # (a) LPS active: a fixed alarm, no score
    rows.append(evaluate_fold("LPS active (current practice)", fold, test,
                              test["lps_alarm"].astype(bool), settings))

    # (b) Low-pressure rule: Atul's class already sets its cut-off from the budget on train
    rule = LowPressureRule(settings.budget_per_operating_hour).fit(
        train["lowest_pressure"], train["label"], train["operating"])
    alarm = rule.predict(test["lowest_pressure"])
    rows.append(evaluate_fold("Low pressure (TP3/Reservoirs)", fold, test, alarm, settings,
                              -test["lowest_pressure"], -rule.pressure_cutoff))

    # Random Forest: budget threshold on cross-fitted training scores, NOT the fixed 0.5
    oof = cross_fitted_scores(train, folds, settings, forest_fit_and_score)
    ok = oof.notna()
    t = threshold_for_budget(oof[ok], train.loc[ok, "label"], train.loc[ok, "operating"], settings)
    scores = pd.Series(forest_fit_and_score(train, test), index=test.index)
    rows.append(evaluate_fold("Random Forest", fold, test, scores >= t, settings, scores, t))
    plot_fold("Random Forest", fold, test, scores >= t, settings, scores, t, folder=config.FIGURES_DIR)

    # Isolation Forest: Atul's detector sets its own budget cut-off on train
    detector = IsolationForestDetector(settings.budget_per_operating_hour).fit(
        train[features], train["label"], train["operating"])
    scores = detector.anomaly_scores(test[features])
    rows.append(evaluate_fold("Isolation Forest", fold, test, detector.predict(test[features]),
                              settings, scores, detector.score_cutoff))

per_fold = pd.DataFrame(rows)
summary = summarise(per_fold)
save_results(per_fold, summary)
print(summary.to_string())
```

The "hold out F4" fold starts at `config.TEST_START` (8 Jun 16:00), so it tests the same minutes as `main.py`. The only difference is that its training data stops 24 h earlier (the purge gap).

### Per teammate

- **Shazia:** keep `make_labels` as 1 / 0 / NaN and the operating-minute rule (`OPERATING_MOTOR_CURRENT_AMPS`, still a placeholder). NaN rows must be dropped before evaluation, which `build_minute_table` already does.
- **Mundram (Atul), rules and Random Forest / Isolation Forest:**
  - `LowPressureRule` and `IsolationForestDetector` plug in unchanged (see above).
  - The Random Forest should use the budget threshold instead of `RANDOM_FOREST_ALARM_PROBABILITY = 0.5`, which the team README says is not yet calibrated.
  - `IsolationForestDetector.fit` sets its cut-off on scores of the same normal minutes it was trained on (in-sample), so the cut-off may be slightly too low. Cross-fitted scores would fix this.
  - `alerts.py` and `evaluate.py` define alerts the same way (cooldown, false alert = alert starting on a normal minute). The one difference is that `detect_failure` reports 0.0 warning hours when a failure is missed, while this module reports blank.
- **Hitesh, CNN+GRU:**
  - Provide a `fit_and_score(fit_rows, score_rows)` that returns one risk score per minute.
  - If his sequences look back more than ~12 h, raise `purge_gap` in `EvalConfig`.
  - The team README gives him `folds.py`. `make_folds` / `split` here already do this (hold out each failure, 24 h purge). Use one or the other, not both, or the results won't be comparable.
  - The team README asks new models for a boolean alarm per minute. `evaluate_fold` accepts that, but a score per minute is also needed for PR-AUC and the budget threshold.

## Settings (`EvalConfig`)

| Setting | Default | `config.py` name | Why |
|---|---|---|---|
| `warning_horizon` | 3 h | `WARNING_HORIZON` | Risk window; must equal the labels' horizon |
| `alert_cooldown` | 3 h | `ALERT_COOLDOWN` | Alarm minutes closer than this are one alert (one inspection) |
| `purge_gap` | 24 h | (none yet) | Longer than any feature (60 min rolling) or sequence look-back |
| `budget_per_operating_hour` | 1/100 | `FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET` | Condition 4: 1 false alert per 100 operating hours |
| `f_beta` | 2 | (none) | A missed failure costs more than a false alarm |
| `unknown_onset` | {"F1"} | (none) | F1 is logged as a whole day, so its warning hours aren't reported |

In the project, always build the settings with `EvalConfig.from_project_config(config)`, so the shared values come from `config.py` and can't drift. Other values can be overridden by team agreement, e.g. `EvalConfig.from_project_config(config, purge_gap=pd.Timedelta(hours=48))`.

## Limitations to state in the report

- Folds 1 to 3 train on failures that happened **later**. This assumes each air leak is a separate, repaired event.
- Four results are a small set, not a statistically powered estimate.
- The F3 test block holds only about 6 days of normal running, so its false-alarm rate is uncertain. This is why rates are pooled across folds.
- F1's start time is a midnight placeholder.

## Rule for the report

Every number in Sections 5 and 6 is copied from `results/`, never retyped from the console.
