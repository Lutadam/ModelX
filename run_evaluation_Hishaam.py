"""Four-fold evaluation of the team's baselines on the real data (Week 1 result).

Owner: Lauthan Muhammad Hishaam Ibn Afzal (Evaluator & Ethics), Team ModelX.

Put this file in the project root, next to main.py, then run:
    python run_evaluation_Hishaam.py            (all methods, ~3 min)
    python run_evaluation_Hishaam.py --no-rf    (skip the Random Forest, ~20 s)

It uses the team's code for the data, labels, features and baselines
(data.load_dataset, main.build_minute_table, models.*), takes the folds and
settings from config.py, and judges every method with evaluation_Hishaam.py:
the threshold of every method is set from the false-alarm budget on the
training blocks only (the Random Forest on cross-fitted scores).
Writes results/per_fold.csv, results/summary.csv, results/results.json and
figures/timeline_<method>_<failure>.png.
"""
import sys
import time

import numpy as np
import pandas as pd

import config
from data import load_dataset
from main import NON_FEATURE_COLUMNS, build_minute_table
from models import IsolationForestDetector, train_random_forest
from evaluation_Hishaam import (EvalConfig, make_folds, split, describe_folds, threshold_for_budget,
                                cross_fitted_scores, evaluate_fold, summarise, save_results, plot_fold)

np.random.seed(config.RANDOM_SEED)
settings = EvalConfig.from_project_config(config)
minutes = build_minute_table(load_dataset())
features = [c for c in minutes.columns if c not in NON_FEATURE_COLUMNS]
folds = make_folds(minutes.index.min(), minutes.index.max(), config.FAILURES)
print(describe_folds(minutes, folds, settings).to_string(), flush=True)
skip_rf = "--no-rf" in sys.argv

def forest_scores(fit, score):
    f = train_random_forest(fit[features], fit["label"])
    return f.predict_proba(score[features])[:, list(f.classes_).index(1)]

rows = []
for fold in folds:
    t = time.time()
    train, test = split(minutes, fold, settings)
    a = test["lps_alarm"].astype(bool)
    rows.append(evaluate_fold("LPS active (current practice)", fold, test, a, settings))
    plot_fold("LPS active (current practice)", fold, test, a, settings)

    tr = threshold_for_budget(-train["lowest_pressure"], train["label"], train["operating"], settings)
    sc = -test["lowest_pressure"]
    rows.append(evaluate_fold("Low pressure rule", fold, test, sc >= tr, settings, sc, tr))
    plot_fold("Low pressure rule", fold, test, sc >= tr, settings, sc, tr)

    det = IsolationForestDetector(settings.budget_per_operating_hour).fit(train[features], train["label"], train["operating"])
    tr_scores = det.anomaly_scores(train[features])
    ti = threshold_for_budget(tr_scores, train["label"], train["operating"], settings)
    si = det.anomaly_scores(test[features])
    rows.append(evaluate_fold("Isolation Forest", fold, test, si >= ti, settings, si, ti))
    plot_fold("Isolation Forest", fold, test, si >= ti, settings, si, ti)

    if not skip_rf:
        oof = cross_fitted_scores(train, folds, settings, forest_scores)
        ok = oof.notna()
        trf = threshold_for_budget(oof[ok], train.loc[ok, "label"], train.loc[ok, "operating"], settings)
        srf = pd.Series(forest_scores(train, test), index=test.index)
        rows.append(evaluate_fold("Random Forest", fold, test, srf >= trf, settings, srf, trf))
        plot_fold("Random Forest", fold, test, srf >= trf, settings, srf, trf)
    print(f"{fold.name} done in {time.time()-t:.0f}s", flush=True)

per_fold = pd.DataFrame(rows)
summary = summarise(per_fold)
save_results(per_fold, summary)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(per_fold[["method","failure","caught","warning_hours","alarm_minutes_in_window","false_alerts",
                    "false_alerts_per_100_operating_hours","within_budget","chance_caught_by_luck","pr_auc","risk_base_rate","f2"]].to_string(index=False))
    print(summary.T.to_string())
