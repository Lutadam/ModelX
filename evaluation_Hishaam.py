"""RailGuard AI: evaluation module (standalone).

Owner: Lauthan Muhammad Hishaam Ibn Afzal (Evaluator & Ethics), Team ModelX.

This is the `evaluate.py` listed as "still to come" in the team README.
Self-contained on purpose: it imports nothing from the rest of the project,
so it can be written and tested now and plugged in once everyone's code is
merged. Everything it needs comes in as arguments. Once merged, build the
settings from the team's config.py so the two never disagree:

    import config
    settings = EvalConfig.from_project_config(config)
    folds = make_folds(start, end, config.FAILURES)

WHAT THE REST OF THE PROJECT MUST HAND OVER (the "contract")
------------------------------------------------------------
1. A per-minute table `minutes` (pandas DataFrame) indexed by timestamp, with:
     label      1 = risk (the N hours before a failure), 0 = normal.
                Failure / maintenance minutes must already be removed.
     operating  1 if the compressor ran in that minute, else 0.
   main.build_minute_table() gives exactly this (plus lps_alarm,
   lowest_pressure and the features).
2. The list of failures: config.FAILURES, or METROPT_FAILURES below (a copy).
3. For each method and fold: a risk score per test minute (higher = riskier),
   or, for a fixed rule, a True/False alarm per test minute.

WHAT IT GIVES BACK
------------------
* Leave-one-failure-out folds with a purge gap      -> make_folds, split
* Threshold from the false-alarm budget             -> threshold_for_budget
* Out-of-sample training scores for that threshold  -> cross_fitted_scores
* Per-fold metrics (event level + window level)     -> evaluate_fold
* One summary row per method across folds           -> summarise
* Results files and timeline figures                -> save_results, plot_fold
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")  # write figures to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score

MINUTES_PER_WEEK = 60 * 24 * 7


# ============================================================ settings

@dataclass
class EvalConfig:
    """Every evaluation setting in one place. Change values here, not in code.

    warning_horizon: length of the risk window before a failure (N hours).
    alert_cooldown: alarm minutes closer than this form ONE alert (one inspection).
    purge_gap: training rows this close to the test block are dropped (leakage).
    budget_per_operating_hour: most false alerts allowed per operating hour
        (feedback condition 4; 1/100 = one per 100 operating hours).
    f_beta: beta of the F-score; 2 weights recall above precision.
    unknown_onset: failures whose start time is a placeholder, so their
        hours of warning are not reported (F1 is logged as a whole day).

    The first, second and fourth defaults equal config.WARNING_HORIZON,
    config.ALERT_COOLDOWN and config.FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET.
    """

    warning_horizon: pd.Timedelta = pd.Timedelta(hours=3)
    alert_cooldown: pd.Timedelta = pd.Timedelta(hours=3)
    purge_gap: pd.Timedelta = pd.Timedelta(hours=24)
    budget_per_operating_hour: float = 1 / 100
    f_beta: float = 2.0
    unknown_onset: set = field(default_factory=lambda: {"F1"})

    @classmethod
    def from_project_config(cls, project_config, **overrides) -> "EvalConfig":
        """Settings taken from the team's config.py module (passed in, not imported).

        Args:
            project_config: The imported `config` module.
            **overrides: Any field to set differently, e.g. purge_gap.

        Returns:
            An EvalConfig whose shared values match config.py.
        """
        shared = {
            "warning_horizon": project_config.WARNING_HORIZON,
            "alert_cooldown": project_config.ALERT_COOLDOWN,
            "budget_per_operating_hour": project_config.FALSE_ALERTS_PER_OPERATING_HOUR_BUDGET,
        }
        return cls(**{**shared, **overrides})


class Failure(NamedTuple):
    """One failure, same fields as config.Failure, so config.FAILURES works here too.

    Rows from `start` up to and including `excluded_until` are dropped.
    `excluded_until` = end of maintenance where known, otherwise the failure end.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    excluded_until: pd.Timestamp


# Copy of config.FAILURES (renumbered: the UCI table lists #1 twice).
# If the team corrects a date, correct it in config.py and pass config.FAILURES.
METROPT_FAILURES = [
    Failure("F1", pd.Timestamp("2020-04-18 00:00"), pd.Timestamp("2020-04-18 23:59"),
            pd.Timestamp("2020-04-18 23:59")),
    Failure("F2", pd.Timestamp("2020-05-29 23:30"), pd.Timestamp("2020-05-30 06:00"),
            pd.Timestamp("2020-05-30 06:00")),
    Failure("F3", pd.Timestamp("2020-06-05 10:00"), pd.Timestamp("2020-06-07 14:30"),
            pd.Timestamp("2020-06-08 16:00")),  # maintenance finished 8 Jun 16:00
    Failure("F4", pd.Timestamp("2020-07-15 14:30"), pd.Timestamp("2020-07-15 19:00"),
            pd.Timestamp("2020-07-16 00:00")),  # maintenance finished 16 Jul 00:00
]


# ============================================================ folds

class Fold(NamedTuple):
    """Test on [test_start, test_end]; train on everything outside it +/- purge gap."""

    name: str
    held_out: Failure
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(data_start: pd.Timestamp, data_end: pd.Timestamp, failures: list) -> list:
    """Leave-one-failure-out: one fold per failure, in time order.

    The timeline is cut into one block per failure: from the end of the
    previous maintenance to the end of this failure's maintenance. The F4
    block starts at config.TEST_START, so it is main.py's original test fold. The last
    block also keeps the normal running after the last failure, so those
    weeks are tested for false alarms too.

        start ---- F1 ][---- F2 ][-- F3 ][---- F4 ---- end
         block 1        block 2  block 3     block 4

    Costs to state in the report: folds 1-3 train on LATER failures (assumes
    each leak is a separate, repaired event); four trainings instead of one;
    four results are a small set, not a statistically powered estimate.
    """
    failures = sorted(failures, key=lambda f: f.start)
    folds, block_start = [], data_start
    for position, failure in enumerate(failures):
        last = position == len(failures) - 1
        block_end = data_end if last else failure.excluded_until
        folds.append(Fold(f"hold out {failure.name}", failure, block_start, block_end))
        block_start = failure.excluded_until + pd.Timedelta(seconds=1)
    return folds


def masks(index: pd.DatetimeIndex, fold: Fold, config: EvalConfig) -> tuple:
    """(train_mask, test_mask) as boolean arrays. Rows in neither are the purge gap."""
    test = (index >= fold.test_start) & (index <= fold.test_end)
    purged_zone = (index >= fold.test_start - config.purge_gap) & (index <= fold.test_end + config.purge_gap)
    return ~purged_zone, test


def split(minutes: pd.DataFrame, fold: Fold, config: EvalConfig) -> tuple:
    """(train, test) rows of `minutes` for one fold."""
    train_mask, test_mask = masks(minutes.index, fold, config)
    return minutes[train_mask], minutes[test_mask]


def describe_folds(minutes: pd.DataFrame, folds: list, config: EvalConfig) -> pd.DataFrame:
    """Row counts per fold (the brief asks for counts to be printed)."""
    rows = []
    for fold in folds:
        train_mask, test_mask = masks(minutes.index, fold, config)
        rows.append({
            "fold": fold.name,
            "test block": f"{fold.test_start:%d %b %H:%M} to {fold.test_end:%d %b %H:%M}",
            "train minutes": int(train_mask.sum()),
            "train risk minutes": int((minutes.loc[train_mask, "label"] == 1).sum()),
            "test minutes": int(test_mask.sum()),
            "test risk minutes": int((minutes.loc[test_mask, "label"] == 1).sum()),
            "test normal days": round(int((minutes.loc[test_mask, "label"] == 0).sum()) / 1440, 1),
            "purged minutes": int((~train_mask & ~test_mask).sum()),
        })
    return pd.DataFrame(rows).set_index("fold")


# ============================================================ alerts

def alert_starts(alarm: pd.Series, config: EvalConfig) -> pd.DatetimeIndex:
    """Start time of each alert. A new alert begins after a quiet spell > cooldown."""
    times = alarm.index[alarm.to_numpy(dtype=bool)]
    if len(times) == 0:
        return times
    new_alert = np.concatenate([[True], np.diff(times.to_numpy()) > config.alert_cooldown.to_timedelta64()])
    return times[new_alert]


def count_false_alerts(alarm: pd.Series, labels: pd.Series, config: EvalConfig) -> int:
    """Alerts that START on a normal minute (an alert starting in a risk window is a hit)."""
    return int((labels.reindex(alert_starts(alarm, config)) == 0).sum())


def normal_operating_hours(labels: pd.Series, operating: pd.Series) -> float:
    """Operating hours inside the normal minutes (one row = one minute)."""
    return float(operating[(labels == 0).to_numpy()].sum()) / 60


def false_alert_rate(alarm: pd.Series, labels: pd.Series, operating: pd.Series, config: EvalConfig) -> float:
    """False alerts per operating hour of normal running."""
    hours = normal_operating_hours(labels, operating)
    return count_false_alerts(alarm, labels, config) / hours if hours else float("nan")


# ============================================================ thresholds

def threshold_for_budget(scores: pd.Series, labels: pd.Series, operating: pd.Series, config: EvalConfig) -> float:
    """Most sensitive threshold (alarm = score >= t) that stays within the budget.

    Uses NORMAL training minutes only, so no failure is spent on tuning
    (feedback condition 4). Returns np.inf (never alarm) if nothing fits.

    The scores MUST be out-of-sample for these rows: a model scored on its own
    training data looks perfect there and gets a threshold that is too low.
    For learned models use cross_fitted_scores().
    """
    normal = (labels == 0).to_numpy()
    if normal_operating_hours(labels, operating) == 0:
        raise ValueError("No normal operating hours to set a threshold on.")
    candidates = np.unique(scores[normal].quantile(np.linspace(0.5, 1.0, 501)))[::-1]  # strictest first
    threshold = np.inf
    for candidate in candidates:
        if false_alert_rate(scores >= candidate, labels, operating, config) > config.budget_per_operating_hour:
            break
        threshold = float(candidate)
    return threshold


def cross_fitted_scores(train: pd.DataFrame, folds: list, config: EvalConfig,
                        fit_and_score: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]) -> pd.Series:
    """Out-of-sample scores for every training row (inner loop of nested CV).

    Each failure block inside `train` is scored by a model fitted on the other
    blocks, with the same purge gap. `fit_and_score(fit_rows, score_rows)`
    must train a FRESH model and return one score per row of score_rows.
    Works for any model: Random Forest, CNN+GRU, ...
    """
    scores = pd.Series(np.nan, index=train.index)
    for fold in folds:
        fit_mask, score_mask = masks(train.index, fold, config)
        if score_mask.sum() == 0 or (train.loc[fit_mask, "label"] == 1).sum() == 0:
            continue
        scores[score_mask] = np.asarray(fit_and_score(train[fit_mask], train[score_mask]))
    return scores


# ============================================================ metrics for one fold

def chance_caught_by_luck(test: pd.DataFrame, config: EvalConfig) -> float:
    """P(a blind detector at the budget rate alarms in the risk window).

    Random alerts ~ Poisson at `budget` per operating hour:
    P(at least one) = 1 - exp(-budget * operating hours in the risk window).
    """
    risk_hours = test.loc[test["label"] == 1, "operating"].sum() / 60
    return 1 - math.exp(-config.budget_per_operating_hour * risk_hours)


def evaluate_fold(method: str, fold: Fold, test: pd.DataFrame, alarm: pd.Series, config: EvalConfig,
                  scores: pd.Series | None = None, threshold: float | None = None) -> dict:
    """Every metric for one method on one fold. Returns one row of the results table.

    Event level (what the depot cares about):
      caught            at least one alarm in the risk window before the failure
      warning_hours     from the first such alarm (blank if missed / unknown onset)
      false_alerts      alerts starting on normal minutes, per 100 h and per week
      within_budget     false-alert rate <= budget
      chance_caught_by_luck  see chance_caught_by_luck()
    Window level (secondary):
      pr_auc            needs scores; compare with risk_base_rate (a random score)
      f2, precision, recall  on the alarms
    Accuracy is deliberately not reported (always "normal" scores ~99.8%).
    """
    labels = test["label"].astype(int)
    alarm = alarm.reindex(test.index).fillna(False).astype(bool)
    failure = fold.held_out
    window_start = failure.start - config.warning_horizon

    in_window = (alarm.index >= window_start) & (alarm.index < failure.start)
    hits = alarm.index[in_window & alarm.to_numpy()]
    caught = len(hits) > 0
    onset_known = failure.name not in config.unknown_onset

    false_alerts = count_false_alerts(alarm, test["label"], config)
    hours = normal_operating_hours(test["label"], test["operating"])
    weeks = int((labels == 0).sum()) / MINUTES_PER_WEEK
    both_classes = labels.nunique() == 2

    return {
        "method": method,
        "fold": fold.name,
        "failure": failure.name,
        "caught": caught,
        "warning_hours": round((failure.start - hits[0]).total_seconds() / 3600, 2)
        if caught and onset_known else float("nan"),
        "onset_known": onset_known,
        "alarm_minutes_in_window": int(alarm[in_window].sum()),
        "false_alerts": false_alerts,
        "normal_operating_hours": round(hours, 1),
        "normal_weeks": round(weeks, 2),
        "false_alerts_per_100_operating_hours": round(100 * false_alerts / hours, 2) if hours else float("nan"),
        "false_alerts_per_week": round(false_alerts / weeks, 2) if weeks else float("nan"),
        "within_budget": bool(hours) and false_alerts / hours <= config.budget_per_operating_hour,
        "chance_caught_by_luck": chance_caught_by_luck(test, config),
        "pr_auc": round(average_precision_score(labels, scores.reindex(test.index)), 4)
        if scores is not None and both_classes else float("nan"),
        "risk_base_rate": round(labels.mean(), 5),
        "f2": round(fbeta_score(labels, alarm, beta=config.f_beta, zero_division=0), 4),
        "precision": round(precision_score(labels, alarm, zero_division=0), 4),
        "recall": round(recall_score(labels, alarm, zero_division=0), 4),
        "threshold": threshold,
        "test_minutes": len(test),
        "test_risk_minutes": int(labels.sum()),
    }


# ============================================================ across folds

def probability_at_least(k: int, chances: list) -> float:
    """P(at least k successes) for independent events with different chances (exact)."""
    distribution = np.zeros(len(chances) + 1)
    distribution[0] = 1.0
    for p in chances:
        distribution[1:] = distribution[1:] * (1 - p) + distribution[:-1] * p
        distribution[0] *= 1 - p
    return float(distribution[k:].sum())


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """One row per method. False-alert rates are POOLED (total / total), not averaged,
    so a short test block does not count as much as a long one."""
    rows = {}
    for method, group in per_fold.groupby("method", sort=False):
        caught = int(group["caught"].sum())
        warnings = group.loc[group["caught"] & group["onset_known"], "warning_hours"]
        hours, weeks = group["normal_operating_hours"].sum(), group["normal_weeks"].sum()
        rows[method] = {
            "failures caught": f"{caught} of {len(group)}",
            "caught": ", ".join(group.loc[group["caught"], "failure"]) or "none",
            "median warning hours (known onset)": round(warnings.median(), 2) if len(warnings) else float("nan"),
            "false alerts": int(group["false_alerts"].sum()),
            "false alerts per 100 operating hours": round(100 * group["false_alerts"].sum() / hours, 2) if hours else float("nan"),
            "false alerts per week": round(group["false_alerts"].sum() / weeks, 2) if weeks else float("nan"),
            "P(this many catches by luck)": f"{probability_at_least(caught, list(group['chance_caught_by_luck'])):.1e}",
            "mean PR-AUC": round(group["pr_auc"].mean(), 4),
            "mean F2": round(group["f2"].mean(), 4),
        }
    return pd.DataFrame(rows).T


# ============================================================ outputs

def save_results(per_fold: pd.DataFrame, summary: pd.DataFrame, folder: Path = Path("results")) -> None:
    """Write the files the report is built from. Never retype numbers from the console."""
    folder.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(folder / "per_fold.csv", index=False)
    summary.to_csv(folder / "summary.csv")
    with open(folder / "results.json", "w") as handle:
        json.dump({"per_fold": per_fold.to_dict(orient="records"),
                   "summary": summary.to_dict(orient="index")}, handle, indent=2, default=str)


def plot_fold(method: str, fold: Fold, test: pd.DataFrame, alarm: pd.Series, config: EvalConfig,
              scores: pd.Series | None = None, threshold: float | None = None,
              folder: Path = Path("figures")) -> Path:
    """48 h before the failure: score, threshold, alarms. Raw material for error analysis
    (did the score never rise, or rise but stay under the threshold?)."""
    failure = fold.held_out
    shown = (test.index >= failure.start - pd.Timedelta(hours=48)) & (test.index < failure.start + pd.Timedelta(hours=2))
    window = test.index[shown]
    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.axvspan(failure.start - config.warning_horizon, failure.start, color="tab:orange", alpha=0.15,
                 label=f"risk window ({config.warning_horizon.total_seconds() / 3600:.0f} h)")
    axis.axvline(failure.start, color="tab:red", linewidth=1, label="failure start")
    if scores is not None:
        axis.plot(window, scores.reindex(window), linewidth=0.8, color="tab:blue", label="risk score")
        if threshold is not None and np.isfinite(threshold):
            axis.axhline(threshold, color="tab:blue", linestyle="--", linewidth=0.8, label="threshold")
    alarm_times = window[alarm.reindex(window).fillna(False).to_numpy(dtype=bool)]
    level = axis.get_ylim()[1] if scores is not None else 1
    axis.scatter(alarm_times, np.full(len(alarm_times), level), marker="|", color="black", s=30, label="alarm")
    axis.set_title(f"{method}, {fold.name}")
    axis.legend(loc="upper left", fontsize=7)
    figure.autofmt_xdate()
    figure.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in method.split(" (")[0])
    path = folder / f"timeline_{safe}_{failure.name}.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path
