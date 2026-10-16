"""Summary figures for the report, made from one run of every method on every fold.

    figures/summary/comparison.png        every method side by side: failures caught, false alerts
    figures/summary/precision_recall.png  precision-recall curve per fold, against the base rate
    figures/alerts/alerts_<failure>.png   the whole test block: when each method raised its alerts
    figures/scores/scores_<method>.png    score on normal vs risk minutes, with the threshold

The per-failure timelines (figures/timelines/<method>/) come from evaluation.core.plot_fold.
"""
import matplotlib

matplotlib.use("Agg")  # write figures to files; no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import config
from evaluation import plot_style as style
from evaluation.core import EvalConfig, alert_starts, file_safe, split


def save(figure, folder, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / name, dpi=style.DPI, facecolor=style.SURFACE)
    plt.close(figure)


def method_colours(per_fold: pd.DataFrame) -> dict:
    """One fixed colour per method, in run order, used in every figure."""
    return {method: style.SERIES[position] for position, method in enumerate(per_fold["method"].unique())}


# ============================================================ every method side by side

def plot_comparison(per_fold: pd.DataFrame, settings: EvalConfig) -> None:
    """Left: failures caught out of 4. Right: false alerts per 100 operating hours, pooled
    over the folds (bar) and per fold (dots), against the budget."""
    methods = list(per_fold["method"].unique())
    positions = np.arange(len(methods))[::-1]  # first method at the top
    budget = 100 * settings.budget_per_operating_hour
    figure, (caught_axis, alert_axis) = plt.subplots(
        1, 2, figsize=(11, 0.6 * len(methods) + 1.8), sharey=True, facecolor=style.SURFACE,
        gridspec_kw={"width_ratios": [1, 1.6]})

    for axis in (caught_axis, alert_axis):
        style.style_axis(axis)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color=style.GRID, linewidth=0.6)

    for position, method in zip(positions, methods):
        rows = per_fold[per_fold["method"] == method]
        caught = int(rows["caught"].sum())
        known = rows[rows["onset_known"]]
        caught_axis.barh(position, caught, height=0.5, color=style.SERIES[0])
        caught_axis.text(caught + 0.08, position, f"{caught} of {len(rows)}  ({int(known['caught'].sum())} of "
                         f"{len(known)} with known onset)", va="center", fontsize=8, color=style.TEXT)

        pooled = 100 * rows["false_alerts"].sum() / rows["normal_operating_hours"].sum()
        alert_axis.barh(position, pooled, height=0.5, color=style.SERIES[0], alpha=0.35)
        alert_axis.scatter(rows["false_alerts_per_100_operating_hours"], np.full(len(rows), position),
                           s=28, color=style.SERIES[0], edgecolor=style.SURFACE, linewidth=1, zorder=3)
        # Folds with the same rate share one dot, so their names share one label ("F1, F2")
        for rate, same in rows.groupby("false_alerts_per_100_operating_hours")["failure"]:
            alert_axis.annotate(", ".join(same), (rate, position), xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=6.5, color=style.TEXT_MUTED)
        alert_axis.text(pooled, position - 0.36, f" {pooled:.2f} pooled", fontsize=7.5, color=style.TEXT,
                        ha="left", va="center",
                        bbox=dict(facecolor=style.SURFACE, edgecolor="none", pad=0.5, alpha=0.9))

    alert_axis.axvline(budget, color=style.TEXT_MUTED, linestyle="--", linewidth=1)
    alert_axis.text(budget, positions[0] + 0.5, f" budget: {budget:g}", fontsize=8, color=style.TEXT_MUTED,
                    va="bottom")
    caught_axis.set_xlim(0, len(per_fold["failure"].unique()) + 2.5)
    caught_axis.set_xticks(range(len(per_fold["failure"].unique()) + 1))
    caught_axis.set_yticks(positions, methods, fontsize=8.5, color=style.TEXT)
    caught_axis.set_ylim(positions[-1] - 0.7, positions[0] + 0.9)
    caught_axis.set_xlabel("failures caught", fontsize=8.5, color=style.TEXT_MUTED)
    alert_axis.set_xlabel("false alerts per 100 operating hours (bar: all folds pooled, dots: each fold)",
                          fontsize=8.5, color=style.TEXT_MUTED)
    style.title(caught_axis, "Failures caught (alarm in the 3 h before)")
    style.title(alert_axis, "False alerts (lower is better; dashed line = budget)")
    figure.tight_layout()
    save(figure, config.FIGURES_SUMMARY_DIR, "comparison.png")


# ============================================================ alerts over the whole test block

def plot_alert_strips(traces: list, minute_table: pd.DataFrame, settings: EvalConfig) -> None:
    """One figure per fold, one row per method: every alert in the test block, not only near the failure.
    Green = alert starting in the risk window, red = false alert. Grey = no data."""
    for fold in dict.fromkeys(trace["fold"] for trace in traces):
        fold_traces = [trace for trace in traces if trace["fold"] == fold]
        _, test = split(minute_table, fold, settings)
        failure = fold.held_out
        risk_start = failure.start - settings.warning_horizon
        no_data = style.on_minute_grid(pd.Series(1.0, index=test.index), test.index[0], test.index[-1]).isna()

        figure, axes = plt.subplots(len(fold_traces), 1, figsize=(11, 0.75 * len(fold_traces) + 1.4),
                                    sharex=True, facecolor=style.SURFACE)
        for axis, trace in zip(axes, fold_traces):
            style.style_axis(axis)
            axis.grid(visible=False)
            axis.set_yticks([])
            axis.spines["left"].set_visible(False)
            axis.fill_between(no_data.index, 0, 1, where=no_data.to_numpy(), color=style.GRID, alpha=0.6,
                              linewidth=0, step="post", label="no data")
            # The 3 h risk window is too narrow to see at this scale: the failure line marks it.
            axis.axvline(failure.start, color=style.FAILURE_LINE, linewidth=1, label="failure start")
            starts = alert_starts(trace["alarm"], settings)
            hit = (starts >= risk_start) & (starts < failure.start)
            labels = test["label"].reindex(starts)
            false = (labels == 0).to_numpy()
            axis.vlines(starts[hit], 0, 1, color=style.CAUGHT, linewidth=2, label="alert in risk window")
            axis.vlines(starts[false], 0, 1, color=style.FALSE_ALERT, linewidth=1.5, label="false alert")
            axis.set_ylim(0, 1)
            axis.text(-0.005, 0.5, trace["method"], transform=axis.transAxes, ha="right", va="center",
                      fontsize=8.5, color=style.TEXT)
            axis.text(1.005, 0.5, f"{int(false.sum())} false\n{'CAUGHT' if hit.any() else 'missed'}",
                      transform=axis.transAxes, ha="left", va="center", fontsize=7.5, color=style.TEXT_MUTED)
        axes[-1].set_xlim(test.index[0], test.index[-1])
        figure.suptitle(f"Every alert in the {fold.name} test block ({test.index[0]:%d %b} to "
                        f"{test.index[-1]:%d %b %Y}). Green: alert in the 3 h before the failure. Red: false alert.",
                        fontsize=10, color=style.TEXT, x=0.01, ha="left")
        figure.autofmt_xdate()
        figure.tight_layout(rect=(0.17, 0.07, 0.93, 0.95))
        # Legend from a row that has every entry (labels are given even when a row has no such alerts)
        style.legend_below(figure, axes[0], columns=5)
        save(figure, config.FIGURES_ALERTS_DIR, f"alerts_{failure.name}.png")


# ============================================================ precision-recall

def plot_precision_recall(traces: list, minute_table: pd.DataFrame, settings: EvalConfig,
                          colours: dict) -> None:
    """One panel per fold, one curve per method with a score. The dashed line is a random score."""
    scored = [trace for trace in traces if trace["scores"] is not None]
    folds = list(dict.fromkeys(trace["fold"] for trace in scored))
    figure, axes = plt.subplots(1, len(folds), figsize=(3.2 * len(folds), 3.6), sharey=True,
                                facecolor=style.SURFACE)
    for axis, fold in zip(np.atleast_1d(axes), folds):
        _, test = split(minute_table, fold, settings)
        labels = test["label"].astype(int)
        style.style_axis(axis)
        for position, trace in enumerate(t for t in scored if t["fold"] == fold):
            precision, recall, _ = precision_recall_curve(labels, trace["scores"].reindex(test.index).fillna(0))
            axis.plot(recall, precision, color=colours[trace["method"]],
                      linestyle=style.LINE_STYLES[position % len(style.LINE_STYLES)], linewidth=1.4,
                      label=trace["method"])
        axis.axhline(labels.mean(), color=style.TEXT_MUTED, linestyle="--", linewidth=1,
                     label="random score (base rate)")
        axis.set_yscale("log")
        axis.set_xlim(0, 1)
        axis.set_xlabel("recall (share of risk minutes flagged)", fontsize=8, color=style.TEXT_MUTED)
        style.title(axis, fold.name)
    np.atleast_1d(axes)[0].set_ylabel("precision (log scale)", fontsize=8.5, color=style.TEXT_MUTED)
    figure.suptitle("Precision-recall per fold, minute by minute. Above the dashed line = better than random.",
                    fontsize=10, color=style.TEXT, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    style.legend_below(figure, np.atleast_1d(axes)[0], columns=5)
    save(figure, config.FIGURES_SUMMARY_DIR, "precision_recall.png")


# ============================================================ score vs threshold

def plot_score_distributions(traces: list, minute_table: pd.DataFrame, settings: EvalConfig) -> None:
    """For each method with a score: where its test scores fall on normal and risk minutes,
    and where the threshold sits. Shows whether a miss is 'the score never rose' or
    'it rose but the threshold was higher'."""
    scored = [trace for trace in traces if trace["scores"] is not None]
    for method in dict.fromkeys(trace["method"] for trace in scored):
        method_traces = [trace for trace in scored if trace["method"] == method]
        figure, axes = plt.subplots(1, len(method_traces), figsize=(3.2 * len(method_traces), 3.4),
                                    facecolor=style.SURFACE)
        for axis, trace in zip(np.atleast_1d(axes), method_traces):
            _, test = split(minute_table, trace["fold"], settings)
            sign = trace["score_sign"]
            shown = sign * trace["scores"].reindex(test.index).fillna(0)
            normal, risk = shown[test["label"] == 0], shown[test["label"] == 1]
            bins = np.histogram_bin_edges(shown, bins=50)
            style.style_axis(axis)
            axis.hist(normal, bins=bins, density=True, histtype="step", linewidth=1.3, color=style.SERIES[0],
                      label="normal minutes")
            axis.hist(risk, bins=bins, density=True, histtype="step", linewidth=1.3, color=style.SERIES[1],
                      linestyle="--", label="risk minutes (3 h before)")
            threshold = trace["threshold"]
            if threshold is not None and np.isfinite(threshold):
                axis.axvline(sign * threshold, color=style.TEXT_MUTED, linewidth=1.2, label="threshold")
            axis.set_yscale("log")
            axis.set_xlabel(trace["score_label"], fontsize=8, color=style.TEXT_MUTED)
            style.title(axis, trace["fold"].name)
        np.atleast_1d(axes)[0].set_ylabel("share of minutes (density, log scale)", fontsize=8,
                                          color=style.TEXT_MUTED)
        side = "left of" if method_traces[0]["score_sign"] < 0 else "right of"
        figure.suptitle(f"{method}: scores on normal and risk minutes. An alarm is a score {side} the threshold.",
                        fontsize=10, color=style.TEXT, x=0.01, ha="left")
        figure.tight_layout(rect=(0, 0.1, 1, 1))
        style.legend_below(figure, np.atleast_1d(axes)[0], columns=3)
        save(figure, config.FIGURES_SCORES_DIR, f"scores_{file_safe(method)}.png")


def make_report_figures(per_fold: pd.DataFrame, traces: list, minute_table: pd.DataFrame,
                        settings: EvalConfig) -> None:
    """Every summary figure. `traces` comes from run_folds.run_all_folds."""
    plot_comparison(per_fold, settings)
    plot_alert_strips(traces, minute_table, settings)
    plot_precision_recall(traces, minute_table, settings, method_colours(per_fold))
    plot_score_distributions(traces, minute_table, settings)
