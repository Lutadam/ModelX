"""Gaps in the data and whether they predict the label (lecturer feedback).

Two questions:
  1. Where do the gaps sit relative to the four failures? A gap after a failure
     is the repair and harmless; a gap in the hours before one eats into its
     risk window.
  2. Do the gaps predict the label? If the logger drops out as a fault develops,
     "the data just came back" carries label information, and a model can learn
     the missingness instead of the signal. The purge gap does not guard against
     this: it stops overlapping windows, not informative missingness.

Why it matters here: several features restart at a gap, so they partly encode
"time since the data came back":
  * motor_run_minutes counts from the last stop OR the last minute with no data;
  * the rolling and duty-cycle features restart in each segment (warm-up rows dropped);
  * the CNN+GRU scores 0 until it has 120 unbroken minutes.

A gap is the definition used everywhere else: more than config.GAP_THRESHOLD
between two readings (data.dataset.load_dataset starts a new segment there).

Runs as step 3 of main.py, or on its own from the project root:
    python -m evaluation.gap_analysis
Writes results/gaps.csv, results/gaps_by_failure.csv and figures/data/gaps.png.
Checks itself first (see sanity_checks) and stops if a check fails.
"""
import sys

import matplotlib

matplotlib.use("Agg")  # write figures to files; no screen needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data.dataset import load_dataset
from data.minute_table import load_minute_table
from evaluation import plot_style as style
from evaluation.console import print_table, print_title, print_values

NEAR_HOURS = config.FIGURE_HOURS_BEFORE_FAILURE  # "near a failure" = within the horizon figure's 48 h
MINUTE = pd.Timedelta(minutes=1)


# ============================================================ question 1: where the gaps are

def find_gaps(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per gap: last reading before it, first reading after it, length."""
    times = readings.index
    jumps = np.flatnonzero(np.diff(times.to_numpy()) > config.GAP_THRESHOLD.to_timedelta64())
    gaps = pd.DataFrame({"last_reading_before": times[jumps], "first_reading_after": times[jumps + 1]})
    gaps["hours"] = (gaps["first_reading_after"] - gaps["last_reading_before"]) / pd.Timedelta(hours=1)
    return gaps


def place_gap(gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> tuple:
    """Where a gap sits relative to the failures: (place, failure name, hours from the gap's end to that failure's start).

    Checked in this order, so a gap gets the most serious place that applies:
    overlaps a risk window > ends in the 48 h before one > during a failure or its repair
    > starts in the 48 h after a repair > far from every failure.
    """
    for failure in config.FAILURES:
        risk_start = failure.start - config.WARNING_HORIZON
        hours_to_failure = round((failure.start - gap_end) / pd.Timedelta(hours=1), 2)
        if gap_start < failure.start and gap_end > risk_start:
            return "overlaps risk window", failure.name, hours_to_failure
    for failure in config.FAILURES:
        risk_start = failure.start - config.WARNING_HORIZON
        if risk_start - pd.Timedelta(hours=NEAR_HOURS) <= gap_end <= risk_start:
            return f"in the {NEAR_HOURS} h before", failure.name, round((failure.start - gap_end) / pd.Timedelta(hours=1), 2)
    for failure in config.FAILURES:
        if gap_start <= failure.excluded_until and gap_end >= failure.start:
            return "during failure or repair", failure.name, float("nan")
    for failure in config.FAILURES:
        if failure.excluded_until < gap_start <= failure.excluded_until + pd.Timedelta(hours=NEAR_HOURS):
            return f"in the {NEAR_HOURS} h after repair", failure.name, float("nan")
    return "far from every failure", "", float("nan")


def gap_table(readings: pd.DataFrame) -> pd.DataFrame:
    gaps = find_gaps(readings)
    places = [place_gap(row.last_reading_before, row.first_reading_after) for row in gaps.itertuples()]
    gaps["place"] = [place for place, _, _ in places]
    gaps["failure"] = [failure for _, failure, _ in places]
    gaps["hours_from_gap_end_to_failure"] = [hours for _, _, hours in places]
    gaps["hours"] = gaps["hours"].round(2)
    return gaps


# ============================================================ question 2: do the gaps predict the label?

def minutes_since_data_resumed(readings: pd.DataFrame, minutes: pd.DatetimeIndex) -> pd.Series:
    """For each minute: minutes since the start of its segment (since the data last came back after a gap).

    The segment start is rounded down to its minute, because the table's rows are labelled by the
    minute they start (a first reading at 22:15:07 belongs to the row 22:15:00).
    """
    segment_start = readings.index.to_series().groupby(readings["segment"]).min().dt.floor(config.FEATURE_STEP)
    segment_per_minute = readings["segment"].resample(config.FEATURE_STEP).max().reindex(minutes)
    resumed_at = segment_per_minute.map(segment_start)
    return pd.Series(((minutes - pd.DatetimeIndex(resumed_at)) / MINUTE).to_numpy(), index=minutes,
                     name="minutes_since_data_resumed")


def by_failure(minute_table: pd.DataFrame, since_resumed: pd.Series) -> pd.DataFrame:
    """Per failure: how close its risk window sits to the data coming back, against normal minutes.

    'share of normal minutes this close' is the key number: if a risk window
    starts closer to a resume than almost all normal minutes do, the gap carries
    label information. With four failures this is a description, not a test.
    """
    normal = since_resumed[minute_table["label"] == 0]
    rows = []
    for failure in config.FAILURES:
        risk_start = failure.start - config.WARNING_HORIZON
        in_window = minute_table.index[(minute_table.index >= risk_start) & (minute_table.index < failure.start)]
        if len(in_window) == 0:
            rows.append({"failure": failure.name, "risk minutes with data": 0})
            continue
        at_window_start = since_resumed[in_window[0]]
        run = minute_table.loc[in_window, "motor_run_minutes"]
        resumed = since_resumed[in_window]
        rows.append({
            "failure": failure.name,
            "risk minutes with data": f"{len(in_window)} of {int(config.WARNING_HORIZON / MINUTE)}",
            "hours since data resumed (window start)": round(at_window_start / 60, 1),
            "share of normal minutes this close": round(float((normal <= at_window_start).mean()), 3),
            # True: the motor run seen in the window began when the data came back, so its length is set by the gap
            "motor run began at data resume": bool(((run - resumed).abs() <= 1).any() and run.max() > 0),
            "longest motor run in window (h)": round(float(run.max()) / 60, 1),
        })
    return pd.DataFrame(rows).set_index("failure")


def pooled_comparison(minute_table: pd.DataFrame, since_resumed: pd.Series) -> pd.DataFrame:
    """Hours since the data resumed: risk minutes vs normal minutes, all failures together."""
    rows = {}
    for name, label in (("normal minutes", 0), ("risk minutes", 1)):
        hours = since_resumed[minute_table["label"] == label] / 60
        rows[name] = {"minutes": len(hours), "median h": round(hours.median(), 1),
                      "share within 3 h": round(float((hours <= 3).mean()), 3),
                      "share within 24 h": round(float((hours <= 24).mean()), 3)}
    return pd.DataFrame(rows).T


# ============================================================ sanity checks

def sanity_checks(readings: pd.DataFrame, gaps: pd.DataFrame, minute_table: pd.DataFrame,
                  since_resumed: pd.Series) -> list:
    """Quick checks that the numbers above can be trusted. Returns the failed checks (empty = all passed)."""
    failed = []
    span = readings.index[-1] - readings.index[0]
    covered = readings.index.to_series().groupby(readings["segment"]).agg(lambda t: t.iloc[-1] - t.iloc[0]).sum()
    gap_time = (gaps["first_reading_after"] - gaps["last_reading_before"]).sum()
    checks = {
        "one gap per new segment": len(gaps) == readings["segment"].nunique() - 1,
        f"every gap is longer than {config.GAP_THRESHOLD}":
            bool(((gaps["first_reading_after"] - gaps["last_reading_before"]) > config.GAP_THRESHOLD).all()),
        "gaps are in time order and do not overlap":
            bool((gaps["first_reading_after"].iloc[:-1].to_numpy() <= gaps["last_reading_before"].iloc[1:].to_numpy()).all()),
        "time with data + time in gaps = whole span": covered + gap_time == span,
        "minutes since resume is never negative": bool((since_resumed.dropna() >= 0).all()),
        "every minute of the table has a segment": bool(since_resumed.notna().all()),
        "risk minutes in the table = sum over failures":
            int((minute_table["label"] == 1).sum()) == sum(
                int(((minute_table.index >= f.start - config.WARNING_HORIZON) & (minute_table.index < f.start)).sum())
                for f in config.FAILURES),
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            failed.append(name)
    return failed


# ============================================================ figure

def draw(gaps: pd.DataFrame, readings: pd.DataFrame) -> None:
    """Every gap on one timeline (bar width = its length), with the failures and their risk windows."""
    figure, axis = plt.subplots(figsize=(12, 2.8), facecolor=style.SURFACE)
    style.style_axis(axis)
    axis.grid(visible=False)
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    near = gaps["place"] != "far from every failure"
    for subset, colour, name in ((gaps[~near], style.AXIS, "gap far from every failure"),
                                 (gaps[near], style.FALSE_ALERT, f"gap within {NEAR_HOURS} h of a failure")):
        starts = mdates.date2num(subset["last_reading_before"])
        widths = mdates.date2num(subset["first_reading_after"]) - starts
        axis.broken_barh(list(zip(starts, widths)), (0.15, 0.7), color=colour, label=name)
    axis.xaxis_date()
    for failure in config.FAILURES:
        axis.axvline(failure.start, color=style.FAILURE_LINE, linewidth=1)
        axis.text(failure.start, 0.95, f" {failure.name}", fontsize=8.5, color=style.TEXT, va="top")
    axis.set_xlim(readings.index[0], readings.index[-1])
    axis.set_ylim(0, 1)
    figure.suptitle(f"The {len(gaps)} gaps in the data (longer than {config.GAP_THRESHOLD}), "
                    "and the four failures (vertical lines)", fontsize=10, color=style.TEXT, x=0.01, ha="left")
    figure.autofmt_xdate()
    figure.tight_layout(rect=(0, 0.12, 1, 0.95))
    style.legend_below(figure, axis, columns=2)
    config.FIGURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.FIGURES_DATA_DIR / "gaps.png", dpi=style.DPI, facecolor=style.SURFACE)
    plt.close(figure)


# ============================================================ report

def make_gap_report(minute_table: pd.DataFrame = None) -> None:
    """Print the gap tables and write results/gaps*.csv and figures/data/gaps.png.

    Raises ValueError, before writing anything, if a sanity check fails.
    `minute_table`: pass it when already loaded (main.py); otherwise it is loaded from the cache.
    """
    readings = load_dataset(config.DATA_FILE)
    if minute_table is None:
        minute_table = load_minute_table(verbose=True)
    gaps = gap_table(readings)
    since_resumed = minutes_since_data_resumed(readings, minute_table.index)

    print("\n  Sanity checks")
    failed = sanity_checks(readings, gaps, minute_table, since_resumed)
    if failed:
        raise ValueError(f"{len(failed)} gap sanity check(s) failed ({'; '.join(failed)}), "
                         "so the gap numbers would not be trustworthy.")

    print_values({"Gaps": len(gaps), "Hours in gaps": f"{gaps['hours'].sum():,.1f}",
                  "Longest gap": f"{gaps['hours'].max():.1f} h"})
    print_table(gaps["place"].value_counts().to_frame("gaps"), "1. Where the gaps sit")
    near = gaps[gaps["place"] != "far from every failure"]
    print_table(near.set_index("failure")[["last_reading_before", "first_reading_after", "hours", "place",
                                           "hours_from_gap_end_to_failure"]],
                f"   Gaps within {NEAR_HOURS} h of a failure")
    failures = by_failure(minute_table, since_resumed)
    print_table(failures,
                "2. Each risk window against normal minutes (share close to 0 = window sits unusually near a resume)")
    print_table(pooled_comparison(minute_table, since_resumed), "   Hours since the data resumed, all failures pooled")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(config.RESULTS_DIR / "gaps.csv", index=False)
    failures.to_csv(config.RESULTS_DIR / "gaps_by_failure.csv")
    draw(gaps, readings)
    print(f"\n  Written: {config.RESULTS_DIR / 'gaps.csv'}, {config.RESULTS_DIR / 'gaps_by_failure.csv'}, "
          f"{config.FIGURES_DATA_DIR / 'gaps.png'}")


if __name__ == "__main__":
    print_title("GAPS IN THE DATA: where they are, and do they predict the label?")
    try:
        make_gap_report()
    except ValueError as error:
        sys.exit(f"STOPPED: {error}")
    except FileNotFoundError:
        sys.exit(f"STOPPED: the MetroPT-3 CSV is not at {config.DATA_FILE}. Change DATA_FILE in config.py.")
    except PermissionError as error:
        sys.exit(f"STOPPED: could not write {error.filename}. Close it in any program that has it open and run again.")
