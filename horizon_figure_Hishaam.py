"""Four-panel figure: what the sensors did in the 48 h before each failure.

Owner: Lauthan Muhammad Hishaam Ibn Afzal (Evaluator & Ethics), Team ModelX.
Week 1 task (refinement 1, condition 3): sanity-check the warning horizon N = 3 h.

One column per failure (F1-F4), two rows of small multiples:
  * continuous compressor run: hours since the motor last stopped (motor
    current above 1 A = running). A leak makes the compressor run without
    stopping to hold pressure, so the run gets unusually long. (A 60-min duty
    cycle cannot show this: it is often 100% for an hour after a normal
    start-up, and it caps at 100% however long the run lasts.)
  * TP3 pressure: 60-min rolling mean, in bar.
The 3-hour risk window is shaded orange; FROZEN stretches are shaded grey:
minutes where pressure does not change at all and the motor is off. There
the train appears switched off and the logger repeats stale values (rows
arrive every ~12 s instead of 10 s); they are ignored when judging departures.
The x-axis is hours relative to the failure start.

It also measures, for each failure, how long before the failure each signal
left its normal range (see run_departure_hours and departure_hours). That
number is the evidence for or against N = 3 h:
  * departures well before -3 h  -> N is conservative; alarms that early count
    as FALSE alerts under our labels, worth saying in the report;
  * departures inside -3 h       -> N fits;
  * no departure                 -> nothing in this signal warns of that failure.

Standalone: needs only the MetroPT-3 CSV.
    python horizon_figure_Hishaam.py "path/to/MetroPT3(AirCompressor).csv"
Writes figures/horizon_four_panel.png and results/horizon_departures.csv.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
WARNING_HORIZON_HOURS = 3          # N, fixed by the team (condition 3)
HOURS_BEFORE = 48                  # how far back each panel looks
HOURS_AFTER = 2                    # a little past the failure start
OPERATING_MOTOR_CURRENT_AMPS = 1.0 # same rule as the team's config.py
ROLLING = "60min"
BASELINE_DAYS = (7, 2)             # normal range = from 7 days to 2 days before the failure
BASELINE_QUANTILES = (0.005, 0.995)  # outside the middle 99% of the baseline = "departed"
SUSTAINED_MINUTES = 30             # must stay outside for this long to count

# Same as config.FAILURES (UCI table, renumbered). F1's start is a midnight
# placeholder: the whole day was logged, so its true onset is unknown.
FAILURES = [
    ("F1", pd.Timestamp("2020-04-18 00:00")),
    ("F2", pd.Timestamp("2020-05-29 23:30")),
    ("F3", pd.Timestamp("2020-06-05 10:00")),
    ("F4", pd.Timestamp("2020-07-15 14:30")),
]
UNKNOWN_ONSET = {"F1"}

# Reference palette (dataviz skill), light mode
SERIES = "#2a78d6"
RISK_BAND = "#eb6834"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


# ------------------------------------------------------------------ data

def load_minutes(csv_path: Path) -> pd.DataFrame:
    """Per-minute signals: continuous run length, pressure, and a frozen flag.

    Only the three needed columns are read, to keep memory low on a laptop.
    """
    readings = pd.read_csv(csv_path, usecols=["timestamp", "TP3", "Motor_current"],
                           parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    motor_on = (readings["Motor_current"] > OPERATING_MOTOR_CURRENT_AMPS).astype(float)
    per_minute = pd.DataFrame({
        "motor_on": motor_on.resample("1min").mean(),
        "tp3": readings["TP3"].resample("1min").mean(),
        "tp3_range": readings["TP3"].resample("1min").agg(lambda x: x.max() - x.min() if len(x) else np.nan),
    })
    has_data = per_minute["motor_on"].notna()

    # Frozen: pressure exactly constant and motor off, for 10+ minutes in a row.
    still = has_data & (per_minute["tp3_range"] == 0) & (per_minute["motor_on"] == 0)
    still_rolling = still.astype(int).rolling(10, min_periods=10).sum()
    frozen = still_rolling.reindex(per_minute.index).fillna(0) >= 10
    frozen = frozen | frozen.shift(-9, fill_value=False).rolling(10, min_periods=1).max().astype(bool)
    frozen &= still

    # Continuous run: minutes since the motor was last off. A minute counts as
    # running if the motor was on for most of it; missing minutes break the run.
    running = (per_minute["motor_on"] > 0.5) & has_data
    run_id = (~running).cumsum()
    run_minutes = running.groupby(run_id).cumsum().where(running, 0)

    out = pd.DataFrame({
        "run_hours": (run_minutes / 60).where(has_data),
        "tp3_bar": per_minute["tp3"].rolling(ROLLING, min_periods=40).mean(),
        "frozen": frozen,
    })
    out.loc[out["frozen"], "tp3_bar"] = np.nan  # stale values are not a measurement
    return out


def run_departure_hours(run_hours: pd.Series, failure_start: pd.Timestamp) -> tuple:
    """Did the compressor run unusually long before the failure, and since when?

    Normal = the longest unbroken run from 7 to 2 days before the failure.
    Returns (hours before the failure that the over-long run began, longest
    run in the 48 h before, longest baseline run). The first is NaN when no
    run in the last 48 h exceeded the baseline maximum.
    """
    baseline = run_hours[(run_hours.index >= failure_start - pd.Timedelta(days=BASELINE_DAYS[0]))
                         & (run_hours.index < failure_start - pd.Timedelta(days=BASELINE_DAYS[1]))]
    recent = run_hours[(run_hours.index >= failure_start - pd.Timedelta(hours=HOURS_BEFORE))
                       & (run_hours.index < failure_start)]
    baseline_max, recent_max = baseline.max(), recent.max()
    if not recent_max > baseline_max:
        return float("nan"), round(float(recent_max), 1), round(float(baseline_max), 1)
    peak_time = recent.idxmax()
    run_began = peak_time - pd.Timedelta(hours=float(recent_max))
    return (round((failure_start - run_began).total_seconds() / 3600, 1),
            round(float(recent_max), 1), round(float(baseline_max), 1))


def departure_hours(signal: pd.Series, failure_start: pd.Timestamp) -> float:
    """Hours before the failure that the signal left its normal range for good.

    Normal range = middle 99% of the same signal from 7 to 2 days before the
    failure. Looking back from the failure start, find the earliest minute of
    the last unbroken run (allowing gaps under SUSTAINED_MINUTES) of out-of-range
    values that reaches the failure. Returns NaN if the signal is in range at
    the failure start (no departure).
    """
    baseline = signal[(signal.index >= failure_start - pd.Timedelta(days=BASELINE_DAYS[0]))
                      & (signal.index < failure_start - pd.Timedelta(days=BASELINE_DAYS[1]))].dropna()
    if len(baseline) < 600:
        return float("nan")
    low, high = baseline.quantile(BASELINE_QUANTILES)
    recent = signal[(signal.index >= failure_start - pd.Timedelta(hours=HOURS_BEFORE))
                    & (signal.index < failure_start)].dropna()
    outside = (recent < low) | (recent > high)
    if len(outside) == 0 or not outside.iloc[-SUSTAINED_MINUTES:].any():
        return float("nan")
    start = None
    last_outside = None
    for time, is_out in outside[::-1].items():  # walk backwards from the failure
        if is_out:
            last_outside = time
            start = time
        elif last_outside is not None and last_outside - time > pd.Timedelta(minutes=SUSTAINED_MINUTES):
            break
    return round((failure_start - start).total_seconds() / 3600, 1) if start is not None else float("nan")


# ------------------------------------------------------------------ figure

def draw(minutes: pd.DataFrame, out_png: Path) -> pd.DataFrame:
    rows = [("run_hours", "Compressor running\nwithout a stop (hours)", None),
            ("tp3_bar", "TP3 pressure (bar)\n60-min mean", None)]
    figure, axes = plt.subplots(2, 4, figsize=(13, 5.2), sharex=True, sharey="row", facecolor=SURFACE)
    departures = []
    for column, (name, start) in enumerate(FAILURES):
        window = minutes[(minutes.index >= start - pd.Timedelta(hours=HOURS_BEFORE))
                         & (minutes.index < start + pd.Timedelta(hours=HOURS_AFTER))]
        hours = (window.index - start).total_seconds() / 3600
        record = {"failure": name, "failure_start": start, "onset_known": name not in UNKNOWN_ONSET}
        for row, (signal, label, limits) in enumerate(rows):
            axis = axes[row, column]
            axis.set_facecolor(SURFACE)
            axis.axvspan(-WARNING_HORIZON_HOURS, 0, color=RISK_BAND, alpha=0.14, linewidth=0)
            frozen_hours = hours[window["frozen"].to_numpy()]
            for block in np.split(frozen_hours, np.flatnonzero(np.diff(frozen_hours) > 0.05) + 1):
                if len(block):
                    axis.axvspan(block[0], block[-1], color=GRID, alpha=0.9, linewidth=0)
            axis.axvline(0, color=TEXT_MUTED, linewidth=1)
            axis.plot(hours, window[signal], color=SERIES, linewidth=1.2)
            axis.grid(axis="y", color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color(GRID)
            axis.tick_params(colors=TEXT_MUTED, labelsize=8)
            if limits:
                axis.set_ylim(*limits)
            if column == 0:
                axis.set_ylabel(label, color=TEXT, fontsize=9)
            if signal == "run_hours":
                gone, longest, normal_longest = run_departure_hours(minutes[signal], start)
                record["longest_run_last_48h_hours"] = longest
                record["longest_run_baseline_hours"] = normal_longest
                record["over_long_run_began_hours_before"] = gone
                note = (f"longest run {longest:g} h (normal max {normal_longest:g} h)" if np.isnan(gone)
                        else f"{longest:g} h run, began {gone:g} h before\n(normal max {normal_longest:g} h)")
            else:
                gone = departure_hours(minutes[signal], start)
                record["pressure_departed_hours_before"] = gone
                note = "no departure" if np.isnan(gone) else f"departs {gone:g} h before"
            frozen_share = window.loc[window.index < start, "frozen"].mean()
            record["frozen_share_last_48h"] = round(float(frozen_share), 2)
            # Run row: label at the top (the line sits on zero); pressure row: at the bottom.
            label_y, label_va = (0.96, "top") if signal == "run_hours" else (0.04, "bottom")
            axis.text(0.03, label_y, note, transform=axis.transAxes, fontsize=8, color=TEXT_MUTED,
                      va=label_va, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9))
        title = f"{name}  ·  {start:%d %b %Y %H:%M}"
        if name in UNKNOWN_ONSET:
            title += "\n(start time is a placeholder)"
        axes[0, column].set_title(title, fontsize=9.5, color=TEXT, loc="left")
        axes[1, column].set_xlabel("hours relative to failure start", fontsize=8.5, color=TEXT_MUTED)
        axes[1, column].set_xlim(-HOURS_BEFORE, HOURS_AFTER)
        axes[1, column].set_xticks([-48, -36, -24, -12, -3, 0])
        departures.append(record)
    figure.suptitle(f"The 48 hours before each failure. Orange: the {WARNING_HORIZON_HOURS}-hour risk window (N). "
                    "Grey: frozen readings (train off). Breaks: no data.",
                    fontsize=11, color=TEXT, x=0.01, ha="left")
    figure.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return pd.DataFrame(departures)


def operating_hours_per_day(csv_path: Path) -> float:
    """Average hours per day the compressor motor is on (condition 4 check).

    The team's budget says 1 false alert per 100 operating hours ~ 1 per week,
    assuming 14 operating hours a day. This is the real figure under the
    code's definition (motor current above 1 A).
    """
    readings = pd.read_csv(csv_path, usecols=["timestamp", "Motor_current"], parse_dates=["timestamp"])
    on_minutes = (readings.set_index("timestamp")["Motor_current"] > OPERATING_MOTOR_CURRENT_AMPS
                  ).astype(float).resample("1min").mean().dropna()
    days = on_minutes.index.normalize().nunique()
    return float(on_minutes.sum() / 60 / days)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('Usage: python horizon_figure_Hishaam.py "path/to/MetroPT3(AirCompressor).csv"')
    csv_path = Path(sys.argv[1])
    minutes = load_minutes(csv_path)
    departures = draw(minutes, Path("figures") / "horizon_four_panel.png")
    Path("results").mkdir(exist_ok=True)
    departures.to_csv(Path("results") / "horizon_departures.csv", index=False)
    print("What each signal did before each failure:")
    print(departures.to_string(index=False))
    hours_a_day = operating_hours_per_day(csv_path)
    print(f"\nCompressor motor on: {hours_a_day:.1f} h per day on average "
          f"-> 1 false alert per 100 operating hours = 1 per {100 / hours_a_day:.1f} days")
    print("Written: figures/horizon_four_panel.png, results/horizon_departures.csv")


if __name__ == "__main__":
    main()
