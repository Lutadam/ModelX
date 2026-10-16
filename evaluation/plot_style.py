"""One look for every figure: colours, axes and the helpers that keep graphs honest.

Palette: the reference palette of the dataviz skill, light mode. Status colours
(caught / false alert) are kept apart from the series colours and always come
with a legend label, never colour alone.
"""
import pandas as pd

# Chart chrome and ink
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
AXIS = "#c3c2b7"

# Series colours, in fixed order (never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]  # second cue next to colour

# Fixed meanings
RISK_BAND = "#eb6834"      # the 3-hour risk window, always shaded
CAUGHT = "#0ca30c"         # status: good
FALSE_ALERT = "#d03b3b"    # status: critical
FAILURE_LINE = TEXT_MUTED

DPI = 150


def style_axis(axis) -> None:
    """Recessive grid and spines, muted ticks."""
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=TEXT_MUTED, labelsize=8)


def title(axis, text: str) -> None:
    axis.set_title(text, fontsize=9.5, color=TEXT, loc="left")


def legend_below(figure, axis, columns: int = 5) -> None:
    """Legend under the plot, so it never covers the data."""
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=min(columns, len(handles)),
                      fontsize=8, frameon=False, labelcolor=TEXT_MUTED)


def on_minute_grid(values: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """`values` on every minute of [start, end), NaN where there is no data.

    Plotting this instead of the raw rows breaks the line at gaps, so a missing
    stretch is not drawn as a straight ramp between its two ends.
    """
    grid = pd.date_range(start.ceil("min"), end, freq="1min", inclusive="left")
    return values.reindex(grid)
