"""Shared chart style, colors and saving, so every chart in the project looks the same."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .data import PLOTS_DIR

INK = "#0b0b0b"
SURFACE = "#fcfcfb"

# Categorical colors, used in this order so neighboring series stay distinguishable
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Return column -> (label, color), so each return type keeps its color across every chart
RETURN_SERIES = {
    "Log_Returns": ("Log returns", "#2a78d6"),
    "Simple_Returns": ("Simple returns", "#eb6834"),
}

# Daily vol measure -> color and legend label
MEASURE_COLORS = {"Close-to-close": "#2a78d6", "Garman-Klass + gap": "#eb6834", "20-day rolling": "#1baf7a"}
MEASURE_LABELS = {
    "Close-to-close": "Close-to-close, |log return| × √252",
    "Garman-Klass + gap": "Garman-Klass + overnight gap",
    "20-day rolling": "20-day rolling vol",
}


def style_charts():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.family": ["Segoe UI", "sans-serif"],
    })


def save(fig, name, folder):
    """Save a chart under plots/<folder>/, or to `folder` itself when it's a full path."""
    directory = PLOTS_DIR / folder if isinstance(folder, str) else folder
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / name, dpi=150)


def format_date_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
