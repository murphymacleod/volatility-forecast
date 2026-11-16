"""Rolling and historic vol over several date ranges, and recent daily vol measures. Charts go to plots/historic_vol/."""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from volforecast.data import load_prices
from volforecast.measures import annualized_vol, period_slice, recent_vol_measures, rolling_vol
from volforecast.plotting import (
    INK,
    MEASURE_COLORS,
    MEASURE_LABELS,
    RETURN_SERIES as SERIES,
    format_date_axis,
    save,
    style_charts,
)

PLOT_FOLDER = "historic_vol"

# Return column -> chart filename, with one chart per return type
CHARTS = {"Log_Returns": "rolling_vol_log.png", "Simple_Returns": "rolling_vol_simple.png"}

# Rolling window in trading days -> color, so each window keeps its color across every panel
WINDOWS = {20: "#2a78d6", 15: "#eb6834", 10: "#1baf7a"}

# Panel title -> years of history, where None means the entire data range
PERIODS = {"Entire range": None, "Last 5 years": 5, "Last 3 years": 3, "Last 1 year": 1}

uso = load_prices("USO")
returns = uso[["Log_Returns", "Simple_Returns"]].dropna()


def plot_rolling_vol(column, filename):
    daily_returns = returns[column]

    # Rolling vol uses the full history, so the shorter panels have no gap at their left edge
    rolling = {window: rolling_vol(daily_returns, window) for window in WINDOWS}

    # Each panel has its own y scale, because 2020 would flatten the shorter periods
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.suptitle(f"USO annualized volatility from {SERIES[column][0].lower()}: rolling windows vs historic vol for each period")

    for ax, (title, years) in zip(axes.flat, PERIODS.items()):
        historic_vol = annualized_vol(period_slice(daily_returns, years))

        # The shortest window goes first, so the smoother longer windows stay visible on top
        for window, color in reversed(WINDOWS.items()):
            series = period_slice(rolling[window], years)
            ax.plot(series.index, series, color=color, linewidth=0.9, label=f"{window}-day rolling")

        ax.axhline(historic_vol, color=INK, linewidth=1.5, linestyle="--", label="Historic vol (period)")
        ax.set_title(f"{title}: historic vol {historic_vol:.1%}", loc="left")
        ax.set_ylabel("Annualized vol")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        format_date_axis(ax)
        ax.margins(x=0.01)

    # Legend entries run from longest to shortest window, then the historic line
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [2, 1, 0, 3]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="outside lower center",
        ncols=4,
        frameon=False,
    )

    save(fig, filename, PLOT_FOLDER)


def plot_recent_vol_measures(days=20):
    historic_vol = annualized_vol(returns["Log_Returns"])
    measures = recent_vol_measures(uso, days)
    print(measures.assign(Historic=historic_vol).round(3))

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    fig.suptitle(f"USO annualized vol from log returns: last {days} trading days")

    for column in ["Close-to-close", "Garman-Klass + gap"]:
        ax.plot(measures.index, measures[column], color=MEASURE_COLORS[column], linewidth=1.5, marker="o", markersize=5, label=MEASURE_LABELS[column])
    ax.plot(measures.index, measures["20-day rolling"], color=MEASURE_COLORS["20-day rolling"], linewidth=2, label=MEASURE_LABELS["20-day rolling"])
    ax.axhline(historic_vol, color=INK, linewidth=1.5, linestyle="--", label=f"Historic vol, entire range ({historic_vol:.1%})")

    ax.set_ylabel("Annualized vol")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    format_date_axis(ax)
    fig.legend(loc="outside lower center", ncols=2, frameon=False)

    save(fig, "recent_vol_measures.png", PLOT_FOLDER)


if __name__ == "__main__":
    style_charts()
    plot_recent_vol_measures()

    for column, filename in CHARTS.items():
        print(SERIES[column][0])
        for title, years in PERIODS.items():
            period = period_slice(returns[column], years)
            print(f"  {title}: {annualized_vol(period):.1%} from {period.index[0]:%Y-%m-%d} ({len(period)} days)")

        plot_rolling_vol(column, filename)

    plt.show()
