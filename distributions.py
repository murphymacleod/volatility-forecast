from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm

DATA_PATH = Path(__file__).parent / "etf_data" / "USO_daily_prices.csv"
PLOTS_DIR = Path(__file__).parent / "plots"

# Column name -> (label, color), so each return type keeps its color across every chart
SERIES = {
    "Log_Returns": ("Log returns", "#2a78d6"),
    "Simple_Returns": ("Simple returns", "#eb6834"),
}
INK = "#0b0b0b"

# Same 0.5% bins for both return types, so their histograms are directly comparable
BINS = np.arange(-0.30, 0.18, 0.005)

uso = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date").sort_index()

# Uses the split-adjusted close, since Raw_Closing_Price has a false jump at the 2020 reverse split
uso["Simple_Returns"] = uso["Closing_Price"].pct_change()

# The first row has no prior close, so its return is NaN
returns = uso[["Log_Returns", "Simple_Returns"]].dropna()


def style_charts():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
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


def save(fig, name):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=150)


def plot_returns_over_time():
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("USO daily returns over time")

    for ax, (column, (label, color)) in zip(axes, SERIES.items()):
        series = returns[column]
        ax.plot(series.index, series, color=color, linewidth=0.7)
        ax.set_title(label, loc="left")
        ax.set_ylabel("Daily return")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        worst = series.idxmin()
        ax.annotate(
            f"{series[worst]:.1%} on {worst:%Y-%m-%d}",
            xy=(worst, series[worst]),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=INK,
            fontsize=9,
        )

    save(fig, "returns_over_time.png")


def plot_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Distribution of USO daily returns")

    for ax, (column, (label, color)) in zip(axes, SERIES.items()):
        ax.hist(returns[column], bins=BINS, color=color, edgecolor="#fcfcfb", linewidth=0.3)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Daily return")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0].set_ylabel("Number of days")

    save(fig, "returns_histograms.png")


def plot_histograms_vs_normal():
    # The bottom row uses a log scale, because the fat tails are too small to see on the linear scale
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey="row", layout="constrained")
    fig.suptitle("USO daily returns vs a normal distribution with the same mean and standard deviation")
    x = np.linspace(BINS[0], BINS[-1], 1000)

    for col, (column, (label, color)) in enumerate(SERIES.items()):
        series = returns[column]
        normal_pdf = norm.pdf(x, series.mean(), series.std())

        for ax in axes[:, col]:
            ax.hist(series, bins=BINS, density=True, color=color, edgecolor="#fcfcfb", linewidth=0.3, label=label)
            ax.plot(x, normal_pdf, color=INK, linewidth=1.5, label="Normal fit")

        # pandas kurt() is excess kurtosis, where a normal distribution scores 0
        axes[0, col].set_title(label, loc="left")
        axes[0, col].text(
            0.03, 0.95,
            f"Excess kurtosis: {series.kurt():.2f}\nNormal: 0",
            transform=axes[0, col].transAxes,
            va="top",
            color=INK,
        )
        axes[0, col].legend(loc="upper right", frameon=False)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(0.03, 40)
        axes[1, col].set_xlabel("Daily return")
        axes[1, col].xaxis.set_major_formatter(PercentFormatter(xmax=1))

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density (log scale)")

    save(fig, "returns_vs_normal.png")


if __name__ == "__main__":
    print(uso.head())
    print(returns.describe())
    print(returns.agg(["skew", "kurt"]))

    style_charts()
    plot_returns_over_time()
    plot_histograms()
    plot_histograms_vs_normal()
    plt.show()
