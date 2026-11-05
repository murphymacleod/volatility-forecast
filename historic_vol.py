import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from distributions import INK, SERIES, returns, save, style_charts, uso

# Return column -> chart filename, with one chart per return type
CHARTS = {"Log_Returns": "rolling_vol_log.png", "Simple_Returns": "rolling_vol_simple.png"}
TRADING_DAYS = 252

# Rolling window in trading days -> color, so each window keeps its color across every panel
WINDOWS = {20: "#2a78d6", 15: "#eb6834", 10: "#1baf7a"}

# Panel title -> years of history, where None means the entire data range
PERIODS = {"Entire range": None, "Last 5 years": 5, "Last 3 years": 3, "Last 1 year": 1}


def annualized_vol(daily_returns):
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def rolling_vol(daily_returns, window):
    return daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def period_slice(data, years):
    if years is None:
        return data
    return data[data.index > data.index[-1] - pd.DateOffset(years=years)]


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
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
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

    save(fig, filename)


def close_to_close_vol(log_returns):
    # A single day's vol is just the size of its return, annualized
    return log_returns.abs() * np.sqrt(TRADING_DAYS)


def garman_klass_gap_vol(prices):
    # Garman-Klass only sees the open to close session, so the overnight gap is added back in
    high_low = np.log(prices["High_Price"] / prices["Low_Price"])
    open_close = np.log(prices["Closing_Price"] / prices["Open_Price"])
    overnight_gap = np.log(prices["Open_Price"] / prices["Closing_Price"].shift())
    daily_variance = 0.5 * high_low**2 - (2 * np.log(2) - 1) * open_close**2 + overnight_gap**2
    return np.sqrt(daily_variance * TRADING_DAYS)


def plot_recent_vol_measures(days=20):
    # Everything is calculated on the full history first, then cut down to the days shown
    log_returns = returns["Log_Returns"]
    historic_vol = annualized_vol(log_returns)
    measures = pd.DataFrame({
        "Close-to-close": close_to_close_vol(log_returns),
        "Garman-Klass + gap": garman_klass_gap_vol(uso),
        "20-day rolling": rolling_vol(log_returns, 20),
    }).tail(days)
    print(measures.assign(Historic=historic_vol).round(3))

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    fig.suptitle(f"USO annualized vol from log returns: last {days} trading days")

    ax.plot(measures.index, measures["Close-to-close"], color="#2a78d6", linewidth=1.5, marker="o", markersize=5, label="Close-to-close, |log return| × √252")
    ax.plot(measures.index, measures["Garman-Klass + gap"], color="#eb6834", linewidth=1.5, marker="o", markersize=5, label="Garman-Klass + overnight gap")
    ax.plot(measures.index, measures["20-day rolling"], color="#1baf7a", linewidth=2, label="20-day rolling vol")
    ax.axhline(historic_vol, color=INK, linewidth=1.5, linestyle="--", label=f"Historic vol, entire range ({historic_vol:.1%})")

    ax.set_ylabel("Annualized vol")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.legend(loc="outside lower center", ncols=2, frameon=False)

    save(fig, "recent_vol_measures.png")


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
