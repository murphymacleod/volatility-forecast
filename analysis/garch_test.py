"""
First test of GARCH(1,1) on USO's daily log returns. Charts go to plots/garch/.

This fits on all of the data, so it shows how the model behaves but is not a
backtest. Its parameters have seen every date that its volatility is compared to.
For out-of-sample results, see backtests/run_backtest.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox

from volforecast.data import load_prices
from volforecast.garch import (
    conditional_vol,
    end_of_day_vol,
    fit_garch,
    forecast_vols as _forecast_vols,
    half_life,
    long_run_vol,
    persistence,
)
from volforecast.measures import annualized_vol, period_slice, recent_vol_measures, rolling_vol
from volforecast.plotting import INK, MEASURE_COLORS, MEASURE_LABELS, format_date_axis, save, style_charts

PLOT_FOLDER = "garch"
FORECAST_DAYS = 60
TARGET_HORIZONS = [1, 10, 20]

uso = load_prices("USO")
returns = uso[["Log_Returns", "Simple_Returns"]].dropna()

GARCH_T_COLOR = "#2a78d6"
GARCH_NORMAL_COLOR = "#eb6834"
ROLLING_COLOR = "#1baf7a"
RAW_RETURNS_COLOR = "#898781"

# The recent vol chart keeps historic_vol.py's colors for the daily measures, so GARCH takes the next two slots there
RECENT_GARCH_COLORS = {"GARCH-t": "#eda100", "GARCH-normal": "#e87ba4"}


def forecast_vols(result, days=FORECAST_DAYS):
    return _forecast_vols(result, days)


def summarize_models(results):
    rows = {}
    for name, result in results.items():
        window_vol = forecast_vols(result)["whole_window"]
        rows[name] = {
            "Converged": result.convergence_flag == 0,
            "mu (% per day)": result.params["mu"],
            "omega (%^2)": result.params["omega"],
            "alpha": result.params["alpha[1]"],
            "beta": result.params["beta[1]"],
            "nu": result.params.get("nu", np.nan),
            "Persistence": persistence(result),
            "Half-life (days)": half_life(result),
            "Long-run vol": long_run_vol(result),
            "AIC": result.aic,
            **{f"{days}-day forecast": window_vol[days] for days in TARGET_HORIZONS},
        }
    return pd.DataFrame(rows)


def plot_conditional_vol(result, log_returns):
    garch_vol = end_of_day_vol(result)
    rolling = rolling_vol(log_returns, 20)
    long_run = long_run_vol(result)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), layout="constrained")
    fig.suptitle("USO GARCH-t vol vs 20-day rolling vol, annualized and fit on all data\nEach line uses data through that day's close")

    for ax, (title, years) in zip(axes, {"Entire range": None, "Last 1 year": 1}.items()):
        ax.plot(period_slice(rolling, years), color=ROLLING_COLOR, linewidth=1, label="20-day rolling vol")
        ax.plot(period_slice(garch_vol, years), color=GARCH_T_COLOR, linewidth=1.2, label="GARCH-t vol")
        ax.axhline(long_run, color=INK, linewidth=1.5, linestyle="--", label=f"GARCH-t long-run vol ({long_run:.1%})")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Annualized vol")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        format_date_axis(ax)
        ax.margins(x=0.01)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    save(fig, "garch_conditional_vol.png", PLOT_FOLDER)


def plot_forecast(results, log_returns):
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    last_date = log_returns.index[-1]
    fig.suptitle(f"USO GARCH forecasts from {last_date:%Y-%m-%d}: vol over the next H trading days")

    forecasts = {name: forecast_vols(result)["whole_window"] for name, result in results.items()}
    colors = {"GARCH-t": GARCH_T_COLOR, "GARCH-normal": GARCH_NORMAL_COLOR}
    # Labels go above the higher curve and below the lower one, so they stay clear of each other
    label_offsets = {"GARCH-normal": 8, "GARCH-t": -14}

    for name, forecast in forecasts.items():
        ax.plot(forecast.index, forecast, color=colors[name], linewidth=2, label=f"{name} forecast")
        ax.axhline(long_run_vol(results[name]), color=colors[name], linewidth=1, linestyle=":", label=f"{name} long-run vol ({long_run_vol(results[name]):.1%})")

        for days in TARGET_HORIZONS:
            ax.plot(days, forecast[days], color=colors[name], marker="o", markersize=6)
            ax.annotate(
                f"{forecast[days]:.1%}",
                xy=(days, forecast[days]),
                xytext=(0, label_offsets[name]),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=9,
            )

    ax.axhline(rolling_vol(log_returns, 20).iloc[-1], color=ROLLING_COLOR, linewidth=1.5, linestyle="--", label=f"Latest 20-day rolling vol ({rolling_vol(log_returns, 20).iloc[-1]:.1%})")
    ax.set_xlabel("Forecast window H (trading days)")
    ax.set_ylabel("Annualized vol")
    ax.set_xlim(0, FORECAST_DAYS + 1)
    ax.set_ylim(0.30, 0.70)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    fig.legend(loc="outside lower center", ncols=3, frameon=False)
    save(fig, "garch_forecast.png", PLOT_FOLDER)


def plot_recent_vol_with_forecast(results, days=20):
    # Same view as historic_vol.py's recent vol chart, extended into the future with GARCH's forecasts
    measures = recent_vol_measures(uso, days)
    historic_vol = annualized_vol(returns["Log_Returns"])
    latest_rolling = measures["20-day rolling"].iloc[-1]
    today = measures.index[-1]

    # The x axis counts trading days from today, because future trading dates aren't known without a holiday calendar
    past_x = np.arange(1 - days, 1)
    future_x = np.arange(0, days + 1)

    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    fig.suptitle(f"USO annualized vol: the last {days} trading days, and GARCH forecasts for the next {days}\nEach line uses data through that day's close")
    ax.axvspan(0, days, color="#f0efec", zorder=0)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.text(days - 0.5, 0.97, "Forecast", transform=ax.get_xaxis_transform(), ha="right", va="top", color="#52514e")
    # GARCH-normal forecasts run higher, so its label goes above the line end and GARCH-t's goes below
    label_offsets = {"GARCH-normal": 15, "GARCH-t": -15}

    for column in ["Close-to-close", "Garman-Klass + gap"]:
        ax.plot(past_x, measures[column], color=MEASURE_COLORS[column], linewidth=1.5, marker="o", markersize=5, label=MEASURE_LABELS[column])

    ax.plot(past_x, measures["20-day rolling"], color=MEASURE_COLORS["20-day rolling"], linewidth=2, label=MEASURE_LABELS["20-day rolling"])
    ax.plot([0, days], [latest_rolling, latest_rolling], color=MEASURE_COLORS["20-day rolling"], linewidth=2, linestyle=":", label=f"Latest 20-day rolling vol, held flat ({latest_rolling:.1%})")

    for name, result in results.items():
        color = RECENT_GARCH_COLORS[name]
        past = end_of_day_vol(result).reindex(measures.index)
        forecast = forecast_vols(result, days)
        window_vol = forecast["whole_window"][days]

        ax.plot(past_x, past, color=color, linewidth=2, label=f"{name} vol")
        # Today's value is already the forecast for day +1, so the line stays flat to +1 before it decays
        ax.plot(future_x, np.r_[past.iloc[-1], forecast["single_day"]], color=color, linewidth=2, linestyle="--", label=f"{name} daily forecast (whole 20-day window: {window_vol:.1%})")
        ax.annotate(
            f"{name}\n20-day: {window_vol:.1%}",
            xy=(days, forecast["single_day"][days]),
            xytext=(6, label_offsets[name]),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
            # A background keeps reference lines from running through the label
            bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1},
            zorder=5,
        )

    ax.axhline(historic_vol, color=INK, linewidth=1.5, linestyle="--", label=f"Historic vol, entire range ({historic_vol:.1%})")

    # Past ticks show dates, and future ticks show trading days ahead
    ticks = np.arange(-15, days + 1, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{measures.index[days - 1 + tick]:%b %d}" if tick <= 0 else f"+{tick}" for tick in ticks])
    ax.set_xlim(1 - days - 0.5, days + 4)
    ax.set_xlabel(f"Date, then trading days after {today:%b %d, %Y}")
    ax.set_ylabel("Annualized vol")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "garch_recent_vol_forecast.png", PLOT_FOLDER)


def squared_autocorrelation(series, lags):
    return [(series**2).autocorr(lag) for lag in lags]


def plot_diagnostics(result, log_returns):
    # A good fit leaves residuals with steady variance, no clustering, and tails matching the t
    residuals = result.std_resid.dropna()
    nu = result.params["nu"]
    # arch's t has a variance of 1, while scipy's has a variance of nu / (nu - 2)
    fitted_t = student_t(nu, scale=np.sqrt((nu - 2) / nu))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle(f"GARCH-t diagnostics: standardized residuals, fit with t degrees of freedom = {nu:.2f}")

    ax = axes[0, 0]
    ax.plot(residuals, color=GARCH_T_COLOR, linewidth=0.6)
    ax.set_title("Residuals over time: should look steady", loc="left")
    ax.set_ylabel("Standardized residual")
    format_date_axis(ax)

    ax = axes[0, 1]
    lags = np.arange(1, 21)
    ax.plot(lags, squared_autocorrelation(log_returns, lags), color=RAW_RETURNS_COLOR, linewidth=2, marker="o", markersize=5, label="Raw log returns")
    ax.plot(lags, squared_autocorrelation(residuals, lags), color=GARCH_T_COLOR, linewidth=2, marker="o", markersize=5, label="GARCH-t residuals")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Autocorrelation of squares: residuals should sit near 0", loc="left")
    ax.set_xticks(np.arange(5, 21, 5))
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 0]
    probabilities = (np.arange(1, len(residuals) + 1) - 0.5) / len(residuals)
    theoretical = fitted_t.ppf(probabilities)
    ax.scatter(theoretical, np.sort(residuals), color=GARCH_T_COLOR, s=10)
    limits = [theoretical.min(), theoretical.max()]
    ax.plot(limits, limits, color=INK, linewidth=1.2, linestyle="--", label="Perfect fit")
    ax.set_title("QQ plot vs fitted t: points should follow the line", loc="left")
    ax.set_xlabel("Fitted t quantile")
    ax.set_ylabel("Residual quantile")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1, 1]
    bins = np.arange(np.floor(residuals.min()), np.ceil(residuals.max()) + 0.25, 0.25)
    x = np.linspace(bins[0], bins[-1], 1000)
    ax.hist(residuals, bins=bins, density=True, color=GARCH_T_COLOR, edgecolor="#fcfcfb", linewidth=0.3, label="GARCH-t residuals")
    ax.plot(x, fitted_t.pdf(x), color=INK, linewidth=1.5, label="Fitted t")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5 / (len(residuals) * 0.25))
    ax.set_title("Residual distribution, on a log scale to show the tails", loc="left")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density (log scale)")
    ax.legend(loc="upper right", frameon=False)

    save(fig, "garch_diagnostics.png", PLOT_FOLDER)


def print_diagnostics(result, log_returns):
    residuals = result.std_resid.dropna()
    stats = pd.DataFrame({
        "Raw log returns": log_returns,
        "GARCH-t residuals": residuals,
    }).agg(["skew", "kurt"])
    print(stats.round(3))

    # Small p-values mean clustering is still left in the series
    tests = {
        "Raw log returns": acorr_ljungbox(log_returns**2, lags=[10, 20]),
        "GARCH-t residuals": acorr_ljungbox(residuals**2, lags=[10, 20]),
    }
    for name, test in tests.items():
        print(f"Ljung-Box on squares, {name}: p-value {test['lb_pvalue'].round(4).to_dict()}")


if __name__ == "__main__":
    log_returns = returns["Log_Returns"]
    results = {"GARCH-t": fit_garch(log_returns, "t"), "GARCH-normal": fit_garch(log_returns, "normal")}

    print(results["GARCH-t"].summary())
    print(summarize_models(results).round(4))
    print_diagnostics(results["GARCH-t"], log_returns)

    style_charts()
    plot_conditional_vol(results["GARCH-t"], log_returns)
    plot_forecast(results, log_returns)
    plot_recent_vol_with_forecast(results)
    plot_diagnostics(results["GARCH-t"], log_returns)
    plt.show()
