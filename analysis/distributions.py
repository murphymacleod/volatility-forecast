"""USO return distributions, and normal vs Student's t fits. Charts go to plots/distributions/."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm, t as student_t

from volforecast.data import load_prices
from volforecast.plotting import INK, RETURN_SERIES as SERIES, save, style_charts

PLOT_FOLDER = "distributions"

# Same 0.5% bins for both return types, so their histograms are directly comparable
BINS = np.arange(-0.30, 0.18, 0.005)

uso = load_prices("USO")

# The first row has no prior close, so its return is NaN
returns = uso[["Log_Returns", "Simple_Returns"]].dropna()


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

    save(fig, "returns_over_time.png", PLOT_FOLDER)


def plot_histograms():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Distribution of USO daily returns")

    for ax, (column, (label, color)) in zip(axes, SERIES.items()):
        ax.hist(returns[column], bins=BINS, color=color, edgecolor="#fcfcfb", linewidth=0.3)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Daily return")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[0].set_ylabel("Number of days")

    save(fig, "returns_histograms.png", PLOT_FOLDER)


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

    save(fig, "returns_vs_normal.png", PLOT_FOLDER)


def fit_models(series):
    return {
        "Normal": norm(*norm.fit(series)),
        "Student's t": student_t(*student_t.fit(series)),
    }


def compare_models(series, models, tail=0.10):
    rows = {"Actual": {"Excess kurtosis": series.kurt(), f"Days beyond Â±{tail:.0%}": (series.abs() > tail).sum()}}

    for name, model in models.items():
        log_likelihood = model.logpdf(series).sum()
        rows[name] = {
            # A t distribution's kurtosis is infinite when its degrees of freedom are 4 or less
            "Excess kurtosis": float(model.stats(moments="k")),
            f"Days beyond Â±{tail:.0%}": len(series) * (model.cdf(-tail) + model.sf(tail)),
            "Log-likelihood": log_likelihood,
            "AIC": 2 * len(model.args) - 2 * log_likelihood,
        }

    return pd.DataFrame(rows).T


def plot_normal_vs_t(column, models, summary, filename):
    # Same layout as plot_histograms_vs_normal, with the log scale row showing the tails
    series = returns[column]
    label, color = SERIES[column]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey="row", layout="constrained")
    fig.suptitle(f"USO {label.lower()}: normal vs Student's t fit")
    x = np.linspace(BINS[0], BINS[-1], 1000)

    for col, (name, model) in enumerate(models.items()):
        for ax in axes[:, col]:
            ax.hist(series, bins=BINS, density=True, color=color, edgecolor="#fcfcfb", linewidth=0.3, label=label)
            ax.plot(x, model.pdf(x), color=INK, linewidth=1.5, label=f"{name} fit")

        model_kurtosis = summary.loc[name, "Excess kurtosis"]
        title = f"{name} (df = {model.args[0]:.2f})" if name == "Student's t" else name
        axes[0, col].set_title(title, loc="left")
        axes[0, col].text(
            0.03, 0.95,
            f"Excess kurtosis\nData: {series.kurt():.2f}\n"
            f"Model: {'infinite' if np.isinf(model_kurtosis) else f'{model_kurtosis:.2f}'}\n"
            f"AIC: {summary.loc[name, 'AIC']:,.0f}",
            transform=axes[0, col].transAxes,
            va="top",
            color=INK,
        )
        axes[0, col].legend(loc="upper right", frameon=False)

        axes[1, col].text(
            0.03, 0.95,
            f"Days beyond Â±10%\nData: {summary.loc['Actual', 'Days beyond Â±10%']:.0f}\n"
            f"Model: {summary.loc[name, 'Days beyond Â±10%']:.1f}",
            transform=axes[1, col].transAxes,
            va="top",
            color=INK,
        )
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(0.03, 40)
        axes[1, col].set_xlabel("Daily return")
        axes[1, col].xaxis.set_major_formatter(PercentFormatter(xmax=1))

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density (log scale)")

    save(fig, filename, PLOT_FOLDER)


if __name__ == "__main__":
    print(uso.head())
    print(returns.describe())
    print(returns.agg(["skew", "kurt"]))

    style_charts()
    plot_returns_over_time()
    plot_histograms()
    plot_histograms_vs_normal()

    for column, filename in [("Simple_Returns", "normal_vs_t.png"), ("Log_Returns", "normal_vs_t_log.png")]:
        models = fit_models(returns[column])
        summary = compare_models(returns[column], models)
        print(SERIES[column][0])
        print(summary)
        plot_normal_vs_t(column, models, summary, filename)

    plt.show()
