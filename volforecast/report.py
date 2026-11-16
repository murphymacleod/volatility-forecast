"""Writing a scored backtest to results/<run>/: the leaderboard, detailed tables and charts."""

import json
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import PROJECT_DIR
from .measures import TRADING_DAYS
from .plotting import INK, PALETTE, format_date_axis, save, style_charts
from .scoring import _label, paired_frame, qlike

TOP_MODELS = 5

# Column -> how to show it in the Markdown tables
PERCENT = ["RMSE vol", "MAE vol", "Bias vol"]
PVALUES = ["DM p", "MCS p", "MZ p", "20d DM p"]


def write_report(run_dir, result, targets, board, tests, by_year, by_regime, without_2020, info):
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    board.to_csv(run_dir / "leaderboard.csv")
    for test, table in tests.items():
        name = _label(test).replace(" ", "_").replace("²", "2")
        table.to_csv(tables_dir / f"{name}.csv")
        by_year[test].to_csv(tables_dir / f"{name}_by_year.csv")
        by_regime[test].to_csv(tables_dir / f"{name}_by_regime.csv")
    result.forecasts.to_csv(run_dir / "forecasts.csv.gz")
    if not result.details.empty:
        result.details.to_csv(run_dir / "garch_fits.csv", index=False)

    info = {**info, "timings_seconds": {k: round(v, 1) for k, v in result.timings.items()}, "git_commit": _git_commit()}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    (run_dir / "leaderboard.md").write_text(_markdown_report(board, tests, info), encoding="utf-8")

    style_charts()
    plot_cumulative_loss(run_dir, result, targets, board, info["benchmark"])
    plot_loss_by_horizon(run_dir, tests, board, info["benchmark"])
    plot_forecasts_vs_realized(run_dir, result, targets, board, info["benchmark"])
    plt.close("all")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def _format(value, column):
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "–"
    if column in PERCENT:
        return f"{value:.1%}"
    if column in PVALUES:
        return f"{value:.3f}"
    if isinstance(value, (int, np.integer)) or column in ("Rank", "Place"):
        return f"{int(value)}"
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"


def _markdown_table(table, columns):
    lines = ["| Model | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for model, row in table.iterrows():
        lines.append(f"| {model} | " + " | ".join(_format(row[c], c) for c in columns) + " |")
    return "\n".join(lines)


def _markdown_report(board, tests, info):
    summary_columns = [c for c in board.columns if not c.startswith(("Daily", "5d", "10d", "15d", "20d r²", "20d GK"))
                       or c.startswith(("20d QLIKE", "20d DM", "20d years", "20d without", "20d calm", "20d normal", "20d high"))]
    rank_columns = [c for c in board.columns if c not in summary_columns]
    detail_columns = ["Rank", "QLIKE", "QLIKE vs benchmark", "DM p", "MCS p", "MSE ratio", "RMSE vol", "Bias vol",
                      "Forecast / realized variance", "MZ slope", "MZ R²", "MZ p"]

    sections = [
        f"# Backtest leaderboard: {info['run_name']}",
        "",
        f"- **Data:** {info['ticker']} through {info['data_end']}",
        f"- **Forecast origins scored:** {info['test_start']} to {info['scored_end']}"
        + (" (held-back period excluded)" if not info["include_holdout"] else " (including the held-back period)"),
        f"- **Benchmark:** {info['benchmark']}",
        f"- **Garman-Klass scale factor:** {info['gk_scale']:.3f} (multiplies Garman-Klass + gap so it matches r² on average)",
        "",
        "## How to read this",
        "",
        "- **QLIKE** is the main score, and lower is better. *QLIKE vs benchmark* below 0 means the model beat the benchmark.",
        "- **DM p** is the Diebold-Mariano p-value for that difference. Below 0.05 means it's unlikely to be luck.",
        "- **MCS p** above 0.10 means the model is in the Model Confidence Set: it can't be told apart from the best model.",
        "- **Regime columns** show 20-day QLIKE vs benchmark in calm, normal and high-vol periods. Negative is better.",
        "- **MZ slope** near 1 and **MZ p** above 0.05 mean the forecasts aren't systematically too high, too low or wrongly scaled.",
        "- **Forecast / realized variance** is the fair bias check: 1 means unbiased. *Bias vol* and *RMSE vol* look worse than they are, "
        "especially for the daily tests, because a noisy realized vol averages below the true vol. Use them for a sense of scale, not for ranking.",
        "- **The Model Confidence Set is cautious with this much noise.** Big losses in 2020 make many models impossible to separate, "
        "so being in the set is a low bar. A model that's out of the set is clearly worse.",
        "",
        "## Overall",
        "",
        _markdown_table(board, summary_columns),
        "",
        "## Rank in each test",
        "",
        _markdown_table(board[["Place"] + rank_columns], rank_columns),
    ]
    for test, table in tests.items():
        sections += [
            "",
            f"## {_label(test)} ({table.attrs['origins']} origins, Newey-West lags {table.attrs['lags']})",
            "",
            _markdown_table(table, detail_columns),
        ]
    return "\n".join(sections) + "\n"


def _chart_models(board, benchmark):
    """The top models, plus the benchmark, each with a fixed color."""
    models = [m for m in board.index if m != benchmark][:TOP_MODELS]
    return {model: PALETTE[i] for i, model in enumerate(models)}


def plot_cumulative_loss(run_dir, result, targets, board, benchmark, proxy="r2", horizon=20):
    F, rv = paired_frame(result.forecasts, targets, proxy, horizon)
    losses = qlike(F, rv.to_numpy()[:, None])
    cumulative = losses.sub(losses[benchmark], axis=0).cumsum()

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    fig.suptitle(f"Cumulative QLIKE vs {benchmark}, {horizon}-day r² test: falling lines are beating the benchmark")
    for model, color in _chart_models(board, benchmark).items():
        ax.plot(cumulative.index, cumulative[model], color=color, linewidth=1.5, label=model)
    ax.axhline(0, color=INK, linewidth=1.2, linestyle="--", label=benchmark)
    ax.set_ylabel("Cumulative QLIKE difference")
    format_date_axis(ax)
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "cumulative_qlike.png", run_dir / "plots")


def plot_loss_by_horizon(run_dir, tests, board, benchmark):
    horizons = sorted({h for _, h in tests})
    proxies = [p for p in ("r2", "gk_gap") if (p, horizons[0]) in tests]

    fig, axes = plt.subplots(1, len(proxies), figsize=(12, 5), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    fig.suptitle(f"QLIKE vs {benchmark} by forecast horizon: below 0 beats the benchmark")
    for ax, proxy in zip(axes, proxies):
        for model, color in _chart_models(board, benchmark).items():
            values = [tests[(proxy, h)].loc[model, "QLIKE vs benchmark"] for h in horizons]
            ax.plot(horizons, values, color=color, linewidth=2, marker="o", markersize=6, label=model)
        ax.axhline(0, color=INK, linewidth=1.2, linestyle="--", label=benchmark)
        ax.set_title("Against r²" if proxy == "r2" else "Against scaled Garman-Klass + gap", loc="left")
        ax.set_xlabel("Horizon (trading days)")
        ax.set_xticks(horizons)
    axes[0].set_ylabel("Average QLIKE difference")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "qlike_by_horizon.png", run_dir / "plots")


def plot_forecasts_vs_realized(run_dir, result, targets, board, benchmark, horizon=20):
    F, rv = paired_frame(result.forecasts, targets, "r2", horizon)
    best_garch = next((m for m in board.index if result.families.get(m) == "garch"), None)
    best_baseline = next((m for m in board.index if result.families.get(m) == "baseline" and m != benchmark), None)
    lines = {m: c for m, c in zip([best_garch, best_baseline], PALETTE[:2]) if m is not None}

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    fig.suptitle(f"{horizon}-day vol forecasts vs what was realized over the next {horizon} days, plotted at the forecast date")
    ax.plot(rv.index, np.sqrt(rv), color="#c3c2b7", linewidth=1, label=f"Realized vol, next {horizon} days")
    ax.plot(F.index, np.sqrt(F[benchmark]), color=INK, linewidth=1, linestyle="--", label=benchmark)
    for model, color in lines.items():
        ax.plot(F.index, np.sqrt(F[model]), color=color, linewidth=1.3, label=model)
    ax.set_ylabel("Annualized vol")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    format_date_axis(ax)
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "forecasts_vs_realized.png", run_dir / "plots")
