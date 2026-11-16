"""
Walk-forward backtest of every vol model on one ticker, with a leaderboard.

Run from the project folder:

    .\\venv\\Scripts\\python.exe backtests\\run_backtest.py

Results go to results/<RUN_NAME>/. To test new models, add them to models() below.

The last stretch of data, from HOLDOUT_START, is held back. Compare and tune models
without it, then run once with --holdout on your final shortlist. Scoring many variants
on the same data makes some look good by luck, and the held-back data checks for that.
"""

import argparse
import time

from volforecast.backtest import run_backtest
from volforecast.data import RESULTS_DIR, load_prices
from volforecast.forecasters import (
    EWMA,
    GarchForecaster,
    HistoricVol,
    RandomWalk,
    RollingGK,
    RollingVol,
    garch_grid,
)
from volforecast.report import write_report
from volforecast.scoring import score_all
from volforecast.targets import gk_scale, model_inputs, realized_targets

RUN_NAME = "uso_garch_grid"
TICKER = "USO"

# Three years of data come first, so every model has history to fit on, and 2020 is in the test
TEST_START = "2019-01-02"
HOLDOUT_START = "2025-11-03"

HORIZON = 20
SCORED_HORIZONS = [1, 5, 10, 15, 20]
BENCHMARK = "Rolling 20d"


def models():
    baselines = [
        HistoricVol(),
        RollingVol(10),
        RollingVol(20),
        RollingVol(60),
        EWMA(0.94),
        RollingGK(20),
        RandomWalk(),
    ]
    garch_specs = garch_grid(
        vol=["GARCH", "GJR"],
        dist=["normal", "t", "skewt"],
        window=[None, 1000],
    )
    return baselines + [GarchForecaster(spec) for spec in garch_specs]


def main(include_holdout):
    prices = load_prices(TICKER)
    inputs = model_inputs(prices)

    # Without --holdout, nothing from the held-back period is forecast or scored.
    # With it, only the held-back period is, as a final out-of-sample check.
    start = HOLDOUT_START if include_holdout else TEST_START
    end = None if include_holdout else inputs.index[inputs.index < HOLDOUT_START][-1]
    scored_inputs = inputs if include_holdout else inputs.loc[:end]

    print(f"Forecasting {TICKER} from {start} to {(end or inputs.index[-1]):%Y-%m-%d}")
    started = time.perf_counter()
    result = run_backtest(inputs, models(), start=start, end=end, horizon=HORIZON)

    # Targets only use the scored period, so 20-day windows can't reach into held-back data
    scale = gk_scale(scored_inputs.loc[start:])
    targets = realized_targets(scored_inputs, SCORED_HORIZONS, scale=scale)
    board, tests, by_year, by_regime, without_2020 = score_all(result.forecasts, targets, inputs, BENCHMARK, SCORED_HORIZONS)

    run_name = RUN_NAME + ("_holdout" if include_holdout else "")
    info = {
        "run_name": run_name,
        "ticker": TICKER,
        "data_end": f"{inputs.index[-1]:%Y-%m-%d}",
        "test_start": start,
        "scored_end": f"{scored_inputs.index[-1]:%Y-%m-%d}",
        "holdout_start": HOLDOUT_START,
        "include_holdout": include_holdout,
        "benchmark": BENCHMARK,
        "horizons": SCORED_HORIZONS,
        "gk_scale": scale,
        "models": list(result.timings),
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    write_report(RESULTS_DIR / run_name, result, targets, board, tests, by_year, by_regime, without_2020, info)

    print(f"\nDone in {info['minutes']} minutes. Report: results/{run_name}/leaderboard.md\n")
    print(board[["Place", "Average rank", "Tests in MCS", "20d QLIKE vs benchmark", "20d DM p"]].round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", action="store_true", help="also forecast and score the held-back period")
    main(parser.parse_args().holdout)
