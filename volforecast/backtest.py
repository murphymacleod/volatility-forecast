"""
The walk-forward engine: run every forecaster over the same origin dates.

It only produces forecasts. Scoring them is scoring.py's job, so forecasts can be saved
once and re-scored as often as needed without refitting anything.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    # Daily variance forecasts, indexed by (model, origin), with columns 1..horizon
    forecasts: pd.DataFrame
    # Seconds each model took
    timings: dict = field(default_factory=dict)
    # Per-model extras, such as GARCH parameter estimates at each refit
    details: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Model name -> family, such as "baseline" or "garch"
    families: dict = field(default_factory=dict)


def run_backtest(inputs, forecasters, start, end=None, horizon=20, verbose=True):
    """
    Forecast from every origin between `start` and `end` (inclusive) with each forecaster.

    Each forecast is checked for the right shape and for impossible values, so a broken
    model fails loudly here instead of quietly skewing the scores.
    """
    origins = inputs.loc[start:end].index
    names = [forecaster.name for forecaster in forecasters]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be unique: {names}")

    forecasts, timings, details, families = {}, {}, [], {}
    for forecaster in forecasters:
        began = time.perf_counter()
        result = forecaster.forecast(inputs, origins, horizon)
        timings[forecaster.name] = time.perf_counter() - began
        families[forecaster.name] = forecaster.family

        _check(forecaster.name, result, origins, horizon)
        forecasts[forecaster.name] = result
        details.append(forecaster.details())
        if verbose:
            print(f"  {forecaster.name}: {timings[forecaster.name]:.1f}s")

    return BacktestResult(
        forecasts=pd.concat(forecasts, names=["model", "origin"]),
        timings=timings,
        details=pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
        families=families,
    )


def _check(name, result, origins, horizon):
    if not result.index.equals(origins):
        raise ValueError(f"{name} returned forecasts for different dates than it was asked for")
    if list(result.columns) != list(range(1, horizon + 1)):
        raise ValueError(f"{name} must return columns 1..{horizon}")
    values = result.to_numpy()
    if (values[~np.isnan(values)] <= 0).any():
        raise ValueError(f"{name} produced a variance of zero or less")
