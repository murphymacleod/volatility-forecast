"""
Every model the backtest can run, behind one interface.

A forecaster takes the model inputs (see targets.model_inputs) and a list of origin dates,
and returns a table of daily variance forecasts: one row per origin, one column per day
ahead (1 to horizon). The forecast in row t may only use rows of the inputs up to and
including t. tests/test_lookahead.py checks this for every forecaster.

To add a model, subclass Forecaster (or FlatForecaster for a single level held flat)
and add it to the model list in backtests/run_backtest.py.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from .garch import SCALE
from .measures import TRADING_DAYS

# 1% annualized vol, as a daily variance
MIN_DAILY_VARIANCE = 0.01**2 / TRADING_DAYS


class Forecaster(ABC):
    name = "unnamed"
    family = "other"

    @abstractmethod
    def forecast(self, inputs, origins, horizon):
        """Daily variance forecasts, as a DataFrame indexed by origin with columns 1..horizon."""

    def details(self):
        """Extra information about the run, such as parameter estimates, for the report."""
        return pd.DataFrame()


class FlatForecaster(Forecaster):
    """A forecaster that predicts one variance level and holds it flat across the horizon."""

    family = "baseline"

    @abstractmethod
    def level(self, inputs):
        """The variance level at each date, using data through that date's close."""

    def forecast(self, inputs, origins, horizon):
        # A zero forecast can't be scored, and happens after an unchanged close, so the level never goes below 1% vol
        level = self.level(inputs).reindex(origins).clip(lower=MIN_DAILY_VARIANCE)
        return pd.DataFrame(np.repeat(level.to_numpy()[:, None], horizon, axis=1), index=origins, columns=range(1, horizon + 1))


class HistoricVol(FlatForecaster):
    """The average squared return over all history so far."""

    name = "Historic vol (expanding)"

    def __init__(self, min_days=250):
        self.min_days = min_days

    def level(self, inputs):
        return inputs["r2"].expanding(self.min_days).mean()


class RollingVol(FlatForecaster):
    """The average squared return over the last `window` days."""

    def __init__(self, window):
        self.window = window
        self.name = f"Rolling {window}d"

    def level(self, inputs):
        return inputs["r2"].rolling(self.window).mean()


class EWMA(FlatForecaster):
    """RiskMetrics exponentially weighted variance: each day's weight shrinks by `lam`."""

    def __init__(self, lam=0.94):
        self.lam = lam
        self.name = f"EWMA {lam}"

    def level(self, inputs):
        return inputs["r2"].ewm(alpha=1 - self.lam, adjust=False).mean()


class RollingGK(FlatForecaster):
    """
    The average Garman-Klass + gap variance over the last `window` days.

    Garman-Klass runs above r² on USO, so it's rescaled by its average ratio to r² over
    all data up to each date. Only past data sets the scale, so this adds no lookahead.
    """

    def __init__(self, window=20):
        self.window = window
        self.name = f"Rolling GK+gap {window}d"

    def level(self, inputs):
        scale = inputs["r2"].expanding(250).mean() / inputs["gk_gap"].expanding(250).mean()
        return inputs["gk_gap"].rolling(self.window).mean() * scale


class RandomWalk(FlatForecaster):
    """Today's squared return, held flat. Expected to lose badly, so it checks the scoring works."""

    name = "Random walk (last r²)"

    def level(self, inputs):
        return inputs["r2"]


@dataclass(frozen=True)
class GarchSpec:
    """
    One GARCH variant. Every option here is something the backtest can compare.

    vol: "GARCH", or "GJR" to let negative returns raise vol more than positive ones
    dist: "normal", "t", "skewt" or "ged"
    mean: "Constant" or "Zero"
    window: None to fit on all history so far, or a number of days for a rolling window
    refit_every: trading days between re-estimating the parameters. Between refits, the
        parameters stay fixed while the vol estimate still updates every day.
    """

    vol: str = "GARCH"
    dist: str = "t"
    mean: str = "Constant"
    p: int = 1
    q: int = 1
    window: int | None = None
    refit_every: int = 21

    @property
    def name(self):
        o = 1 if self.vol == "GJR" else 0
        lags = f"{self.p},{o},{self.q}" if o else f"{self.p},{self.q}"
        window = "expanding" if self.window is None else f"roll{self.window}"
        mean = "" if self.mean == "Constant" else f"-{self.mean.lower()}mean"
        return f"{self.vol}({lags})-{self.dist}{mean}-{window}-refit{self.refit_every}"


class GarchForecaster(Forecaster):
    """
    A GARCH-family model, re-estimated every `refit_every` origins.

    At each refit origin, parameters are estimated on data through that origin's close.
    For the origins until the next refit, those parameters stay fixed, and arch updates the
    variance each day from the returns through that origin only. The model is never given
    data after the last origin in a block, so later prices can't leak into a forecast.

    Forecasts are of E[r²] = variance + mean², because the targets are raw squared returns.
    """

    family = "garch"

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.name
        self._fits = []

    def _model(self, returns):
        return arch_model(
            returns,
            mean=self.spec.mean,
            vol="GARCH",
            p=self.spec.p,
            o=1 if self.spec.vol == "GJR" else 0,
            q=self.spec.q,
            dist=self.spec.dist,
        )

    def forecast(self, inputs, origins, horizon):
        returns = SCALE * inputs["r"]
        positions = inputs.index.get_indexer(origins)
        if (positions < 0).any():
            raise ValueError("Every origin must be a date in the inputs")

        blocks = []
        previous = None
        for start in range(0, len(positions), self.spec.refit_every):
            block = positions[start:start + self.spec.refit_every]
            refit_at = block[0]
            first_obs = 0 if self.spec.window is None else max(0, refit_at + 1 - self.spec.window)

            # The model only holds data through the block's last origin
            model = self._model(returns.iloc[: block[-1] + 1])
            result, converged = self._fit(model, first_obs, refit_at + 1, previous)
            previous = result

            mean = result.params.get("mu", 0.0) / SCALE
            variance = result.forecast(horizon=horizon, start=refit_at, reindex=False).variance / SCALE**2
            variance = variance + mean**2
            variance.columns = range(1, horizon + 1)
            blocks.append(variance)

            self._fits.append({
                "model": self.name,
                "refit_date": inputs.index[refit_at],
                "observations": refit_at + 1 - first_obs,
                "converged": converged,
                **result.params.to_dict(),
            })

        forecasts = pd.concat(blocks)
        forecasts.index = pd.DatetimeIndex(forecasts.index)
        return forecasts.reindex(origins)

    def _fit(self, model, first_obs, last_obs, previous):
        """Fit, and if the optimizer fails, keep the previous parameters rather than use a bad fit."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = model.fit(first_obs=first_obs, last_obs=last_obs, disp="off")
                if result.convergence_flag == 0:
                    return result, True
            except (ValueError, np.linalg.LinAlgError):
                result = None

        if previous is not None:
            return model.fix(previous.params, first_obs=first_obs, last_obs=last_obs), False
        if result is not None:
            return result, False
        raise RuntimeError(f"{self.name} could not be fit at its first refit")

    def details(self):
        return pd.DataFrame(self._fits)


def garch_grid(**options):
    """
    Every combination of the options given, as GarchSpecs.

    For example, garch_grid(vol=["GARCH", "GJR"], dist=["normal", "t"]) gives four specs.
    """
    specs = [{}]
    for field, values in options.items():
        specs = [{**spec, field: value} for spec in specs for value in values]
    return [GarchSpec(**spec) for spec in specs]
