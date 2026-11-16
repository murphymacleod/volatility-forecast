"""
What forecasts are scored against.

Everything here is a daily variance of log returns in decimal units, so 0.0004 is a
2% daily move. Reports annualize by multiplying by 252. All targets assume a zero mean,
matching the planned target RV(t,h) = sqrt((252/h) * sum of r² over t+1..t+h).
"""

import numpy as np
import pandas as pd

from .measures import garman_klass_gap_variance

# Daily realized measure -> description, used in reports
PROXIES = {
    "r2": "squared log return",
    "gk_gap": "Garman-Klass + overnight gap, scaled to match r² on average",
}


def model_inputs(prices):
    """
    The data every forecaster gets: each day's log return, its square, and its Garman-Klass + gap variance.

    Row t holds values known at day t's close, so a forecast made at origin t may use rows up to and including t.
    """
    log_returns = np.log(prices["Closing_Price"] / prices["Closing_Price"].shift())
    return pd.DataFrame({
        "r": log_returns,
        "r2": log_returns**2,
        "gk_gap": garman_klass_gap_variance(prices),
    }).dropna()


def future_mean(daily, horizon):
    """
    At each origin t, the average of `daily` over the next `horizon` days, t+1 through t+horizon.

    Origins without a full window of later days get NaN, so they can't be scored by accident.
    """
    return daily[::-1].rolling(horizon).mean()[::-1].shift(-1)


def gk_scale(inputs):
    """
    The factor that makes Garman-Klass + gap match r² on average over the days given.

    Range-based estimates run above r² on USO, mostly because prices overshoot and partly
    reverse within the day in turbulent markets. Scaling removes the average gap but not
    the part that varies by regime, which the regime tables show.
    """
    return inputs["r2"].mean() / inputs["gk_gap"].mean()


def realized_targets(inputs, horizons, scale=None):
    """
    Realized variance for every origin and horizon, for each proxy.

    Returns columns named like ("r2", 20): the average daily variance from t+1 to t+20.
    Horizon 1 is the daily test. The Garman-Klass proxy is multiplied by `scale`, which
    defaults to its average ratio to r² over `inputs`.
    """
    scale = gk_scale(inputs) if scale is None else scale
    proxies = {"r2": inputs["r2"], "gk_gap": inputs["gk_gap"] * scale}
    return pd.DataFrame({
        (proxy, horizon): future_mean(series, horizon)
        for proxy, series in proxies.items()
        for horizon in horizons
    })
