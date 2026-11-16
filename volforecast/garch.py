"""Fitting GARCH models with arch, and converting their output to annualized vol."""

import numpy as np
import pandas as pd
from arch import arch_model

from .measures import TRADING_DAYS

# Returns are in percent while fitting, which helps the optimizer converge
SCALE = 100


def fit_garch(log_returns, dist):
    model = arch_model(SCALE * log_returns, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    return model.fit(disp="off")


def persistence(result):
    return result.params["alpha[1]"] + result.params.get("gamma[1]", 0) / 2 + result.params["beta[1]"]


def long_run_vol(result):
    long_run_variance = result.params["omega"] / (1 - persistence(result))
    return np.sqrt(long_run_variance * TRADING_DAYS) / SCALE


def half_life(result):
    return np.log(0.5) / np.log(persistence(result))


def conditional_vol(result):
    # Each date's value only uses returns from before that date, because it's GARCH's forecast for that date
    return result.conditional_volatility * np.sqrt(TRADING_DAYS) / SCALE


def forecast_vols(result, days):
    daily_variance = result.forecast(horizon=days).variance.iloc[-1].to_numpy() / SCALE**2
    horizons = np.arange(1, days + 1)

    # A window's vol averages the daily variances first, because variance is what adds up across days
    return pd.DataFrame({
        "single_day": np.sqrt(daily_variance * TRADING_DAYS),
        "whole_window": np.sqrt(np.cumsum(daily_variance) / horizons * TRADING_DAYS),
    }, index=horizons)


def end_of_day_vol(result):
    """
    GARCH vol using data through each day's close, so it lines up with rolling vol and the daily measures.

    The value made after day t's close is GARCH's forecast for day t+1, so each forecast moves back one day
    to the day whose data it uses. The last day's value is the first step of the forecast. Scoring forecasts
    still has to compare each value with the next day's realized vol, not that same day's.
    """
    next_day = forecast_vols(result, 1)["single_day"].iloc[0]
    vol = conditional_vol(result).shift(-1)
    vol.iloc[-1] = next_day
    return vol
