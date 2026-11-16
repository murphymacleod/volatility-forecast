"""Ways of measuring volatility from past prices."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_vol(daily_returns):
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def rolling_vol(daily_returns, window):
    return daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def period_slice(data, years):
    """The last `years` of data, or all of it when years is None."""
    if years is None:
        return data
    return data[data.index > data.index[-1] - pd.DateOffset(years=years)]


def close_to_close_vol(log_returns):
    # A single day's vol is just the size of its return, annualized
    return log_returns.abs() * np.sqrt(TRADING_DAYS)


def garman_klass_gap_variance(prices):
    """
    Each day's variance from its open, high, low and close, plus the overnight gap, as a daily decimal.

    Garman-Klass only sees the session from open to close, so the squared gap from the
    previous close is added back in to cover the full day that close-to-close returns cover.
    """
    high_low = np.log(prices["High_Price"] / prices["Low_Price"])
    open_close = np.log(prices["Closing_Price"] / prices["Open_Price"])
    overnight_gap = np.log(prices["Open_Price"] / prices["Closing_Price"].shift())
    return 0.5 * high_low**2 - (2 * np.log(2) - 1) * open_close**2 + overnight_gap**2


def garman_klass_gap_vol(prices):
    return np.sqrt(garman_klass_gap_variance(prices) * TRADING_DAYS)


def recent_vol_measures(prices, days=20):
    # Everything is calculated on the full history first, then cut down to the days shown
    log_returns = prices["Log_Returns"]
    return pd.DataFrame({
        "Close-to-close": close_to_close_vol(log_returns),
        "Garman-Klass + gap": garman_klass_gap_vol(prices),
        "20-day rolling": rolling_vol(log_returns, 20),
    }).tail(days)
