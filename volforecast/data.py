"""Where the project's files live, and loading the daily price data."""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = PROJECT_DIR / "etf_data"
PLOTS_DIR = PROJECT_DIR / "plots"
RESULTS_DIR = PROJECT_DIR / "results"


def load_prices(ticker="USO"):
    """Daily prices sorted by date, with simple returns added."""
    prices = pd.read_csv(PRICES_DIR / f"{ticker}_daily_prices.csv", parse_dates=["Date"], index_col="Date").sort_index()

    # Uses the split-adjusted close, since Raw_Closing_Price has a false jump at the 2020 reverse split
    prices["Simple_Returns"] = prices["Closing_Price"].pct_change()
    return prices


def load_returns(ticker="USO"):
    """Log and simple returns, without the first day, which has no prior close."""
    return load_prices(ticker)[["Log_Returns", "Simple_Returns"]].dropna()
