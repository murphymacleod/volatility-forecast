"""
Download USO's price history and current option chain for building the vol forecaster.

Run it from the project folder:

    .\\venv\\Scripts\\python.exe data_download.py

This saves two CSVs, replacing any earlier copies:

    etf_data/USO_daily_prices.csv       One row per trading day since 2016
    options_data/USO_option_chain.csv   One row per contract, as of this run

The results are also available in this script as the uso_prices and uso_options
DataFrames.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from coins_data.alpaca import _get_all_pages, get_current_price

TICKER = "USO"

PROJECT_DIR = Path(__file__).resolve().parent
PRICES_PATH = PROJECT_DIR / "etf_data" / f"{TICKER}_daily_prices.csv"
OPTIONS_PATH = PROJECT_DIR / "options_data" / f"{TICKER}_option_chain.csv"

DATA_URL = "https://data.alpaca.markets"
# Open interest comes from the trading API, which has separate paper and live hosts.
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"

# The earliest date Alpaca has for this data.
PRICE_START = "2016-01-01"
# Stop at yesterday so a day still being traded never ends up in the history.
PRICE_END = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def get_daily_bars(adjustment: str) -> pd.DataFrame:
    """
    Fetch USO's daily bars from SIP, the feed covering all US exchanges.

    The team's get_price_bars_df uses IEX, which covers one exchange and only has
    USO from mid-2020, so it would miss the 2020 oil crash.
    """
    pages = _get_all_pages(
        f"{DATA_URL}/v2/stocks/{TICKER}/bars",
        params={
            "timeframe": "1Day",
            "start": PRICE_START,
            "end": PRICE_END,
            "limit": 10000,
            "adjustment": adjustment,
            "feed": "sip",
            "sort": "asc",
        },
    )
    bars = pd.DataFrame([bar for page in pages for bar in (page.get("bars") or [])])
    bars["Date"] = pd.to_datetime(bars["t"]).dt.tz_localize(None).dt.normalize()
    return bars


def get_open_interest() -> dict[str, dict]:
    """Return open interest (contracts currently held open) for each USO contract."""
    # Without a far-off expiration_date_lte, Alpaca returns only the nearest expiries.
    params = {"underlying_symbols": TICKER, "expiration_date_lte": "2099-12-31", "limit": 1000}
    try:
        pages = _get_all_pages(f"{PAPER_TRADING_URL}/v2/options/contracts", params=params)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 401:
            raise
        # Live account keys are rejected by the paper host, so try the live one.
        pages = _get_all_pages(f"{LIVE_TRADING_URL}/v2/options/contracts", params=params)

    return {
        contract["symbol"]: contract
        for page in pages
        for contract in (page.get("option_contracts") or [])
    }


# ----- Daily price history -----

# USO did a 1-for-8 reverse split on 2020-04-29: one share became eight times the
# price with no real gain. Split-adjusted prices remove that false jump from the
# returns, so they are the ones to use for volatility.
adjusted = get_daily_bars("split")
raw = get_daily_bars("raw")

uso_prices = pd.DataFrame({
    "Date": adjusted["Date"],
    "Open_Price": adjusted["o"],
    "High_Price": adjusted["h"],
    "Low_Price": adjusted["l"],
    "Closing_Price": adjusted["c"],
    "Volume": adjusted["v"],
    "VWAP": adjusted["vw"],
    "Trade_Count": adjusted["n"],
    "Log_Returns": np.log(adjusted["c"]).diff(),
})
# The price as actually quoted that day, which is what old option strikes compare to.
uso_prices = uso_prices.merge(
    raw[["Date", "c"]].rename(columns={"c": "Raw_Closing_Price"}),
    on="Date",
    how="left",
    validate="one_to_one",
)

# ----- Current option chain -----

pages = _get_all_pages(
    f"{DATA_URL}/v1beta1/options/snapshots/{TICKER}",
    params={"feed": "indicative", "limit": 1000},
)
open_interest = get_open_interest()

rows = []
for page in pages:
    for contract, snap in (page.get("snapshots") or {}).items():
        quote = snap.get("latestQuote") or {}
        trade = snap.get("latestTrade") or {}
        daily_bar = snap.get("dailyBar") or {}
        greeks = snap.get("greeks") or {}
        interest = open_interest.get(contract) or {}
        rows.append({
            "contract": contract,
            # An OCC symbol ends in YYMMDD + C or P + the strike times 1000.
            "expiry": contract[-15:-9],
            "type": "call" if contract[-9] == "C" else "put",
            "strike": int(contract[-8:]) / 1000,
            "bid": quote.get("bp"),
            "bid_size": quote.get("bs"),
            "ask": quote.get("ap"),
            "ask_size": quote.get("as"),
            "quote_time": quote.get("t"),
            "last_price": trade.get("p"),
            "last_size": trade.get("s"),
            "last_trade_time": trade.get("t"),
            # A thin contract's latest bar can be from an earlier day, so check this
            # date before treating volume as today's.
            "bar_date": (daily_bar.get("t") or "")[:10] or None,
            "high": daily_bar.get("h"),
            "low": daily_bar.get("l"),
            "volume": daily_bar.get("v"),
            "trade_count": daily_bar.get("n"),
            "open_interest": interest.get("open_interest"),
            "open_interest_date": interest.get("open_interest_date"),
            "iv": snap.get("impliedVolatility"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "rho": greeks.get("rho"),
        })

uso_options = pd.DataFrame(rows)
uso_options["expiry"] = pd.to_datetime(uso_options["expiry"], format="%y%m%d")
uso_options["quote_time"] = pd.to_datetime(uso_options["quote_time"], format="ISO8601", utc=True)
uso_options["last_trade_time"] = pd.to_datetime(uso_options["last_trade_time"], format="ISO8601", utc=True)
uso_options["bar_date"] = pd.to_datetime(uso_options["bar_date"])
uso_options["open_interest_date"] = pd.to_datetime(uso_options["open_interest_date"])

# Whole-number columns would otherwise be saved as 4.0 whenever some rows are blank.
for column in ["bid_size", "ask_size", "last_size", "volume", "trade_count", "open_interest"]:
    uso_options[column] = pd.to_numeric(uso_options[column]).astype("Int64")

# Record when the chain was captured and where USO was trading at that moment,
# since implied volatility and moneyness both depend on the ETF's price.
uso_options.insert(0, "snapshot_time", uso_options["quote_time"].max())
uso_options.insert(1, "underlying_price", get_current_price(TICKER))
uso_options = uso_options.sort_values(["expiry", "type", "strike"]).reset_index(drop=True)

# ----- Save -----

PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
OPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
uso_prices.to_csv(PRICES_PATH, index=False)
uso_options.to_csv(OPTIONS_PATH, index=False)

print(f"uso_prices: {len(uso_prices)} days, {uso_prices['Date'].min():%Y-%m-%d} to {uso_prices['Date'].max():%Y-%m-%d}")
print(f"  saved to {PRICES_PATH}")
print(f"uso_options: {len(uso_options)} contracts across {uso_options['expiry'].nunique()} expiries")
print(f"  saved to {OPTIONS_PATH}")
