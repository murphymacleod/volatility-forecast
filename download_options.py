"""
Save today's full option chain for each club ETF to local CSV files.

Alpaca does not offer historical option quotes, so a day that is not saved here
can never be downloaded later. Run this once per trading day, after 4:15pm ET:

    .\\venv\\Scripts\\python.exe download_options.py

Files are saved to data/options/<TICKER>/<TICKER>_<YYYY-MM>.csv, one file per
ticker per month, because USO alone would pass GitHub's 100 MB file limit in
about four months as a single file. Use load_options() to get every month back
as one DataFrame. See data/options/README.md for what each column means.

Running it twice for the same trading day is safe: the newer download replaces
that day's rows instead of adding duplicates.
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from coins_data.alpaca import _get_all_pages

TICKERS = ["WEAT", "CORN", "SOYB", "USO", "BNO", "UGA", "UNG", "IAU", "SLV", "CPER"]

OPTIONS_DIR = Path(__file__).resolve().parent / "data" / "options"

DATA_URL = "https://data.alpaca.markets"
# Open interest comes from the trading API, which has separate paper and live hosts.
PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"

NEW_YORK = "America/New_York"

# Option prices on this feed are delayed about 15 minutes, so the final prices of
# the day are not available until 15 minutes after the 4:00pm close.
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
FINAL_PRICES_READY = pd.Timedelta(hours=16, minutes=15)

DATE_COLUMNS = ["snapshot_date", "expiry", "quote_time", "last_trade_time", "bar_date", "open_interest_date"]


def market_may_be_open(now: pd.Timestamp) -> bool:
    """Return True if a download now would capture a day that is still trading."""
    if now.weekday() >= 5:
        return False
    time_of_day = now - now.normalize()
    return MARKET_OPEN <= time_of_day < FINAL_PRICES_READY


def _get_open_interest(ticker: str) -> dict[str, dict]:
    """
    Return open interest (contracts currently held open) for each contract.

    Open interest helps separate real markets from thin ones. It is only reported
    once a day and lags, so open_interest_date can be a few days before the snapshot.
    """
    # Without a far-off expiration_date_lte, Alpaca returns only the nearest expiries.
    params = {"underlying_symbols": ticker, "expiration_date_lte": "2099-12-31", "limit": 1000}
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


def _get_underlying_close(ticker: str, snapshot_date: str) -> float:
    """
    Return the ETF's closing price on snapshot_date, which IV calculations need.

    This uses the full-market SIP feed because thin ETFs can go days without a
    trade on IEX. Prices are raw, not split-adjusted, to match the option strikes.
    """
    # Free accounts cannot request SIP data from the last 15 minutes.
    latest_allowed = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    pages = _get_all_pages(
        f"{DATA_URL}/v2/stocks/{ticker}/bars",
        params={
            "timeframe": "1Day",
            "start": snapshot_date,
            "end": latest_allowed.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": "sip",
            "adjustment": "raw",
        },
    )
    for page in pages:
        for bar in page.get("bars") or []:
            if bar["t"][:10] == snapshot_date:
                return float(bar["c"])
    raise ValueError(f"No {ticker} closing price found for {snapshot_date}")


def download_snapshot(ticker: str) -> pd.DataFrame:
    """Download a ticker's full option chain as one row per contract."""
    pages = _get_all_pages(
        f"{DATA_URL}/v1beta1/options/snapshots/{ticker}",
        params={"feed": "indicative", "limit": 1000},
    )

    try:
        open_interest = _get_open_interest(ticker)
    except requests.RequestException as exc:
        # Prices can't be recovered later, so still save them without open interest.
        print(f"  WARNING: {ticker} open interest unavailable, leaving it blank: {exc}")
        open_interest = {}

    rows = []
    for page in pages:
        for contract, snap in (page.get("snapshots") or {}).items():
            quote = snap.get("latestQuote") or {}
            trade = snap.get("latestTrade") or {}
            bar = snap.get("dailyBar") or {}
            greeks = snap.get("greeks") or {}
            interest = open_interest.get(contract) or {}
            rows.append({
                "contract": contract,
                # An OCC symbol ends in YYMMDD + C or P + the strike times 1000.
                "root": contract[:-15],
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
                "bar_date": bar.get("t"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "vwap": bar.get("vw"),
                "volume": bar.get("v"),
                "trade_count": bar.get("n"),
                "open_interest": interest.get("open_interest"),
                "open_interest_date": interest.get("open_interest_date"),
                "iv": snap.get("impliedVolatility"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "rho": greeks.get("rho"),
            })

    if not rows:
        raise ValueError(f"Alpaca returned no option contracts for {ticker}")

    df = pd.DataFrame(rows)

    # Label the rows with the trading day the quotes came from, not the day this
    # ran, so a Saturday run still counts as Friday's data.
    quote_times = pd.to_datetime(df["quote_time"], format="ISO8601", utc=True)
    snapshot_date = quote_times.max().tz_convert(NEW_YORK).strftime("%Y-%m-%d")
    df.insert(0, "snapshot_date", snapshot_date)
    df.insert(1, "underlying_close", _get_underlying_close(ticker, snapshot_date))

    df["expiry"] = pd.to_datetime(df["expiry"], format="%y%m%d").dt.strftime("%Y-%m-%d")
    df["quote_time"] = quote_times.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["last_trade_time"] = (
        pd.to_datetime(df["last_trade_time"], format="ISO8601", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    df["bar_date"] = df["bar_date"].str[:10]

    # Whole-number columns would otherwise be saved as 4.0 whenever some rows are blank.
    for column in ["bid_size", "ask_size", "last_size", "volume", "trade_count", "open_interest"]:
        df[column] = pd.to_numeric(df[column]).astype("Int64")

    # Six decimals is plenty of precision and keeps the files noticeably smaller.
    for column in ["iv", "delta", "gamma", "theta", "vega", "rho"]:
        df[column] = pd.to_numeric(df[column]).round(6)

    return df.sort_values(["expiry", "type", "strike", "contract"]).reset_index(drop=True)


def save_snapshot(snapshot: pd.DataFrame, ticker: str, output_dir: Path = OPTIONS_DIR) -> Path:
    """Add a snapshot to the ticker's CSV for that month and return the file's path."""
    snapshot_date = snapshot["snapshot_date"].iloc[0]
    path = Path(output_dir) / ticker / f"{ticker}_{snapshot_date[:7]}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Everything is handled as text here so rows already saved are never reformatted,
    # which keeps each day's Git commit limited to the rows that actually changed.
    new_rows = pd.read_csv(io.StringIO(snapshot.to_csv(index=False)), dtype=str, keep_default_na=False)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        kept = existing[existing["snapshot_date"] != snapshot_date]
        new_rows = pd.concat([kept, new_rows], ignore_index=True).fillna("")
        new_rows = new_rows.sort_values("snapshot_date", kind="stable")

    # Writing to a temporary file first means a crash can't damage the existing data.
    temporary_path = path.with_suffix(".csv.tmp")
    new_rows.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)
    return path


def load_options(ticker: str, data_dir: Path = OPTIONS_DIR) -> pd.DataFrame:
    """Load every saved day for a ticker into one DataFrame with real date columns."""
    paths = sorted((Path(data_dir) / ticker).glob(f"{ticker}_*.csv"))
    if not paths:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], format="ISO8601")
    return df


def main() -> int:
    now = pd.Timestamp.now(tz=NEW_YORK)
    if market_may_be_open(now):
        print(f"It is {now:%H:%M} ET, so today's final prices aren't ready. Nothing was saved.")
        print("Run this again after 4:15pm ET.")
        return 1

    failed = []
    for ticker in TICKERS:
        try:
            snapshot = download_snapshot(ticker)
            path = save_snapshot(snapshot, ticker)
            print(f"{ticker}: saved {len(snapshot)} contracts for {snapshot['snapshot_date'].iloc[0]} to {path}")
        # One failed ticker shouldn't cost the rest of the club's data for the day.
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"{ticker}: FAILED: {exc}")
            failed.append(ticker)

    if failed:
        print(f"Failed: {', '.join(failed)}. Run this again before the next market open to fill them in.")
        return 1

    print("All tickers saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
