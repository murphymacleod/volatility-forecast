# Daily Option Chain Data

Created by `download_options.py`, which saves each ticker's full option chain once per trading day.
Alpaca has no historical option quotes, so these files are the only record of these prices.

## Layout

    data/options/<TICKER>/<TICKER>_<YYYY-MM>.csv

There is one file per ticker per month, with one row per contract per trading day.
Load all months for a ticker as a single DataFrame with:

    from download_options import load_options
    uso = load_options("USO")

## Columns

| Column | Meaning |
|---|---|
| `snapshot_date` | Trading day the row belongs to |
| `underlying_close` | ETF's closing price that day (all exchanges, not split-adjusted) |
| `contract` | Full OCC option symbol |
| `root` | Symbol prefix. If it differs from the ticker (like `WEAT1`), see the warning below |
| `expiry`, `type`, `strike` | Expiration date, `call` or `put`, and strike price |
| `bid`, `ask` | Latest quote at the time of the download |
| `bid_size`, `ask_size` | Number of contracts offered at the bid and ask |
| `quote_time` | When that quote was made (UTC) |
| `last_price`, `last_size`, `last_trade_time` | Most recent trade, which may be from days earlier (UTC) |
| `bar_date` | Date of the bar in the next six columns |
| `open`, `high`, `low`, `close`, `vwap` | Trade prices for `bar_date` |
| `volume`, `trade_count` | Contracts and trades for `bar_date` |
| `open_interest` | Contracts currently held open. Blank when Alpaca reports none, which is common on thin contracts |
| `open_interest_date` | Date that open interest was reported, which can be a few days before `snapshot_date` |
| `iv` | Implied volatility from Alpaca, annualized as a decimal (0.50 = 50%) |
| `delta`, `gamma`, `theta`, `vega`, `rho` | Greeks from Alpaca |

## Before using this data

- **Check `bar_date` before using volume.** Alpaca sends a contract's most recent bar, which can be from an
  earlier day. Volume only applies to `snapshot_date` when `bar_date` equals `snapshot_date`, and a blank
  bar means Alpaca returned no recent bar for that contract.
- **Check `quote_time` for stale quotes.** It can be well before the close on thin contracts.
- **Quotes are from Alpaca's `indicative` feed.** They are delayed and approximate, not exact exchange quotes.
- **Exclude contracts where `root` differs from the ticker** unless you handle them specifically. These usually
  come from past corporate actions, so one contract may not represent the usual 100 shares of the ETF.
- **Some contracts have no `iv` or Greeks.** Those cells are blank.
