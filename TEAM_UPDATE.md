# Volatility Forecast: Team Catch-Up

*Last updated November 3, 2026. This is a snapshot of where things stand, and everything under next steps and roles is a suggestion, so please change whatever doesn't fit.*

## The short version

- **The goal:** give the tool a ticker and a horizon (10 to 20 trading days), and it returns a *distribution* of future realized volatility (RV). The main signal is where implied volatility (IV) sits inside that distribution: near the top suggests options look rich, and near the bottom suggests they look cheap.
- **Where we are:** USO data is ready to use, and we've taken a first look at its return distribution. No forecasting model has been built yet.
- **What we've learned:** USO returns have very fat tails, so a Student's t fits far better than a normal.
- **What's next:** agree on how we define and score a forecast, then fit GARCH and test it against simple baselines.

## Where things stand

| Area | Status | Notes |
|---|---|---|
| USO price history | Done | 2,724 trading days, 2016-01-04 to 2026-11-02 |
| USO option chain | One snapshot | 6,226 contracts, captured at 10:14am ET on 2026-11-03 |
| Daily options download, all 10 tickers | Set up, not yet proven | First real run is scheduled for 6pm ET on 2026-11-03 |
| Return distribution analysis | First pass done | Normal vs Student's t, on returns only |
| Target definition and conventions (Phase 0) | Not started | **Suggested next step** |
| Baselines and walk-forward testing | Not started | |
| GARCH and other models | Not started | Only the formula notes in `GARCHFORMULA` so far |
| IV history | Not started | See [The IV history problem](#the-iv-history-problem) |
| Seasonality and event analysis | Not started | |

The club's tickers are WEAT, CORN, SOYB, USO, BNO, UGA, UNG, IAU, SLV and CPER. We're starting with USO because it has the most liquid options of those we compared, with a near-the-money bid/ask spread of about 7%, versus 15% for UNG and 22% for UGA.

## What's in the repo

| Path | What it is |
|---|---|
| `etf_data/USO_daily_prices.csv` | Daily USO prices. **Start here for modeling** |
| `options_data/USO_option_chain.csv` | The single current USO chain snapshot |
| `data/options/` | Where the daily snapshots for all tickers will build up, with a column guide in its README |
| `data_download.py` | Re-downloads the two USO CSVs above and **overwrites them** |
| `download_options.py`, `run_daily_download.bat` | The daily snapshot downloader and the script that launches it |
| `distributions.py` | The normal vs Student's t analysis |
| `plots/` | Charts from `distributions.py` |

You don't need an Alpaca account to work with the committed CSVs:

```python
import pandas as pd

prices = pd.read_csv("etf_data/USO_daily_prices.csv", parse_dates=["Date"], index_col="Date")
returns = prices["Log_Returns"].dropna()
```

## Things to know before using the data

**Prices**

- Use `Closing_Price` and `Log_Returns` for volatility. They're adjusted for USO's 1-for-8 reverse split on 2020-04-29.
- `Raw_Closing_Price` has a false jump at that split. Only use it to compare against strikes from the time, since old option strikes were quoted in unadjusted prices.
- The prices come from Alpaca's SIP feed, which covers all US exchanges. The `coins_data` function `get_price_bars_df` uses IEX, one exchange at about 1% of USO's volume, with history only from mid-2020. It would miss the 2020 crash and return raw prices with the split in them.
- Other tickers need the same split check. UNG, for example, had 1-for-4 reverse splits on 2018-01-05 and 2024-01-24.

**Options**

- `options_data/USO_option_chain.csv` is from the middle of a trading day, not the close, so don't treat it as an end-of-day price.
- In that snapshot, 18.6% of contracts have no IV or Greeks, and 26.6% have no open interest. Filter before using them.
- Check `bar_date` before trusting volume, because a thin contract's latest bar can be from an earlier day.
- Quote timestamps are current, but trades on Alpaca's `indicative` feed are delayed about 15 minutes.
- USO options are American-style. We haven't checked how Alpaca calculates its IV, so it's worth confirming before relying on those numbers.

**Daily snapshots**

- The scheduled task runs on one person's laptop on weekdays at 6pm and 9pm ET, so the laptop has to be on and logged in by then or soon after.
- **A missed day can't be recovered**, because Alpaca has no historical option quotes.
- Snapshots stay on that laptop until someone commits them.

## What we've learned so far

### Student's t beats normal for USO returns

From `distributions.py`, using simple returns (`plots/normal_vs_t.png`). Excess kurtosis is a measure of tail fatness, where a normal scores 0.

| | Actual data | Normal | Student's t (df = 3.76) |
|---|---|---|---|
| Excess kurtosis | 11.66 | 0 | Infinite (df is 4 or less) |
| Days beyond ±10% | 16 | 0.2 expected | 13.8 expected |
| AIC (lower is better) | | -12,342 | -12,926 |

Some caveats:

- **This fit ignores volatility clustering.** Calm and wild periods mixed together create some of these fat tails on their own. Once GARCH handles the clustering, the t's degrees of freedom will probably rise, perhaps to around 5 to 8, though that's an estimate until we fit it.
- **Our target uses log returns, and this table uses simple returns.** Log returns give the same conclusion (t df = 3.69, AIC -12,920 vs -12,265), but their excess kurtosis is higher at 16.37. We should use log returns from here on.
- **The worst days should stay in the data.** They were -25.3% on 2020-03-09 and -24.9% on 2020-04-21. Those are extreme even for a t, but they're real events the model has to cope with.
- **The choice of distribution matters mostly for tails.** It should change point forecasts of variance fairly little, but it matters a lot for us because our output is a distribution.

### How GARCH is likely to fit in

- Fit GARCH to **daily returns**, not a rolling volatility series. Overlapping windows are autocorrelated and break the likelihood.
- For a multi-day forecast, add up the forecast daily variances over the horizon, then annualize and take the square root. That gives the volatility of the whole window, not just the last day.
- GARCH forecasts drift back toward long-run variance, with a half-life of `ln(0.5) / ln(α+β)`. For example, α+β = 0.98 gives about 34 days, which affects how much a 20-day forecast depends on today's conditions.
- A reasonable starting setup is a 3 to 5 year window (750 to 1,250 days), refit weekly or monthly. It's worth also testing an expanding window, since a 3 to 5 year window will eventually drop 2020 entirely.

### Where IV fits

Standard GARCH doesn't use IV. There are three ways to bring it in:

1. **As the signal:** compare the forecast to IV. IV usually sits above later realized vol, and that gap is called the variance risk premium.
2. **As an input:** add IV as an extra term in the variance equation (GARCH-X).
3. **As a benchmark:** a model that can't beat IV may not be adding much.

### The IV history problem

Testing an IV signal needs IV history, and we have little of it so far:

| Source | History | Cost | Catch |
|---|---|---|---|
| **OVX** (FRED: `OVXCLS`) | Back to 2007 | Free | Fixed at 30 days, and it reflects the whole smile, not just at-the-money options. We haven't pulled it yet |
| **Alpaca option bars** | From Jan 2024 | Free | Based on trades, not quotes. Only about 35 non-overlapping 20-day windows |
| **Our daily snapshots** | Starting now | Free | Takes a long time to build up |
| **OptionMetrics, ORATS, Cboe DataShop, Polygon, Databento** | Varies | Usually paid | Worth asking whether VT has OptionMetrics through WRDS |

OVX is computed from USO options, and 30 calendar days is about 21 trading days. That makes it a good match for our 20-day horizon but a weaker one for 10 days, since crude's volatility term structure can invert under stress. A 10-day comparison against OVX mixes model error with that mismatch.

One suggestion is to test in two tiers: price-based models on 2016 onward, and IV-based models on the shorter period where IV exists.

Our own constant-maturity IV would follow these steps:

1. Clean the chain.
2. Find at-the-money using the forward price.
3. Interpolate between expiries in **total variance** (σ²T), not IV:

   `σ²(T*) = [(T₂−T*)·σ₁²T₁ + (T*−T₁)·σ₂²T₂] / [(T₂−T₁)·T*]`

## Suggested next steps

Roughly in order, though steps 3 and 4 can run side by side:

1. **Write down our conventions and the target.** Add a short conventions doc and a `targets.py` with tests you can check by hand. The planned target, using log returns, is:

   `RV(t,h) = sqrt( (252/h) * sum_{i=1..h} r_{t+i}^2 )`

   We'd model its log at h = 10, 15 and 20. Doing this first means all of us are scoring against the same number.
2. **Build the baselines and a walk-forward test harness.** Baselines are trailing 10-day and 20-day vol, EWMA (λ = 0.94), HAR-RV and OVX. Everything runs through one harness, so results are comparable.
3. **Fit GARCH-normal and GARCH-t on USO.** Compare log-likelihood, AIC and BIC, and QQ-plot the standardized residuals. Try skewed-t if they look skewed. The `arch` library handles this with `dist='normal'`, `'t'` or `'skewt'`.
4. **Pull OVX from FRED** and do a first variance risk premium comparison.
5. **Produce the full distribution.** Simulate GARCH-t price paths, calculate each path's RV, and read off the percentiles. Compare that to the HAR regression route from the project plan.
6. **Then try the extensions:** GARCH-X with OVX, GJR or EGARCH, HMM regimes, seasonality and events.

The full plan is several weeks of work. A version covering Phases 0 to 3 plus a HAR model with IV would already be a defensible tool, so that makes a sensible first milestone.

**Small cleanups:**

- Add a `requirements.txt`, because the repo doesn't have one yet.
- Install `arch`, which isn't in the current environment. We also haven't tested it on Python 3.14, so we may want to agree on a Python version.
- Correct the line in `data/options/README.md` that calls quotes delayed. Only trades are.
- Consider adding `.idea/` and `.vscode/` to `.gitignore`.

## Things to think about

**Modeling**

- **How do we score a distribution?** QLIKE and Mincer-Zarnowitz only judge point forecasts. Since the output is a distribution, we also need pinball loss, PIT histograms and interval coverage (does an 80% band contain realized vol about 80% of the time?). A Kupiec VaR test checks daily return tails, which is useful but a different question.
- **Lookahead is the biggest risk.** Every input has to be something we'd actually have known on that date. Common traps are smoothed HMM probabilities, seasonal adjustments using full-sample averages, and scaling with full-sample statistics.
- **Overlapping targets inflate significance.** Daily 20-day forecasts share most of their data, so we need an embargo gap between training and test data, plus Newey-West standard errors. 2,724 days is far fewer independent observations than it sounds.
- **Is 2020 a different regime?** From memory, USO changed which futures it holds around April to May 2020, on top of the split. We haven't verified this, so it needs confirming before treating pre-2020 and post-2020 returns as the same process.
- **Do inventory reports need special handling?** Plain GARCH tends to underreact to jumps around EIA reports and OPEC meetings.

**IV and thin markets**

- **A wide quote can look like a signal.** IV estimates on thin contracts can be distorted, so a "mispricing" may just be a wide spread. The tool should flag low-quality IV days and fall back to non-IV inputs.
- **One snapshot isn't evidence.** The near-the-money 25 to 35 day IV was about 58%, versus about 38% for trailing 20-day RV. It's an interesting gap, but it comes from a single moment and is compared with a *past* window.
- **Timing has to match.** Comparing an option's price with an ETF price from a different time gives wrong IV. Some `coins_data` examples, such as `hist_iv.py` and `atm_iv.py`, mix an option's VWAP with the underlying's close, so be careful reusing them.

**Data and process**

- **Where should daily snapshots live?** All 10 tickers come to about 3.4 MB a day, or roughly 860 MB a year uncompressed. Committing that to Git gets unwieldy, and the monthly files keep it under GitHub's 100 MB per-file limit but don't solve overall repo growth. Compression, Parquet or a shared drive are all options.
- **Who backs up the download?** A second machine running the same download would protect against a laptop being off, since missed days are gone for good.
- **How do we handle shared data changes?** `data_download.py` overwrites the shared CSVs, so if each of us re-runs it, our results will drift apart. We could agree on one person or a set schedule for refreshing the committed data.
- **What will other tools need?** Position sizing, limit pricing and the options parity strategy are all expected to use this tool, so it's worth keeping `vol_forecast(ticker, horizon)` and the evaluation harness reusable.

## If we split the work three ways

This is just one possible way to divide things, so that we each own something that moves the project forward. Please swap, combine or rewrite these based on what you're interested in or want to learn. Nobody is signed up for anything by this document, and it would make sense to talk it through as a group.

### Possible role: Data and IV

*Could be a good fit if you enjoy data pipelines and want to learn how options markets work.*

- Keep the daily snapshots healthy, and decide where they're stored.
- Pull OVX and look into OptionMetrics access.
- Build chain cleaning and the constant-maturity IV series.
- Work through split and data checks as we add tickers beyond USO.

### Possible role: Models

*Could be a good fit if you like statistics and want to go deep on volatility models.*

- Fit and diagnose GARCH-normal and GARCH-t.
- Simulate paths to get the forecast distribution.
- Try HAR regressions, GJR or EGARCH, GARCH-X and later HMMs.
- Keep track of which models are worth continuing with.

### Possible role: Evaluation and tool

*Could be a good fit if you care about rigorous testing and building something others can use.*

- Write `targets.py` and its tests.
- Build the walk-forward harness, scoring metrics and lookahead checks.
- Keep the baselines as the bar every model has to beat.
- Shape the final `vol_forecast(ticker, horizon)` output and its plots.

### Working across roles

Some things would probably work best together, whichever split we choose:

- **Conventions doc:** we should all agree on this, since all our work depends on it.
- **Handoffs:** Data supplies clean inputs, Models produces forecasts, and Evaluation scores them. Agreeing early on one simple forecast format (date, horizon, percentiles) would let everyone work in parallel.
- **Code review:** having someone from another role look over each other's work is a good way to catch lookahead bugs and for all of us to learn the whole project.

## If you want to re-download data

You only need this to refresh the CSVs. Everything else works from the committed files.

1. Create a virtual environment and install `coins_data` from a local clone of `coins-data`, plus `scipy`, which its dependencies leave out.
2. Set the environment variables `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. The `coins-data` guide lists different names, but these are the ones the code reads. Keep your keys out of the repo.
3. Run `data_download.py` with the venv's Python, not your system one, and restart your terminal or app if it can't see newly saved keys.

Alpaca's free plan can't query the most recent 15 minutes of data. Also, `coins_data` installed this way doesn't update itself, so pull and reinstall to get changes.
