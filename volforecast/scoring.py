"""
Scoring forecasts against realized variance, and comparing models fairly.

A "test" is one realized measure at one horizon, such as ("r2", 20): the average squared
return over the next 20 days. Horizon 1 is the daily test. Within a test, every model is
scored on exactly the same origins, so no model gets an easier sample.

All variances are annualized here (daily variance × 252) so the numbers read like vol².
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS

from .measures import TRADING_DAYS


def qlike(forecast, realized):
    """
    QLIKE loss: log(F) + RV / F. Lower is better.

    It's the main ranking metric because it ranks models correctly even when the realized
    measure is a noisy estimate of true variance, as long as that measure is unbiased.
    It also works when realized variance is zero, which happens on unchanged-close days.
    """
    return np.log(forecast) + realized / forecast


def newey_west_lags(n, horizon):
    """Lags for autocorrelation-robust errors: at least horizon - 1, because overlapping windows share days."""
    return max(horizon - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def paired_frame(forecasts, targets, proxy, horizon):
    """
    Forecasts and realized values for one test, on the origins where every model and the target exist.

    Returns (F, rv): F has one column per model, and both are annualized variances.
    """
    window_mean = forecasts[list(range(1, horizon + 1))].mean(axis=1)
    F = window_mean.unstack("model") * TRADING_DAYS
    rv = targets[(proxy, horizon)].reindex(F.index) * TRADING_DAYS
    keep = F.notna().all(axis=1) & rv.notna()
    return F[keep], rv[keep]


def _hac_mean(series, lags):
    """Mean of a series with a Newey-West standard error, as (mean, t-stat, p-value)."""
    fit = sm.OLS(series.to_numpy(), np.ones(len(series))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.params[0], fit.tvalues[0], fit.pvalues[0]


def mincer_zarnowitz(forecast, realized, lags):
    """
    Regress realized on forecast. A perfect forecast has intercept 0, slope 1.

    Returns the intercept, slope, R² and the p-value for both being (0, 1) together.
    A small p-value means the forecasts are biased or wrongly scaled. A forecast that never
    changes has no slope to estimate, so it gets NaN.
    """
    if forecast.std() < 1e-12 * max(abs(forecast.mean()), 1e-12):
        return np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(forecast.to_numpy(), has_constant="add")
    fit = sm.OLS(realized.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joint = fit.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True)
    return fit.params[0], fit.params[1], fit.rsquared, float(joint.pvalue)


def model_confidence_set(losses, horizon, size=0.10, reps=1000, seed=7):
    """
    p-values from Hansen's Model Confidence Set.

    Models with a p-value above `size` can't be statistically told apart from the best
    model, which is the honest way to pick winners from many similar variants. The
    bootstrap uses blocks so overlapping windows keep their dependence.
    """
    mcs = MCS(losses, size=size, reps=reps, block_size=max(horizon, 10), method="R", seed=seed)
    mcs.compute()
    return mcs.pvalues["Pvalue"]


def score_one(F, rv, benchmark, horizon):
    """Every metric for one test, one row per model."""
    losses = qlike(F, rv.to_numpy()[:, None])
    lags = newey_west_lags(len(rv), horizon)
    mcs_pvalues = model_confidence_set(losses, horizon)
    vol_forecast, vol_realized = np.sqrt(F), np.sqrt(rv)

    rows = {}
    for model in F.columns:
        difference = losses[model] - losses[benchmark]
        if model == benchmark:
            dm_t, dm_p = np.nan, np.nan
        else:
            _, dm_t, dm_p = _hac_mean(difference, lags)
        intercept, slope, r_squared, mz_p = mincer_zarnowitz(F[model], rv, lags)
        vol_error = vol_forecast[model] - vol_realized

        rows[model] = {
            "QLIKE": losses[model].mean(),
            "QLIKE vs benchmark": difference.mean(),
            "DM t": dm_t,
            "DM p": dm_p,
            "MCS p": mcs_pvalues[model],
            "MSE ratio": ((F[model] - rv) ** 2).mean() / ((F[benchmark] - rv) ** 2).mean(),
            "RMSE vol": np.sqrt((vol_error**2).mean()),
            "MAE vol": vol_error.abs().mean(),
            "Bias vol": vol_error.mean(),
            "Forecast / realized variance": F[model].mean() / rv.mean(),
            "MZ intercept": intercept,
            "MZ slope": slope,
            "MZ R²": r_squared,
            "MZ p": mz_p,
        }

    table = pd.DataFrame(rows).T.sort_values("QLIKE")
    table.insert(0, "Rank", range(1, len(table) + 1))
    table.attrs.update(origins=len(rv), lags=lags)
    return table


def origin_groups(inputs, origins):
    """
    Labels for splitting origins into subperiods: calendar year, and the vol regime at the time.

    The regime uses trailing 20-day vol known at each origin, split into thirds over the test.
    """
    trailing = np.sqrt(inputs["r2"].rolling(20).mean() * TRADING_DAYS).reindex(origins)
    regime = pd.qcut(trailing, 3, labels=["Calm", "Normal", "High vol"])
    return pd.DataFrame({"year": origins.year, "regime": regime.astype(str)}, index=origins)


def subperiod_scores(F, rv, groups, benchmark):
    """
    Average QLIKE vs the benchmark within each year and regime, plus all data without 2020.

    Negative numbers mean the model beat the benchmark in that subperiod.
    """
    losses = qlike(F, rv.to_numpy()[:, None])
    difference = losses.sub(losses[benchmark], axis=0)
    groups = groups.reindex(F.index)

    by_year = difference.groupby(groups["year"]).mean().T
    by_regime = difference.groupby(groups["regime"]).mean().T[["Calm", "Normal", "High vol"]]
    without_2020 = difference[groups["year"] != 2020].mean().rename("Without 2020")
    return by_year, by_regime, without_2020


def leaderboard(test_tables, by_year, by_regime, without_2020):
    """
    One row per model across every test: its rank in each, its average rank, and how
    often it's in the Model Confidence Set or beats the benchmark.
    """
    ranks = pd.DataFrame({_label(test): table["Rank"] for test, table in test_tables.items()})
    in_mcs = pd.DataFrame({_label(test): table["MCS p"] > 0.10 for test, table in test_tables.items()})

    board = pd.DataFrame({
        "Average rank": ranks.mean(axis=1),
        "Tests in MCS": in_mcs.sum(axis=1).astype(int).astype(str) + f" of {len(test_tables)}",
    })
    main = ("r2", 20)
    if main in test_tables:
        board["20d QLIKE vs benchmark"] = test_tables[main]["QLIKE vs benchmark"]
        board["20d DM p"] = test_tables[main]["DM p"]
        years = by_year[main]
        board["20d years beating benchmark"] = (years < 0).sum(axis=1).astype(str) + f" of {years.shape[1]}"
        board["20d without 2020"] = without_2020[main]
        for regime in by_regime[main].columns:
            board[f"20d {regime.lower()}"] = by_regime[main][regime]
    board = board.join(ranks).sort_values("Average rank")
    board.insert(0, "Place", range(1, len(board) + 1))
    return board


def _label(test):
    proxy, horizon = test
    return f"{'Daily' if horizon == 1 else f'{horizon}d'} {'r²' if proxy == 'r2' else 'GK'}"


def score_all(forecasts, targets, inputs, benchmark, horizons, proxies=("r2", "gk_gap")):
    """Score every test, and build the leaderboard and subperiod tables."""
    tests, by_year, by_regime, without_2020 = {}, {}, {}, {}
    origins = forecasts.index.get_level_values("origin").unique()
    groups = origin_groups(inputs, origins)

    for proxy in proxies:
        for horizon in horizons:
            F, rv = paired_frame(forecasts, targets, proxy, horizon)
            key = (proxy, horizon)
            tests[key] = score_one(F, rv, benchmark, horizon)
            by_year[key], by_regime[key], without_2020[key] = subperiod_scores(F, rv, groups, benchmark)

    board = leaderboard(tests, by_year, by_regime, without_2020)
    return board, tests, by_year, by_regime, without_2020
