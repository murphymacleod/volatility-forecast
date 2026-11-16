import numpy as np
import pandas as pd
import pytest

from volforecast.scoring import mincer_zarnowitz, qlike, score_one


def test_qlike_is_lowest_at_the_true_variance():
    rng = np.random.default_rng(1)
    true_variance = 0.16
    realized = true_variance * rng.chisquare(1, 20_000)
    candidates = np.linspace(0.08, 0.32, 49)
    losses = [qlike(c, realized).mean() for c in candidates]
    assert candidates[int(np.argmin(losses))] == pytest.approx(true_variance, abs=0.01)


def test_qlike_handles_zero_realized_variance():
    assert np.isfinite(qlike(0.1, 0.0))


def test_mincer_zarnowitz_on_a_perfect_forecast():
    forecast = pd.Series(np.linspace(0.05, 0.5, 200))
    intercept, slope, r_squared, _ = mincer_zarnowitz(forecast, forecast.copy(), lags=5)
    assert intercept == pytest.approx(0, abs=1e-10)
    assert slope == pytest.approx(1)
    assert r_squared == pytest.approx(1)


def test_score_one_ranks_the_true_model_first():
    rng = np.random.default_rng(2)
    n = 1500
    true = pd.Series(0.1 + 0.3 * np.abs(np.sin(np.arange(n) / 50)), index=pd.date_range("2020-01-01", periods=n))
    realized = true * rng.chisquare(1, n)
    F = pd.DataFrame({"truth": true, "flat": true.mean(), "too high": true * 1.5})

    table = score_one(F, realized, benchmark="flat", horizon=1)
    assert table.index[0] == "truth"
    assert table.loc["flat", "QLIKE vs benchmark"] == 0
    assert table.loc["truth", "DM p"] < 0.05
    assert table.loc["truth", "MCS p"] > 0.10
    assert table.loc["too high", "Forecast / realized variance"] > 1.3
