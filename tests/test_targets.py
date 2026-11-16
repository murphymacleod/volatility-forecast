import numpy as np
import pandas as pd
import pytest

from volforecast.targets import future_mean, gk_scale, model_inputs, realized_targets


def test_future_mean_uses_the_next_days_only():
    daily = pd.Series(np.arange(1.0, 11.0), index=pd.date_range("2024-01-01", periods=10))
    result = future_mean(daily, 3)

    # At the first origin, the next three days hold 2, 3 and 4
    assert result.iloc[0] == pytest.approx(3.0)
    assert result.iloc[6] == pytest.approx((8 + 9 + 10) / 3)
    # The last three origins don't have three later days
    assert result.iloc[7:].isna().all()


def test_horizon_one_is_the_next_day():
    daily = pd.Series([4.0, 1.0, 9.0], index=pd.date_range("2024-01-01", periods=3))
    assert future_mean(daily, 1).tolist()[:2] == [1.0, 9.0]


def test_model_inputs_match_hand_calculation():
    prices = pd.DataFrame({
        "Open_Price": [100.0, 102.0],
        "High_Price": [101.0, 105.0],
        "Low_Price": [99.0, 101.0],
        "Closing_Price": [100.0, 104.0],
    }, index=pd.date_range("2024-01-01", periods=2))
    row = model_inputs(prices).iloc[0]

    r = np.log(104 / 100)
    gk = 0.5 * np.log(105 / 101) ** 2 - (2 * np.log(2) - 1) * np.log(104 / 102) ** 2 + np.log(102 / 100) ** 2
    assert row["r"] == pytest.approx(r)
    assert row["r2"] == pytest.approx(r**2)
    assert row["gk_gap"] == pytest.approx(gk)


def test_log_returns_match_the_csv(prices, inputs):
    assert np.allclose(inputs["r"], prices["Log_Returns"].reindex(inputs.index))


def test_scaled_gk_matches_r2_on_average(inputs):
    sample = inputs.loc["2019":"2024"]
    targets = realized_targets(sample, [1], scale=gk_scale(sample))
    daily = targets[[("r2", 1), ("gk_gap", 1)]].dropna()
    # Both are shifted by one day, so they average the same days
    assert daily[("gk_gap", 1)].mean() == pytest.approx(daily[("r2", 1)].mean(), rel=0.01)


def test_targets_never_reach_past_the_data_given(inputs):
    cut = inputs.loc[:"2020-06-30"]
    targets = realized_targets(cut, [20])
    assert targets[("r2", 20)].iloc[-20:].isna().all()
    assert targets[("r2", 20)].iloc[-21:-20].notna().all()
