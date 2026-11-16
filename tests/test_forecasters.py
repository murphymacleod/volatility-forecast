import numpy as np
import pytest

from volforecast.forecasters import EWMA, GarchSpec, HistoricVol, RandomWalk, RollingGK, RollingVol, garch_grid


def _origins(inputs):
    return inputs.loc["2018-01-02":"2018-03-30"].index


def test_rolling_vol_averages_the_last_days_including_the_origin(short_inputs):
    origins = _origins(short_inputs)
    forecast = RollingVol(20).forecast(short_inputs, origins, 5)

    position = short_inputs.index.get_loc(origins[10])
    expected = short_inputs["r2"].iloc[position - 19: position + 1].mean()
    assert forecast.loc[origins[10], 1] == pytest.approx(expected)


@pytest.mark.parametrize("forecaster", [HistoricVol(), RollingVol(20), EWMA(), RollingGK(), RandomWalk()])
def test_baselines_hold_one_level_flat(short_inputs, forecaster):
    forecast = forecaster.forecast(short_inputs, _origins(short_inputs), 20)
    assert np.allclose(forecast.to_numpy(), forecast[[1]].to_numpy())
    assert (forecast.to_numpy() > 0).all()


def test_historic_vol_is_the_average_of_all_history(short_inputs):
    origin = _origins(short_inputs)[0]
    forecast = HistoricVol().forecast(short_inputs, [origin], 1)
    assert forecast.iloc[0, 0] == pytest.approx(short_inputs["r2"].loc[:origin].mean())


def test_ewma_follows_its_recursion(short_inputs):
    origins = _origins(short_inputs)
    forecast = EWMA(0.94).forecast(short_inputs, origins, 1)[1]
    previous_day = short_inputs.index[short_inputs.index.get_loc(origins[5]) - 1]
    previous = EWMA(0.94).forecast(short_inputs, [previous_day], 1).iloc[0, 0]
    expected = 0.94 * previous + 0.06 * short_inputs["r2"].loc[origins[5]]
    assert forecast.loc[origins[5]] == pytest.approx(expected)


def test_garch_grid_makes_every_combination_with_unique_names():
    specs = garch_grid(vol=["GARCH", "GJR"], dist=["normal", "t", "skewt"], window=[None, 1000])
    assert len(specs) == 12
    assert len({spec.name for spec in specs}) == 12


def test_garch_spec_names_describe_the_model():
    assert GarchSpec().name == "GARCH(1,1)-t-expanding-refit21"
    assert GarchSpec(vol="GJR", dist="skewt", window=1000).name == "GJR(1,1,1)-skewt-roll1000-refit21"
