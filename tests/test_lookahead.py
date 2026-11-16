"""
The most important check: no forecast may use data from after its origin.

Each forecaster runs twice, once on the real data and once with every return after a
cutoff date scrambled. Forecasts made on or before the cutoff must not change at all.
"""

import numpy as np
import pytest

from volforecast.forecasters import (
    EWMA,
    GarchForecaster,
    GarchSpec,
    HistoricVol,
    RandomWalk,
    RollingGK,
    RollingVol,
)

CUTOFF = "2019-04-15"

FORECASTERS = [
    lambda: HistoricVol(),
    lambda: RollingVol(20),
    lambda: EWMA(0.94),
    lambda: RollingGK(20),
    lambda: RandomWalk(),
    lambda: GarchForecaster(GarchSpec(dist="normal", refit_every=10)),
    lambda: GarchForecaster(GarchSpec(vol="GJR", dist="t", window=500, refit_every=10)),
]


def _scramble_after(inputs, cutoff):
    changed = inputs.copy()
    later = changed.index > cutoff
    rng = np.random.default_rng(0)
    changed.loc[later, "r"] = changed.loc[later, "r"] * 5 + rng.normal(0, 0.02, later.sum())
    changed.loc[later, "r2"] = changed.loc[later, "r"] ** 2
    changed.loc[later, "gk_gap"] = changed.loc[later, "gk_gap"] * 7
    return changed


@pytest.mark.parametrize("make", FORECASTERS)
def test_forecasts_ignore_data_after_their_origin(short_inputs, make):
    origins = short_inputs.loc["2019-01-02":"2019-06-28"].index
    real = make().forecast(short_inputs, origins, 20)
    scrambled = make().forecast(_scramble_after(short_inputs, CUTOFF), origins, 20)

    before = origins[origins <= CUTOFF]
    after = origins[origins > CUTOFF]
    np.testing.assert_allclose(real.loc[before].to_numpy(), scrambled.loc[before].to_numpy(), rtol=1e-10)

    # The scramble should change later forecasts, or this test proves nothing
    assert not np.allclose(real.loc[after].to_numpy(), scrambled.loc[after].to_numpy())
