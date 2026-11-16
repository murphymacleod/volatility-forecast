"""
End-to-end check on simulated data, where the right answer is known.

Returns are simulated from a GARCH(1,1) with Student's t shocks. The backtest should
recover the parameters, and a GARCH-t should beat rolling vol on every test.
"""

import numpy as np
import pandas as pd
import pytest
from arch.univariate import GARCH, ConstantMean, StudentsT

from volforecast.backtest import run_backtest
from volforecast.forecasters import EWMA, GarchForecaster, GarchSpec, RollingVol
from volforecast.scoring import score_all
from volforecast.targets import realized_targets

TRUE = {"mu": 0.02, "omega": 0.05, "alpha": 0.08, "beta": 0.90, "nu": 7.0}


@pytest.fixture(scope="module")
def simulated():
    model = ConstantMean(None, volatility=GARCH(), distribution=StudentsT(seed=np.random.default_rng(11)))
    sim = model.simulate([TRUE["mu"], TRUE["omega"], TRUE["alpha"], TRUE["beta"], TRUE["nu"]], nobs=3000, burn=500)
    r = sim["data"].to_numpy() / 100
    index = pd.bdate_range("2010-01-04", periods=len(r))
    # Simulated data has no intraday prices, so the Garman-Klass column just repeats r²
    return pd.DataFrame({"r": r, "r2": r**2, "gk_gap": r**2}, index=index)


def test_garch_recovers_the_true_parameters(simulated):
    garch = GarchForecaster(GarchSpec(dist="t", refit_every=250))
    garch.forecast(simulated, simulated.index[1500:], 5)
    last = garch.details().iloc[-1]
    assert last["alpha[1]"] == pytest.approx(TRUE["alpha"], abs=0.04)
    assert last["beta[1]"] == pytest.approx(TRUE["beta"], abs=0.05)
    assert last["nu"] == pytest.approx(TRUE["nu"], abs=3)


def test_the_true_model_wins_the_backtest(simulated):
    models = [RollingVol(20), EWMA(0.94), GarchForecaster(GarchSpec(dist="t", refit_every=63))]
    result = run_backtest(simulated, models, start=simulated.index[1500], horizon=20, verbose=False)
    targets = realized_targets(simulated, [1, 20], scale=1.0)
    board, tests, *_ = score_all(result.forecasts, targets, simulated, "Rolling 20d", [1, 20], proxies=("r2",))

    garch = models[2].name
    assert board.index[0] == garch
    for test in tests.values():
        assert test.loc[garch, "QLIKE vs benchmark"] < 0
