import pytest

from volforecast.data import load_prices
from volforecast.targets import model_inputs


@pytest.fixture(scope="session")
def prices():
    return load_prices("USO")


@pytest.fixture(scope="session")
def inputs(prices):
    return model_inputs(prices)


@pytest.fixture(scope="session")
def short_inputs(inputs):
    """2016 through 2019, which keeps GARCH tests fast."""
    return inputs.loc[:"2019-12-31"]
