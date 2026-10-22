# Volatility-Forecast

## Project Objective
The goal of this project is to develop a tool that forecasts future volatility for commodity ETFs to improve our risk assessment, options valuation, and trading decisions.
Rather than predicting the exact path of volatility, the objective is to forecast a probability distribution of future realized volatility over a specified time horizon.
This will allow us to estimate the range of likely volatility outcomes and compare our forecasts against market-implied volatility to identify potential trading opportunities.

### 1. Implied vs. Historical Volatility
Understand the differences between implied volatility (IV) and historical/realized volatility (RV).
IV will likely be a useful input, but it is not necessarily an unbiased forecast of future realized volatility.
Many of our commodity ETF options markets are thinly traded, so IV estimates may be distorted.
Consider how to incorporate IV without relying on it exclusively.

### 2. Seasonality and Commodity-Specific Events
Commodity volatility may exhibit seasonal patterns due to weather, supply and demand cycles, etc.
Analyze historical volatility over comparable periods in previous years to determine whether meaningful seasonal patterns exist.
Seasonality will likely vary significantly across commodities and should be evaluated separately for each asset.
Think of how to incorporate volatility seasonality into the model.
Consider whether incorporating events like WASDE, EIA and FED reports directly into the forecasting model would improve its accuracy.
Alternatively, it may be more useful to leave event-specific volatility to analysts and instead provide historical volatility data surrounding these events.

### 3. Volatility Distributions
Volatility is nonnegative and frequently exhibits a right-skewed distribution rather than a normal distribution.
Taking the natural log of volatility values may produce a distribution closer to normal, but this must be tested for each asset.

### 4. Volatility Clustering and Mean Reversion
Volatility observations are not independent. High-volatility periods tend to be followed by high-volatility periods, and similarly for low volatility.
Volatility also frequently exhibits mean-reverting behavior.
Consider models such as GARCH or Hidden Markov Models to account for volatility clustering.

### 5. Forecast Horizons
Ideally forecast the volatility out for 10 to 20 trading days, recognize that the longer the forecast the more inaccurate.
The forecast distribution should represent volatility over the entire future period, not volatility on the final day.
Ensure that historical and implied volatility comparisons use consistent time horizons and annualization conventions.

### 6. Other Information to Consider
Think about modelling out specific day volatility or a rolling volatility and what window.
An alternative strategy may be forcasting out price paths and then taking the volatility of those price paths to get a distribution.







