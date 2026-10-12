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
Think of how to incorporate volatility seasonality into the model
Consider whether incorporating events like WASDE, EIA and FED reports directly into the forecasting model would improve its accuracy
Alternatively, it may be more useful to leave event-specific adjustments to analysts and instead provide historical volatility data surrounding these events
This would allow analysts to evaluate how volatility has historically responded to similar announcements and incorporate that information into their trading decisions

### 3. Volatility Distributions
Volatility is nonnegative and frequently exhibits a right-skewed distribution rather than a normal distribution.
Taking the natural log of volatility values may produce a distribution closer to normal, but this must be tested for each asset.
Consider whether alternative distributions better represent historical volatility, particularly during periods of extreme market movement.
Remember that fitting a historical distribution alone does not account for how volatility changes over time.

### 4. Volatility Clustering and Mean Reversion
Volatility observations are not independent. High-volatility periods tend to be followed by high-volatility periods, and similarly for low volatility.
Volatility also frequently exhibits mean-reverting behavior
Consider models such as GARCH or Hidden Markov Models to account for volatility clustering

### 5. Forecast Horizons
Ideally forecast the volatility out for 10 to 20 trading days, recognize that the longer the forecast the more inaccurate.
The forecast distribution should represent volatility over the entire future period, not volatility on the final day.
Ensure that historical and implied volatility comparisons use consistent time horizons and annualization conventions.

### Expected Output
The final tool should ideally provide:
Forecast distribution: Expected volatility, median volatility, and probability intervals for the selected forecast horizon.
Historical comparison: Historical realized volatility for comparable periods, including seasonal benchmarks where appropriate.
Implied volatility comparison: Current IV for a comparable horizon, allowing analysts to assess potential differences between market pricing and our forecast.
Visualization: A clear display of the forecast distribution and relevant historical comparisons.

