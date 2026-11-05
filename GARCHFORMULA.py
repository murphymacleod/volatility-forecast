from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arch


DATA_PATH = Path(__file__).parent / "etf_data" / "USO_daily_prices.csv"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#30946e"
INK = "#0b0b0b"

# --------------------------------------------------
# Load data and calculate returns
# --------------------------------------------------
uso = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
).sort_index()

# Calculate directly from split-adjusted prices.
prices = pd.to_numeric(uso["Closing_Price"], errors="raise")

if (prices.dropna() <= 0).any():
    raise ValueError("Log returns require positive prices.")

returns = (
    100 * np.log(prices / prices.shift(1))
).replace([np.inf, -np.inf], np.nan).dropna()

# --------------------------------------------------
# Fit GARCH(1,1)
# --------------------------------------------------
model = arch.arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal",
    rescale=False
)

result = model.fit(disp="off")
print(result.summary())

if result.convergence_flag != 0:
    print("\nWARNING: Optimization did not converge. Check the fit.")

mu = result.params["mu"]
omega = result.params["omega"]
alpha = result.params["alpha[1]"]
beta = result.params["beta[1]"]

# --------------------------------------------------
# Collect fitted series and equation components
# --------------------------------------------------
diagnostics = pd.DataFrame({
    "return": returns,
    "residual": result.resid,
    "volatility": result.conditional_volatility,
    "standardized_residual": result.std_resid
})

diagnostics["variance"] = diagnostics["volatility"] ** 2

diagnostics["baseline"] = omega
diagnostics["shock_term"] = (
    alpha * diagnostics["residual"].shift(1) ** 2
)
diagnostics["persistence_term"] = (
    beta * diagnostics["variance"].shift(1)
)

components = diagnostics[
    ["baseline", "shock_term", "persistence_term"]
].dropna()

reconstructed_variance = components.sum(axis=1)

peak_date = diagnostics["volatility"].idxmax()
peak_volatility = diagnostics.loc[peak_date, "volatility"]

# Zoom to 40 observations on each side of the volatility peak.
peak_position = diagnostics.index.get_loc(peak_date)
zoom = diagnostics.iloc[
    max(0, peak_position - 40):
    min(len(diagnostics), peak_position + 41)
]

print("\nEstimated parameters")
print(f"mu:           {mu:.6f}")
print(f"omega:        {omega:.6f}")
print(f"alpha:        {alpha:.6f}")
print(f"beta:         {beta:.6f}")
print(f"alpha + beta: {alpha + beta:.6f}")

print(f"\nMaximum daily volatility: {peak_volatility:.2f}%")
print(f"Peak date: {peak_date:%Y-%m-%d}")

print("\nObservations around the volatility peak:")
print(
    zoom[
        ["return", "volatility", "shock_term", "persistence_term"]
    ].loc[:peak_date].tail(6).round(3)
)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------
# Window 1: charts 1–4
fig, top_axes = plt.subplots(2, 2, figsize=(13, 8))

# Window 2: charts 5–6
fig_detail, detail_axes = plt.subplots(1, 2, figsize=(13, 5))

# Keep the existing axes references working.
axes = np.empty((3, 2), dtype=object)
axes[:2, :] = top_axes
axes[2, :] = detail_axes

# 1. Returns and conditional standard-deviation bands
ax = axes[0, 0]
ax.plot(
    diagnostics.index,
    diagnostics["return"],
    color=BLUE,
    linewidth=0.6,
    alpha=0.7,
    label="Daily log return"
)
ax.plot(
    diagnostics.index,
    mu + 2 * diagnostics["volatility"],
    color=ORANGE,
    linewidth=0.9,
    label="Mean ± 2 conditional standard deviations"
)
ax.plot(
    diagnostics.index,
    mu - 2 * diagnostics["volatility"],
    color=ORANGE,
    linewidth=0.9
)
ax.axhline(mu, color=INK, linewidth=0.7)
ax.set_title("1. Returns versus fitted fluctuation scale")
ax.set_ylabel("Daily log return (%)")
ax.legend(fontsize=8)

# 2. Fitted conditional volatility
ax = axes[0, 1]
ax.plot(
    diagnostics.index,
    diagnostics["volatility"],
    color=ORANGE,
    label="GARCH conditional volatility"
)
ax.scatter(
    [peak_date],
    [peak_volatility],
    color=INK,
    zorder=5
)
ax.annotate(
    f"{peak_volatility:.2f}% on {peak_date:%Y-%m-%d}",
    xy=(peak_date, peak_volatility),
    xytext=(8, -20),
    textcoords="offset points",
    fontsize=9
)
ax.set_title("2. Estimated daily volatility")
ax.set_ylabel("Daily standard deviation (%)")
ax.legend(fontsize=8)

# 3. Contributions to variance
ax = axes[1, 0]
ax.stackplot(
    components.index,
    components["baseline"].to_numpy(),
    components["shock_term"].to_numpy(),
    components["persistence_term"].to_numpy(),
    labels=[
        "Baseline: omega",
        "Previous shock: alpha × residual²",
        "Previous variance: beta × variance"
    ],
    colors=[GREEN, ORANGE, BLUE],
    alpha=0.8
)
ax.plot(
    diagnostics.index,
    diagnostics["variance"],
    color=INK,
    linewidth=0.7,
    label="Fitted variance"
)
ax.set_title("3. What builds the conditional variance?")
ax.set_ylabel("Variance (percentage points squared)")
ax.legend(fontsize=8, loc="upper left")

# 4. Verify the recursion
ax = axes[1, 1]
ax.plot(
    diagnostics.index,
    diagnostics["variance"],
    color=INK,
    linewidth=1.5,
    label="Fitted variance"
)
ax.plot(
    reconstructed_variance.index,
    reconstructed_variance,
    color=ORANGE,
    linestyle="--",
    linewidth=1,
    label="omega + alpha × lagged residual² + beta × lagged variance"
)
ax.set_title("4. Equation check: the lines should overlap")
ax.set_ylabel("Variance (percentage points squared)")
ax.legend(fontsize=8)

# 5. Peak-period close-up
ax = axes[2, 0]
ax.bar(
    zoom.index,
    zoom["return"].abs(),
    color=BLUE,
    alpha=0.45,
    width=1.5,
    label="Absolute daily log return"
)
ax.plot(
    zoom.index,
    zoom["volatility"],
    color=ORANGE,
    linewidth=2,
    label="Conditional volatility"
)
ax.axvline(peak_date, color=INK, linestyle=":", linewidth=1)
ax.set_title("5. Close-up around peak volatility")
ax.set_ylabel("Percentage points")
ax.legend(fontsize=8)

# 6. Standardized residuals
ax = axes[2, 1]
ax.plot(
    diagnostics.index,
    diagnostics["standardized_residual"],
    color=BLUE,
    linewidth=0.6
)
ax.axhline(0, color=INK, linewidth=0.7)
ax.axhline(2, color=ORANGE, linestyle="--", linewidth=0.8)
ax.axhline(-2, color=ORANGE, linestyle="--", linewidth=0.8)
ax.set_title("6. Shocks after dividing by modeled volatility")
ax.set_ylabel("Standardized residual (unitless)")

for ax in axes.flat:
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", labelrotation=25)

fig.suptitle(
    "USO: Inside the GARCH(1,1) Model\n"
    f"omega={omega:.4f} | alpha={alpha:.4f} | "
    f"beta={beta:.4f} | alpha+beta={alpha + beta:.4f}",
    fontsize=13
)

fig_detail.suptitle(
    "USO: Peak Volatility and Standardized Shocks",
    fontsize=13
)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig_detail.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig(
    PLOTS_DIR / "USO_garch_overview.png",
    dpi=200,
    bbox_inches="tight"
)

fig_detail.savefig(
    PLOTS_DIR / "USO_garch_details.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()