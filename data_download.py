from coins_data.alpaca import *

df = get_options_hist_bars_df(
    occ_symbol="SPY261218C00700000",
    start="2026-09-01",
    end="2026-10-21"
)

print(df.head())