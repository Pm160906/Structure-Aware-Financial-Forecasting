import pandas as pd
import numpy as np

# load merged data
df = pd.read_csv('merged_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

assets = ['TCS', 'BTC', 'OIL', 'FX']

# microfeature extraction: these describe behavior of each individual asset.
## useful for short-term movement and local instability.
for a in assets:
    price = f'{a}_Price'
    high = f'{a}_High'
    low = f'{a}_Low'
    openp = f'{a}_Open'

    # daily return, log return, intraday price range, candle body
    df[f'{a}_ret'] = df[price].pct_change()                       # measures day-to-day gain or loss speed
    df[f'{a}_logret'] = np.log(df[price] / df[price].shift(1))    # preferred in finance for statistical stability
    df[f'{a}_range'] = df[high] - df[low]                         # measures volatility inside one trading period
    df[f'{a}_body'] = df[price] - df[openp]                       # shows bullish or bearish dominance

    # moving averages [5 and 10 days]
    df[f'{a}_ma5'] = df[price].rolling(5).mean()                  # captures short-term trend
    df[f'{a}_ma10'] = df[price].rolling(10).mean()                # slightly smoother trend estimate

    # volatility (rolling) and momentum (5 days)
    df[f'{a}_vol5'] = df[f'{a}_ret'].rolling(5).std()             # standard deviation of returns (high values indicate instability)
    df[f'{a}_mom5'] = df[price] - df[price].shift(5)              # measures directional strength over recent days


# macrofeature extraction: describe relationships between markets
## useful for systemic stress and hidden regime shifts

# rolling correlations between assets: detect coupling / decoupling behavior
df['corr_tcs_btc'] = df['TCS_ret'].rolling(10).corr(df['BTC_ret'])
df['corr_tcs_oil'] = df['TCS_ret'].rolling(10).corr(df['OIL_ret'])
df['corr_tcs_fx'] = df['TCS_ret'].rolling(10).corr(df['FX_ret'])
df['corr_btc_oil'] = df['BTC_ret'].rolling(10).corr(df['OIL_ret'])

# relative valuation ratios, helps capture cross-market imbalance
df['btc_oil_ratio'] = df['BTC_Price'] / df['OIL_Price']
df['tcs_fx_ratio'] = df['TCS_Price'] / df['FX_Price']

# combined stress index: sum of recent volatilities across markets, indicating overall market tension
df['stress_index'] = (
    df['TCS_vol5'].fillna(0) +
    df['BTC_vol5'].fillna(0) +
    df['OIL_vol5'].fillna(0) +
    df['FX_vol5'].fillna(0)
)

df['btc_minus_tcs_ret'] = df['BTC_ret'] - df['TCS_ret']   # relative performance spread

# clean nans
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()
df = df.iloc[10:].reset_index(drop=True)

# save output
df.to_csv('features_data.csv', index=False)

print('Saved: features_data.csv')
print('Shape:', df.shape)
print(df.head())