import pandas as pd
import numpy as np

# 1. preprocessing
tcs_file = r'C:\\Users\\Parkhi\\Desktop\\college\\etps\\year 2\\sem four\\mis iv\\code\\datasets\\tata consultancy stock price history.csv'
btc_file = r'C:\\Users\\Parkhi\\Desktop\\college\\etps\\year 2\\sem four\\mis iv\\code\\datasets\\btc_usd binance historical data.csv'
oil_file = r'C:\\Users\\Parkhi\\Desktop\\college\\etps\\year 2\\sem four\\mis iv\\code\\datasets\\crude oil wti futures historical data.csv'
fx_file  = r'C:\\Users\\Parkhi\\Desktop\\college\\etps\\year 2\\sem four\\mis iv\\code\\datasets\\usd_inr historical data.csv'


# clean numeric values (remove commas, %, K/M/B suffixes)
def clean_numeric(x):
    if pd.isna(x):
        return np.nan

    x = str(x).replace(',', '').replace('%', '').strip()

    multiplier = 1
    if 'K' in x:
        multiplier = 1_000
        x = x.replace('K', '')

    elif 'M' in x:
        multiplier = 1_000_000
        x = x.replace('M', '')

    elif 'B' in x:
        multiplier = 1_000_000_000
        x = x.replace('B', '')

    try:
        return float(x) * multiplier
    except:
        return np.nan

def preprocess_dataset(path, prefix):
    df = pd.read_csv(path)

    # strip spaces
    df.columns = [c.strip() for c in df.columns]

    # rename volume/change columns if weird names
    for c in df.columns:
        if 'Vol' in c:
            df.rename(columns={c: 'Vol'}, inplace=True)
        if 'Change' in c:
            df.rename(columns={c: 'Change'}, inplace=True)

    ## date parse
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    ## sort ascending
    df = df.sort_values('Date')

    ## remove duplicate dates
    df = df.drop_duplicates(subset='Date')

    ## clean numeric columns
    numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol', 'Change']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    ## missing values fill
    df = df.ffill().bfill()

    ## add prefixes
    rename_map = {}
    for col in df.columns:
        if col != 'Date':
            rename_map[col] = f'{prefix}_{col}'

    df.rename(columns=rename_map, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

tcs = preprocess_dataset(tcs_file, 'TCS')
btc = preprocess_dataset(btc_file, 'BTC')
oil = preprocess_dataset(oil_file, 'OIL')
fx  = preprocess_dataset(fx_file,  'FX')

print('TCS Shape:', tcs.shape)
print('BTC Shape:', btc.shape)
print('OIL Shape:', oil.shape)
print('FX Shape :', fx.shape)

print('\nTCS Sample:')
print(tcs.head())


# 2. merge all datasets on date (using inner join to keep only common dates)
merged_df = tcs.merge(btc, on='Date', how='inner')
merged_df = merged_df.merge(oil, on='Date', how='inner')
merged_df = merged_df.merge(fx, on='Date', how='inner')

## sort by date after merge
merged_df = merged_df.sort_values('Date').reset_index(drop=True)
merged_df = merged_df.drop(columns=['FX_Vol'], errors='ignore')

print('Merged Shape:', merged_df.shape)

print('\nColumns:')
print(merged_df.columns.tolist())

print('\nDate Range:')
print('Start:', merged_df['Date'].min())
print('End  :', merged_df['Date'].max())

print('\nSample Rows:')
print(merged_df.head())

## check missing values
print('\nMissing Values Per Column:')
print(merged_df.isna().sum())

## save output
merged_df.to_csv('merged_data.csv', index=False)

print('Saved: merged_data.csv')
print('Shape:', merged_df.shape)
print(merged_df.head())