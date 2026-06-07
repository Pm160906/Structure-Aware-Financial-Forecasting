import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# constants
time_steps = 20             # number of past rows used to study short-term structure 
hankel_window = 8           # internal hankel embedding size, converts each rolling window into lagged matrix form


# hankel matrix
## converts sequential data into structured lag matrix, exposing hidden temporal dependencies
def hankelize_window(X_window, h_window):
    rows = []

    n = len(X_window)
    for i in range(n - h_window + 1):
        block = X_window[i:i+h_window].flatten()
        rows.append(block)

    return np.array(rows)


# svd structure score: decomposes Hankel matrix into dominant patterns.
def structure_score(X_window):
    H = hankelize_window(X_window, hankel_window)

    try:
        _, S, _ = np.linalg.svd(H, full_matrices=False)

        # 1. ratio = dominance of strongest pattern
        if np.sum(S) == 0:
            return np.array([0,0,0], dtype=np.float32)

        ratio = S[0] / np.sum(S)

        # 2. spread of dynamics across patterns (entropy)
        entropy = -np.sum(
            (S/np.sum(S)) * np.log((S/np.sum(S)) + 1e-8)
        )
  
        # 3. number of active dynamics (how many hidden patterns exist)
        rank_proxy = np.sum(S > 0.1)

        return np.array(
            [ratio, entropy, rank_proxy],
            dtype=np.float32
        )

    except:
        return np.array([0,0,0], dtype=np.float32)

# prepare data
def prepare_hankel_data():
    df = pd.read_csv('features_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.reset_index(drop=True)

    macro_cols = [
        'corr_tcs_btc',
        'corr_tcs_oil',
        'corr_tcs_fx',
        'corr_btc_oil',
        'btc_oil_ratio',
        'tcs_fx_ratio',
        'stress_index',
        'btc_minus_tcs_ret'
    ]

    micro_cols = [c for c in df.columns if c not in (
        ['Date', 'target'] + macro_cols
    )]

    # clean
    df[micro_cols] = df[micro_cols].replace(
        [np.inf, -np.inf], np.nan
    ).ffill().bfill().fillna(0)

    df[macro_cols] = df[macro_cols].replace(
        [np.inf, -np.inf], np.nan
    ).ffill().bfill().fillna(0)

    # scale
    X_micro = StandardScaler().fit_transform(df[micro_cols].values)
    X_macro = StandardScaler().fit_transform(df[macro_cols].values)

    y = None

    # rolling structure scores
    micro_scores, macro_scores = [], []
    for i in range(time_steps, len(df)):
        micro_window, macro_window = X_micro[i-time_steps:i], X_macro[i-time_steps:i]

        micro_scores.append(structure_score(micro_window))
        macro_scores.append(structure_score(macro_window))

    # convert into sequences for LSTM
    seq_len = 30
    Xm_seq, XM_seq, y_final = [], [], []
    for i in range(seq_len, len(micro_scores) - 1):
        Xm_seq.append(
        np.array(micro_scores[i-seq_len:i])
        )

        XM_seq.append(
        np.array(macro_scores[i-seq_len:i])
        )

        next_score = (
            np.mean(micro_scores[i+1]) +
            np.mean(macro_scores[i+1])
        ) / 2
        y_final.append(next_score)

    Xm_seq = np.array(Xm_seq, dtype=np.float32)
    XM_seq = np.array(XM_seq, dtype=np.float32)
    y_final = np.array(y_final, dtype=np.float32)

    split = int(len(Xm_seq) * 0.8)

    return (
    Xm_seq[:split], Xm_seq[split:],
    XM_seq[:split], XM_seq[split:],
    y_final[:split], y_final[split:]
    )