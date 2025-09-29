import pandas as pd


def create_lag_features(df: pd.DataFrame, target_col='occ', lags=(1, 7, 14, 30), roll_windows=(7, 30)) -> pd.DataFrame:
    df_f = pd.DataFrame(index=df.index)
    df_f['y'] = df[target_col].astype(float)
    for l in lags:
        df_f[f'lag_{l}'] = df[target_col].shift(l)
    for w in roll_windows:
        df_f[f'roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        df_f[f'roll_std_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    df_f['dayofweek'] = df_f.index.dayofweek
    df_f['month'] = df_f.index.month
    df_f['is_weekend'] = df_f['dayofweek'].isin([5, 6]).astype(int)
    if 'adr' in df.columns:
        df_f['lag_1_adr'] = df['adr'].shift(1)
    if 'revpar' in df.columns:
        df_f['lag_1_revpar'] = df['revpar'].shift(1)
    return df_f
