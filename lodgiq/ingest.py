import os
import pandas as pd


def ingest_data_simple(src_path="market.csv") -> pd.DataFrame:
    """Simple ingestion used by pipeline. Reads CSV, sets daily index, imputes forward/backward.
    Expected columns: stay_date, occ, adr, revpar
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    df = pd.read_csv(src_path, parse_dates=['stay_date'])
    df = df[['stay_date', 'occ', 'adr', 'revpar']].copy()
    df = df.drop_duplicates(subset=['stay_date'])
    df = df.set_index('stay_date').sort_index().asfreq('D')
    df = df.ffill().bfill()
    return df
