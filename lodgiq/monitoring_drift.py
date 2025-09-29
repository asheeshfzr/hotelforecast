import logging
import os
from typing import Dict
import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from .config import (
    MONITOR_DIR,
    DRIFT_WINDOW_DAYS,
    HIST_WINDOW_DAYS,
    KS_PVALUE_THRESHOLD,
    DRIFT_KS_STAT_THRESHOLD,
    DRIFT_MEAN_DIFF_THRESHOLD,
    DRIFT_COHEN_D_THRESHOLD,
    DRIFT_REQUIRE_BOTH_EFFECTS,
)
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def ks_test_drift(series_hist: pd.Series, series_recent: pd.Series):
    if HAS_SCIPY:
        try:
            stat, pvalue = ks_2samp(series_hist.dropna().values, series_recent.dropna().values)
            return pvalue, stat
        except Exception as e:
            logger.exception("KS test failed: %s", e)
            return 1.0, 0.0
    # Fallback
    hist = series_hist.dropna().values
    rec = series_recent.dropna().values
    if len(hist) < 10 or len(rec) < 10:
        return 1.0, 0.0
    stat = abs(np.mean(hist) - np.mean(rec)) / (np.std(hist) + 1e-9)
    pvalue = float(np.exp(-stat))
    return pvalue, float(stat)


def detect_data_drift(df: pd.DataFrame, run_ts: str) -> Dict:
    ensure_dir(MONITOR_DIR)
    if 'occ' not in df.columns or len(df.dropna(subset=['occ'])) < (DRIFT_WINDOW_DAYS + 30):
        report = {'run_ts': run_ts, 'error': 'insufficient_data_for_drift', 'drift': False}
        path = os.path.join(MONITOR_DIR, f"drift_report_{run_ts}.json")
        save_json(report, path, logger)
        logger.info("Saved drift report: %s (insufficient data)", path)
        return report

    last_date = df.dropna(subset=['occ']).index.max()
    recent_start = last_date - pd.Timedelta(days=DRIFT_WINDOW_DAYS - 1)
    hist_start = last_date - pd.Timedelta(days=HIST_WINDOW_DAYS - 1)
    recent = df.loc[recent_start:last_date]['occ'].dropna()

    recent_months = set(pd.Index(recent.index).month)
    hist_seg = df.loc[hist_start:recent_start - pd.Timedelta(days=1)]['occ']
    hist_seg = hist_seg[hist_seg.index.month.isin(recent_months)].dropna()
    hist = hist_seg if len(hist_seg) >= max(30, len(recent)) else df.loc[hist_start:recent_start - pd.Timedelta(days=1)]['occ'].dropna()

    pvalue, ks_stat = ks_test_drift(hist, recent)

    mean_diff = float(abs(recent.mean() - hist.mean()))
    std_pool = float((recent.std(ddof=1) + hist.std(ddof=1)) / 2.0) if (recent.std(ddof=1) + hist.std(ddof=1)) > 0 else 0.0
    cohen_d = float(mean_diff / std_pool) if std_pool > 0 else 0.0

    passes_ks = ks_stat >= DRIFT_KS_STAT_THRESHOLD
    passes_mean = mean_diff >= DRIFT_MEAN_DIFF_THRESHOLD
    passes_cohen = cohen_d >= DRIFT_COHEN_D_THRESHOLD
    if DRIFT_REQUIRE_BOTH_EFFECTS:
        effect_gate = (passes_ks and passes_cohen) or (passes_mean and passes_cohen)
    else:
        effect_gate = passes_ks or passes_mean or passes_cohen

    drift_flag = (pvalue < KS_PVALUE_THRESHOLD) and effect_gate

    report = {
        'run_ts': run_ts,
        'last_date': str(last_date),
        'hist_start': str(hist_start),
        'recent_start': str(recent_start),
        'recent_months': sorted(list(recent_months)),
        'ks_pvalue': float(pvalue),
        'ks_statistic': float(ks_stat),
        'mean_recent': float(recent.mean()) if len(recent) else None,
        'mean_hist': float(hist.mean()) if len(hist) else None,
        'mean_abs_diff': mean_diff,
        'cohen_d': cohen_d,
        'n_recent': int(len(recent)),
        'n_hist': int(len(hist)),
        'gates': {
            'KS_PVALUE_THRESHOLD': KS_PVALUE_THRESHOLD,
            'KS_STAT_THRESHOLD': DRIFT_KS_STAT_THRESHOLD,
            'MEAN_DIFF_THRESHOLD': DRIFT_MEAN_DIFF_THRESHOLD,
            'cohen_d_threshold': DRIFT_COHEN_D_THRESHOLD,
            'require_both_effects': DRIFT_REQUIRE_BOTH_EFFECTS,
            'passes': {
                'ks_stat': bool(passes_ks),
                'mean_diff': bool(passes_mean),
                'cohen_d': bool(passes_cohen),
            },
        },
        'drift': bool(drift_flag),
    }

    path = os.path.join(MONITOR_DIR, f"drift_report_{run_ts}.json")
    save_json(report, path, logger)
    logger.info(
        "Saved drift report: %s (drift=%s p=%.3g stat=%.3g mean_diff=%.3g cohen_d=%.3g n_rec=%d n_hist=%d)",
        path,
        report['drift'],
        report['ks_pvalue'],
        report['ks_statistic'],
        report['mean_abs_diff'],
        report['cohen_d'],
        report['n_recent'],
        report['n_hist'],
    )
    return report


def detect_feature_drift(X_hist: pd.DataFrame, X_recent: pd.DataFrame, feature_list=None):
    drifted = {}
    features = feature_list or list(set(X_hist.columns) & set(X_recent.columns))
    for f in features:
        if f not in X_hist.columns or f not in X_recent.columns:
            continue
        pvalue, stat = ks_test_drift(X_hist[f], X_recent[f])
        drifted[f] = {'pvalue': pvalue, 'stat': stat, 'drift': pvalue < KS_PVALUE_THRESHOLD}
    return drifted
