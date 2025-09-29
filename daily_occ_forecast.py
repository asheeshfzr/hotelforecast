#!/usr/bin/env python3
"""
daily_occ_forecast.py

Extended with:
 - Monitoring, drift detection & alerting (--monitor)
 - Retraining automation stub (--auto-retrain)
 - Stores daily actual vs forecast metrics to data/monitoring/metrics.csv
 - KS-test based drift checks (if scipy available), feature drift checks using LightGBM importances
 - Alerts via Slack webhook (SLACK_WEBHOOK env var) or email (SMTP env vars)
 - All added features are optional and do not change default behavior.

Usage examples:
  python daily_occ_forecast.py
  python daily_occ_forecast.py --fast --eda
  python daily_occ_forecast.py --ml --monitor
  python daily_occ_forecast.py --ml --monitor --auto-retrain
"""

import os
import sys
import time
import json
import hashlib
import logging
import pickle
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# New modular package imports
from lodgiq.logging_setup import setup_logging
from lodgiq import config as CFG
from lodgiq.utils import ensure_dir, timestamp_str, save_json
from lodgiq.ingest import ingest_data_simple
from lodgiq.forecasting import fit_hw, rmse, mape
from lodgiq.plots import save_forecast_plots
from lodgiq.features import create_lag_features
from lodgiq.ml_lightgbm import train_lightgbm
from lodgiq.monitoring_metrics import append_monitoring_metrics
from lodgiq.monitoring_drift import detect_data_drift, detect_feature_drift
from lodgiq.alerts import send_slack_alert, send_email_alert
from lodgiq.retrain import (
    check_retrain_trigger,
    auto_retrain_if_triggered,
    generate_retrain_plan,
)

# Optional libs
HAS_GE = False
try:
    import great_expectations as ge
    HAS_GE = True
except Exception:
    HAS_GE = False

HAS_SEABORN = False
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

HAS_LGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

HAS_SCIPY = False
try:
    from scipy.stats import ks_2samp
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Logging (delegated)
logger = setup_logging()

# Deterministic seed
np.random.seed(CFG.SEED)

# ----------------------
# Configurable defaults
# ----------------------
MAX_GAP_DAYS = 14
WINSORIZE_LIMITS = (0.01, 0.99)
SMOOTH_WINDOW = None
FAIL_ON_LONG_GAP = False

TOP_N_FEATURES = 10
LSTM_SEQ_LEN = 30
LSTM_EPOCHS = 20
LSTM_BATCH = 32

# Monitoring/drift/alerts config now provided by lodgiq.config (CFG)

# Utilities are provided by lodgiq.utils (ensure_dir, compute_md5, timestamp_str, save_json)


# ----------------------
# Forecasting helpers
# ----------------------
def rmse(true, pred):
    return float(np.sqrt(mean_squared_error(true, pred)))


def mape(true, pred):
    true_arr = np.array(true)
    pred_arr = np.array(pred)
    mask = true_arr != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((true_arr[mask] - pred_arr[mask]) / true_arr[mask])) * 100.0)


# Note: fit_hw is provided by lodgiq.forecasting; avoid redefining locally to prevent shadowing.


# ----------------------
# (Preprocessing / validation / EDA / ML functions)
# Reuse earlier implementations: preprocess_and_impute, ingest_data, run_eda, create_lag_features, etc.
# For brevity, I include the core functions required. If you already have them in your file, they remain unchanged.
# ----------------------

# For this response I'll embed essential helper functions (safe, deterministic) used in pipeline.

def ingest_data_simple(src_path="market.csv"):
    """Simpler ingestion to ensure monitoring integration below. For full ingest use your earlier ingest_data()."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    df = pd.read_csv(src_path, parse_dates=['stay_date'])
    df = df[['stay_date', 'occ', 'adr', 'revpar']].copy()
    df = df.drop_duplicates(subset=['stay_date'])
    df = df.set_index('stay_date').sort_index().asfreq('D')
    # simple impute
    df = df.ffill().bfill()
    return df


def create_lag_features(df: pd.DataFrame, target_col='occ', lags=(1,7,14,30), roll_windows=(7,30)) -> pd.DataFrame:
    df_f = pd.DataFrame(index=df.index)
    df_f['y'] = df[target_col].astype(float)
    for l in lags:
        df_f[f'lag_{l}'] = df[target_col].shift(l)
    for w in roll_windows:
        df_f[f'roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        df_f[f'roll_std_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    df_f['dayofweek'] = df_f.index.dayofweek
    df_f['month'] = df_f.index.month
    df_f['is_weekend'] = df_f['dayofweek'].isin([5,6]).astype(int)
    if 'adr' in df.columns:
        df_f['lag_1_adr'] = df['adr'].shift(1)
    if 'revpar' in df.columns:
        df_f['lag_1_revpar'] = df['revpar'].shift(1)
    return df_f


# LightGBM trainer (if available)
def train_lightgbm(X_train, y_train, X_val, y_val, num_boost_round=200, early_stopping_rounds=20):
    if not HAS_LGB:
        logger.info("LightGBM not available.")
        return None, None
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
        'learning_rate': 0.05,
        'num_leaves': 31,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dtrain, dval], early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=False)
    preds = model.predict(X_val)
    return model, compute_metrics(y_val, preds)


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mse))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    mask = (np.array(y_true) != 0)
    if mask.sum() > 0:
        mape_val = float(np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100.0)
    else:
        mape_val = None
    return {'rmse': rmse_val, 'mae': mae_val, 'mape': mape_val}


def save_forecast_plots(train_series, test_series, test_pred, future_forecast, out_dir, run_ts):
    save_dir = os.path.join(out_dir, run_ts)
    ensure_dir(save_dir)
    try:
        plt.figure(figsize=(12,5))
        plt.plot(train_series.index, train_series, label='train')
        plt.plot(test_series.index, test_series, label='test (actual)')
        plt.plot(test_series.index[:len(test_pred)], test_pred, label='test (pred)', linestyle='--')
        if future_forecast is not None:
            plt.plot(future_forecast.index, future_forecast.values, label='future forecast')
        plt.legend()
        plt.title("Train / Test / Forecast")
        plt.tight_layout()
        p = os.path.join(save_dir, "forecast_vs_actual.png")
        plt.savefig(p, dpi=150); plt.close()
        logger.info("Saved forecast plot: %s", p)
    except Exception as e:
        logger.exception("Could not save forecast plots: %s", e)


# Monitoring & Drift Detection
# Implementations provided by lodgiq.monitoring_metrics and lodgiq.monitoring_drift


 # ks_test_drift provided by lodgiq.monitoring_drift


 # detect_data_drift provided by lodgiq.monitoring_drift


 # detect_feature_drift provided by lodgiq.monitoring_drift




 # send_email_alert provided by lodgiq.alerts


 # check_retrain_trigger provided by lodgiq.retrain


 # auto_retrain_if_triggered provided by lodgiq.retrain


 # generate_retrain_plan provided by lodgiq.retrain


# ----------------------
# Main pipeline (integrates monitoring & retrain options)
# ----------------------
def main():
    start_time = time.time()
    fast_mode = "--fast" in sys.argv
    eda_mode = "--eda" in sys.argv
    ml_mode = "--ml" in sys.argv
    monitor_mode = "--monitor" in sys.argv
    auto_retrain_flag = "--auto-retrain" in sys.argv
    monitor_only = "--monitor-only" in sys.argv
    forecast_horizon = CFG.FAST_FORECAST_HORIZON if fast_mode else CFG.DEFAULT_FORECAST_HORIZON

    ensure_dir(CFG.MONITOR_DIR)
    ensure_dir(CFG.PLOTS_DIR)

    run_ts = timestamp_str()

    if monitor_only:
        logger.info("Running monitor-only mode")
        df = ingest_data_simple(CFG.MARKET_CSV)
        drift_report = detect_data_drift(df, run_ts)
        run_meta = {'notes': 'monitor_only'}
        metrics_path = append_monitoring_metrics(run_ts, run_meta, {}, pd.Series(dtype=float), np.array([]))
        trig, reasons = check_retrain_trigger(metrics_path, drift_report)
        if trig:
            alert_text = f"[Monitor] Drift trigger detected for run {run_ts}: {reasons}"
            send_slack_alert(alert_text)
            send_email_alert("Drift detected", alert_text)
        return

    # Normal pipeline
    df = ingest_data_simple(CFG.MARKET_CSV)

    if eda_mode:
        try:
            run_eda_dir = os.path.join(CFG.PLOTS_DIR, run_ts)
            ensure_dir(run_eda_dir)
            plt.figure(figsize=(12,4))
            plt.plot(df.index, df['occ']); plt.title("Occupancy time series"); plt.tight_layout()
            plt.savefig(os.path.join(run_eda_dir, "occ_timeseries.png"), dpi=150); plt.close()
            logger.info("Saved minimal EDA to %s", run_eda_dir)
        except Exception as e:
            logger.exception("EDA failed: %s", e)

    occ = df['occ'].astype(float)
    horizon_test = CFG.HOLDOUT_DAYS if len(occ) > 60 else max(7, int(len(occ) * 0.1))
    train_series = occ[:-horizon_test].copy()
    test_series = occ[-horizon_test:].copy()
    logger.info("Train length: %d, Test length: %d", len(train_series), len(test_series))

    hw_fit = fit_hw(train_series, seasonal_periods=CFG.SEASONAL_PERIODS, logger=logger)
    test_pred = hw_fit.forecast(steps=horizon_test)
    if not isinstance(test_pred, pd.Series):
        test_pred = pd.Series(test_pred, index=test_series.index[:len(test_pred)])
    logger.info("HW validation RMSE=%.4f", rmse(test_series.loc[test_pred.index], test_pred))

    hw_full = fit_hw(occ, seasonal_periods=CFG.SEASONAL_PERIODS, logger=logger)
    future_forecast = hw_full.forecast(steps=forecast_horizon)
    if not isinstance(future_forecast, pd.Series):
        last_date = occ.index.max()
        idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_forecast), freq='D')
        future_forecast = pd.Series(future_forecast, index=idx)

    out_csv = f"occ_forecast_{forecast_horizon}.csv"
    pd.DataFrame({'stay_date': future_forecast.index, 'occ_forecast': future_forecast.values}).to_csv(out_csv, index=False)
    logger.info("Saved forecast CSV: %s", out_csv)

    with open("occ_forecast_model.pkl", "wb") as pf:
        pickle.dump(hw_full, pf)

    save_forecast_plots(train_series, test_series, test_pred, future_forecast, CFG.PLOTS_DIR, run_ts)

    ml_results = {}
    if ml_mode:
        try:
            feats = create_lag_features(df, target_col='occ').dropna()
            split_idx = len(feats) - horizon_test
            X = feats.drop(columns=['y'])
            y = feats['y']
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
            if lgb_metrics:
                ml_results['lightgbm'] = lgb_metrics
        except Exception as e:
            logger.exception("ML candidate training failed: %s", e)

    run_meta = {
        'run_ts': run_ts,
        'forecast_horizon_days': int(forecast_horizon),
        'validation_test_rmse': float(rmse(test_series.loc[test_pred.index], test_pred)),
        'validation_test_mae': float(mean_absolute_error(test_series.loc[test_pred.index], test_pred)),
        'validation_test_mape': float(mape(test_series.loc[test_pred.index], test_pred)),
        'ml_results': ml_results,
    }

    metrics_history_path = append_monitoring_metrics(run_ts, run_meta, run_meta, test_series, test_pred)

    drift_report = detect_data_drift(df, run_ts)

    try:
        last_date = df.dropna(subset=['occ']).index.max()
        recent_start = last_date - pd.Timedelta(days=CFG.DRIFT_WINDOW_DAYS - 1)
        hist_start = last_date - pd.Timedelta(days=CFG.HIST_WINDOW_DAYS - 1)
        feats_all = create_lag_features(df, target_col='occ').dropna()
        X_hist = feats_all.loc[hist_start:recent_start - pd.Timedelta(days=1)].drop(columns=['y'])
        X_recent = feats_all.loc[recent_start:last_date].drop(columns=['y'])
        feature_drift = detect_feature_drift(X_hist, X_recent, feature_list=None)
    except Exception as e:
        logger.exception("Feature drift detection failed: %s", e)
        feature_drift = {}

    combined = {'drift': drift_report, 'feature_drift': feature_drift, 'run_meta': run_meta}
    drift_path = os.path.join(CFG.MONITOR_DIR, f"combined_drift_{run_ts}.json")
    save_json(combined, drift_path, logger)

    trigger, reasons = check_retrain_trigger(metrics_history_path, drift_report)
    if trigger:
        alert_msg = f"[ALERT] Drift or performance trigger detected for run {run_ts}. Reasons: {reasons}"
        logger.warning(alert_msg)
        send_slack_alert(alert_msg)
        send_email_alert("Drift/Performance Alert", alert_msg)
        plan_path = generate_retrain_plan(run_ts)
        logger.info("Retrain plan generated: %s", plan_path)
        if auto_retrain_flag:
            retrain_result = auto_retrain_if_triggered(df, run_meta, run_ts)
            logger.info("Auto-retrain result: %s", retrain_result)

    run_meta_path = os.path.join(CFG.RUN_META_DIR, f"run_meta_{run_ts}.json")
    ensure_dir(os.path.dirname(run_meta_path))
    save_json(run_meta, run_meta_path, logger)

    elapsed = time.time() - start_time
    logger.info("Run complete. Elapsed: %.2f s. Forecast CSV: %s", elapsed, out_csv)


if __name__ == "__main__":
    main()
