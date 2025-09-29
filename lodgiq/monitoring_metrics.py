import logging
from datetime import datetime
import os
import numpy as np
import pandas as pd

from .config import MONITOR_DIR
from .forecasting import rmse, mape
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def append_monitoring_metrics(run_ts: str, run_meta: dict, baseline_metrics: dict, test_series: pd.Series, test_pred: np.ndarray):
    ensure_dir(MONITOR_DIR)
    metrics_path = os.path.join(MONITOR_DIR, "metrics.csv")
    metrics_row = {
        'run_ts': run_ts,
        'run_time_utc': datetime.utcnow().isoformat(),
        'forecast_horizon_days': run_meta.get('forecast_horizon_days'),
        'rmse': baseline_metrics.get('rmse') if isinstance(baseline_metrics, dict) else float(rmse(test_series, test_pred)),
        'mae': baseline_metrics.get('mae') if isinstance(baseline_metrics, dict) else float(np.nan if len(test_series)==0 else np.mean(np.abs(test_series - test_pred))),
        'mape': baseline_metrics.get('mape') if isinstance(baseline_metrics, dict) else float(mape(test_series, test_pred)),
        'n_test': len(test_series),
        'notes': run_meta.get('notes', ''),
    }
    df_row = pd.DataFrame([metrics_row])
    if os.path.exists(metrics_path):
        df_row.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(metrics_path, index=False)
    logger.info("Appended monitoring metrics to %s", metrics_path)
    return metrics_path
