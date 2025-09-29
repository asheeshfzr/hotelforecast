import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def save_forecast_plots(train_series, test_series, test_pred, future_forecast, out_dir, run_ts):
    save_dir = os.path.join(out_dir, run_ts)
    ensure_dir(save_dir)
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(train_series.index, train_series, label='train')
        plt.plot(test_series.index, test_series, label='test (actual)')
        plt.plot(test_series.index[:len(test_pred)], test_pred, label='test (pred)', linestyle='--')
        if future_forecast is not None:
            plt.plot(future_forecast.index, future_forecast.values, label='future forecast')
        plt.legend()
        plt.title("Train / Test / Forecast")
        plt.tight_layout()
        p = os.path.join(save_dir, "forecast_vs_actual.png")
        plt.savefig(p, dpi=150)
        plt.close()
        logger.info("Saved forecast plot: %s", p)
    except Exception as e:
        logger.exception("Could not save forecast plots: %s", e)
