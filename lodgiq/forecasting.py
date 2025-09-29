import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def rmse(true, pred):
    return float(np.sqrt(mean_squared_error(true, pred)))


def mape(true, pred):
    true_arr = np.array(true)
    pred_arr = np.array(pred)
    mask = true_arr != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((true_arr[mask] - pred_arr[mask]) / true_arr[mask])) * 100.0)


def fit_hw(train_series, seasonal_periods=365, logger=None):
    if logger:
        logger.info("Fitting Holt-Winters ExponentialSmoothing")
    try:
        model = ExponentialSmoothing(
            train_series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods,
            initialization_method='estimated',
        )
        fit = model.fit(optimized=True)
        return fit
    except Exception as e:
        if logger:
            logger.warning("HW seasonal fit failed (%s). Falling back to additive trend only.", e)
        model = ExponentialSmoothing(train_series, trend='add', seasonal=None, initialization_method='estimated')
        fit = model.fit(optimized=True)
        return fit
