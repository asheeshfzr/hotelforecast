import logging
from typing import Tuple, Dict, Any

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse_val = float(np.sqrt(mse))
    mae_val = float(mean_absolute_error(y_true, y_pred))
    mask = (np.array(y_true) != 0)
    if mask.sum() > 0:
        mape_val = float(np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100.0)
    else:
        mape_val = None
    return {'rmse': rmse_val, 'mae': mae_val, 'mape': mape_val}


def train_lightgbm(X_train, y_train, X_val, y_val, num_boost_round=200, early_stopping_rounds=20):
    if not HAS_LGB:
        logging.getLogger(__name__).info("LightGBM not available.")
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
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    preds = model.predict(X_val)
    return model, compute_metrics(y_val, preds)
