import os

# General
SEED = 42
MARKET_CSV = os.getenv("MARKET_CSV", "market.csv")
PLOTS_DIR = os.getenv("PLOTS_DIR", os.path.join("reports", "plots"))
MONITOR_DIR = os.path.join("data", "monitoring")
RUN_META_DIR = os.path.join("data", "run_meta")
RETRAIN_DIR = os.path.join("data", "retrain")
MODELS_DIR = os.path.join("models")

# Forecast
DEFAULT_FORECAST_HORIZON = 365
FAST_FORECAST_HORIZON = 10
HOLDOUT_DAYS = 30
SEASONAL_PERIODS = 365

# Monitoring
DRIFT_WINDOW_DAYS = int(os.getenv("DRIFT_WINDOW_DAYS", 90))
HIST_WINDOW_DAYS = int(os.getenv("HIST_WINDOW_DAYS", 365 * 2))
KS_PVALUE_THRESHOLD = float(os.getenv("KS_PVALUE_THRESHOLD", 0.01))
MAE_DRIFT_MULTIPLIER = float(os.getenv("MAE_DRIFT_MULTIPLIER", 1.5))

# Drift effect-size thresholds (configurable via env)
DRIFT_KS_STAT_THRESHOLD = float(os.getenv("DRIFT_KS_STAT_THRESHOLD", 0.3))
DRIFT_MEAN_DIFF_THRESHOLD = float(os.getenv("DRIFT_MEAN_DIFF_THRESHOLD", 0.05))
DRIFT_COHEN_D_THRESHOLD = float(os.getenv("DRIFT_COHEN_D_THRESHOLD", 0.5))
DRIFT_REQUIRE_BOTH_EFFECTS = os.getenv("DRIFT_REQUIRE_BOTH_EFFECTS", "true").lower() in ("1","true","yes")

# Alerts
ALERT_SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
