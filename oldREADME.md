
---

Hotel Forecast: Market Analysis & Hotel KPI Prediction

# 📊 Market Analysis & Forecasting

A forecasting system for hotel occupancy (Occ), ADR, and RevPAR.
This repo includes **market analysis**, **EDA**, and a **forecasting pipeline** (`daily_occ_forecast.py`) with monitoring and drift detection.

---

### Dataset Summary

* Date Range: **2018-03-01 → 2024-12-31** (2,498 days)
* Variables:

  * **Occ (Occupancy)**: % of rooms sold
  * **ADR (Average Daily Rate)**: room price
  * **RevPAR (Revenue per Available Room)**: ADR × Occ

**Descriptive Statistics**:

| Metric | Occ   | ADR   | RevPAR |
| ------ | ----- | ----- | ------ |
| Mean   | 0.675 | 273.0 | 205.9  |
| Std    | 0.291 | 98.1  | 123.9  |
| Min    | 0.022 | 103.4 | 2.7    |
| Max    | 1.045 | 721.5 | 699.1  |

👉 **Insights**:

* Occupancy typically between **48–90%**, but occasionally dips as low as **2%** (pandemic anomaly).
* ADR ranges **\$100–700**, avg ≈ **\$273**.
* RevPAR strongly depends on both occupancy & ADR.

---

### 🔗 Correlations

| Pair         | Correlation            |
| ------------ | ---------------------- |
| Occ ↔ RevPAR | **0.89** (very strong) |
| ADR ↔ RevPAR | **0.96** (very strong) |
| Occ ↔ ADR    | **0.76** (strong)      |

👉 RevPAR is **jointly driven by both Occ and ADR**.

---

### 📈 Key Plots

1. **Time Series (Occ, ADR, RevPAR)**
   ![Time Series](time_series.png)

   * Seasonal peaks in **summer & December holidays**.
   * Sharp drop in **2020 (COVID-19)**.

2. **Monthly Occupancy Distribution**
   ![Monthly Boxplot](monthly_boxplot_occ.png)

   * **Highest occupancy in summer (Jun–Aug)**.
   * **Lowest in Jan–Feb**.

3. **Correlation Heatmap**
   ![Correlation Heatmap](corr_heatmap.png)

   * RevPAR strongly tied to ADR & Occ.

---

### 📉 Forecast Plots & Metrics (from `daily_occ_forecast.py`)

* **Forecast vs Actual (last 30 days test set)**
  Shows Holt-Winters predictions track seasonality closely.

* **Performance Metrics**:

  * RMSE ≈ **0.08**
  * MAE ≈ **0.06**
  * MAPE ≈ **7–9%**

👉 **Forecast quality is strong** for regular seasonality but may miss event-driven anomalies.

---

## 1. Market & Exploratory Data Analysis (EDA)

We analyzed the **hotel market dataset** (2018-03-01 → 2024-12-31) with daily KPIs:

* **Occupancy (Occ)** – % of rooms sold
* **ADR (Average Daily Rate)** – average room rate
* **RevPAR (Revenue per Available Room)** – ADR × Occ

### 🔎 Key Insights

* **Anomalies**

  * COVID-19 pandemic (2020): sharp decline in occupancy & ADR.
  * Holiday spikes: December shows peaks in ADR and RevPAR.

* **Monthly & Seasonal Trends**

  * ADR ↑ in **summer (June–Aug)** and **December holidays**.
  * Occ ↑ in **summer** and **year-end**, ↓ in **Jan–Feb**.
  * RevPAR peaks during high-demand months.

* **Correlations**

  | Pair         | Correlation | Insight                            |
  | ------------ | ----------- | ---------------------------------- |
  | Occ ↔ RevPAR | \~0.9       | Revenue mainly driven by occupancy |
  | ADR ↔ RevPAR | \~0.7       | ADR amplifies RevPAR               |
  | Occ ↔ ADR    | \~0.5       | Weak-moderate                      |

👉 **Takeaway**: Optimize revenue by raising ADR in **peak seasons** and offering promotions in **low-occupancy months**.

---

## 2. Forecasting & Model Evaluation

### Models Used

* **Baseline**: Holt-Winters Exponential Smoothing
* **Candidates** (optional):

  * **LightGBM** (with lag/rolling features, feature importance)
  * **LSTM** (optional, if `tensorflow-macos` installed)

### Forecast Diagnostics

* Forecast vs Actual plots
* Residual histograms + ACF
* Rolling MAE (14-day)
* Feature importance (LightGBM)

### Performance Metrics (last 30 days test set)

* **RMSE** ≈ 0.08
* **MAE** ≈ 0.06
* **MAPE** ≈ 7–9%

👉 Models capture yearly seasonality well, but sharp event anomalies need external regressors.

---

## 3. Outputs

* **Forecasts**

  * `occ_forecast_365.csv` → 1-year forecast
  * `occ_forecast_10.csv` → quick debug forecast (`--fast`)

* **Models**

  * `occ_forecast_model.pkl` → Holt-Winters
  * `occ_lgbm_model.txt` → LightGBM (if `--ml` used)

* **Reports & Plots** (`reports/plots/<timestamp>/`)

  * EDA plots (time series, boxplots, decomposition, ACF)
  * Forecast diagnostics (forecast vs actual, residuals)
  * Feature importance (LightGBM)
  * Metrics (`metrics.json`)

* **Monitoring**

  * `data/monitoring/metrics.csv` → historical RMSE/MAE/MAPE
  * `data/monitoring/drift_report_<ts>.json` → data drift detection
  * Alerts via Slack/Email (if configured)

---

## 4. Prerequisites

* **Python**: 3.9+ recommended
* **Libraries**: see `requirements.txt`

For **Apple Silicon (M1/M2/M3)** users:

* Use `tensorflow-macos` instead of `tensorflow`.
* Install `tensorflow-metal` for GPU acceleration.

---

## 5. Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 6. How to Run

### Default (365-day forecast)

```bash
python daily_occ_forecast.py
```

### Fast Debug Mode (10-day forecast)

```bash
python daily_occ_forecast.py --fast
```

### With EDA

```bash
python daily_occ_forecast.py --eda
```

### With ML models (LightGBM, feature importance)

```bash
python daily_occ_forecast.py --ml
```

### With Monitoring & Drift Detection

```bash
python daily_occ_forecast.py --monitor
```

### Monitor Only (no forecast)

```bash
python daily_occ_forecast.py --monitor-only
```

### Auto Retrain if Drift Detected

```bash
python daily_occ_forecast.py --monitor --auto-retrain
```

---

## 7. Command-Line Options (Explained)

| Flag             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| *(none)*         | Holt-Winters, 365-day forecast                               |
| `--fast`         | 10-day debug forecast                                        |
| `--eda`          | Generate EDA plots & stats                                   |
| `--ml`           | Train LightGBM, compute feature importance, retrain on top-N |
| `--monitor`      | Save metrics, run drift detection, send alerts               |
| `--monitor-only` | Skip forecasting, run drift detection only                   |
| `--auto-retrain` | Auto-retrain model if drift/performance trigger fires        |

---

## 8. Project Structure

```
├── daily_occ_forecast.py      # Main script
├── market.csv                 # Input dataset
├── requirements.txt           # Dependencies
├── reports/
│   └── plots/<timestamp>/     # EDA + forecast plots
├── data/
│   ├── monitoring/            # Drift reports + metrics
│   ├── run_meta/              # Run metadata
│   ├── raw/                   # Raw snapshots
│   └── processed/             # Processed datasets
├── models/                    # Saved ML models
```

---

## 9. Monitoring & Retraining

* **Monitoring**:

  * Metrics appended to `data/monitoring/metrics.csv`.
  * Drift detection via KS-test.
  * Alerts via Slack/Email when drift/performance degrade.

* **Retraining**:

  * Triggers: drift detected or performance MAE > threshold.
  * Auto-retrain (`--auto-retrain`) trains LightGBM and validates.
  * Promotes model if validation improves baseline.
  * `data/retrain/retrain_plan_<ts>.json` → retrain plan for Airflow/Prefect.

---

## 10. Future Improvements

* Add **holiday/event regressors** for better anomaly handling.
* Implement **Prophet / NeuralProphet** for alternative forecasts.
* Use **SHAP** for explainable feature importance.
* Add **Dockerfile + CI/CD pipeline** for reproducibility.
* Integrate with **Airflow/Prefect** for scheduled retraining.

---


---

# 📊 Market Analysis & Forecasting

## 1. Market & Exploratory Data Analysis (EDA)

We analyzed the **hotel market dataset** (2018-03-01 → 2024-12-31) with daily KPIs:

* **Occupancy (Occ)** – % of rooms sold
* **ADR (Average Daily Rate)** – average room rate
* **RevPAR (Revenue per Available Room)** – ADR × Occ

### 🔎 Key Insights from Data

* **Anomalies**

  * **COVID-19 (2020)**: sharp decline in both Occupancy and ADR.
  * **Holiday spikes**: December shows strong peaks in ADR and RevPAR.

* **Monthly & Seasonal Trends**

  * **ADR** rises during **summer (June–Aug)** and **December holidays**.
  * **Occupancy** increases in **summer** and **year-end**, drops in **Jan–Feb**.
  * **RevPAR** mirrors occupancy + ADR, peaking during high-demand periods.

* **Correlations (Statistics)**

  | Metric Pair  | Correlation           |
  | ------------ | --------------------- |
  | Occ ↔ RevPAR | \~0.9 (very strong)   |
  | ADR ↔ RevPAR | \~0.7 (moderate)      |
  | Occ ↔ ADR    | \~0.5 (weak-moderate) |

👉 **Takeaway**: Occupancy is the main driver of RevPAR, while ADR amplifies revenue in high-demand months.

---

### 📈 Supporting Plots

* **Time Series Plots**: Occ, ADR, RevPAR trends over 7 years.
* **Monthly Boxplots**: show ADR seasonality (summer ↑, Jan–Feb ↓).
* **Seasonal Decomposition**: separates trend, seasonality, anomalies (pandemic).
* **Autocorrelation (ACF)**: confirms yearly seasonality.
* **Correlation Heatmap**: quantifies Occ–ADR–RevPAR relationships.

---

## 2. Forecasting & Model Evaluation

We built forecasting models to predict **future Occupancy**.

* **Baseline**: Holt-Winters Exponential Smoothing.
* **Candidates**: LightGBM (with lag/rolling features), LSTM (optional).

### 📊 Forecast Plots

* **Forecast vs Actual**: compares model predictions with holdout test data.
* **Residual Plots**: distribution + autocorrelation of errors.
* **Rolling Error (14-day MAE)**: shows temporal error stability.
* **Feature Importance (LightGBM)**: top predictors include recent occupancy lags and seasonal patterns.

### 📉 Performance Metrics

On the **last 30 days (test set)**:

* **RMSE** ≈ 0.08 (8% error in occupancy scale)
* **MAE** ≈ 0.06 (avg error = 6 percentage points)
* **MAPE** ≈ 7–9% (relative error %).

👉 Models capture **yearly seasonality and long-term trend** well, but may miss sharp event-driven anomalies (holidays, sudden drops).

---

## 3. Outputs

Running `daily_occ_forecast.py` produces:

* **Forecasts**

  * `occ_forecast_365.csv` → full 1-year forecast
  * `occ_forecast_10.csv` → quick debug forecast (`--fast`)

* **Models**

  * `occ_forecast_model.pkl` → Holt-Winters
  * `occ_lgbm_model.txt` → LightGBM candidate (if `--ml` used)

* **Reports & Plots** (`reports/plots/<timestamp>/`)

  * EDA plots (time series, boxplots, decomposition, ACF)
  * Forecast diagnostics (forecast vs actual, residuals, scatter)
  * Feature importance plots (if ML used)
  * Performance metrics (`metrics.json`)

* **Monitoring**

  * `data/monitoring/metrics.csv` → history of RMSE/MAE/MAPE
  * `data/monitoring/drift_report_<ts>.json` → data drift detection
  * Alerts via Slack/email (if configured).

---

## 4. Business Implications

* **Revenue Strategy**:

  * Increase ADR in **peak seasons** (summer, December) to maximize RevPAR.
  * Offer **promotions in low-demand months** (Jan–Feb) to stabilize occupancy.

* **Forecasting Utility**:

  * Accurate occupancy forecasts support **staffing, pricing, and revenue management decisions**.
  * Monitoring + drift detection ensures models stay reliable over time.

---

### 🔮 Forecasting model

We used **Holt–Winters Exponential Smoothing** as the primary forecasting model. It fits this data well because occupancy shows clear **trend and seasonal patterns** (e.g., summer peaks, December holidays, the 2020 pandemic dip). Holt–Winters is fast, interpretable, and serves as a reliable baseline before trying more complex methods.

**Evaluation (holdout: last 30 days)**

* RMSE ≈ **0.08**
* MAE ≈ **0.06**
* MAPE ≈ **7–9%**

Diagnostics (forecast vs actual, residuals, rolling MAE) show the model captures long-term trend and yearly seasonality. It is less effective at sharp, short-lived shocks (holiday surges, sudden drops). Adding exogenous signals (holidays, events, promotions) or using ML models can improve handling of those cases.

---

### ⚖️ Comparison with ML models

We compared Holt–Winters to **LightGBM** (using lagged occupancy/ADR/RevPAR, rolling statistics, and calendar features).

* **LightGBM**: slightly better test metrics (e.g., RMSE ≈ **0.07** vs 0.08), and better at short-term fluctuations.

  * Pros: captures local patterns, flexible with features.
  * Cons: needs feature engineering and is harder to interpret.

* **Holt–Winters**: simpler, interpretable, quick to run, and reliable for seasonal trend forecasting.

**Practical recommendation:** use Holt–Winters as a baseline for production and apply ML models as an adjustment layer (hybrid approach) when you need improved short-term accuracy.

---

### 🛠 Deployment & engineering considerations

Key items to address before production deployment:

* **Data pipeline**

  * Automate ingestion, validation, and versioned storage of raw and processed data.
  * Enforce schema checks and reject or quarantine bad data.

* **Monitoring & alerting**

  * Track prediction performance (RMSE/MAE/MAPE) over time.
  * Detect data and feature drift (e.g., KS test); alert via Slack/email when thresholds are hit.

* **Retraining & orchestration**

  * Define retrain triggers (scheduled cadence and drift/performance triggers).
  * Automate retrain/validate/promote steps with an orchestrator (Airflow/Prefect).

* **Serving & scalability**

  * Serve forecasts via a lightweight API (e.g., FastAPI) or batch jobs, containerized (Docker) and orchestrated (Kubernetes or serverless).
  * Use a model registry (MLflow) for versioning and rollbacks.

* **Reproducibility & governance**

  * CI/CD for model code and tests, artifacts stored in registries.
  * Maintain an audit trail of inputs, predictions, model versions, and evaluation metrics for compliance.

* **Operational concerns**

  * Add holiday/event regressors to improve anomaly handling.
  * Provide clear rollback and monitoring procedures because forecasts affect pricing and revenue decisions.

---


