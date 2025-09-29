# 🏨 Hotel Forecast: Market Analysis & KPI Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-EDA-blue)
![Forecasting](https://img.shields.io/badge/Forecasting-Holt--Winters%20%7C%20LightGBM%20%7C%20LSTM-green)

**Hotel Forecast** is an end-to-end system for **market analysis, exploratory data analysis (EDA), forecasting, and monitoring** of key hotel performance indicators:

* **Occupancy (Occ)** → % of rooms sold
* **Average Daily Rate (ADR)** → average room price
* **Revenue per Available Room (RevPAR)** → ADR × Occ

The project includes:
✅ **EDA & Market Insights**
✅ **Forecasting Pipeline (`daily_occ_forecast.py`)**
✅ **Monitoring + Drift Detection**
✅ **Business Strategy Implications**

---

## 📊 Dataset Overview

* **Date Range:** 2018-03-01 → 2024-12-31 (2,498 days)
* **Metrics:** Occ, ADR, RevPAR

| Metric | Occ   | ADR   | RevPAR |
| ------ | ----- | ----- | ------ |
| Mean   | 0.675 | 273.0 | 205.9  |
| Std    | 0.291 | 98.1  | 123.9  |
| Min    | 0.022 | 103.4 | 2.7    |
| Max    | 1.045 | 721.5 | 699.1  |

👉 **Key Insights**

* Occ typically **48–90%**, with anomalies (COVID dip to ~2%).
* ADR ranges **$100–700**, avg ≈ $273.
* RevPAR highly correlated with both Occ (**0.89**) and ADR (**0.96**).

---

## 🔎 Market & EDA Highlights

* **Seasonality**:

  * Summer (Jun–Aug) & December holidays → peaks in Occ + ADR.
  * Jan–Feb → lowest demand.

* **Pandemic Anomaly**:

  * 2020 showed unprecedented drops in both ADR and Occ.

* **Correlations**:

  * Occ ↔ RevPAR ≈ **0.9** (main driver).
  * ADR ↔ RevPAR ≈ **0.7** (amplifier).
  * Occ ↔ ADR ≈ **0.5** (moderate).

👉 **Takeaway:** Raise ADR in peak seasons, offer promotions in low-occupancy months.

---

## 📈 Forecasting

### Models Implemented

* **Baseline:** Holt–Winters Exponential Smoothing
* **Candidates:**

  * LightGBM → uses lagged & rolling features
  * LSTM → (requires `tensorflow-macos` on Apple Silicon)

### Metrics (last 30 test days)

* RMSE ≈ **0.08**
* MAE ≈ **0.06**
* MAPE ≈ **7–9%**

👉 Models capture **seasonality & trend** well but may miss **event-driven anomalies**.

---

## 🗂 Project Structure

```
├── daily_occ_forecast.py      # Main forecasting script
├── market.csv                 # Input dataset
├── requirements.txt           # Dependencies
├── reports/
│   └── plots/<timestamp>/     # EDA + forecast plots
├── data/
│   ├── monitoring/            # Drift reports + metrics
│   ├── raw/                   # Raw data snapshots
│   ├── processed/             # Processed datasets
│   └── run_meta/              # Run metadata
├── models/                    # Saved ML models
```

---

## ⚙️ Installation

### Prerequisites

* Python **3.9+**
* See `requirements.txt` for dependencies.

For Apple Silicon (M1/M2/M3):

```bash
pip install tensorflow-macos tensorflow-metal
```

### Setup

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate    # Mac/Linux
# .venv\Scripts\activate     # Windows

# Upgrade pip & install deps
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Default (365-day forecast)

```bash
python daily_occ_forecast.py
```

### Fast Debug (10-day forecast)

```bash
python daily_occ_forecast.py --fast
```

### With EDA

```bash
python daily_occ_forecast.py --eda
```

### With ML (LightGBM, feature importance)

```bash
python daily_occ_forecast.py --ml
```

### With Monitoring & Drift Detection

```bash
python daily_occ_forecast.py --monitor
```

### Auto Retrain on Drift

```bash
python daily_occ_forecast.py --monitor --auto-retrain
```

---

## 📉 Monitoring & Drift Detection

* **Performance tracking:** RMSE, MAE, MAPE → logged in `metrics.csv`.
* **Data drift detection:** KS-test → reports in `drift_report_<ts>.json`.
* **Alerts:** Slack/Email (configurable).
* **Auto-Retrain:** LightGBM retrains & promotes if performance improves.

---

## 🔮 Business Impact

* **Revenue Optimization**: Adjust ADR in high/low demand seasons.
* **Operational Planning**: Staffing & resource allocation based on forecasted occupancy.
* **Decision Support**: Data-driven pricing strategies for maximizing RevPAR.

---

## 🛠 Future Improvements

* Add **holiday/event regressors** for anomaly handling.
* Experiment with **Prophet / NeuralProphet**.
* Integrate **SHAP** for explainable ML forecasts.
* Containerize with **Docker + Kubernetes**.
* Automate retraining via **Airflow/Prefect**.
* Add CI/CD pipeline & model registry (MLflow).

---

## 📜 License

MIT License © 2025 — Contributions welcome 🚀

---
