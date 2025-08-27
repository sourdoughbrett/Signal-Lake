# **SignalLake** — Local MVP

Real-time, single-ticker forecasting pipeline that ingests minute bars, engineers features, trains ARIMA + XGBoost, writes predictions/metrics to Postgres, and serves an interactive Streamlit dashboard.

## **Overview**

*SignalLake is a compact, production-style demo that brings data engineering, ML modeling, and analytics together in one codebase.*

**Ingestion:** pulls 1-minute OHLCV from Alpaca and keeps a rolling 7-day history in Postgres (idempotent upserts).

**Feature engineering:** adds technical indicators (e.g., MACD, RSI) via features.py.

**Modeling:** trains ARIMA and XGBoost on the latest history; Ensemble = mean(ARIMA, XGB).

**Metrics:** computes rolling RMSE and stores to forecast_metrics.

**Analytics:** Streamlit dashboard with Today / Last 6h / Last 24h chart views, forecast overlays, and KPI cards.

**Limitations:** This MVP is tuned for one symbol at a time to stay simple and fast.

## **Features & Flow**
Architecture (high level)

Alpaca → Ingest
src/data_ingest.py fetches minute bars (rolling 7 days), flattens and enriches with indicators, and upserts into Postgres market_data.

Model Train + Forecast
src/models.py reads the most-recent rows (default TRAIN_ROWS=10000), trains ARIMA (5,1,0) and XGBoost with sliding windows, generates 60-minute forecasts, and upserts into predictions. It then writes rolling RMSE for each model to forecast_metrics.

Dashboard
app/dashboard.py queries Postgres and renders:

Intraday price chart (Today / Last 6h / Last 24h)

Forecast lines (ensemble, ARIMA, XGB)

KPIs: latest price, best model’s exact-horizon forecast, best rolling RMSE (n=200)

**Tech Stack**

- Python 3.9+ (Anaconda or venv)
- Postgres 13+ (local or managed)
- Alpaca Market Data API (must be paper)
- pandas, SQLAlchemy/psycopg2
- statsmodels (ARIMA), xgboost
- Streamlit + Plotly
- Data Model
- market_data(ticker, ts, open, high, low, close, volume, macd, rsi, …features…)
- predictions(ticker, ts, horizon_min, model, pred_value) (PK on all 4)
-forecast_metrics(ticker, ts, model, rmse_100)

## **Quick Start (Local)**
**0) Prerequisites**

Postgres running locally (e.g., localhost:5432)

Python 3.9+

Alpaca API key/secret (***PAPER ONLY***)

**1) Clone & install**
```plaintext
git clone https://github.com/<you>/signal-lake.git
cd signal-lake
pip install -r requirements.txt
```

Conda option:
```plaintext
conda create -n signallake python=3.10 && conda activate signallake && pip install -r requirements.txt
```

**2) Create the database**

In pgAdmin or psql:

CREATE DATABASE signallake_db;

**3) Configure secrets**

Create a .env file in the project root (do not commit real secrets):
```plaintext
# .env (local only)
PG_HOST=localhost
PG_USER=postgres
PG_PASSWORD=signalpass
PG_DB=signallake_db

# Alpaca (paper trading keys)
ALPACA_KEY=YOUR_KEY
ALPACA_SECRET=YOUR_SECRET
```

Edit config/settings.yml for your ticker (keep it single-ticker):
```plaintext
# config/settings.yml
API_KEY: ''         # leave blank (read from env)
SECRET_KEY: ''      # leave blank (read from env)
api_base_url: 'https://paper-api.alpaca.markets'

# Dates are not used by ingest anymore (rolling week is computed automatically)
start_time: "2025-04-01"
end_time: "2025-12-31"

position_size: 5000

symbols:
  - AMD    # set exactly one symbol for local MVP
```
We intentionally read Alpaca keys from environment (.env) and ignore the YAML values for safety.


4) Initialize schema & start the worker

The schema is created automatically at startup.

```plaintext
# terminal #1 — run the ingestion + forecasting loop
python -m src.main
```

You should see logs like:
```plaintext
INFO: Database schema ensured.
INFO: Fetching bars for ['AMD'] | window: [...]
INFO: Upserting N enriched rows into Postgres
INFO: Ingestion step complete ✅
INFO: Generating forecasts…
INFO: Upserted 240 forecast rows.
```

The loop runs ingest every 60s and forecast every 5m.

5) Start the dashboard

In another terminal:

```plaintext
streamlit run app/dashboard.py
```

Open the URL (usually http://localhost:8501
). Use the sidebar to switch between Today / Last 6h / Last 24h.

Important: Keep the worker (src.main) running so the dashboard has fresh data.

---

## **Usage Notes**

Change ticker: update config/settings.yml (symbols: [<TICKER>]), then restart:

1. Stop the worker (src.main) and the dashboard

2. Start python -m src.main again

3. Start streamlit run app/dashboard.py

Backfill behavior: ingest maintains a rolling 7-day window in market_data for the current ticker. This guarantees a full intraday chart on restart.

Ensemble: simple mean of ARIMA and XGB; dashboard highlights best RMSE model and uses it for the main KPI forecast.

RMSE windows: dashboard shows n=200 rolling RMSE per model (aligned realized points).

---

# **Repository Layout**

```plaintext
signal-lake/
├─ app/
│  └─ dashboard.py          # Streamlit app
├─ config/
│  └─ settings.yml          # human config (ticker, etc.)
├─ src/
│  ├─ __init__.py
│  ├─ config.py             # loads settings + env overrides
│  ├─ db.py                 # Postgres helpers & upserts
│  ├─ indicators.py         # TA helpers (TALIB-based)
│  ├─ features.py           # build_feature_df(...) → adds indicators
│  ├─ data_ingest.py        # rolling-week ingest + feature upsert
│  ├─ models.py             # ARIMA + XGB + ensemble + RMSE
│  └─ main.py               # scheduler: ingest/forecast loops
├─ .env.example             # sample env file (no secrets)
├─ requirements.txt
└─ README.md                # you are here
```

---

# **Common Commands**
```plaintext
# run the worker (ingest + forecast)
python -m src.main

# run the dashboard
streamlit run app/dashboard.py

# quick, one-off ingest (debug)
python -m src.data_ingest

# quick, one-off forecast (debug)
python -m src.models
```

---

## FAQ

Dashboard is blank / no chart?
Ensure python -m src.main is running; verify Postgres creds in .env; confirm market_data has rows:
```plaintext
SELECT COUNT(*) FROM market_data WHERE ticker = 'AMD';
```

If zero, check Alpaca keys/permissions.

Switched ticker but still seeing the old one?
After editing config/settings.yml, restart both the worker and the dashboard. The ingest trims data to the current ticker automatically.

RMSE shows 0.000 or NaN
RMSE needs realized pairs (prediction timestamps that now have realized closes). Let the loop run; it will fill in.

Can I show more indicators in the dashboard?
Yes—extend features.py and add new columns to the FEATURE_COLS list in db.py (schema must match). The dashboard currently reads close and predictions, but you can add secondary charts.

LSTM?
The MVP removed LSTM for robustness and speed. It can be reintroduced later if desired.

*Happy forecasting!*
