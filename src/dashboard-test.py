# app/dashboard.py
# streamlit run app/dashboard.py
import os
import sys, pathlib
from typing import Optional

# --- make project root importable (quick, portable fix) -----------------
ROOT = pathlib.Path(__file__).resolve().parent.parent  # …/signal-lake
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from streamlit_autorefresh import st_autorefresh

from src.config import settings

import pytz
from datetime import datetime, time

ET = pytz.timezone("America/New_York")

def today_open_utc() -> datetime:
    now_utc = datetime.now(pytz.UTC)
    now_et = now_utc.astimezone(ET)
    open_et = ET.localize(datetime.combine(now_et.date(), time(9, 30)))
    return open_et.astimezone(pytz.UTC)


# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="SignalLake", layout="wide")
st.title("📈 SignalLake — Real-Time Forecasts")

# ── auto-refresh every 30 s ───────────────────────────────────────────────
st_autorefresh(interval=30_000, limit=100, key="datarefresh")

# ── DB connection (cached) ────────────────────────────────────────────────
@st.cache_resource(ttl=60)
def get_engine():
    pg_user = os.getenv("PG_USER", "postgres")
    pg_pwd  = os.getenv("PG_PASSWORD", "signalpass")
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_db   = os.getenv("PG_DB", "signallake_db")
    url = f"postgresql+psycopg2://{pg_user}:{pg_pwd}@{pg_host}/{pg_db}"
    return create_engine(url, pool_pre_ping=True)

engine = get_engine()
ticker = settings.primary_symbol   # single-symbol mode

# ── sidebar controls (add a range picker) ────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Ticker:** `{ticker}`")
    horizon = st.slider("Forecast horizon (minutes)", 1, 60, 60)
    model_options = ["ensemble", "arima", "xgb"]
    models_selected = st.multiselect("Models to show", model_options, default=["ensemble"])
    range_choice = st.radio("Show range", ["Today", "Last 6h", "Last 24h"], index=0)
    st.caption("Data refreshes every 30 seconds.")

# ── SQL depending on range (NY trading day for 'Today') ───────────────────
if range_choice == "Today":
    bars_sql = """
        SELECT ts, close
        FROM market_data
        WHERE ticker = %(ticker)s
          AND (ts AT TIME ZONE 'America/New_York')::date =
              (now() AT TIME ZONE 'America/New_York')::date
        ORDER BY ts
    """
    pred_sql = """
        SELECT ts, horizon_min, model, pred_value
        FROM predictions
        WHERE ticker = %(ticker)s
          AND horizon_min <= %(horizon)s
          AND (ts AT TIME ZONE 'America/New_York')::date =
              (now() AT TIME ZONE 'America/New_York')::date
    """
    params = {"ticker": ticker, "horizon": horizon}

elif range_choice == "Last 24h":
    bars_sql = """
        SELECT ts, close
        FROM market_data
        WHERE ticker = %(ticker)s
          AND ts >= NOW() - INTERVAL '24 hours'
        ORDER BY ts
    """
    pred_sql = """
        SELECT ts, horizon_min, model, pred_value
        FROM predictions
        WHERE ticker = %(ticker)s
          AND horizon_min <= %(horizon)s
          AND ts >= NOW() - INTERVAL '24 hours'
    """
    params = {"ticker": ticker, "horizon": horizon}

else:  # "Last 6h"
    bars_sql = """
        SELECT ts, close
        FROM market_data
        WHERE ticker = %(ticker)s
          AND ts >= NOW() - INTERVAL '6 hours'
        ORDER BY ts
    """
    pred_sql = """
        SELECT ts, horizon_min, model, pred_value
        FROM predictions
        WHERE ticker = %(ticker)s
          AND horizon_min <= %(horizon)s
          AND ts >= NOW() - INTERVAL '6 hours'
    """
    params = {"ticker": ticker, "horizon": horizon}

bars = pd.read_sql(bars_sql, engine, params=params)
pred  = pd.read_sql(pred_sql,  engine, params=params)

# ── reshape predictions tidy → wide for plotting ─────────────────────────
if not pred.empty:
    # ensure only our 3 models
    pred = pred[pred["model"].isin(model_options)]
    # apply user subset for the chart
    pred_chart = pred[pred["model"].isin(models_selected)]
    pred_wide = (
        pred_chart.pivot_table(index="ts", columns="model", values="pred_value")
        .sort_index()
        .reset_index()
    )
else:
    pred_wide = pd.DataFrame()

# ── main chart ────────────────────────────────────────────────────────────
fig = go.Figure()
if not bars.empty:
    fig.add_trace(
        go.Scatter(
            x=bars["ts"],
            y=bars["close"],
            name="Close",
            line=dict(color="#1696d2"),
        )
    )

color_map = {"ensemble": "black", "arima": "#2ca02c", "xgb": "#d62728"}
for model in models_selected:
    if (not pred_wide.empty) and (model in pred_wide.columns):
        fig.add_trace(
            go.Scatter(
                x=pred_wide["ts"],
                y=pred_wide[model],
                name=f"{model} forecast",
                line=dict(dash="dash", color=color_map.get(model)),
            )
        )

fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title=None,
    yaxis_title="Price",
)
st.plotly_chart(fig, use_container_width=True)

# ── KPI cards ─────────────────────────────────────────────────────────────
latest_price = bars["close"].iloc[-1] if not bars.empty else None

col1, col2, col3 = st.columns(3)
col1.metric("Latest price", f"${latest_price:,.2f}" if latest_price is not None else "—")

# ── rolling RMSE (per model), last 200 realized points for selected horizon ──
rmse_sql = """
WITH ranked AS (
  SELECT
    p.model,
    p.ts,
    p.pred_value,
    m.close,
    ROW_NUMBER() OVER (PARTITION BY p.model ORDER BY p.ts DESC) AS rn
  FROM predictions p
  JOIN market_data m
    ON m.ticker = p.ticker
   AND m.ts     = p.ts
  WHERE p.ticker = %(ticker)s
    AND p.horizon_min = %(horizon)s
    AND p.model IN ('ensemble','arima','xgb')
)
SELECT model, sqrt(avg( (pred_value - close)^2 )) AS rmse
FROM ranked
WHERE rn <= 200
GROUP BY model;
"""
rmse_df = pd.read_sql(rmse_sql, engine, params={"ticker": ticker, "horizon": horizon})

def rmse_badge(val: Optional[float]) -> str:
    if val is None: return "🟦"
    if val < 0.50:  return "🟢"
    if val < 1.00:  return "🟠"
    return "🔴"

best_model: Optional[str] = None 
best_rmse: Optional[float] = None 
if not rmse_df.empty: 
    best_idx = rmse_df["rmse"].idxmin() 
    best_model = str(rmse_df.loc[best_idx, "model"]) 
    best_rmse = float(rmse_df.loc[best_idx, "rmse"])


# ── KPI #2: best-model forecast at the exact selected horizon ─────────────
best_pred_sql = """
SELECT pred_value
FROM predictions
WHERE ticker = %(ticker)s
  AND model  = %(model)s
  AND horizon_min = %(horizon)s
ORDER BY ts DESC
LIMIT 1;
"""

best_forecast_val: Optional[float] = None
if best_model is not None:
    row = pd.read_sql(
        best_pred_sql, engine,
        params={"ticker": ticker, "model": best_model, "horizon": horizon}
    )
    if not row.empty and row.iloc[0, 0] is not None:
        best_forecast_val = float(row.iloc[0, 0])

# If no best model (no data yet), fall back to ensemble
if best_model is None:
    best_model = "ensemble"
    row = pd.read_sql(
        best_pred_sql, engine,
        params={"ticker": ticker, "model": best_model, "horizon": horizon}
    )
    if not row.empty and row.iloc[0, 0] is not None:
        best_forecast_val = float(row.iloc[0, 0])

# Show KPI using best model & its latest exact-horizon forecast
if latest_price is not None and best_forecast_val is not None:
    pct = (best_forecast_val - latest_price) / latest_price * 100
    col2.metric(
        f"{horizon}-min {best_model.upper()} forecast",
        f"${best_forecast_val:,.2f}",
        f"{pct:+.2f}%",
    )
else:
    col2.metric(f"{horizon}-min forecast (best model)", "—")

# ── KPI #3: show best RMSE with model name and badge ──────────────────────
if best_rmse is not None and best_model is not None:
    col3.metric(f"Best rolling RMSE (n=200) — {best_model.upper()}", f"{best_rmse:.3f}")
    col3.write(rmse_badge(best_rmse))
else:
    col3.metric("Best rolling RMSE (n=200)", "—")
    col3.write(rmse_badge(None))
