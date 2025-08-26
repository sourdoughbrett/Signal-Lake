# app/dashboard.py
# streamlit run app/dashboard.py
import os
import sys, pathlib
from typing import Optional
from datetime import datetime, time, timedelta

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
ET  = pytz.timezone("America/New_York")
UTC = pytz.UTC


def et_bounds_for_today_utc() -> tuple[datetime, datetime]:
    """Return UTC datetimes for today's midnight ET and 23:59:59 ET."""
    now_et = datetime.now(ET)
    start_et = ET.localize(datetime.combine(now_et.date(), time(0, 0)))
    end_et   = ET.localize(datetime.combine(now_et.date(), time(23, 59, 59)))
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


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
    model_options = ["ensemble", "arima", "xgb"]  # no LSTM
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
    pred = pred[pred["model"].isin(model_options)]      # guard against legacy rows
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

# Shade pre-market (00:00–09:30 ET) and after-hours (16:00–23:59 ET) for “Today”
if range_choice == "Today" and not bars.empty:
    start_utc, end_utc = et_bounds_for_today_utc()
    pre_start = start_utc
    pre_end   = ET.localize(datetime.combine(datetime.now(ET).date(), time(9, 30))).astimezone(UTC)
    aft_start = ET.localize(datetime.combine(datetime.now(ET).date(), time(16, 0))).astimezone(UTC)
    aft_end   = end_utc
    # pre-market
    fig.add_vrect(x0=pre_start, x1=pre_end, fillcolor="LightGray", opacity=0.15, line_width=0)
    # after-hours
    fig.add_vrect(x0=aft_start, x1=aft_end, fillcolor="LightGray", opacity=0.15, line_width=0)

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

# Rolling RMSE per model (last 200 realized pairs at selected horizon) + counts
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
SELECT model,
       COUNT(*) AS n,
       sqrt(avg( (pred_value - close)^2 )) AS rmse
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
best_n: Optional[int] = None
if not rmse_df.empty:
    # prefer models with at least 10 realized pairs
    eligible = rmse_df[rmse_df["n"] >= 10]
    pick_df = eligible if not eligible.empty else rmse_df
    best_idx = pick_df["rmse"].idxmin()
    best_model = str(pick_df.loc[best_idx, "model"])
    best_rmse  = float(pick_df.loc[best_idx, "rmse"])
    best_n     = int(pick_df.loc[best_idx, "n"])

# KPI #2: best-model forecast at the exact selected horizon
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
    row = pd.read_sql(best_pred_sql, engine, params={"ticker": ticker, "model": best_model, "horizon": horizon})
    if not row.empty and row.iloc[0, 0] is not None:
        best_forecast_val = float(row.iloc[0, 0])
else:
    # fall back to ensemble if nothing yet
    row = pd.read_sql(best_pred_sql, engine, params={"ticker": ticker, "model": "ensemble", "horizon": horizon})
    if not row.empty and row.iloc[0, 0] is not None:
        best_model = "ensemble"
        best_forecast_val = float(row.iloc[0, 0])

if latest_price is not None and best_forecast_val is not None and best_model is not None:
    pct = (best_forecast_val - latest_price) / latest_price * 100
    label = f"{horizon}-min {best_model.upper()} forecast"
    col2.metric(label, f"${best_forecast_val:,.2f}", f"{pct:+.2f}%")
else:
    col2.metric(f"{horizon}-min forecast (best model)", "—")

# KPI #3: show best RMSE with model name, badge, and n
if best_rmse is not None and best_model is not None:
    n_txt = f" • n={best_n}" if best_n is not None else ""
    col3.metric(f"Best rolling RMSE {n_txt}", f"{best_rmse:.3f}", help=f"Best model: {best_model.upper()}")
    col3.write(rmse_badge(best_rmse))
else:
    col3.metric("Best rolling RMSE ", "—")
    col3.write(rmse_badge(None))

# (Optional) tiny freshness footer
if not bars.empty:
    st.caption(f"Data freshness — last bar: {pd.to_datetime(bars['ts'].iloc[-1]).tz_convert('US/Eastern'):%Y-%m-%d %H:%M ET}")
