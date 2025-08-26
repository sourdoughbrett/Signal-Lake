# src/models.py
"""
Model training & forecasting for SignalLake.
Now uses ARIMA + XGBoost only. Ensemble = mean(ARIMA, XGB).

Upgrades:
  • ARIMA: pick best (p,d,q) by AIC from a small grid on recent data
  • XGBoost: early stopping with a small validation split + sturdier defaults
  • Metrics: compute rolling RMSE for the configured HORIZON for consistency
"""

from __future__ import annotations

import logging
from datetime import datetime as dt, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import settings
from src.db import get_conn, upsert_preds, upsert_metrics

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Hyper-params
# ----------------------------------------------------------------------
SEQ_LEN     = 100        # window length for XGB rolling features
HORIZON     = 60         # minutes into the future to forecast
TRAIN_ROWS  = 10_000     # most-recent rows used for training/forecasting

# ARIMA search limits (keep tiny so it’s fast)
ARIMA_TRAIN_MAX = 5_000
ARIMA_GRID: Tuple[Tuple[int,int,int], ...] = (
    (1,1,0), (2,1,0), (3,1,0), (5,1,0),
    (1,1,1), (2,1,1), (3,1,1)
)

# XGBoost params (robust defaults; early stopping will pick n_estimators)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=2_000,      # upper bound; early-stopping will cut this
    random_state=42,
    n_jobs=-1,
)
XGB_VAL_FRAC = 0.1
XGB_EARLY_STOP = 50

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _fit_best_arima(close: np.ndarray) -> ARIMA:
    """
    Fit a tiny grid of ARIMA orders and return the model with lowest AIC.
    Uses the last ARIMA_TRAIN_MAX points to keep runtime bounded.
    Falls back to (5,1,0) if all fail.
    """
    y = close[-ARIMA_TRAIN_MAX:] if len(close) > ARIMA_TRAIN_MAX else close
    best = None
    best_order = None
    best_aic = np.inf

    for order in ARIMA_GRID:
        try:
            m = ARIMA(y, order=order).fit()
            aic = m.aic if np.isfinite(m.aic) else np.inf
            if aic < best_aic:
                best, best_order, best_aic = m, order, aic
        except Exception as e:
            # benign: some combos may not converge
            continue

    if best is None:
        logger.warning("ARIMA grid failed; falling back to (5,1,0).")
        best = ARIMA(y, order=(5,1,0)).fit()
        best_order = (5,1,0)
        best_aic = best.aic

    logger.info("ARIMA best order=%s (AIC=%.2f)", best_order, best_aic)
    return best

def _sliding_window(arr: np.ndarray, window: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window : i])
        y.append(arr[i])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

# ----------------------------------------------------------------------
# Train ARIMA + XGB
# ----------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> Dict[str, object]:
    """
    Trains ARIMA and XGBoost on 'close' prices.
    Returns dict: {'arima': arima_model, 'xgb': xgb_model}
    """
    import xgboost as xgb

    close = df["close"].values.astype(float)

    # ---- ARIMA (auto-pick) ----
    arima = _fit_best_arima(close)

    # ---- XGBoost with sliding window + early stopping ----
    X_all, y_all = _sliding_window(close, window=SEQ_LEN)
    if len(X_all) < 200:   # sanity guard
        # not enough examples to hold out a validation set
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_all, y_all, verbose=False)
        xgb_model = model
    else:
        n = len(X_all)
        val_n = max(100, int(n * XGB_VAL_FRAC))
        X_tr, y_tr = X_all[:-val_n], y_all[:-val_n]
        X_val, y_val = X_all[-val_n:], y_all[-val_n:]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose=False
        )
        xgb_model = model

    return {"arima": arima, "xgb": xgb_model}

# ----------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------
def generate_forecast(
    models: Dict[str, object],
    recent_df: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Produce horizon-minute forecasts using trained models.
    Returns tidy DF: ticker, ts, horizon_min, model, pred_value
    """
    close_series = recent_df["close"].values.astype(float)
    ts_start = recent_df["ts"].iloc[-1]

    # ---- ARIMA ----
    arima_preds = np.array(models["arima"].forecast(steps=horizon), dtype=float)

    # ---- XGB (rolling one-step) ----
    window = close_series[-SEQ_LEN:]
    xgb_preds = []
    for _ in range(horizon):
        pred = float(models["xgb"].predict(window.reshape(1, -1))[0])
        xgb_preds.append(pred)
        window = np.append(window[1:], pred)
    xgb_preds = np.array(xgb_preds, dtype=float)

    # ---- Ensemble = mean(ARIMA, XGB) ----
    ensemble = (arima_preds + xgb_preds) / 2.0

    # ---- Tidy output ----
    rows = []
    for i in range(horizon):
        ts_pred = ts_start + timedelta(minutes=i + 1)
        rows.append((settings.primary_symbol, ts_pred, i + 1, "arima",    float(arima_preds[i])))
        rows.append((settings.primary_symbol, ts_pred, i + 1, "xgb",      float(xgb_preds[i])))
        rows.append((settings.primary_symbol, ts_pred, i + 1, "ensemble", float(ensemble[i])))

    return pd.DataFrame(
        rows, columns=["ticker", "ts", "horizon_min", "model", "pred_value"]
    )

# ----------------------------------------------------------------------
# RMSE query (last 100 aligned points) — horizon-specific for consistency
# ----------------------------------------------------------------------
_RMSE_SQL = """
WITH pairs AS (
  SELECT p.pred_value, m.close
  FROM predictions p
  JOIN market_data m
    ON m.ticker = p.ticker
   AND m.ts     = p.ts
  WHERE p.ticker = %(ticker)s
    AND p.model  = %(model)s
    AND p.horizon_min = %(horizon)s
  ORDER BY p.ts DESC
  LIMIT 100
)
SELECT sqrt(avg((pred_value - close)^2)) AS rmse_100
FROM pairs;
"""

# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def generate_and_store_forecast() -> None:
    """
    Pulls recent bars, trains ARIMA+XGB, generates forecasts,
    upserts predictions, and stores rolling RMSE for the configured horizon.
    """
    logger.info("Generating forecasts…")

    with get_conn() as conn:
        recent_df = pd.read_sql(
            f"""
            SELECT *
            FROM market_data
            WHERE ticker = %s
            ORDER BY ts DESC
            LIMIT {TRAIN_ROWS};
            """,
            conn,
            params=(settings.primary_symbol,),
        ).sort_values("ts")

    if recent_df.empty:
        logger.warning("No market_data yet, skipping forecast.")
        return
    if len(recent_df) < (SEQ_LEN + 1):
        logger.warning("Not enough rows (%d) < SEQ_LEN+1 (%d). Skipping.", len(recent_df), SEQ_LEN + 1)
        return

    models = train_models(recent_df)
    preds_df = generate_forecast(models, recent_df, horizon=HORIZON)
    upsert_preds(preds_df)
    logger.info("Upserted %d forecast rows.", len(preds_df))

    # write rolling RMSE for ensemble + base models at this HORIZON
    metric_models = ["ensemble", "arima", "xgb"]
    metrics_rows = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for mdl in metric_models:
                cur.execute(_RMSE_SQL, {
                    "ticker": settings.primary_symbol,
                    "model": mdl,
                    "horizon": HORIZON
                })
                res = cur.fetchone()
                if res and res[0] is not None:
                    metrics_rows.append(
                        {"ticker": settings.primary_symbol, "ts": dt.utcnow(), "model": mdl, "rmse_100": float(res[0])}
                    )

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        upsert_metrics(metrics_df)
        logger.info(
            "Rolling RMSE@%dmin saved: %s",
            HORIZON,
            ", ".join(f"{r['model']}={r['rmse_100']:.3f}" for r in metrics_rows),
        )

if __name__ == "__main__":
    generate_and_store_forecast()
