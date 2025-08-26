# src/models.py
"""
Model training & forecasting for SignalLake.
Uses LSTM, ARIMA, and XGBoost, then publishes an ensemble forecast.
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
SEQ_LEN     = 100       # sequence length for LSTM/XGB windows
HORIZON     = 60        # minutes into the future to forecast
TRAIN_RATIO = 0.7
TRAIN_ROWS  = 10000      # pull this many most-recent rows for training/forecasting


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _train_test_split(series: np.ndarray, seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    split_idx = int(len(series) * TRAIN_RATIO)
    train = series[:split_idx]
    test  = series[split_idx - seq_len :]
    return train, test


# ----------------------------------------------------------------------
# Main public API
# ----------------------------------------------------------------------
def train_models(df: pd.DataFrame) -> Dict[str, object]:
    """
    Retrain LSTM, ARIMA, XGB on dataframe with column 'close'.
    Returns a dict of fitted models and scalers needed for inference.
    """
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import xgboost as xgb

    close = df["close"].values.astype(float)
    train, _ = _train_test_split(close)

    # ---------------- LSTM ----------------
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))

    def create_xy(arr):
        X, y = [], []
        for i in range(SEQ_LEN, len(arr)):
            X.append(arr[i - SEQ_LEN : i])
            y.append(arr[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_xy(train_scaled)

    lstm = Sequential(
        [
            LSTM(128, input_shape=(SEQ_LEN, 1), return_sequences=True, activation="relu"),
            Dropout(0.3),
            LSTM(64, activation="relu"),
            Dropout(0.3),
            Dense(1),
        ]
    )
    lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse")
    lstm.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    # ---------------- ARIMA ----------------
    arima = ARIMA(close, order=(5, 1, 0)).fit()

    # ---------------- XGBoost --------------
    def sliding_window(arr, window: int = SEQ_LEN):
        X, y = [], []
        for i in range(window, len(arr)):
            X.append(arr[i - window : i])
            y.append(arr[i])
        return np.array(X), np.array(y)

    X_train_xgb, y_train_xgb = sliding_window(close)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05)
    xgb_model.fit(X_train_xgb, y_train_xgb)

    return {
        "lstm": (lstm, scaler),
        "arima": arima,
        "xgb": xgb_model,
    }


def generate_forecast(
    models: Dict[str, object],
    recent_df: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Produce horizon-minute forecasts using trained models.
    Returns tidy DF: ticker, ts (forecast start), horizon_min, model, pred_value
    """
    close_series = recent_df["close"].values.astype(float)
    ts_start = recent_df["ts"].iloc[-1]

    # ---- LSTM ----
    lstm, scaler = models["lstm"]
    seq = scaler.transform(close_series[-SEQ_LEN :].reshape(-1, 1))
    lstm_preds = []
    seq_copy = seq.copy()
    for _ in range(horizon):
        pred = lstm.predict(seq_copy[np.newaxis, :, :], verbose=0)[0]
        lstm_preds.append(pred[0])
        seq_copy = np.append(seq_copy[1:], pred).reshape(-SEQ_LEN, 1)
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1)).flatten()

    # ---- ARIMA ---
    arima_preds = models["arima"].forecast(steps=horizon)

    # ---- XGB -----
    window = close_series[-SEQ_LEN:]
    xgb_preds = []
    for _ in range(horizon):
        pred = models["xgb"].predict(window.reshape(1, -1))[0]
        xgb_preds.append(pred)
        window = np.append(window[1:], pred)
    xgb_preds = np.array(xgb_preds)

    # ---- Simple ensemble (equal weights)
    ensemble = (lstm_preds + arima_preds + xgb_preds) / 3.0

    # ---- Tidy output
    rows = []
    for i in range(horizon):
        ts_pred = ts_start + timedelta(minutes=i + 1)
        rows.extend(
            [
                (settings.primary_symbol, ts_pred, i + 1, "lstm", lstm_preds[i]),
                (settings.primary_symbol, ts_pred, i + 1, "arima", arima_preds[i]),
                (settings.primary_symbol, ts_pred, i + 1, "xgb", xgb_preds[i]),
                (settings.primary_symbol, ts_pred, i + 1, "ensemble", ensemble[i]),
            ]
        )
    return pd.DataFrame(
        rows, columns=["ticker", "ts", "horizon_min", "model", "pred_value"]
    )


_RMSE_SQL = """
SELECT
    p.model,
    sqrt(avg( (p.pred_value - m.close)^2 )) AS rmse_100
FROM (
    SELECT *
    FROM predictions
    WHERE ticker = %(ticker)s
      AND model = %(model)s
    ORDER BY ts DESC
    LIMIT 100
) p
JOIN market_data m
  ON p.ticker = m.ticker AND p.ts = m.ts
GROUP BY p.model;
"""

# ----------------------------------------------------------------------
# Convenience wrapper for scheduler
# ----------------------------------------------------------------------
def generate_and_store_forecast() -> None:
    """
    Pulls recent bars, (re)trains if necessary, generates forecast,
    and upserts predictions to Postgres. Also stores rolling RMSE metrics.
    """
    logger.info("Generating forecasts…")

    # pull *more* history than SEQ_LEN*3
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

    # ── train (could cache to disk later) ─────────────────────────────
    models = train_models(recent_df)

    # ── forecast ──────────────────────────────────────────────────────
    preds_df = generate_forecast(models, recent_df, horizon=HORIZON)

    # ── store predictions ─────────────────────────────────────────────
    upsert_preds(preds_df)
    logger.info("Upserted %d forecast rows.", len(preds_df))

    # ── rolling RMSE per model (last 100 aligned rows) ───────────────
    metrics_rows = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for mdl in ["ensemble", "lstm", "arima", "xgb"]:
                cur.execute(_RMSE_SQL, {"ticker": settings.primary_symbol, "model": mdl})
                res = cur.fetchone()
                if res and res[1] is not None:
                    metrics_rows.append(
                        {
                            "ticker": settings.primary_symbol,
                            "ts": dt.utcnow(),
                            "model": mdl,
                            "rmse_100": float(res[1]),
                        }
                    )

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        upsert_metrics(metrics_df)
        logger.info(
            "Rolling RMSE saved: %s",
            ", ".join(f"{r['model']}={r['rmse_100']:.3f}" for r in metrics_rows),
        )


# ----------------------------------------------------------------------
# CLI test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    generate_and_store_forecast()
