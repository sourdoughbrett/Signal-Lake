"""
src/db.py
PostgreSQL helpers for SignalLake.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from src.config import settings

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ema_fast","ema_slow","upper_band","lower_band","atr","adx",
    "slowk","slowd","parabolic_sar","momentum_histogram","rolling_avg_volume"
]


# ----------------------------------------------------------------------
# Connection helper
# ----------------------------------------------------------------------
def get_conn():
    """
    Returns a new psycopg2 connection.
    """
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "signalpass"),
        dbname=os.getenv("PG_DB", "signallake_db"),
    )

# ----------------------------------------------------------------------
# Schema (idempotent)
# ----------------------------------------------------------------------
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS market_data (
    ticker   TEXT        NOT NULL,
    ts       TIMESTAMPTZ NOT NULL PRIMARY KEY,
    open     NUMERIC,
    high     NUMERIC,
    low      NUMERIC,
    close    NUMERIC,
    volume   BIGINT,
    macd     NUMERIC,
    rsi      NUMERIC
);

CREATE TABLE IF NOT EXISTS predictions (
    ticker       TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    horizon_min  INT         NOT NULL,
    model        TEXT        NOT NULL,
    pred_value   NUMERIC,
    PRIMARY KEY (ticker, ts, horizon_min, model)
);
"""

MIGRATE_SQL = """
ALTER TABLE market_data
  ADD COLUMN IF NOT EXISTS ema_fast NUMERIC,
  ADD COLUMN IF NOT EXISTS ema_slow NUMERIC,
  ADD COLUMN IF NOT EXISTS upper_band NUMERIC,
  ADD COLUMN IF NOT EXISTS lower_band NUMERIC,
  ADD COLUMN IF NOT EXISTS atr NUMERIC,
  ADD COLUMN IF NOT EXISTS adx NUMERIC,
  ADD COLUMN IF NOT EXISTS slowk NUMERIC,
  ADD COLUMN IF NOT EXISTS slowd NUMERIC,
  ADD COLUMN IF NOT EXISTS parabolic_sar NUMERIC,
  ADD COLUMN IF NOT EXISTS momentum_histogram NUMERIC,
  ADD COLUMN IF NOT EXISTS rolling_avg_volume NUMERIC;
"""


METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_metrics (
    ticker   TEXT        NOT NULL,
    ts       TIMESTAMPTZ NOT NULL,
    model    TEXT        NOT NULL,
    rmse_100 DOUBLE PRECISION,
    PRIMARY KEY (ticker, ts, model)
);
"""

def init_schema() -> None:
    """Ensure all tables exist."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(CREATE_SQL)
        cur.execute(MIGRATE_SQL)
        cur.execute(METRICS_SCHEMA)          # ← NEW
        conn.commit()
    logger.info("Database schema ensured.")

# ----------------------------------------------------------------------
# Generic bulk‑upsert helper
# ----------------------------------------------------------------------
def _execute_upsert(sql: str, rows: Iterable[tuple]) -> None:
    if not rows:
        return
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows)
        conn.commit()

# ----------------------------------------------------------------------
# market_data upsert
# ----------------------------------------------------------------------
def upsert_bars(df: pd.DataFrame) -> None:
    # allow FEATURE_COLS to be empty without breaking SQL
    feature_cols = list(FEATURE_COLS) if 'FEATURE_COLS' in globals() else []

    # ensure all expected columns exist
    base_cols = ["ticker","ts","open","high","low","close","volume","macd","rsi"]
    for c in base_cols + feature_cols:
        if c not in df.columns:
            df[c] = np.nan  # requires: import numpy as np

    insert_cols = base_cols + feature_cols
    insert_cols_sql = ", ".join(insert_cols)

    # do not update primary key columns
    update_cols = [c for c in insert_cols if c not in ("ticker","ts")]
    update_set_sql = ",\n        ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    sql = f"""
    INSERT INTO market_data (
        {insert_cols_sql}
    ) VALUES %s
    ON CONFLICT (ticker, ts)
    DO UPDATE SET
        {update_set_sql};
    """

    rows = list(df[insert_cols].itertuples(index=False, name=None))
    _execute_upsert(sql, rows)
    logger.debug("Upserted %d bar rows.", len(rows))

# ----------------------------------------------------------------------
# predictions upsert
# ----------------------------------------------------------------------
def upsert_preds(df: pd.DataFrame) -> None:
    """
    DataFrame columns: ticker, ts, horizon_min, model, pred_value
    """
    sql = """
    INSERT INTO predictions (
        ticker, ts, horizon_min, model, pred_value
    ) VALUES %s
    ON CONFLICT (ticker, ts, horizon_min, model)
    DO UPDATE SET pred_value = EXCLUDED.pred_value;
    """
    rows = list(
        df[["ticker", "ts", "horizon_min", "model", "pred_value"]]
        .itertuples(index=False, name=None)
    )
    _execute_upsert(sql, rows)
    logger.debug("Upserted %d prediction rows.", len(rows))

# ----------------------------------------------------------------------
# NEW: forecast_metrics upsert
# ----------------------------------------------------------------------
def upsert_metrics(df: pd.DataFrame) -> None:
    """
    DataFrame columns: ticker, ts, model, rmse_100
    """
    sql = """
    INSERT INTO forecast_metrics (
        ticker, ts, model, rmse_100
    ) VALUES %s
    ON CONFLICT (ticker, ts, model)
    DO UPDATE SET rmse_100 = EXCLUDED.rmse_100;
    """
    rows = list(
        df[["ticker", "ts", "model", "rmse_100"]]
        .itertuples(index=False, name=None)
    )
    _execute_upsert(sql, rows)
    logger.debug("Upserted %d metric rows.", len(rows))
