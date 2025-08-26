# src/data_ingest.py
"""
Ingestion loop for SignalLake:
  • keep a rolling week of 1-minute bars in Postgres
  • backfill last 7 days if none exist for the ticker
  • thereafter, incrementally sync (last 2 days → now)
  • build indicator features and upsert (idempotent)
"""

from __future__ import annotations

import logging
from datetime import datetime as dt, timedelta, time
from typing import Tuple

import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.config import settings
from src.features import build_feature_df
from src.db import upsert_bars, get_conn

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
UTC = pytz.UTC


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _week_bounds_utc(now_utc: dt | None = None) -> Tuple[dt, dt]:
    """
    Return (start_utc, end_utc) covering the last 7 calendar days in ET,
    with end at tomorrow 00:00 ET to ensure we capture the full current day.
    """
    if now_utc is None:
        now_utc = dt.now(UTC)
    now_et = now_utc.astimezone(ET)

    start_et = ET.localize(dt.combine((now_et - timedelta(days=7)).date(), time(0, 0)))
    end_et   = ET.localize(dt.combine((now_et + timedelta(days=1)).date(), time(0, 0)))

    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def _incremental_bounds_utc(now_utc: dt | None = None) -> Tuple[dt, dt]:
    """
    Return a small window (last 2 days → now) in UTC for quick top-ups and gap repair.
    """
    if now_utc is None:
        now_utc = dt.now(UTC)
    return now_utc - timedelta(days=2), now_utc


def get_stock_bars_data(
    tickers: list[str],
    start_dt_utc: dt,
    end_dt_utc: dt,
    timeframe: TimeFrame = TimeFrame.Minute,
) -> pd.DataFrame:
    """
    Fetch bars from Alpaca between [start_dt_utc, end_dt_utc) with tz-aware UTC datetimes.
    Returns a DataFrame with a MultiIndex (symbol, timestamp) → use .reset_index() to flatten.
    """
    client = StockHistoricalDataClient(settings.API_KEY, settings.SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=timeframe,
        start=start_dt_utc,
        end=end_dt_utc,
    )
    bars = client.get_stock_bars(req)
    return bars.df


# ----------------------------------------------------------------------
# Core: decide window → fetch → features → upsert
# ----------------------------------------------------------------------
def fetch_and_store() -> None:
    ticker = settings.primary_symbol
    logger.info("Ingesting bars for %s", [ticker])

    # Decide whether to do a full week backfill or just an incremental sync
    latest_ts = None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT max(ts) FROM market_data WHERE ticker=%s", (ticker,))
            row = cur.fetchone()
            latest_ts = row[0] if row and row[0] is not None else None

    if latest_ts is None:
        start_utc, end_utc = _week_bounds_utc()
        logger.info("No history found — backfilling last week [%s → %s)", start_utc, end_utc)
    else:
        start_utc, end_utc = _incremental_bounds_utc()
        logger.info("Incremental sync window [%s → %s)", start_utc, end_utc)

    # Fetch
    raw = get_stock_bars_data(
        tickers=[ticker],
        start_dt_utc=start_utc,
        end_dt_utc=end_utc,
        timeframe=TimeFrame.Minute,
    )

    if raw.empty:
        logger.warning("No data returned from Alpaca.")
        return

    # Flatten index and normalize expected column names
    raw = raw.reset_index()  # brings 'symbol' and 'timestamp' out of the index
    if "symbol" in raw.columns and "ticker" not in raw.columns:
        raw.rename(columns={"symbol": "ticker"}, inplace=True)
    if "ticker" not in raw.columns:
        raw["ticker"] = ticker
    raw.rename(columns={"timestamp": "ts"}, inplace=True)
    raw.sort_values("ts", inplace=True)

    # Build indicator features (drops NaNs from rolling windows)
    feat = build_feature_df(raw)

    # Upsert
    logger.info("Upserting %d enriched rows into Postgres", len(feat))
    upsert_bars(feat)
    logger.info("Ingestion step complete ✅")


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from src.db import init_schema
    init_schema()      # create tables if first run
    fetch_and_store()  # one-off ingest
