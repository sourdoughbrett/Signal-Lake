# src/features.py
"""
Transforms a raw 1-minute bar DataFrame into an
indicator-enriched DataFrame for SignalLake.

Expected input columns:
    ticker, ts, open, high, low, close, volume
"""

from __future__ import annotations
import pandas as pd

from src.indicators import (
    calculate_macd,
    calculate_rsi,
    calculate_atr,
    calculate_adx,
    calculate_parabolic_SAR,
    calculate_stochastic,
    calculate_bollinger_bands_intense,
    calculate_bollinger_bands_tame,
    calculate_sma_fast,
    calculate_ema_fast,
    calculate_ema_slow,
    calculate_rolling_average_volume,
    calculate_rolling_average_high_price,
    calculate_rolling_average_low_price,
    calculate_momentum,
)


def build_feature_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw : DataFrame
        Must contain at least ['ticker','ts','open','high','low','close','volume'].

    Returns
    -------
    DataFrame
        Same index as input, enriched with indicator columns.
        NaNs dropped so it is ready for DB upsert.
    """
    df = raw.copy()

    # --- LazyBear Momentum -------------------------------------------------
    mom = calculate_momentum(df, column="close", bollinger_period=20, keltner_period=20, momentum_period=14)
    df = pd.concat([df, mom], axis=1)

    # --- MACD --------------------------------------------------------------
    macd = calculate_macd(df, column="close", fast_period=12, slow_period=26, signal_period=7)
    df = pd.concat([df, macd], axis=1)

    # --- RSI ---------------------------------------------------------------
    df["RSI"] = calculate_rsi(df, column="close", period=14)

    # --- ATR ---------------------------------------------------------------
    df["atr"] = calculate_atr(df, atr_period=14, multiplier=0.2)["atr"]

    # --- ADX ---------------------------------------------------------------
    adx = calculate_adx(df, period=10)
    df = pd.concat([df, adx], axis=1)

    # --- Parabolic SAR -----------------------------------------------------
    df["parabolic_sar"] = calculate_parabolic_SAR(df)["parabolic_sar"]

    # --- Stochastic --------------------------------------------------------
    stoch = calculate_stochastic(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df = pd.concat([df, stoch], axis=1)

    # --- Bollinger Bands ---------------------------------------------------
    bb_intense = calculate_bollinger_bands_intense(df, column="close")
    bb_tame    = calculate_bollinger_bands_tame(df, column="close")
    df = pd.concat([df, bb_intense, bb_tame], axis=1)

    # --- Moving Averages ---------------------------------------------------
    df["SMA_Fast"]  = calculate_sma_fast(df, column="close", period=10)
    df["EMA_Fast"]  = calculate_ema_fast(df, column="close", period=5)
    df["EMA_Slow"]  = calculate_ema_slow(df, column="close", period=20)

    # --- Rolling volume / price levels ------------------------------------
    df["rolling_avg_volume"]      = calculate_rolling_average_volume(df, period=14)
    df["rolling_avg_high_price"]  = calculate_rolling_average_high_price(df, period=14)
    df["rolling_avg_low_price"]   = calculate_rolling_average_low_price(df, period=14)

    # ----------------------------------------------------------------------
    # Final cleanup: drop rows with any NaN values created by rolling windows
    # ----------------------------------------------------------------------
    df.dropna(inplace=True)
    # standardise column names for DB
    df.rename(columns={"RSI": "rsi"}, inplace=True)

    return df


# quick self-test
if __name__ == "__main__":
    import datetime as dt

    now = dt.datetime.utcnow()
    sample = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 120,
            "ts": pd.date_range(end=now, periods=120, freq="T"),
            "open": 100 + pd.Series(range(120)),
            "high": 101 + pd.Series(range(120)),
            "low": 99 + pd.Series(range(120)),
            "close": 100 + pd.Series(range(120)),
            "volume": 1_000,
        }
    )

    enriched = build_feature_df(sample)
    print("Enriched columns:", list(enriched.columns)[:15], "...")
    print("Rows after dropna:", len(enriched))
