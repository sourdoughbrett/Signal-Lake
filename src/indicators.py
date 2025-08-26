# src/indicators.py
"""
Pure TA-Lib indicator helpers used by SignalLake.

Usage
-----
>>> from src.indicators import calculate_macd
>>> enriched = calculate_macd(df)
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
import talib

# ──────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Hyper-parameters (tweak here or load from YAML later)
# ──────────────────────────────────────────────────────────────
macd_fast_period_val = 8
macd_slow_period_val = 18
macd_signal_period_val = 2

ema_fast_period_val = 10
ema_mod_period_val = 50
ema_slow_period_val = 200
ema_20_period_val = 20

sma_fast_period_val = 20
sma_slow_period_val = 50

bb_dev_factor_intense_val = 3.0
bb_dev_factor_tame_val = 1.0

psar_acceleration_dev_factor = 0.02
psar_maximum_dev_factor = 0.2

stoch_slowK_period_val = 3
stoch_slowD_period_val = 3
stoch_fastK_period_val = 14

# ──────────────────────────────────────────────────────────────
# Indicator functions
# ──────────────────────────────────────────────────────────────


# ---------- MACD ----------------------------------------------------------
def calculate_macd(
    df: pd.DataFrame,
    column: str = "close",
    fast_period: int = macd_fast_period_val,
    slow_period: int = macd_slow_period_val,
    signal_period: int = macd_signal_period_val,
) -> pd.DataFrame:
    close = df[column].astype(float).values
    macd, signal, hist = talib.MACD(
        close,
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    return pd.DataFrame(
        {"macd": macd, "signal": signal, "histogram": hist}, index=df.index
    )


# ---------- RSI -----------------------------------------------------------
def calculate_rsi(
    df: pd.DataFrame,
    column: str = "close",
    period: int = 14,
) -> pd.Series:
    rsi = talib.RSI(df[column].astype(float).values, timeperiod=period)
    return pd.Series(rsi, index=df.index, name="RSI")


# ---------- ATR -----------------------------------------------------------
def calculate_atr(
    df: pd.DataFrame,
    atr_period: int = 14,
    multiplier: float = 0.2,
) -> pd.DataFrame:
    atr_vals = talib.ATR(
        df["high"].astype(float).values,
        df["low"].astype(float).values,
        df["close"].astype(float).values,
        timeperiod=atr_period,
    )
    return pd.DataFrame({"atr": atr_vals * multiplier}, index=df.index)


# ---------- ADX / DI ------------------------------------------------------
def calculate_adx(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
) -> pd.DataFrame:
    high = df[high_col].astype(float).values
    low = df[low_col].astype(float).values
    close = df[close_col].astype(float).values

    di_plus = talib.PLUS_DI(high, low, close, timeperiod=period)
    di_minus = talib.MINUS_DI(high, low, close, timeperiod=period)
    adx = talib.ADX(high, low, close, timeperiod=period)
    adx_hist = di_plus - di_minus

    return pd.DataFrame(
        {
            "DI_plus": di_plus,
            "DI_minus": di_minus,
            "ADX": adx,
            "ADX_Histogram": adx_hist,
        },
        index=df.index,
    )


# ---------- Parabolic SAR -------------------------------------------------
def calculate_parabolic_SAR(
    df: pd.DataFrame,
    acceleration: float = psar_acceleration_dev_factor,
    maximum: float = psar_maximum_dev_factor,
) -> pd.DataFrame:
    sar = talib.SAR(
        df["high"].astype(float).values,
        df["low"].astype(float).values,
        acceleration=acceleration,
        maximum=maximum,
    )
    return pd.DataFrame({"parabolic_sar": sar}, index=df.index)


# ---------- Stochastic ----------------------------------------------------
def calculate_stochastic(
    df: pd.DataFrame,
    fastk_period: int = stoch_fastK_period_val,
    slowk_period: int = stoch_slowK_period_val,
    slowd_period: int = stoch_slowD_period_val,
) -> pd.DataFrame:
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    slowk, slowd = talib.STOCH(
        high,
        low,
        close,
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return pd.DataFrame({"SlowK": slowk, "SlowD": slowd}, index=df.index)


# ---------- Bollinger Bands ----------------------------------------------
def calculate_bollinger_bands_intense(
    df: pd.DataFrame,
    column: str = "close",
    period: int = 20,
    dev_factor: float = bb_dev_factor_intense_val,
) -> pd.DataFrame:
    up, mid, low = talib.BBANDS(
        df[column].astype(float).values,
        timeperiod=period,
        nbdevup=dev_factor,
        nbdevdn=dev_factor,
    )
    return pd.DataFrame({"upper_band": up, "middle_band": mid, "lower_band": low}, index=df.index)


def calculate_bollinger_bands_tame(
    df: pd.DataFrame,
    column: str = "close",
    period: int = 20,
    dev_factor: float = bb_dev_factor_tame_val,
) -> pd.DataFrame:
    up, mid, low = talib.BBANDS(
        df[column].astype(float).values,
        timeperiod=period,
        nbdevup=dev_factor,
        nbdevdn=dev_factor,
    )
    return pd.DataFrame({"upper_b": up, "middle_b": mid, "lower_b": low}, index=df.index)


# ---------- SMA / EMA -----------------------------------------------------
def calculate_sma_fast(df: pd.DataFrame, column="close", period: int = sma_fast_period_val) -> pd.Series:
    sma = talib.SMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(sma, index=df.index, name="SMA_Fast")


def calculate_sma_slow(df: pd.DataFrame, column="close", period: int = sma_slow_period_val) -> pd.Series:
    sma = talib.SMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(sma, index=df.index, name="SMA_Slow")


def calculate_ema_fast(df: pd.DataFrame, column="close", period: int = ema_fast_period_val) -> pd.Series:
    ema = talib.EMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(ema, index=df.index, name="EMA_Fast")


def calculate_ema_mod(df: pd.DataFrame, column="close", period: int = ema_mod_period_val) -> pd.Series:
    ema = talib.EMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(ema, index=df.index, name="EMA_Mod")


def calculate_ema_slow(df: pd.DataFrame, column="close", period: int = ema_slow_period_val) -> pd.Series:
    ema = talib.EMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(ema, index=df.index, name="EMA_Slow")


def calculate_ema_20(df: pd.DataFrame, column="close", period: int = ema_20_period_val) -> pd.Series:
    ema = talib.EMA(df[column].astype(float).values, timeperiod=period)
    return pd.Series(ema, index=df.index, name="EMA_20")


# ---------- VWAP Drawdown -------------------------------------------------
def calculate_vwap_drawdown(
    df: pd.DataFrame,
    column: str = "close",
    vwap_column: str = "vwap",
    drawdown_threshold: float = 30.0,
) -> pd.DataFrame:
    close = df[column].astype(float).values
    vwap = df[vwap_column].astype(float).values
    drawdown = ((vwap - close) / vwap) * 100
    return pd.DataFrame(
        {"drawdown": drawdown, "below_threshold": drawdown > drawdown_threshold},
        index=df.index,
    )


# ---------- LazyBear Momentum --------------------------------------------
def calculate_momentum(
    df: pd.DataFrame,
    column: str = "close",
    bollinger_period: int = 20,
    keltner_period: int = 20,
    momentum_period: int = 14,
) -> pd.DataFrame:
    df = df.copy()
    df["bollinger_mid"] = df[column].rolling(window=bollinger_period).mean()
    df["bollinger_std"] = df[column].rolling(window=bollinger_period).std()
    df["bollinger_upper"] = df["bollinger_mid"] + 2 * df["bollinger_std"]
    df["bollinger_lower"] = df["bollinger_mid"] - 2 * df["bollinger_std"]

    df["keltner_mid"] = df[column].rolling(window=keltner_period).mean()
    df["keltner_atr"] = df["bollinger_std"].rolling(window=keltner_period).mean()
    df["keltner_upper"] = df["keltner_mid"] + 1.5 * df["keltner_atr"]
    df["keltner_lower"] = df["keltner_mid"] - 1.5 * df["keltner_atr"]

    df["momentum"] = df[column].diff(momentum_period)
    df["momentum_histogram"] = df["momentum"].rolling(window=5).mean()

    return df[
        [
            "bollinger_upper",
            "bollinger_lower",
            "keltner_upper",
            "keltner_lower",
            "momentum_histogram",
        ]
    ]


# ---------- Rate of Change & Rolling Averages ----------------------------
def calculate_roc_and_avg(df, column="close", period: int = 14) -> pd.DataFrame:
    roc = talib.ROC(df[column].astype(float).values, timeperiod=period)
    avg_roc = pd.Series(roc).rolling(window=period).mean().values
    return pd.DataFrame({"ROC": roc, "avg_ROC": avg_roc}, index=df.index)


def calculate_rolling_average_volume(df, column="volume", period: int = 14) -> pd.Series:
    return df[column].rolling(window=period).mean()


def calculate_rolling_average_low_price(df, column="low", period: int = 3) -> pd.Series:
    return df[column].rolling(window=period).min().rolling(window=period).mean()


def calculate_rolling_average_high_price(df, column="high", period: int = 3) -> pd.Series:
    return df[column].rolling(window=period).max().rolling(window=period).mean()


# ──────────────────────────────────────────────────────────────
# __all__ for clean exports
# ──────────────────────────────────────────────────────────────
__all__: list[str] = [name for name in globals() if name.startswith("calculate_")]
