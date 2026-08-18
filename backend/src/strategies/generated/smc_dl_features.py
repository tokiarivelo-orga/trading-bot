import numpy as np
import pandas as pd
from typing import Any, Dict

def compute_smc_features(candles: dict[str, Any], symbol: str, lookback: int = 20) -> dict[str, float]:
    """Compute all SMC features for the current bar. Returns a flat dict of feature_name -> value."""
    df_m5 = candles.get('M5')
    if df_m5 is None or len(df_m5) < lookback + 50:
        return {}
    
    # We will just extract the last row from a batched computation for simplicity
    df_m15 = candles.get('M15', pd.DataFrame())
    df_h1 = candles.get('H1', pd.DataFrame())
    df_h4 = candles.get('H4', pd.DataFrame())
    
    features_df = compute_features_batch(df_m5, df_m15, df_h1, df_h4, lookback=lookback)
    if features_df.empty:
        return {}
        
    last_row = features_df.iloc[-1].to_dict()
    return last_row

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_features_batch(candles_m5: pd.DataFrame, candles_m15: pd.DataFrame, candles_h1: pd.DataFrame, candles_h4: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute features for every bar in the M5 DataFrame. Returns DataFrame with feature columns."""
    df = candles_m5.copy()
    if df.empty or len(df) < 50:
        return pd.DataFrame()
        
    # Price features
    df['atr_5'] = _compute_atr(df, 5)
    df['atr_14'] = _compute_atr(df, 14)
    df['atr_50'] = _compute_atr(df, 50)
    df['atr_ratio'] = df['atr_5'] / df['atr_50'].replace(0, np.nan)
    
    df['returns'] = df['close'].pct_change()
    df['realized_vol_5'] = df['returns'].rolling(5).std()
    df['realized_vol_20'] = df['returns'].rolling(20).std()
    
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    total_range = (df['high'] - df['low']).replace(0, np.nan)
    df['body_ratio'] = df['body_size'] / total_range
    df['wick_ratio_upper'] = df['upper_wick'] / total_range
    df['wick_ratio_lower'] = df['lower_wick'] / total_range
    
    for i in [1, 3, 5, 10]:
        df[f'return_{i}'] = df['close'].pct_change(i)
        
    # Market structure (simplified)
    df['swing_high'] = (df['high'] == df['high'].rolling(5, center=True).max()).astype(float)
    df['swing_low'] = (df['low'] == df['low'].rolling(5, center=True).min()).astype(float)
    df['trend_encoding'] = np.where(df['close'] > df['close'].rolling(20).mean(), 1, -1)
    df['bos'] = (df['close'] > df['high'].rolling(20).max().shift()).astype(float)
    df['choch'] = (df['close'] < df['low'].rolling(20).min().shift()).astype(float)
    df['mss'] = df['choch'] # Simplified
    df['displacement'] = df['body_size']
    df['displacement_atr'] = df['displacement'] / df['atr_14']
    
    # SMC Zones
    df['fvg_bullish'] = (df['low'].shift(-1) > df['high'].shift(1)).astype(float)
    df['fvg_bearish'] = (df['high'].shift(-1) < df['low'].shift(1)).astype(float)
    df['fvg_size'] = np.where(df['fvg_bullish'], df['low'].shift(-1) - df['high'].shift(1), 0)
    df['fvg_size'] = np.where(df['fvg_bearish'], df['low'].shift(1) - df['high'].shift(-1), df['fvg_size'])
    df['fvg_age'] = 0.0 # Placeholder
    df['fvg_fill_percentage'] = 0.0
    df['distance_to_fvg'] = 0.0
    
    df['ob_size'] = df['body_size'].rolling(3).mean()
    df['ob_age'] = 0.0
    df['distance_to_ob'] = 0.0
    df['premium_discount'] = np.where(df['close'] > df['close'].rolling(50).mean(), 1.0, -1.0)
    
    # Liquidity
    df['equal_highs'] = (df['high'].diff().abs() < df['atr_14'] * 0.1).astype(float)
    df['equal_lows'] = (df['low'].diff().abs() < df['atr_14'] * 0.1).astype(float)
    df['liquidity_sweep'] = df['equal_highs'] # Simplified
    df['distance_to_liquidity'] = df['atr_14']
    
    # Session features
    if hasattr(df.index, 'hour'):
        ts_hour = df.index.hour
        ts_dow = df.index.dayofweek
    elif 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        ts_hour = ts.dt.hour
        ts_dow = ts.dt.dayofweek
    elif 'close_time' in df.columns:
        ts = pd.to_datetime(df['close_time'])
        ts_hour = ts.dt.hour
        ts_dow = ts.dt.dayofweek
    else:
        ts_hour = pd.Series([0] * len(df), index=df.index)
        ts_dow = pd.Series([0] * len(df), index=df.index)
    
    df['asian_session'] = ((ts_hour >= 0) & (ts_hour < 8)).astype(float)
    df['london_session'] = ((ts_hour >= 8) & (ts_hour < 16)).astype(float)
    df['ny_session'] = ((ts_hour >= 13) & (ts_hour < 21)).astype(float)
    
    df['sin_hour'] = np.sin(2 * np.pi * ts_hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * ts_hour / 24)
    df['sin_dow'] = np.sin(2 * np.pi * ts_dow / 7)
    df['cos_dow'] = np.cos(2 * np.pi * ts_dow / 7)
    
    # Multi-timeframe: actually compute trend/BOS/CHoCH from each HTF,
    # then forward-fill (reindex) to align with M5 bars.
    def _apply_htf(htf_df: pd.DataFrame, prefix: str) -> None:
        if htf_df is None or htf_df.empty or len(htf_df) < 22:
            df[f'htf_trend_{prefix}'] = 0.0
            df[f'htf_bos_{prefix}']   = 0.0
            df[f'htf_choch_{prefix}'] = 0.0
            return
        h = htf_df.copy()
        h['_trend'] = np.where(h['close'] > h['close'].rolling(20).mean(), 1.0, -1.0)
        h['_bos']   = (h['close'] > h['high'].rolling(20).max().shift()).astype(float)
        h['_choch'] = (h['close'] < h['low'].rolling(20).min().shift()).astype(float)
        for col, feat in [('_trend', 'trend'), ('_bos', 'bos'), ('_choch', 'choch')]:
            merged = h[col].reindex(df.index, method='ffill')
            df[f'htf_{feat}_{prefix}'] = merged.fillna(0.0)

    _apply_htf(candles_m15, 'm15')
    _apply_htf(candles_h1,  'h1')
    _apply_htf(candles_h4,  'h4')
    
    # Base detection
    df['base_type'] = 0.0
    
    # Clean up and return
    df = df.fillna(0.0)
    
    # Drop raw OHLCV and time columns so they don't become features
    drop_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread_points', 'time', 'timestamp', 'close_time', 'open_time']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Return only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df
