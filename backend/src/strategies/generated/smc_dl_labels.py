import numpy as np
import pandas as pd

def generate_labels(candles: pd.DataFrame, atr_series: pd.Series, tp_atr_mult: float = 1.5, sl_atr_mult: float = 1.0, max_holding_bars: int = 50) -> pd.DataFrame:
    """
    Generate labels for each bar: direction, hit_tp_before_sl, mfe, mae, bars_to_resolution.
    Uses a triple barrier method.
    """
    labels = pd.DataFrame(index=candles.index)
    n = len(candles)
    
    direction = np.zeros(n)
    hit_tp_before_sl = np.zeros(n)
    mfe = np.zeros(n)
    mae = np.zeros(n)
    bars_to_resolution = np.full(n, max_holding_bars)
    
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    atrs = atr_series.values
    
    for i in range(n):
        if np.isnan(atrs[i]) or atrs[i] == 0:
            continue
            
        c_price = closes[i]
        c_atr = atrs[i]
        
        tp_long = c_price + c_atr * tp_atr_mult
        sl_long = c_price - c_atr * sl_atr_mult
        
        tp_short = c_price - c_atr * tp_atr_mult
        sl_short = c_price + c_atr * sl_atr_mult
        
        long_mfe = 0.0
        long_mae = 0.0
        short_mfe = 0.0
        short_mae = 0.0
        
        hit_long_tp = False
        hit_long_sl = False
        hit_short_tp = False
        hit_short_sl = False
        
        bar_count = 0
        
        for j in range(i + 1, min(i + 1 + max_holding_bars, n)):
            bar_count += 1
            h = highs[j]
            l = lows[j]
            
            # Long MFE/MAE
            if h > c_price:
                long_mfe = max(long_mfe, (h - c_price) / c_atr)
            if l < c_price:
                long_mae = min(long_mae, (l - c_price) / c_atr)
                
            # Short MFE/MAE
            if l < c_price:
                short_mfe = max(short_mfe, (c_price - l) / c_atr)
            if h > c_price:
                short_mae = min(short_mae, (c_price - h) / c_atr)
                
            if not hit_long_tp and not hit_long_sl:
                if l <= sl_long and h >= tp_long:
                    hit_long_sl = True # assume SL hits first on huge bar
                elif l <= sl_long:
                    hit_long_sl = True
                    long_bars = bar_count
                elif h >= tp_long:
                    hit_long_tp = True
                    long_bars = bar_count
                    
            if not hit_short_tp and not hit_short_sl:
                if h >= sl_short and l <= tp_short:
                    hit_short_sl = True
                elif h >= sl_short:
                    hit_short_sl = True
                    short_bars = bar_count
                elif l <= tp_short:
                    hit_short_tp = True
                    short_bars = bar_count
                    
            if (hit_long_tp or hit_long_sl) and (hit_short_tp or hit_short_sl):
                break
                
        # Determine direction and labels
        if hit_long_tp and not hit_long_sl:
            direction[i] = 1
            hit_tp_before_sl[i] = 1
            mfe[i] = long_mfe
            mae[i] = long_mae
            bars_to_resolution[i] = long_bars
        elif hit_short_tp and not hit_short_sl:
            direction[i] = -1
            hit_tp_before_sl[i] = 1
            mfe[i] = short_mfe
            mae[i] = short_mae
            bars_to_resolution[i] = short_bars
        else:
            direction[i] = 0
            hit_tp_before_sl[i] = 0
            mfe[i] = max(long_mfe, short_mfe)
            mae[i] = min(long_mae, short_mae)
            bars_to_resolution[i] = max_holding_bars
            
    labels['direction'] = direction
    labels['hit_tp_before_sl'] = hit_tp_before_sl
    labels['mfe'] = mfe
    labels['mae'] = mae
    labels['bars_to_resolution'] = bars_to_resolution
    
    return labels
