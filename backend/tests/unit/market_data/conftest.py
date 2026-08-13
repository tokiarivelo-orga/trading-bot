from datetime import UTC, datetime

import pytest

from src.market_data.domain.models import Candle, Timeframe


@pytest.fixture
def candle_factory():
    def make(time: datetime, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M5, **kw):
        assert time.tzinfo is UTC
        defaults = dict(
            open=2400.0,
            high=2401.0,
            low=2399.0,
            close=2400.5,
            tick_volume=1000,
            spread_points=25,
            real_volume=0,
        )
        return Candle(symbol=symbol, timeframe=timeframe, time=time, **{**defaults, **kw})

    return make
