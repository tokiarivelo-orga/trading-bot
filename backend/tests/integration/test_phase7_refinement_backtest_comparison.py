"""Phase 7 integration: the refinement loop's core safety property is that a
proposed strategy is always backtested against a baseline over the same
period before any apply decision (§8.2). This exercises the real
`run_backtest` call twice — once per registry, exactly how
`RefinementLoopService._propose_refinement` does it — over a synthetic M5
history containing one clean breakout episode, and confirms the two reports
genuinely diverge (a strategy that never trades vs. one that does).

Mirrors `test_phase5_backtest_flow.py`'s fixture style. This is the closest
thing to a "paper-mode" check available for this feature: the refinement
loop introduces no new broker/order-service code of its own — it only
reuses `run_backtest`'s existing paper-broker composition root (covered by
Phase 5's own tests) and `activate_version` (covered by Phase 6's), so what's
actually novel and worth an integration test is this comparison step.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.application.run_backtest import run_backtest
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base
from src.strategies.domain.models import StrategySpec
from src.strategies.generated.breakout_v1 import BreakoutV1
from src.strategies.registry import StrategyRegistry

M5_STEP = timedelta(minutes=5)
START = datetime(2025, 1, 1, tzinfo=UTC)


def m5(i: int, *, open: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        time=START + i * M5_STEP,
        open=open,
        high=high,
        low=low,
        close=close,
        tick_volume=1000,
        spread_points=25,
    )


def build_m5_candles() -> list[Candle]:
    bars = [m5(i, open=2400.0, high=2401.0, low=2399.0, close=2400.0) for i in range(20)]
    bars.append(m5(20, open=2401.0, high=2411.0, low=2400.5, close=2410.0))  # BUY breakout
    bars.append(m5(21, open=2410.0, high=2440.0, low=2405.0, close=2408.0))  # clears TP
    return bars


def build_htf_candles(timeframe: Timeframe, step: timedelta, count: int = 5) -> list[Candle]:
    return [
        Candle(
            symbol="XAUUSD",
            timeframe=timeframe,
            time=START - (count - i) * step,
            open=2400.0,
            high=2401.0,
            low=2399.0,
            close=2400.5,
            tick_volume=1000,
            spread_points=25,
        )
        for i in range(count)
    ]


@pytest.fixture
def database_url(tmp_path) -> str:
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    repository = CandleRepository(session_factory)
    repository.upsert_many(build_m5_candles())
    repository.upsert_many(build_htf_candles(Timeframe.H1, timedelta(hours=1)))
    repository.upsert_many(build_htf_candles(Timeframe.H4, timedelta(hours=4)))
    return url


class NeverTradesStrategy:
    """Stands in for "the baseline strategy, before refinement" — deliberately
    inert so its backtest report is unambiguously different from a strategy
    that actually trades the breakout episode."""

    def __init__(self) -> None:
        self.spec = StrategySpec(
            name="breakout_v1",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params={},
        )

    def evaluate(self, ctx):
        return None


async def test_baseline_and_candidate_backtests_reflect_different_signals(database_url):
    baseline_registry = StrategyRegistry()
    baseline_registry.register("breakout_v1", NeverTradesStrategy())
    baseline_report = await run_backtest(
        "breakout_v1",
        "XAUUSD",
        "2025-01:2025-01",
        strategy_source=baseline_registry,
        database_url=database_url,
    )

    candidate_registry = StrategyRegistry()
    candidate_registry.register("breakout_v1", BreakoutV1())
    candidate_report = await run_backtest(
        "breakout_v1",
        "XAUUSD",
        "2025-01:2025-01",
        strategy_source=candidate_registry,
        database_url=database_url,
    )

    assert len(baseline_report.trades) == 0
    assert baseline_report.avg_r == 0.0

    assert len(candidate_report.trades) == 1
    assert candidate_report.trades[0].side == "buy"
    assert candidate_report.avg_r != baseline_report.avg_r
