"""Paper-mode integration test for `xauusd_regime_router_m1_v3` (Phase 4 of
the "XAUUSD regime-switching meta-strategy, DL-guided" plan) — shaped exactly
like `test_phase5_backtest_flow.py`: synthetic OHLC candle history seeded
into a temp SQLite DB via `CandleRepository.upsert_many`, then replayed
through the real `run_backtest()` driving the exact same
`TradeEngine`/`RiskManager`/`PositionManager`/`OrderService` pipeline live
trading uses, against `PaperBroker` — genuine paper-mode, never live.

Uses the real `configs/` (risk.yaml, symbols/xauusd.yaml, regime.yaml,
volatility.yaml) since those are the project's actual checked-in trading
config, same convention `test_phase5_backtest_flow.py` follows — no fixture
config needed.

**The ACTIVE-only registry gate**: this strategy is deliberately still
`VALIDATED`, not `ACTIVE` (activation is a manual step the user takes later —
see the plan), so `run_backtest`'s default strategy resolution
(`_default_registry` -> `StrategyVersionService(...).load_active_into_registry()`)
would never find it; it only loads DB versions with status ACTIVE.
`run_backtest` accepts a `strategy_source: StrategyRegistry` override for
exactly this case — already exercised by
`test_backtest_rejects_symbol_the_strategy_does_not_trade` in
test_phase5_backtest_flow.py — so this test builds its own `StrategyRegistry`
and registers `XauusdRegimeRouterM1()` by hand, bypassing the DB's
ACTIVE-only gate entirely. This is a standard test-harness pattern for
validating the mechanism end-to-end, not a real deployment: it never calls
`activate_version`, and the strategy's real DB row (status VALIDATED) is
left untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.application.run_backtest import run_backtest
from src.backtest.domain.models import BacktestReport
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base
from src.strategies.generated.xauusd_regime_router_m1_v3 import XauusdRegimeRouterM1
from src.strategies.registry import StrategyRegistry
from tests.unit.strategies.test_xauusd_regime_router_m1 import _build_zigzag

M1_STEP = timedelta(minutes=1)
M5_STEP = timedelta(minutes=5)
# Wednesday 13:00 UTC — 5-minute aligned (minute=0) so the M5 resample below
# groups cleanly, and within the overlap session (12-16 UTC per
# configs/regime.yaml), matching the fixture's own known-firing convention
# in test_xauusd_regime_router_m1.py.
START = datetime(2025, 1, 1, 13, 0, tzinfo=UTC)


def _m1_candles() -> list[Candle]:
    """Same clean-uptrend zigzag `test_mode_a_fires_on_clean_uptrend_continuation`
    uses for Mode A (trend_continuation) — proven to fire — with a much
    longer strong tail appended so there is plenty of room *after* the entry
    for the take-profit / secure-base trail to actually resolve the position
    live, rather than relying solely on the backtest's own forced
    end-of-run close."""
    highs, lows = _build_zigzag(
        n_legs=11, leg=3.0, pull=1.0, bars_per_leg=4, tail_bars=90, tail_step=0.4
    )
    closes = (highs + lows) / 2.0
    candles: list[Candle] = []
    for i in range(len(highs)):
        t = START + i * M1_STEP
        candles.append(
            Candle(
                symbol="XAUUSD",
                timeframe=Timeframe.M1,
                time=t,
                open=float(closes[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                tick_volume=500,
                spread_points=15,
            )
        )
    return candles


def _resample_m5(m1_candles: list[Candle]) -> list[Candle]:
    """Downsamples the M1 series into real M5 bars. `ReplayMarketDataPort`
    derives bid/ask from the *M5* stream's own close (its module docstring:
    "Bid/ask are derived from the current M5 bar's close"), not from the M1
    entry frame this M1 strategy actually reads — so M5 has to track the
    same price path the M1 series does, or fills would land nowhere near the
    strategy's own SL/TP levels. A flat/insufficient-history stub (the
    convention `test_phase5_backtest_flow.py::build_htf_candles` uses for
    *confirmation* timeframes the strategy under test never reads) would be
    wrong here specifically because M5 also drives execution pricing."""
    bars: list[Candle] = []
    # A leading M5 bar closing exactly at the first M1 bar's *open* time:
    # the very first M1 bar's own close (at START + 1 minute) needs an M5
    # bar already closed at-or-before it (`current_m5_candle` raises
    # otherwise), and the real first 5-minute group doesn't close until
    # START + 5 minutes.
    first = m1_candles[0]
    bars.append(
        Candle(
            symbol="XAUUSD",
            timeframe=Timeframe.M5,
            time=first.time - M5_STEP,
            open=first.open,
            high=first.open,
            low=first.open,
            close=first.open,
            tick_volume=1,
            spread_points=15,
        )
    )
    full_groups = len(m1_candles) - len(m1_candles) % 5
    for start in range(0, full_groups, 5):
        group = m1_candles[start : start + 5]
        bars.append(
            Candle(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                time=group[0].time,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                tick_volume=sum(c.tick_volume for c in group),
                spread_points=15,
            )
        )
    return bars


def database_url_for(tmp_path) -> str:
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    repository = CandleRepository(session_factory)
    m1_candles = _m1_candles()
    repository.upsert_many(m1_candles)
    repository.upsert_many(_resample_m5(m1_candles))
    # No H1 seeded on purpose: `XauusdRegimeRouterM1.evaluate()` only ever
    # reads its own M1 entry frame (see the strategy's module docstring —
    # confirmation_timeframes=("M5", "H1") is declared solely so the
    # engine's own HTF-veto pipeline has native M5/H1 to check against, and
    # that gate is currently bypassed engine-wide per CLAUDE.md/the plan's
    # scope). `ReplayMarketDataPort.get_candles` on an unseeded timeframe
    # returns an empty list rather than raising, so this is safe.
    return url


async def test_regime_router_paper_backtest_runs_end_to_end(tmp_path) -> None:
    """Real `TradeEngine`/`RiskManager`/`PositionManager`/`OrderService`
    against `PaperBroker`, with `XauusdRegimeRouterM1` injected into a
    test-local registry (bypassing the DB's ACTIVE-only gate — see module
    docstring). Asserts the pipeline runs clean end-to-end and produces at
    least one real trade; it does not assert specific PF/win-rate numbers —
    this is synthetic data, not a real market, so those numbers aren't
    meaningful here (Phase 5 of the plan is the real backtest against real
    XAUUSD history)."""
    database_url = database_url_for(tmp_path)
    registry = StrategyRegistry()
    strategy = XauusdRegimeRouterM1()
    registry.register(strategy.spec.name, strategy)

    report = await run_backtest(
        strategy.spec.name,
        "XAUUSD",
        "2025-01:2025-01",
        database_url=database_url,
        strategy_source=registry,
    )

    assert isinstance(report, BacktestReport)
    assert report.strategy == strategy.spec.name
    assert report.symbol == "XAUUSD"

    # The pipeline actually produced a real trade, not just an empty report.
    assert len(report.trades) >= 1
    trade = report.trades[0]
    assert trade.side == "buy"  # the zigzag fixture is a clean uptrend
    assert trade.pattern == "trend_continuation"
    assert trade.volume > 0.0

    # Report fields the plan documents (BacktestReport: profit_factor,
    # win_rate, trades, equity_curve, ...) are all populated, not NaN/None —
    # confirming the whole metrics pipeline ran, not just position opening.
    assert isinstance(report.profit_factor, float)
    assert isinstance(report.win_rate, float)
    assert 0.0 <= report.win_rate <= 1.0
    assert len(report.equity_curve) >= 1
    assert report.ending_balance != report.starting_balance
    assert report.activity_log
    assert any("SIGNAL" in e.message and "buy" in e.message for e in report.activity_log)
    assert any("ENTRY OPENED" in e.message for e in report.activity_log)
