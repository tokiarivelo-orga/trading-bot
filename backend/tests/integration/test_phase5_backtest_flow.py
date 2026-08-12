"""Phase 5 end-to-end: a synthetic M5 candle history containing two clean
breakout episodes (mirrors test_phase4_engine_flow.py's fixture style) is
seeded into a temp SQLite DB and replayed through `run_backtest`, driving the
exact same TradeEngine/RiskManager/PositionManager/OrderService pipeline live
trading uses. Confirms one trade closes via TP and the other via SL, and the
report's derived metrics match by direct calculation.

Uses the real `configs/` (risk.yaml, symbols/xauusd.yaml, app.yaml) since
those are the project's actual, checked-in trading config — no fixture
config needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.application.run_backtest import NoSymbolSpecError, run_backtest
from src.engine.domain.volatility import VolatilityConfig
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.replay import SymbolSpec
from src.market_data.adapters.symbol_spec_repository import SymbolSpecRepository
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base
from src.strategies.registry import StrategyRegistry
from src.strategies.generated.xauusd_snd_qm_structure_m5_v1 import XauusdSndQmStructureM5

M5_STEP = timedelta(minutes=5)
START = datetime(2025, 1, 1, tzinfo=UTC)


def m5(i: int, *, open: float, high: float, low: float, close: float, spread: int = 25) -> Candle:
    return Candle(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        time=START + i * M5_STEP,
        open=open,
        high=high,
        low=low,
        close=close,
        tick_volume=1000,
        spread_points=spread,
    )


def build_m5_candles() -> list[Candle]:
    bars: list[Candle] = []
    # Episode 1: 20-bar flat range, a bullish breakout, then a bar whose high
    # clears the strategy's TP.
    bars += [m5(i, open=2400.0, high=2401.0, low=2399.0, close=2400.0) for i in range(20)]
    bars.append(m5(20, open=2401.0, high=2411.0, low=2400.5, close=2410.0))  # BUY breakout
    # Wick reaches the TP (2434.325) but the close stays below bar 20's high
    # (2411) so this bar doesn't *also* look like a fresh breakout signal.
    bars.append(m5(21, open=2410.0, high=2440.0, low=2405.0, close=2408.0))  # clears TP

    # Episode 2: a fresh 20-bar flat range, a bearish breakout, then a bar
    # whose high clears the strategy's SL (stopping the short out at a loss).
    bars += [m5(22 + i, open=2440.0, high=2441.0, low=2439.0, close=2440.0) for i in range(20)]
    bars.append(m5(42, open=2439.0, high=2439.5, low=2429.0, close=2430.0))  # SELL breakout
    bars.append(m5(43, open=2430.0, high=2445.0, low=2428.0, close=2432.0))  # clears SL
    return bars


def build_htf_candles(timeframe: Timeframe, step: timedelta, count: int = 5) -> list[Candle]:
    """Deliberately fewer bars than mtf_confirm's slow EMA period needs, so
    HTF confirmation is skipped (insufficient history) rather than vetoing —
    same trick test_phase4_engine_flow.py uses."""
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


async def test_backtest_closes_one_trade_via_tp_and_one_via_sl(database_url, monkeypatch):
    # This test's real subject is breakout_v1's TP/SL/sizing math against the
    # real configs/risk.yaml + configs/symbols/xauusd.yaml (see the module
    # docstring), not the volatility guard (Phase B; covered by its own unit
    # tests in tests/unit/engine/). build_m5_candles()'s textbook "quiet
    # range, then breakout" shape is exactly what that guard is designed to
    # flag as a volatility spike, so it's neutralized here the same way
    # `_minimal_configs_dir` neutralizes it below (atr_period exceeding the
    # fixture's bar count -> "insufficient history" NORMAL fallback).
    monkeypatch.setattr(
        "src.backtest.application.run_backtest.load_volatility_config",
        lambda configs_dir: VolatilityConfig(atr_period=999),
    )
    # Frictionless fills: this test pins the *nominal* R multiples (2.2 / -1.0)
    # that breakout_v1's TP_RR and stop define. Broker-constraint simulation
    # (OBSERVABILITY_PLAN.md Phase 4, on by default) slips the entry off the
    # requested price, so realized R lands near but not on those numbers —
    # correct behaviour, covered by tests/unit/backtest/ and
    # tests/integration/test_phase4_backtest_realism.py rather than here.
    report = await run_backtest(
        "breakout_v1",
        "XAUUSD",
        "2025-01:2025-01",
        database_url=database_url,
        simulate_broker_constraints=False,
    )

    assert len(report.trades) == 2
    tp_trade, sl_trade = report.trades

    assert tp_trade.side == "buy"
    assert tp_trade.profit > 0
    assert tp_trade.r_multiple == pytest.approx(2.2, rel=1e-3)  # breakout_v1's TP_RR

    assert sl_trade.side == "sell"
    assert sl_trade.profit < 0
    assert sl_trade.r_multiple == pytest.approx(-1.0, rel=1e-3)

    assert report.win_rate == pytest.approx(0.5)
    assert report.worst_losing_streak == 1
    assert report.ending_balance == pytest.approx(
        report.starting_balance + tp_trade.profit + sl_trade.profit
    )
    assert report.profit_factor == pytest.approx(tp_trade.profit / -sl_trade.profit)
    assert report.avg_r == pytest.approx((tp_trade.r_multiple + sl_trade.r_multiple) / 2)

    # Activity log: the decision trail explaining the two trades, timestamped
    # on the simulated clock (not wall-clock), not just their final outcome.
    # (The very first line — "backtest starting" — is logged before the
    # replay loop sets the clock to the first candle, so it's still stamped
    # at the pre-warmup history_start; every line from the replay itself is
    # bounded by the candle range.)
    messages = [e.message for e in report.activity_log]
    assert any("SIGNAL" in m and "buy" in m for m in messages)
    assert any("ENTRY OPENED" in m for m in messages)
    replay_entries = [e for e in report.activity_log if "backtest starting" not in e.message]
    assert replay_entries
    assert all(START <= e.time <= START + 44 * M5_STEP for e in replay_entries)


async def test_run_backtest_min_rr_override(database_url):
    """configs/symbols/xauusd.yaml's real min_rr=1.5 lets both signals from
    `test_backtest_closes_one_trade_via_tp_and_one_via_sl` through. Overriding
    min_rr to something far stricter for this call only should reject both at
    the spread/RR gate instead, without touching the file."""
    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url, min_rr=10.0
    )
    assert len(report.trades) == 0
    assert report.min_rr == 10.0
    assert any(
        "spread/RR gate" in e.message and "min_rr=10" in e.message for e in report.activity_log
    )


async def test_run_backtest_records_configured_min_rr_without_override(database_url):
    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url
    )
    assert report.min_rr == 1.5  # configs/symbols/xauusd.yaml's real value


async def test_run_backtest_records_configured_risk_caps(database_url):
    """The full RiskCaps actually enforced for the run is recorded on the
    report, not just min_rr. Compared against configs/risk.yaml as loaded,
    not hardcoded values — that file is user-owned and retuned at will."""
    from src.shared.config.loaders import load_risk_caps
    from src.shared.config.settings import CONFIGS_DIR

    caps = load_risk_caps(CONFIGS_DIR)
    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url
    )
    assert report.risk_per_trade_pct == caps.risk_per_trade_pct
    assert report.daily_loss_limit_pct == caps.daily_loss_limit_pct
    assert report.max_open_positions == caps.max_open_positions
    assert report.max_trades_per_day_enabled == caps.max_trades_per_day_enabled
    assert report.consecutive_loss_pause == caps.consecutive_loss_pause
    assert report.min_lot_fallback_enabled is caps.min_lot_fallback_enabled
    assert report.max_risk_per_trade_pct == caps.max_risk_per_trade_pct


async def test_run_backtest_records_risk_override(database_url):
    report = await run_backtest(
        "breakout_v1",
        "XAUUSD",
        "2025-01:2025-01",
        database_url=database_url,
        min_lot_fallback_enabled=False,
        max_risk_per_trade_pct=7.5,
    )
    assert report.min_lot_fallback_enabled is False
    assert report.max_risk_per_trade_pct == 7.5
    # Untouched by the override.
    assert report.risk_per_trade_pct == 0.5


async def test_backtest_raises_when_no_history(database_url):
    from src.backtest.application.run_backtest import NoHistoryError

    with pytest.raises(NoHistoryError):
        await run_backtest("breakout_v1", "XAUUSD", "2030-01:2030-01", database_url=database_url)


async def test_backtest_raises_when_history_only_partially_covers_the_period(database_url):
    """`build_m5_candles()` only seeds 2025-01-01 onward — requesting a period
    that starts well before that (e.g. 2024-06) must raise instead of
    silently replaying just the 2025-01 slice that happens to exist, which is
    what made a request for a year of history quietly turn into a report
    covering only a couple of days."""
    from src.backtest.application.run_backtest import NoHistoryError

    with pytest.raises(NoHistoryError, match="only goes back to"):
        await run_backtest("breakout_v1", "XAUUSD", "2024-06:2025-01", database_url=database_url)


async def test_backtest_rejects_unknown_strategy(database_url):
    with pytest.raises(ValueError, match="unknown strategy"):
        await run_backtest("not_a_strategy", "XAUUSD", "2025-01:2025-01", database_url=database_url)


async def test_backtest_rejects_symbol_the_strategy_does_not_trade(database_url):
    registry = StrategyRegistry()
    strategy = XauusdSndQmStructureM5()
    registry.register(strategy.spec.name, strategy)
    with pytest.raises(ValueError, match="does not trade"):
        await run_backtest(strategy.spec.name, "EURUSD", "2025-01:2025-01", database_url=database_url, strategy_source=registry)


def _minimal_configs_dir(tmp_path: Path, *, xauusd_yaml: bool) -> Path:
    """A fixture configs/ containing only risk.yaml + app.yaml + volatility.yaml
    (all required unconditionally by run_backtest) and, optionally, a legacy
    symbols/xauusd.yaml — for exercising the DB-backed SymbolSpec sourcing
    without depending on the project's real checked-in config.

    volatility.yaml's atr_period (999) deliberately exceeds every fixture's
    bar count in this file, so the volatility guard (Phase B) always takes
    its "insufficient history" NORMAL fallback here — these tests exercise
    symbol-spec sourcing and risk sizing, not the volatility guard (see
    tests/unit/engine/test_trade_loop.py and test_position_manager.py for
    that), and build_m5_candles()'s textbook "quiet range, then breakout"
    shape is exactly what the guard is designed to flag as a volatility
    spike."""
    configs_dir = tmp_path / "configs"
    (configs_dir / "symbols").mkdir(parents=True)
    (configs_dir / "risk.yaml").write_text(
        "risk_per_trade_pct: 0.5\n"
        "daily_loss_limit_pct: 2.0\n"
        "max_open_positions: 100\n"
        "max_trades_per_day_enabled: false\n"
        "consecutive_loss_pause: 10\n"
    )
    (configs_dir / "app.yaml").write_text('timezone: "UTC"\n')
    (configs_dir / "volatility.yaml").write_text("atr_period: 999\n")
    if xauusd_yaml:
        (configs_dir / "symbols" / "xauusd.yaml").write_text(
            "symbol: XAUUSD\n"
            "max_spread_points: 35\n"
            "min_rr: 1.5\n"
            "contract_size: 100.0\n"
            "point: 0.01\n"
            "digits: 2\n"
            "stops_level: 0\n"
            "volume_min: 0.01\n"
            "volume_max: 50\n"
            "volume_step: 0.01\n"
        )
    return configs_dir


def _spec(contract_size: float = 100.0) -> SymbolSpec:
    return SymbolSpec(
        point=0.01,
        digits=2,
        stops_level=0,
        contract_size=contract_size,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
    )


async def test_backtest_uses_db_backed_symbol_spec_without_any_yaml(tmp_path, database_url):
    """No configs/symbols/xauusd.yaml at all — the symbol_specs DB row
    (as populated by POST /market-data/backfill in production) is enough on
    its own to run a backtest."""
    configs_dir = _minimal_configs_dir(tmp_path, xauusd_yaml=False)
    engine = create_engine(database_url)
    SymbolSpecRepository(sessionmaker(bind=engine, expire_on_commit=False)).upsert(
        "XAUUSD", _spec()
    )

    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url,
        configs_dir=configs_dir,
    )

    assert len(report.trades) == 2


async def test_run_backtest_min_lot_fallback_override(tmp_path, database_url):
    """`_minimal_configs_dir`'s risk.yaml has no min_lot_fallback_enabled key
    (defaults False) — a $50 balance is too small for breakout_v1's normal
    risk % to reach volume_min, so sizing rejects both signals with the file
    default. Passing the override params to run_backtest() turns the
    fallback on for this call only, without touching the file, and the same
    two signals now open."""
    configs_dir = _minimal_configs_dir(tmp_path, xauusd_yaml=False)
    engine = create_engine(database_url)
    SymbolSpecRepository(sessionmaker(bind=engine, expire_on_commit=False)).upsert(
        "XAUUSD", _spec()
    )

    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url,
        configs_dir=configs_dir, starting_balance=50.0,
    )
    assert len(report.trades) == 0

    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url,
        configs_dir=configs_dir, starting_balance=50.0,
        min_lot_fallback_enabled=True, max_risk_per_trade_pct=25.0,
    )
    assert len(report.trades) == 2
    assert any("min-lot fallback" in e.message for e in report.activity_log)


async def test_backtest_raises_no_symbol_spec_without_db_row_or_yaml(tmp_path, database_url):
    configs_dir = _minimal_configs_dir(tmp_path, xauusd_yaml=False)

    with pytest.raises(NoSymbolSpecError, match="XAUUSD"):
        await run_backtest(
            "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url,
            configs_dir=configs_dir,
        )


async def test_db_symbol_spec_takes_precedence_over_legacy_yaml(tmp_path, database_url):
    """The legacy YAML has a normal volume_min (0.01, same as the main
    fixture test above, which trades fine). The DB row's volume_min is set
    absurdly high (1000) — risk-based position sizing can never produce a
    viable lot size that large, so no trade opens. If the YAML were still
    winning over the DB row, trades would go through exactly like the main
    test; zero trades here proves the DB row is the one actually used."""
    configs_dir = _minimal_configs_dir(tmp_path, xauusd_yaml=True)
    engine = create_engine(database_url)
    repository = SymbolSpecRepository(sessionmaker(bind=engine, expire_on_commit=False))
    repository.upsert(
        "XAUUSD",
        SymbolSpec(
            point=0.01,
            digits=2,
            stops_level=0,
            contract_size=100.0,
            volume_min=1000.0,
            volume_max=2000.0,
            volume_step=0.01,
        ),
    )

    report = await run_backtest(
        "breakout_v1", "XAUUSD", "2025-01:2025-01", database_url=database_url,
        configs_dir=configs_dir,
    )

    assert report.trades == ()
