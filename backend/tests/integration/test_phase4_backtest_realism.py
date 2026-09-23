"""Backtest realism end-to-end (OBSERVABILITY_PLAN.md Phase 4).

Replays the same synthetic two-breakout history `test_phase5_backtest_flow.py`
uses — one trade closing at TP, one at SL — through `run_backtest`, and checks
that the simulated broker now behaves like a real one:

* an entry whose stop sits inside the symbol's `stops_level` is **refused and
  counted**, not reported as a trade;
* lot sizes land on the broker's volume grid;
* the report carries the Phase 2 split outcome vocabulary rather than the
  collapsed one the log-scraper produced;
* identical inputs still produce byte-identical trades.

The `stops_level` used in the rejection tests is Volatility 75 Index's real
one (10770 points on a 0.01 point = a 107.70-unit minimum stop distance),
applied to the fixture's XAUUSD history because `breakout_v1` is the strategy
with a deterministic two-trade fixture. The figure is what matters: every M1
scalp stop this project ran on VIX75 was inside it, and every one was refused
live with MT5 retcode 10016 while the backtest called them winners.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.application.run_backtest import run_backtest
from src.backtest.domain.models import BacktestReport
from src.broker.domain.broker_constraints import (
    REASON_STOPS_LEVEL,
    REASON_VOLUME_BELOW_MIN,
    RETCODE_INVALID_STOPS,
)
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.replay import SymbolSpec
from src.market_data.adapters.symbol_spec_repository import SymbolSpecRepository
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base

M5_STEP = timedelta(minutes=5)
START = datetime(2025, 1, 1, tzinfo=UTC)
PERIOD = "2025-01:2025-01"

# XAUUSD's real broker facts, as snapshotted in this project's `symbol_specs`.
XAUUSD_SPEC = SymbolSpec(
    point=0.01,
    digits=2,
    stops_level=20,
    contract_size=100.0,
    volume_min=0.01,
    volume_max=10.0,
    volume_step=0.01,
)
# Volatility 75 Index's real minimum stop distance, in points on a 0.01 point.
VIX75_STOPS_LEVEL = 10770


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
    """Two clean breakout episodes: the first resolves at TP, the second at SL."""
    bars: list[Candle] = []
    bars += [m5(i, open=2400.0, high=2401.0, low=2399.0, close=2400.0) for i in range(20)]
    bars.append(m5(20, open=2401.0, high=2411.0, low=2400.5, close=2410.0))  # BUY breakout
    bars.append(m5(21, open=2410.0, high=2440.0, low=2405.0, close=2408.0))  # clears TP
    bars += [m5(22 + i, open=2440.0, high=2441.0, low=2439.0, close=2440.0) for i in range(20)]
    bars.append(m5(42, open=2439.0, high=2439.5, low=2429.0, close=2430.0))  # SELL breakout
    bars.append(m5(43, open=2430.0, high=2445.0, low=2428.0, close=2432.0))  # clears SL
    return bars


def build_htf_candles(timeframe: Timeframe, step: timedelta, count: int = 5) -> list[Candle]:
    """Too few bars for mtf_confirm's slow EMA, so HTF confirmation is skipped
    rather than vetoing — the same trick the Phase 5 flow test uses."""
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


def seed_database(tmp_path, spec: SymbolSpec) -> str:
    url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    repository = CandleRepository(session_factory)
    repository.upsert_many(build_m5_candles())
    repository.upsert_many(build_htf_candles(Timeframe.H1, timedelta(hours=1)))
    repository.upsert_many(build_htf_candles(Timeframe.H4, timedelta(hours=4)))
    SymbolSpecRepository(session_factory).upsert("XAUUSD", spec)
    return url


@pytest.fixture
def database_url(tmp_path) -> str:
    return seed_database(tmp_path, XAUUSD_SPEC)


@pytest.fixture
def vix75_stops_database_url(tmp_path) -> str:
    """Same history, but the symbol demands VIX75's 107.70-unit minimum stop —
    far wider than anything `breakout_v1` places on this fixture."""
    return seed_database(tmp_path, replace(XAUUSD_SPEC, stops_level=VIX75_STOPS_LEVEL))


async def run(url: str, **kwargs) -> BacktestReport:
    return await run_backtest("breakout_v1", "XAUUSD", PERIOD, database_url=url, **kwargs)


class TestStopsLevelRejection:
    async def test_entries_the_broker_would_refuse_are_not_reported_as_trades(
        self, vix75_stops_database_url
    ) -> None:
        report = await run(vix75_stops_database_url)
        assert report.trades == ()
        assert report.broker_realism.rejected_count > 0
        assert report.broker_realism.accepted_count == 0
        assert report.broker_realism.acceptance_rate == 0.0

    async def test_the_rejections_are_counted_by_reason_with_the_mt5_retcode(
        self, vix75_stops_database_url
    ) -> None:
        report = await run(vix75_stops_database_url)
        (rejection,) = report.broker_realism.rejections
        assert rejection.reason == REASON_STOPS_LEVEL
        assert rejection.retcode == RETCODE_INVALID_STOPS
        assert rejection.count == report.broker_realism.rejected_count
        assert "107.70" in rejection.example

    async def test_the_rejected_signals_show_as_broker_rejected_on_the_trail(
        self, vix75_stops_database_url
    ) -> None:
        report = await run(vix75_stops_database_url)
        assert any(s.outcome == "broker_rejected" for s in report.signals)

    async def test_disabling_the_simulation_reproduces_the_old_fictional_result(
        self, vix75_stops_database_url
    ) -> None:
        """The size of the correction, made explicit: the same history, the
        same strategy, and the pre-Phase-4 run happily reports trades the
        broker would have refused every single time."""
        honest = await run(vix75_stops_database_url)
        fictional = await run(vix75_stops_database_url, simulate_broker_constraints=False)
        assert len(fictional.trades) == 2
        assert len(honest.trades) == 0
        assert fictional.broker_realism.enabled is False

    async def test_clamping_trades_them_instead_but_flags_the_wider_risk(
        self, vix75_stops_database_url
    ) -> None:
        report = await run(vix75_stops_database_url, clamp_stops=True)
        assert report.broker_realism.clamp_stops is True
        assert report.broker_realism.clamped_count > 0
        assert report.broker_realism.rejected_count == 0
        # The stops really moved — from ~11 units away to ~107.7, i.e. the
        # position now risks an order of magnitude more than the risk manager
        # sized it for, which is why this is research-only. The distance is
        # measured from the pre-slippage reference price the broker validates
        # against, so the realized distance from the fill differs by the
        # slippage draw; a 1-unit tolerance covers that without hiding a
        # clamp that did not happen.
        for trade in report.trades:
            assert trade.sl is not None
            assert abs(trade.open_price - trade.sl) >= 107.70 - 1.0


class TestNormalSymbolStillTrades:
    async def test_xauusds_real_stops_level_refuses_nothing(self, database_url) -> None:
        """A regression guard on the guard: enforcing `stops_level` must not
        start rejecting ordinary trades on symbols whose minimum is small."""
        report = await run(database_url)
        assert len(report.trades) == 2
        assert report.broker_realism.rejected_count == 0
        assert report.broker_realism.accepted_count == 2
        assert report.broker_realism.acceptance_rate == 1.0

    async def test_lot_sizes_land_on_the_brokers_volume_grid(self, database_url) -> None:
        report = await run(database_url)
        assert report.trades
        for trade in report.trades:
            steps = trade.volume / XAUUSD_SPEC.volume_step
            assert steps == pytest.approx(round(steps))
            assert trade.volume >= XAUUSD_SPEC.volume_min

    async def test_a_sub_minimum_lot_is_refused_rather_than_silently_traded(self, tmp_path) -> None:
        """A 1.0-lot step: the risk manager's 0.04-lot sizing is perfectly
        valid on its own terms, rounds down to 0.0 on the broker's grid, and
        must then be **refused** — never quietly bumped up to `volume_min`,
        which would trade 25x the size the risk manager approved."""
        url = seed_database(tmp_path, replace(XAUUSD_SPEC, volume_step=1.0))
        report = await run(url)
        assert report.trades == ()
        (rejection,) = report.broker_realism.rejections
        assert rejection.reason == REASON_VOLUME_BELOW_MIN

    async def test_slippage_makes_fills_worse_than_the_frictionless_ones(
        self, database_url
    ) -> None:
        """Calibrated on 40 identical 0.10-unit live fills, so the model has
        no variance and the effect is exact rather than a lucky draw: a buy
        pays 0.10 more and a sell receives 0.10 less than the frictionless
        quote, and both earn less as a result."""
        honest = await run(database_url, slippage_samples=[0.10] * 40)
        frictionless = await run(database_url, simulate_broker_constraints=False)
        buy_honest, sell_honest = honest.trades
        buy_free, sell_free = frictionless.trades
        assert buy_honest.open_price == pytest.approx(buy_free.open_price + 0.10)
        assert sell_honest.open_price == pytest.approx(sell_free.open_price - 0.10)
        assert buy_honest.profit < buy_free.profit
        assert sell_honest.profit < sell_free.profit

    async def test_the_slippage_model_documents_its_own_calibration(self, database_url) -> None:
        report = await run(database_url)
        # No live fills supplied, so the documented fallback applies and says so.
        assert report.broker_realism.slippage_source == "fallback"
        assert report.broker_realism.slippage_sample_count == 0
        assert report.broker_realism.slippage_mean > 0.0

    async def test_live_samples_calibrate_the_model_instead_of_the_fallback(
        self, database_url
    ) -> None:
        report = await run(database_url, slippage_samples=[0.07] * 40)
        assert report.broker_realism.slippage_source == "live"
        assert report.broker_realism.slippage_sample_count == 40
        assert report.broker_realism.slippage_mean == pytest.approx(0.07)


class TestOutcomeVocabulary:
    async def test_signals_use_the_phase_2_split_vocabulary_not_the_collapsed_one(
        self, database_url
    ) -> None:
        from src.activity.domain.models import SIGNAL_OUTCOMES

        report = await run(database_url)
        assert report.signals
        assert all(s.outcome in SIGNAL_OUTCOMES for s in report.signals)
        # The legacy log-scraper's collapsed bucket must no longer be emitted.
        assert not any(s.outcome == "risk_rejected" for s in report.signals)
        assert sum(1 for s in report.signals if s.outcome == "opened") == len(report.trades)

    async def test_signals_carry_the_reference_price_the_engine_saw(self, database_url) -> None:
        report = await run(database_url)
        assert all(s.price is not None for s in report.signals)


class TestDeterminism:
    """Backtest reproducibility was previously verified via trade fingerprints.
    Simulating slippage introduces an RNG, which is exactly the kind of change
    that can silently break it — so it is pinned here."""

    @staticmethod
    def fingerprint(report: BacktestReport) -> list[tuple]:
        return [
            (
                t.side,
                t.volume,
                t.open_time,
                t.open_price,
                t.sl,
                t.tp,
                t.close_time,
                t.close_price,
                t.profit,
                t.r_multiple,
            )
            for t in report.trades
        ]

    async def test_identical_inputs_produce_identical_trades(self, database_url) -> None:
        first = await run(database_url)
        second = await run(database_url)
        assert self.fingerprint(first) == self.fingerprint(second)
        assert first.ending_balance == second.ending_balance
        assert first.profit_factor == second.profit_factor
        assert first.equity_curve == second.equity_curve

    async def test_the_rejection_counts_are_reproducible_too(
        self, vix75_stops_database_url
    ) -> None:
        first = await run(vix75_stops_database_url)
        second = await run(vix75_stops_database_url)
        assert first.broker_realism.rejections == second.broker_realism.rejections

    async def test_a_different_slippage_seed_changes_the_fills(self, database_url) -> None:
        """Determinism must come from the seed, not from the RNG being inert —
        otherwise the test above would pass on a broken model."""
        default = await run(database_url)
        reseeded = await run(database_url, slippage_seed=424242)
        assert default.broker_realism.slippage_stddev > 0.0
        assert self.fingerprint(default) != self.fingerprint(reseeded)
