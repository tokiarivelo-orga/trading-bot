"""Pure aggregation logic for the analytics dashboard (journal/domain/analytics.py)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

from src.journal.domain.analytics import (
    compute_bot_analytics,
    compute_daily_pnl,
    compute_regime_analytics,
    compute_symbol_analytics,
)
from src.journal.domain.models import CandleSnapshot, TradeAnalyticsRecord, TradeRecord


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def make_record(id: str, symbol: str = "XAUUSD", **kw) -> TradeRecord:
    defaults = dict(
        id=id,
        symbol=symbol,
        side="buy",
        volume=0.1,
        open_price=2400.35,
        open_time=utc(2026, 7, 10, 14, 0),
        sl=2390.0,
        tp=2420.0,
        spread_points_at_entry=25,
        comment="",
    )
    return TradeRecord(**{**defaults, **kw})


# ── compute_symbol_analytics ────────────────────────────────────────────────


def test_symbol_analytics_aggregates_wins_losses_and_open_trades():
    trades = [
        make_record(
            "1",
            symbol="XAUUSD",
            skill="normal/xauusd/a",
            open_time=utc(2026, 7, 10, 14, 0),
            close_time=utc(2026, 7, 10, 15, 0),
            profit=10.0,
        ),
        make_record(
            "2",
            symbol="XAUUSD",
            skill="normal/xauusd/b",
            open_time=utc(2026, 7, 10, 15, 0),
            close_time=utc(2026, 7, 10, 16, 0),
            profit=-4.0,
        ),
        make_record("3", symbol="XAUUSD", open_time=utc(2026, 7, 10, 16, 0)),  # still open
        make_record(
            "4",
            symbol="EURUSD",
            open_time=utc(2026, 7, 10, 17, 0),
            close_time=utc(2026, 7, 10, 18, 0),
            profit=2.0,
        ),
    ]

    results = compute_symbol_analytics(trades)

    xau = next(r for r in results if r.symbol == "XAUUSD")
    assert xau.trade_count == 3
    assert xau.open_count == 1
    assert xau.closed_count == 2
    assert xau.win_count == 1
    assert xau.loss_count == 1
    assert xau.win_rate == 0.5
    assert xau.total_profit == 6.0
    assert xau.gross_profit == 10.0
    assert xau.gross_loss == 4.0
    assert xau.profit_factor == 2.5
    assert xau.bot_count == 2

    eur = next(r for r in results if r.symbol == "EURUSD")
    assert eur.closed_count == 1
    assert eur.win_rate == 1.0


def test_symbol_analytics_sorted_by_total_profit_descending():
    trades = [
        make_record("1", symbol="A", close_time=utc(2026, 7, 10, 15, 0), profit=1.0),
        make_record("2", symbol="B", close_time=utc(2026, 7, 10, 15, 0), profit=50.0),
    ]

    results = compute_symbol_analytics(trades)

    assert [r.symbol for r in results] == ["B", "A"]


def test_symbol_analytics_profit_factor_none_with_no_losses():
    trades = [make_record("1", close_time=utc(2026, 7, 10, 15, 0), profit=5.0)]

    result = compute_symbol_analytics(trades)[0]

    assert result.profit_factor is None


# ── slim TradeAnalyticsRecord projection (optimization: repository.py's
#    get_all_for_analytics skips the JSON snapshot/structure columns this
#    aggregation never reads) ─────────────────────────────────────────────


def _to_analytics_record(record: TradeRecord) -> TradeAnalyticsRecord:
    """Mirrors what `JournalRepository.get_all_for_analytics` projects from a
    full row — only the fields `compute_symbol_analytics`/`compute_bot_analytics`
    actually touch."""
    return TradeAnalyticsRecord(
        id=record.id,
        symbol=record.symbol,
        volume=record.volume,
        open_time=record.open_time,
        close_time=record.close_time,
        profit=record.profit,
        skill=record.skill,
        strategy_version=record.strategy_version,
        regime_volatility=record.regime_volatility,
        regime_trend=record.regime_trend,
        regime_session=record.regime_session,
        transaction_cost=record.transaction_cost,
    )


def _make_full_trades() -> list[TradeRecord]:
    snapshot = (
        CandleSnapshot(
            time=utc(2026, 7, 10, 13, 55), open=1, high=2, low=0.5, close=1.5, tick_volume=100
        ),
    )
    return [
        make_record(
            "1",
            symbol="XAUUSD",
            skill="normal/xauusd/a",
            strategy_version="breakout_v1:v1",
            volume=0.2,
            open_time=utc(2026, 7, 10, 14, 0),
            close_time=utc(2026, 7, 10, 15, 0),
            profit=10.0,
            m5_entry_snapshot=snapshot,
            h1_entry_snapshot=snapshot,
            m5_exit_snapshot=snapshot,
            h1_exit_snapshot=snapshot,
            reason="RBR base retest",
            confidence=0.82,
            zone_kind="demand",
            structure=(("HL", 2397.2, utc(2026, 7, 10, 13, 30)),),
        ),
        make_record(
            "2",
            symbol="XAUUSD",
            skill="normal/xauusd/b",
            strategy_version="breakout_v1:v1",
            volume=0.1,
            open_time=utc(2026, 7, 10, 15, 0),
            close_time=utc(2026, 7, 10, 16, 0),
            profit=-4.0,
        ),
        make_record(
            "3",
            symbol="XAUUSD",
            open_time=utc(2026, 7, 10, 16, 0),
        ),  # still open
        make_record(
            "4",
            symbol="EURUSD",
            skill="normal/eurusd/a",
            strategy_version="meanrev_v2:v1",
            open_time=utc(2026, 7, 10, 17, 0),
            close_time=utc(2026, 7, 10, 18, 0),
            profit=2.0,
        ),
    ]


def test_compute_symbol_analytics_bit_identical_for_slim_records():
    full_trades = _make_full_trades()
    slim_trades = [_to_analytics_record(t) for t in full_trades]

    assert compute_symbol_analytics(slim_trades) == compute_symbol_analytics(full_trades)


def test_compute_bot_analytics_bit_identical_for_slim_records():
    full_trades = _make_full_trades()
    slim_trades = [_to_analytics_record(t) for t in full_trades]

    assert compute_bot_analytics(slim_trades) == compute_bot_analytics(full_trades)


def test_trade_analytics_record_has_no_json_snapshot_or_structure_fields():
    """The slim projection must not carry the four JSON columns
    (m5/h1_entry/exit_snapshot, structure) that analytics.py never reads —
    that's the whole point of `get_all_for_analytics` over `get_all`."""
    field_names = {f.name for f in dataclasses.fields(TradeAnalyticsRecord)}
    assert field_names == {
        "id",
        "symbol",
        "volume",
        "open_time",
        "close_time",
        "profit",
        "skill",
        "strategy_version",
        # Execution telemetry / excursion (OBSERVABILITY_PLAN.md Phase 3) —
        # scalar columns the per-bot aggregates read, still no JSON.
        "slippage",
        "execution_latency_ms",
        "broker_retcode",
        "mfe",
        "mfe_time",
        "mae",
        "mae_time",
        # Regime tagging + transaction cost (OBSERVABILITY_PLAN.md Phase 6) —
        # bucket strings + cost only, still no JSON.
        "regime_volatility",
        "regime_trend",
        "regime_session",
        "transaction_cost",
    }
    excluded = {
        "m5_entry_snapshot",
        "h1_entry_snapshot",
        "m5_exit_snapshot",
        "h1_exit_snapshot",
        "structure",
    }
    assert field_names.isdisjoint(excluded)


def test_symbol_analytics_empty_input_returns_empty_list():
    assert compute_symbol_analytics([]) == []


# ── compute_bot_analytics ───────────────────────────────────────────────────


def test_bot_analytics_excludes_trades_with_no_skill():
    trades = [
        make_record(
            "1", skill="normal/xauusd/a", close_time=utc(2026, 7, 10, 15, 0), profit=5.0
        ),
        make_record("2", skill=None, close_time=utc(2026, 7, 10, 15, 0), profit=100.0),
    ]

    results = compute_bot_analytics(trades)

    assert len(results) == 1
    assert results[0].skill == "normal/xauusd/a"


def test_bot_analytics_computes_equity_curve_and_drawdown():
    trades = [
        make_record(
            "1",
            skill="normal/xauusd/a",
            open_time=utc(2026, 7, 10, 14, 0),
            close_time=utc(2026, 7, 10, 15, 0),
            profit=10.0,
        ),
        make_record(
            "2",
            skill="normal/xauusd/a",
            open_time=utc(2026, 7, 10, 15, 0),
            close_time=utc(2026, 7, 10, 16, 0),
            profit=-15.0,
        ),
        make_record(
            "3",
            skill="normal/xauusd/a",
            open_time=utc(2026, 7, 10, 16, 0),
            close_time=utc(2026, 7, 10, 17, 0),
            profit=8.0,
        ),
    ]

    bot = compute_bot_analytics(trades)[0]

    assert [p.cumulative_profit for p in bot.equity_curve] == [10.0, -5.0, 3.0]
    # peak 10 -> trough -5 => drawdown 15
    assert bot.max_drawdown == 15.0
    assert bot.total_profit == 3.0
    assert bot.win_count == 2
    assert bot.loss_count == 1
    assert bot.expectancy == 1.0
    assert bot.avg_trade_duration_seconds == 3600.0


def test_bot_analytics_bot_name_and_symbol_derived_from_skill_and_latest_trade():
    trades = [
        make_record(
            "1",
            symbol="XAUUSD",
            skill="normal/xauusd/breakout_v1",
            strategy_version="breakout_v1:v1",
            open_time=utc(2026, 7, 10, 14, 0),
            close_time=utc(2026, 7, 10, 15, 0),
            profit=1.0,
        )
    ]

    bot = compute_bot_analytics(trades)[0]

    assert bot.bot_name == "breakout_v1"
    assert bot.symbol == "XAUUSD"
    assert bot.strategy_version == "breakout_v1:v1"


def test_bot_analytics_sorted_by_total_profit_descending():
    trades = [
        make_record(
            "1", skill="normal/a/x", close_time=utc(2026, 7, 10, 15, 0), profit=1.0
        ),
        make_record(
            "2", skill="normal/b/y", close_time=utc(2026, 7, 10, 15, 0), profit=99.0
        ),
    ]

    results = compute_bot_analytics(trades)

    assert [r.skill for r in results] == ["normal/b/y", "normal/a/x"]


def test_bot_analytics_max_drawdown_zero_when_curve_never_dips():
    trades = [
        make_record(
            "1", skill="normal/a/x", close_time=utc(2026, 7, 10, 15, 0), profit=1.0
        ),
        make_record(
            "2", skill="normal/a/x", close_time=utc(2026, 7, 10, 16, 0), profit=2.0
        ),
    ]

    bot = compute_bot_analytics(trades)[0]

    assert bot.max_drawdown == 0.0


# ── execution-quality aggregates (OBSERVABILITY_PLAN.md Phase 3) ────────────


def telemetry_record(id: str, **kw) -> TradeRecord:
    """A closed, bot-attributed trade carrying execution telemetry."""
    defaults = dict(
        skill="normal/xauusd/a",
        close_time=utc(2026, 7, 10, 15, 0),
        profit=10.0,
    )
    return make_record(id, **{**defaults, **kw})


def test_bot_analytics_averages_slippage_latency_and_excursion():
    bots = compute_bot_analytics(
        [
            telemetry_record("1", slippage=0.10, execution_latency_ms=200.0, mfe=8.0, mae=2.0),
            telemetry_record("2", slippage=0.30, execution_latency_ms=400.0, mfe=4.0, mae=6.0),
        ]
    )

    assert bots[0].avg_slippage == 0.20
    assert bots[0].measured_slippage_count == 2
    assert bots[0].avg_execution_latency_ms == 300.0
    assert bots[0].avg_mfe == 6.0
    assert bots[0].avg_mae == 4.0
    assert bots[0].mfe_mae_ratio == 1.5


def test_unmeasured_trades_are_skipped_rather_than_counted_as_zero():
    """Trades journaled before Phase 3 carry no telemetry; folding them in as
    zeros would drag every average toward 0 the longer a bot's history is."""
    bots = compute_bot_analytics(
        [
            telemetry_record("1", slippage=0.40),
            telemetry_record("2"),  # pre-Phase-3 row: slippage is None
        ]
    )

    assert bots[0].avg_slippage == 0.40
    assert bots[0].measured_slippage_count == 1


def test_a_bot_with_no_telemetry_at_all_reports_none_not_zero():
    bots = compute_bot_analytics([telemetry_record("1")])

    assert bots[0].avg_slippage is None
    assert bots[0].avg_execution_latency_ms is None
    assert bots[0].avg_mfe is None
    assert bots[0].mfe_mae_ratio is None
    assert bots[0].retcode_histogram == ()


def test_retcode_histogram_counts_codes_most_frequent_first():
    bots = compute_bot_analytics(
        [
            telemetry_record("1", broker_retcode=10009),
            telemetry_record("2", broker_retcode=10016),
            telemetry_record("3", broker_retcode=10016),
            telemetry_record("4"),  # no code reported — contributes no bucket
        ]
    )

    assert bots[0].retcode_histogram == ((10016, 2), (10009, 1))


def test_excursion_is_split_by_outcome_to_answer_tp_and_sl_questions():
    """avg_mfe_on_losers answers "are my TPs too far", avg_mae_on_winners
    answers "are my SLs too tight" — each must read only its own subset."""
    bots = compute_bot_analytics(
        [
            telemetry_record("1", profit=10.0, mfe=12.0, mae=1.0),
            telemetry_record("2", profit=-5.0, mfe=9.0, mae=7.0),
            telemetry_record("3", profit=-5.0, mfe=11.0, mae=8.0),
        ]
    )

    assert bots[0].avg_mfe_on_losers == 10.0
    assert bots[0].avg_mae_on_winners == 1.0


# ── cost-as-%-of-gross-edge (OBSERVABILITY_PLAN.md Phase 6) ─────────────────


def test_bot_analytics_cost_fields_averaged_over_measured_closed_trades():
    bots = compute_bot_analytics(
        [
            telemetry_record("1", profit=10.0, transaction_cost=2.0),
            telemetry_record("2", profit=10.0, transaction_cost=4.0),
        ]
    )

    assert bots[0].total_transaction_cost == 6.0
    assert bots[0].avg_transaction_cost_per_trade == 3.0
    # gross_edge = total_profit (20) + total_transaction_cost (6) = 26
    assert bots[0].cost_pct_of_gross_edge == 6.0 / 26.0


def test_bot_analytics_cost_fields_none_when_unmeasured():
    bots = compute_bot_analytics([telemetry_record("1", profit=10.0)])

    assert bots[0].total_transaction_cost is None
    assert bots[0].avg_transaction_cost_per_trade is None
    assert bots[0].cost_pct_of_gross_edge is None


def test_cost_pct_of_gross_edge_none_when_gross_edge_not_positive():
    """A bot that lost more than its cost drag (gross_edge <= 0) reports
    `cost_pct_of_gross_edge=None` — undefined, like `profit_factor` with no
    losses, rather than a negative or divide-by-zero artifact."""
    bots = compute_bot_analytics(
        [telemetry_record("1", profit=-10.0, transaction_cost=2.0)]
    )

    # gross_edge = total_profit (-10) + total_transaction_cost (2) = -8 <= 0
    assert bots[0].total_transaction_cost == 2.0
    assert bots[0].cost_pct_of_gross_edge is None


# ── compute_regime_analytics (OBSERVABILITY_PLAN.md Phase 6) ────────────────


def regime_record(id: str, **kw) -> TradeRecord:
    defaults = dict(
        skill="normal/xauusd/a",
        close_time=utc(2026, 7, 10, 15, 0),
        profit=10.0,
    )
    return make_record(id, **{**defaults, **kw})


def test_regime_analytics_groups_by_bot_and_bucket_per_dimension():
    trades = [
        regime_record("1", profit=10.0, regime_volatility="high", regime_trend="trending",
                       regime_session="london"),
        regime_record("2", profit=-4.0, regime_volatility="high", regime_trend="ranging",
                       regime_session="london"),
        regime_record("3", profit=6.0, regime_volatility="low", regime_trend="trending",
                       regime_session="asian"),
    ]

    results = compute_regime_analytics(trades)

    by_key = {(r.dimension, r.bucket): r for r in results}
    assert by_key[("volatility", "high")].trade_count == 2
    assert by_key[("volatility", "high")].total_profit == 6.0
    assert by_key[("volatility", "low")].trade_count == 1
    assert by_key[("trend", "trending")].trade_count == 2
    assert by_key[("trend", "ranging")].trade_count == 1
    assert by_key[("session", "london")].trade_count == 2
    assert by_key[("session", "asian")].trade_count == 1
    # Three dimensions, one bucket-group per dimension present in the trades.
    assert {r.dimension for r in results} == {"volatility", "trend", "session"}


def test_regime_analytics_skips_trades_with_no_skill_or_untagged_dimension():
    trades = [
        regime_record("1", skill=None, regime_volatility="high"),  # no skill: excluded entirely
        regime_record("2", regime_volatility=None, regime_trend="trending"),  # untagged volatility
    ]

    results = compute_regime_analytics(trades)

    assert all(r.dimension != "volatility" for r in results)
    assert any(r.dimension == "trend" and r.bucket == "trending" for r in results)


def test_regime_analytics_sorted_by_bot_name_dimension_bucket():
    trades = [
        regime_record("1", skill="normal/xauusd/b", regime_volatility="low"),
        regime_record("2", skill="normal/xauusd/a", regime_volatility="high"),
        regime_record("3", skill="normal/xauusd/a", regime_session="london"),
    ]

    results = compute_regime_analytics(trades)

    keys = [(r.bot_name, r.dimension, r.bucket) for r in results]
    assert keys == sorted(keys)


def test_regime_analytics_win_rate_and_profit_factor_within_bucket():
    trades = [
        regime_record("1", profit=10.0, regime_volatility="high"),
        regime_record("2", profit=-5.0, regime_volatility="high"),
    ]

    results = compute_regime_analytics(trades)
    high = next(r for r in results if r.dimension == "volatility" and r.bucket == "high")

    assert high.win_count == 1
    assert high.loss_count == 1
    assert high.win_rate == 0.5
    assert high.profit_factor == 2.0
    assert high.expectancy == 2.5
    assert high.total_profit == 5.0


def test_regime_analytics_empty_input_returns_empty_list():
    assert compute_regime_analytics([]) == []


# ── compute_daily_pnl (Phase 6 Part A) ───────────────────────────────────────


def test_daily_pnl_groups_by_close_date_across_two_days():
    trades = [
        # Day 1 (2026-07-10): one win, one loss.
        make_record(
            "1",
            open_time=utc(2026, 7, 10, 8, 0),
            close_time=utc(2026, 7, 10, 9, 0),
            profit=10.0,
        ),
        make_record(
            "2",
            open_time=utc(2026, 7, 10, 10, 0),
            close_time=utc(2026, 7, 10, 11, 0),
            profit=-4.0,
        ),
        # Day 2 (2026-07-11): two wins.
        make_record(
            "3",
            open_time=utc(2026, 7, 11, 8, 0),
            close_time=utc(2026, 7, 11, 9, 0),
            profit=6.0,
        ),
        make_record(
            "4",
            open_time=utc(2026, 7, 11, 10, 0),
            close_time=utc(2026, 7, 11, 11, 0),
            profit=3.0,
        ),
        # Still open — no close_time, must be excluded from every day.
        make_record("5", open_time=utc(2026, 7, 11, 12, 0)),
    ]

    results = compute_daily_pnl(trades)

    assert [r.date for r in results] == [date(2026, 7, 10), date(2026, 7, 11)]

    day1 = results[0]
    assert day1.pnl == 6.0  # 10.0 - 4.0
    assert day1.trade_count == 2
    assert day1.win_count == 1
    assert day1.loss_count == 1
    assert day1.breakeven_count == 0
    assert day1.win_rate == 0.5
    assert day1.gross_profit == 10.0
    assert day1.gross_loss == 4.0
    assert day1.profit_factor == 2.5
    assert day1.avg_win == 10.0
    assert day1.avg_loss == 4.0
    assert day1.largest_win == 10.0
    assert day1.largest_loss == -4.0

    day2 = results[1]
    assert day2.pnl == 9.0  # 6.0 + 3.0
    assert day2.trade_count == 2
    assert day2.win_count == 2
    assert day2.loss_count == 0
    assert day2.win_rate == 1.0
    assert day2.profit_factor is None  # no losses that day
    assert day2.avg_loss == 0.0


def test_daily_pnl_sorted_ascending_regardless_of_input_order():
    trades = [
        make_record("1", close_time=utc(2026, 7, 12, 9, 0), profit=1.0),
        make_record("2", close_time=utc(2026, 7, 10, 9, 0), profit=2.0),
        make_record("3", close_time=utc(2026, 7, 11, 9, 0), profit=3.0),
    ]

    results = compute_daily_pnl(trades)

    assert [r.date for r in results] == [
        date(2026, 7, 10),
        date(2026, 7, 11),
        date(2026, 7, 12),
    ]


def test_daily_pnl_empty_input_returns_empty_list():
    assert compute_daily_pnl([]) == []
