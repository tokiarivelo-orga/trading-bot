from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.journal.adapters.orm import TradeRow
from src.journal.adapters.repository import JournalRepository
from src.journal.domain.models import CandleSnapshot, TradeRecord
from src.shared.db.base import Base


@pytest.fixture
def repository(tmp_path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def make_record(id: str, symbol: str = "XAUUSD", open_time=None, **kw) -> TradeRecord:
    defaults = dict(
        id=id,
        symbol=symbol,
        side="buy",
        volume=0.1,
        open_price=2400.35,
        open_time=open_time or utc(2026, 7, 10, 14, 0),
        sl=2390.0,
        tp=2420.0,
        spread_points_at_entry=25,
        comment="",
    )
    return TradeRecord(**{**defaults, **kw})


def test_roundtrip_preserves_trade(repository):
    record = make_record("1")
    repository.save(record)
    assert repository.get("1") == record


def test_roundtrip_preserves_snapshots(repository):
    snapshot = (
        CandleSnapshot(
            time=utc(2026, 7, 10, 13, 55), open=1, high=2, low=0.5, close=1.5, tick_volume=100
        ),
    )
    record = make_record("1", m5_entry_snapshot=snapshot, h1_entry_snapshot=snapshot)
    repository.save(record)
    stored = repository.get("1")
    assert stored.m5_entry_snapshot == snapshot
    assert stored.h1_entry_snapshot == snapshot


def test_roundtrip_preserves_decision_context(repository):
    record = make_record(
        "1",
        reason="RBR base retest + M15 bullish engulf",
        confidence=0.82,
        zone_kind="demand",
        zone_price_low=2395.0,
        zone_price_high=2398.5,
        zone_time_start=utc(2026, 7, 10, 10, 0),
        zone_time_end=utc(2026, 7, 10, 13, 45),
        zone_pattern="RBR",
        pattern="bullish_engulfing",
        structure=(
            ("HL", 2397.2, utc(2026, 7, 10, 13, 30)),
            ("HH", 2401.0, utc(2026, 7, 10, 13, 45)),
        ),
        indicators=(
            ("RSI", 62.3, 50.0, ">", True),
            ("EMA_FAST_VS_SLOW", 2401.1, 2398.4, ">", True),
            ("VOLUME", 120.0, 95.5, ">", True),
        ),
    )
    repository.save(record)
    assert repository.get("1") == record


def test_roundtrip_preserves_regime_and_transaction_cost(repository):
    """Regime tagging + transaction cost (OBSERVABILITY_PLAN.md Phase 6)."""
    record = make_record(
        "1",
        regime_volatility="high",
        regime_volatility_percentile=82.5,
        regime_trend="trending",
        regime_adx=27.3,
        regime_session="london",
        transaction_cost=1.85,
    )
    repository.save(record)
    assert repository.get("1") == record


def test_regime_fields_default_to_none(repository):
    """Trades journaled before Phase 6 (or via `make_record`'s defaults)
    carry no regime tag — must round-trip as None, not a fabricated bucket."""
    repository.save(make_record("1"))
    stored = repository.get("1")
    assert stored.regime_volatility is None
    assert stored.regime_volatility_percentile is None
    assert stored.regime_trend is None
    assert stored.regime_adx is None
    assert stored.regime_session is None
    assert stored.transaction_cost is None


def test_get_returns_none_for_unknown_id(repository):
    assert repository.get("missing") is None


def test_get_by_id_returns_matching_record(repository):
    repository.save(make_record("1"))
    assert repository.get_by_id("1") == make_record("1")


def test_get_by_id_returns_none_for_unknown_id(repository):
    assert repository.get_by_id("missing") is None


def test_get_by_id_scopes_to_account(repository):
    repository.save(make_record("1"), account_id="ftmo-1")

    assert repository.get_by_id("1", account_id="ftmo-1") is not None
    assert repository.get_by_id("1", account_id="ftmo-2") is None
    assert repository.get_by_id("1") is None  # default account


def test_save_upserts_same_id(repository):
    repository.save(make_record("1"))
    repository.save(make_record("1", close_price=2410.0, close_time=utc(2026, 7, 10, 15, 0)))

    stored = repository.get("1")
    assert stored.close_price == 2410.0
    assert stored.is_open is False


def test_get_last_n_orders_by_open_time_desc(repository):
    for i, minute in enumerate((0, 5, 10)):
        repository.save(make_record(str(i), open_time=utc(2026, 7, 10, 14, minute)))

    last_two = repository.get_last_n("XAUUSD", 2)
    assert [r.id for r in last_two] == ["2", "1"]


def test_get_markers_filters_by_time_range(repository):
    for i, minute in enumerate((0, 5, 10)):
        repository.save(make_record(str(i), open_time=utc(2026, 7, 10, 14, minute)))

    frm = int(utc(2026, 7, 10, 14, 5).timestamp())
    markers = repository.get_markers("XAUUSD", frm=frm)
    assert [r.id for r in markers] == ["1", "2"]


def test_get_markers_filters_by_skill(repository):
    repository.save(make_record("1", skill="normal/xauusd/breakout_v1"))
    repository.save(make_record("2", skill="normal/xauusd/mean_reversion"))
    repository.save(make_record("3", skill=None))

    markers = repository.get_markers("XAUUSD", skill="normal/xauusd/breakout_v1")

    assert [r.id for r in markers] == ["1"]


def test_count_closed_only_counts_closed_trades(repository):
    repository.save(make_record("1"))
    repository.save(
        make_record("2", close_price=2410.0, close_time=utc(2026, 7, 10, 15, 0), profit=9.65)
    )
    assert repository.count_closed("XAUUSD") == 1


def test_get_last_n_closed_orders_by_close_time_desc(repository):
    repository.save(
        make_record("1", close_price=2410.0, close_time=utc(2026, 7, 10, 15, 0), profit=9.65)
    )
    repository.save(
        make_record("2", close_price=2405.0, close_time=utc(2026, 7, 10, 16, 0), profit=4.65)
    )
    repository.save(make_record("3"))  # still open, excluded

    closed = repository.get_last_n_closed("XAUUSD", 10)
    assert [r.id for r in closed] == ["2", "1"]


def _seed_history(repository) -> None:
    repository.save(
        make_record(
            "1",
            symbol="XAUUSD",
            side="buy",
            strategy_version="breakout_v1:v1",
            skill="normal/xauusd",
            open_time=utc(2026, 7, 10, 14, 0),
            close_price=2410.0,
            close_time=utc(2026, 7, 10, 15, 0),
            profit=9.65,
        )
    )
    repository.save(
        make_record(
            "2",
            symbol="XAUUSD",
            side="sell",
            strategy_version="breakout_v1:v1",
            skill="news/xauusd",
            open_time=utc(2026, 7, 10, 16, 0),
            close_price=2415.0,
            close_time=utc(2026, 7, 10, 17, 0),
            profit=-4.20,
        )
    )
    repository.save(
        make_record(
            "3",
            symbol="EURUSD",
            side="buy",
            strategy_version="meanrev_v2:v1",
            skill=None,
            open_time=utc(2026, 7, 10, 18, 0),
            close_price=1.1000,
            close_time=utc(2026, 7, 10, 18, 30),
            profit=0.0,
        )
    )
    repository.save(
        make_record(
            "4",
            symbol="EURUSD",
            side="buy",
            open_time=utc(2026, 7, 10, 19, 0),
        )
    )  # still open


def test_search_with_no_filters_returns_everything_and_total(repository):
    _seed_history(repository)
    items, total, total_profit = repository.search()
    assert total == 4
    assert total_profit == pytest.approx(5.45)
    assert [r.id for r in items] == ["4", "3", "2", "1"]  # open_time desc


def test_search_filters_by_symbol(repository):
    _seed_history(repository)
    items, total, _ = repository.search(symbol="EURUSD")
    assert total == 2
    assert {r.id for r in items} == {"3", "4"}


def test_search_filters_by_side_and_strategy_version(repository):
    _seed_history(repository)
    items, total, _ = repository.search(side="sell", strategy_version="breakout_v1:v1")
    assert total == 1
    assert items[0].id == "2"


def test_search_filters_by_skill(repository):
    _seed_history(repository)
    items, total, _ = repository.search(skill="news/xauusd")
    assert total == 1
    assert items[0].id == "2"


def test_search_filters_by_strategy_version_substring(repository):
    _seed_history(repository)
    items, total, _ = repository.search(strategy_version="breakout")
    assert total == 2
    assert {r.id for r in items} == {"1", "2"}

    items_rev, total_rev, _ = repository.search(strategy_version="meanrev")
    assert total_rev == 1
    assert items_rev[0].id == "3"


def test_search_filters_by_skill_substring(repository):
    _seed_history(repository)
    items, total, _ = repository.search(skill="xauusd")
    assert total == 2
    assert {r.id for r in items} == {"1", "2"}

    items_norm, total_norm, _ = repository.search(skill="normal")
    assert total_norm == 1
    assert items_norm[0].id == "1"


def test_search_outcome_win_loss_breakeven_open(repository):
    _seed_history(repository)
    assert [r.id for r in repository.search(outcome="win")[0]] == ["1"]
    assert [r.id for r in repository.search(outcome="loss")[0]] == ["2"]
    assert [r.id for r in repository.search(outcome="breakeven")[0]] == ["3"]
    assert [r.id for r in repository.search(outcome="open")[0]] == ["4"]


def test_search_filters_by_open_time_range(repository):
    _seed_history(repository)
    frm = int(utc(2026, 7, 10, 16, 0).timestamp())
    to = int(utc(2026, 7, 10, 18, 0).timestamp())
    items, total, _ = repository.search(open_from=frm, open_to=to)
    assert total == 2
    assert {r.id for r in items} == {"2", "3"}


def test_search_filters_by_close_time_range(repository):
    _seed_history(repository)
    frm = int(utc(2026, 7, 10, 15, 0).timestamp())
    to = int(utc(2026, 7, 10, 17, 0).timestamp())
    items, total, _ = repository.search(close_from=frm, close_to=to)
    assert total == 2
    assert {r.id for r in items} == {"1", "2"}


def test_search_orders_by_profit_asc(repository):
    _seed_history(repository)
    items, total, _ = repository.search(outcome=None, order_by="profit", order_dir="asc")
    # open trade has profit=None, sorts first under ascending NULLS-first (sqlite default)
    assert total == 4
    assert items[-1].id == "1"  # highest profit (9.65) last when ascending


def test_search_paginates_with_limit_and_offset(repository):
    _seed_history(repository)
    page1, total, _ = repository.search(limit=2, offset=0)
    page2, _, _ = repository.search(limit=2, offset=2)
    assert total == 4
    assert [r.id for r in page1] == ["4", "3"]
    assert [r.id for r in page2] == ["2", "1"]


def test_save_defaults_to_default_account(repository):
    repository.save(make_record("1"))
    assert repository.get_last_n("XAUUSD", 10, account_id="default") == [make_record("1")]


def test_queries_do_not_leak_across_accounts(repository):
    repository.save(make_record("1"), account_id="ftmo-1")
    repository.save(make_record("2"), account_id="ftmo-2")

    assert [r.id for r in repository.get_last_n("XAUUSD", 10, account_id="ftmo-1")] == ["1"]
    assert [r.id for r in repository.get_last_n("XAUUSD", 10, account_id="ftmo-2")] == ["2"]
    assert repository.get_last_n("XAUUSD", 10, account_id="default") == []


def test_search_scopes_to_account_id(repository):
    repository.save(make_record("1"), account_id="ftmo-1")
    repository.save(make_record("2"), account_id="ftmo-2")

    items, total, _ = repository.search(account_id="ftmo-1")
    assert total == 1
    assert items[0].id == "1"


def test_get_open_and_get_last_n_closed_and_count_closed_scope_to_account(repository):
    repository.save(make_record("1"), account_id="ftmo-1")
    repository.save(
        make_record("2", close_price=2410.0, close_time=utc(2026, 7, 10, 15, 0), profit=1.0),
        account_id="ftmo-1",
    )
    repository.save(make_record("3"), account_id="ftmo-2")

    assert [r.id for r in repository.get_open(account_id="ftmo-1")] == ["1"]
    assert [r.id for r in repository.get_open(account_id="ftmo-2")] == ["3"]
    assert [r.id for r in repository.get_last_n_closed("XAUUSD", 10, account_id="ftmo-1")] == ["2"]
    assert repository.get_last_n_closed("XAUUSD", 10, account_id="ftmo-2") == []
    assert repository.count_closed("XAUUSD", account_id="ftmo-1") == 1
    assert repository.count_closed("XAUUSD", account_id="ftmo-2") == 0


def test_get_markers_scopes_to_account(repository):
    repository.save(make_record("1"), account_id="ftmo-1")
    repository.save(make_record("2"), account_id="ftmo-2")

    assert [r.id for r in repository.get_markers("XAUUSD", account_id="ftmo-1")] == ["1"]


def test_get_all_for_analytics_matches_get_all_core_fields(repository):
    """The slim analytics query (`get_all_for_analytics`) must agree with the
    full `get_all` fetch on every field analytics.py reads, even though it
    doesn't fetch the JSON snapshot/structure columns at all."""
    snapshot = (
        CandleSnapshot(
            time=utc(2026, 7, 10, 13, 55), open=1, high=2, low=0.5, close=1.5, tick_volume=100
        ),
    )
    repository.save(
        make_record(
            "1",
            skill="normal/xauusd/breakout_v1",
            strategy_version="breakout_v1:v1",
            volume=0.2,
            close_price=2410.0,
            close_time=utc(2026, 7, 10, 15, 0),
            profit=9.65,
            m5_entry_snapshot=snapshot,
            h1_entry_snapshot=snapshot,
            structure=(("HL", 2397.2, utc(2026, 7, 10, 13, 30)),),
            regime_volatility="high",
            regime_trend="trending",
            regime_session="london",
            transaction_cost=1.85,
        )
    )
    repository.save(make_record("2", symbol="EURUSD"))  # still open

    full = {r.id: r for r in repository.get_all()}
    slim = {r.id: r for r in repository.get_all_for_analytics()}

    assert slim.keys() == full.keys()
    for trade_id, slim_record in slim.items():
        full_record = full[trade_id]
        assert slim_record.symbol == full_record.symbol
        assert slim_record.volume == full_record.volume
        assert slim_record.open_time == full_record.open_time
        assert slim_record.close_time == full_record.close_time
        assert slim_record.profit == full_record.profit
        assert slim_record.skill == full_record.skill
        assert slim_record.strategy_version == full_record.strategy_version
        assert slim_record.is_open == full_record.is_open
        assert slim_record.regime_volatility == full_record.regime_volatility
        assert slim_record.regime_trend == full_record.regime_trend
        assert slim_record.regime_session == full_record.regime_session
        assert slim_record.transaction_cost == full_record.transaction_cost


def test_get_all_for_analytics_omits_json_snapshot_and_structure_fields(repository):
    """Structural proof the slim query drops the four JSON columns
    analytics.py never reads — `TradeAnalyticsRecord` has no such attributes
    at all, unlike the `TradeRecord` rows `get_all` returns."""
    repository.save(make_record("1"))

    slim_record = repository.get_all_for_analytics()[0]

    for attr in (
        "m5_entry_snapshot",
        "h1_entry_snapshot",
        "m5_exit_snapshot",
        "h1_exit_snapshot",
        "structure",
    ):
        assert not hasattr(slim_record, attr)


def test_get_all_for_analytics_scopes_to_account(repository):
    repository.save(make_record("1"), account_id="ftmo-1")
    repository.save(make_record("2"), account_id="ftmo-2")

    assert [r.id for r in repository.get_all_for_analytics(account_id="ftmo-1")] == ["1"]
    assert [r.id for r in repository.get_all_for_analytics(account_id="ftmo-2")] == ["2"]
    assert repository.get_all_for_analytics(account_id="default") == []


def test_get_all_for_analytics_filters_by_open_time(repository):
    t2 = int(utc(2026, 7, 11, 10, 0).timestamp())
    repository.save(make_record("1", open_time=utc(2026, 7, 10, 10, 0)))
    repository.save(make_record("2", open_time=utc(2026, 7, 11, 10, 0)))
    repository.save(make_record("3", open_time=utc(2026, 7, 12, 10, 0)))

    assert {r.id for r in repository.get_all_for_analytics(open_from=t2)} == {"2", "3"}
    assert {r.id for r in repository.get_all_for_analytics(open_to=t2)} == {"1", "2"}
    assert {r.id for r in repository.get_all_for_analytics(open_from=t2, open_to=t2)} == {"2"}


def test_trade_row_declares_composite_account_symbol_close_index():
    """Every hot query (`get_last_n`, `get_markers`, `get_open`, `count_closed`,
    `search`) filters on account_id AND symbol together; without a composite
    index SQLite can only use one of the single-column indexes and
    row-filters the rest. Guards against the index declaration being lost or
    its column order changed."""
    index = next(
        (idx for idx in TradeRow.__table__.indexes if idx.name == "ix_trades_account_symbol_close"),
        None,
    )
    assert index is not None
    assert [col.name for col in index.columns] == ["account_id", "symbol", "close_time"]


def test_composite_index_is_created_in_sqlite(tmp_path):
    """Proves the ORM `Index(...)` declaration actually materializes as a
    real SQLite index (not just a Python-side artifact) when the schema is
    created — same `create_all` path the `repository` fixture above uses."""
    engine = create_engine(f"sqlite:///{tmp_path}/index_check.db")
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        index_names = {
            row[1] for row in conn.exec_driver_sql("PRAGMA index_list('trades')").fetchall()
        }
        assert "ix_trades_account_symbol_close" in index_names
        columns = [
            row[2]
            for row in conn.exec_driver_sql(
                "PRAGMA index_info('ix_trades_account_symbol_close')"
            ).fetchall()
        ]
        assert columns == ["account_id", "symbol", "close_time"]
