"""Covers `scripts/backfill_candle_enrichment.py`'s core logic: enriching
every stored symbol/timeframe for one account, leaving other accounts
untouched, and being a safe no-op on a second run.

Note: `enrich_missing`'s default `atr_period=14` means the leading 13 bars
of any symbol/timeframe's own history can never get a real trailing-context
ATR (there's nothing before them to compute a rolling window from) — that's
inherent to ATR, not a script bug, so these tests assert "every row that can
be enriched, is" rather than "literally zero NULL rows anywhere".
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from scripts.backfill_candle_enrichment import backfill_account, distinct_symbol_timeframes, main
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.orm import CandleRow
from src.market_data.domain.models import Candle, Timeframe
from src.shared.db.base import Base

ATR_PERIOD = 14


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def make_bars(
    n: int, *, symbol: str, timeframe: Timeframe, start: datetime
) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            time=start + i * timedelta(seconds=timeframe.seconds),
            open=100.0 + i * 0.1,
            high=100.0 + i * 0.1 + 1.0,
            low=100.0 + i * 0.1 - 1.0,
            close=100.0 + i * 0.1,
            tick_volume=10,
            spread_points=5,
        )
        for i in range(n)
    ]


@pytest.fixture
def engine_and_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/scratch.db")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def null_atr_count(session_factory, account_id: str | None = None) -> int:
    query = select(func.count()).select_from(CandleRow).where(CandleRow.atr_14.is_(None))
    if account_id is not None:
        query = query.where(CandleRow.account_id == account_id)
    with session_factory() as session:
        return session.execute(query).scalar_one()


def test_distinct_symbol_timeframes_returns_pairs_for_account(engine_and_session_factory):
    _, session_factory = engine_and_session_factory
    repository = CandleRepository(session_factory)
    repository.upsert_many(
        make_bars(20, symbol="XAUUSD", timeframe=Timeframe.M5, start=utc(2026, 8, 3, 0, 0)),
        account_id="default",
    )
    repository.upsert_many(
        make_bars(20, symbol="BTCUSD", timeframe=Timeframe.M15, start=utc(2026, 8, 3, 0, 0)),
        account_id="default",
    )
    repository.upsert_many(
        make_bars(20, symbol="XAUUSD", timeframe=Timeframe.M5, start=utc(2026, 8, 3, 0, 0)),
        account_id="other",
    )

    pairs = distinct_symbol_timeframes(session_factory, "default")

    assert set(pairs) == {("XAUUSD", Timeframe.M5), ("BTCUSD", Timeframe.M15)}


def test_backfill_account_enriches_every_enrichable_row_and_leaves_other_accounts_alone(
    engine_and_session_factory,
):
    _, session_factory = engine_and_session_factory
    repository = CandleRepository(session_factory)
    n = 40
    for symbol, timeframe in (("XAUUSD", Timeframe.M5), ("BTCUSD", Timeframe.M15)):
        repository.upsert_many(
            make_bars(n, symbol=symbol, timeframe=timeframe, start=utc(2026, 8, 3, 0, 0)),
            account_id="default",
        )
    repository.upsert_many(
        make_bars(n, symbol="XAUUSD", timeframe=Timeframe.M5, start=utc(2026, 8, 3, 0, 0)),
        account_id="other",
    )

    total = backfill_account(repository, session_factory, "default")

    # Every bar past the leading (period - 1) warm-up gets enriched, for
    # each of the two symbol/timeframe combinations under "default".
    assert total == (n - (ATR_PERIOD - 1)) * 2
    assert null_atr_count(session_factory, "default") == (ATR_PERIOD - 1) * 2
    # "other" was never touched.
    assert null_atr_count(session_factory, "other") == n


def test_backfill_account_second_run_is_a_no_op(engine_and_session_factory):
    _, session_factory = engine_and_session_factory
    repository = CandleRepository(session_factory)
    repository.upsert_many(
        make_bars(40, symbol="XAUUSD", timeframe=Timeframe.M5, start=utc(2026, 8, 3, 0, 0)),
        account_id="default",
    )

    first = backfill_account(repository, session_factory, "default")
    assert first > 0
    remaining_after_first = null_atr_count(session_factory, "default")

    second = backfill_account(repository, session_factory, "default")

    assert second == 0
    assert null_atr_count(session_factory, "default") == remaining_after_first


def test_main_runs_end_to_end_against_configured_database_url(
    engine_and_session_factory, monkeypatch
):
    engine, session_factory = engine_and_session_factory
    repository = CandleRepository(session_factory)
    repository.upsert_many(
        make_bars(40, symbol="XAUUSD", timeframe=Timeframe.M5, start=utc(2026, 8, 3, 0, 0)),
        account_id="default",
    )

    monkeypatch.setenv("TB_DATABASE_URL", str(engine.url))
    monkeypatch.setattr("sys.argv", ["backfill_candle_enrichment", "--account", "default"])

    exit_code = main()

    assert exit_code == 0
    assert null_atr_count(session_factory, "default") == ATR_PERIOD - 1

    # Re-running (e.g. a retried cron invocation) is safe and changes nothing.
    exit_code_again = main()
    assert exit_code_again == 0
    assert null_atr_count(session_factory, "default") == ATR_PERIOD - 1
