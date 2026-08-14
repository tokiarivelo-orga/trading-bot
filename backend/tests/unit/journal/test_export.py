"""GET /journal/export/dataset — SMC v2 training-dataset export.

Covers the Phase 2 rewrite of `journal/api/export.py`: v2 features present in
the output, the corrected `profit_r` formula, new enrichment fields (regime
tags, transaction cost, candle real_volume/atr_14/day_of_week) present when
the underlying data has them and null (not fabricated) when it doesn't, the
order-book join present only when a snapshot actually exists for a trade's
`signal_id`, and both CSV and JSON formats working end-to-end.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.journal.adapters.repository import JournalRepository
from src.journal.api.routes import router
from src.journal.application.trade_journal import TradeJournalService
from src.journal.domain.models import MarketSnapshot, TradeRecord
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.replay import SymbolSpec
from src.market_data.domain.models import Candle, Timeframe
from src.order_book.domain.models import BookLevel, BookSide, OrderBookSnapshot
from src.shared.db.base import Base
from src.shared.events.bus import EventBus

M5_SECONDS = 300
ORIGIN = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
N_CANDLES = 200


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def build_m5_candles(n: int = N_CANDLES) -> list[Candle]:
    """Deterministic, non-degenerate OHLCV series (sine wave + drift) — real
    enough that ATR/EMA/etc. are always well-defined (no exact-zero true
    range), so the v2 feature set doesn't NaN out rows past its warmup."""
    candles = []
    for i in range(n):
        drift = i * 0.01
        wave = 3.0 * math.sin(i / 7.0)
        close = 2000.0 + drift + wave
        open_ = close - 0.3 * math.cos(i / 5.0)
        high = max(open_, close) + 0.8
        low = min(open_, close) - 0.8
        candles.append(
            Candle(
                symbol="XAUUSD",
                timeframe=Timeframe.M5,
                time=ORIGIN + timedelta(seconds=i * M5_SECONDS),
                open=open_,
                high=high,
                low=low,
                close=close,
                tick_volume=100 + (i % 5),
                spread_points=25,
                real_volume=50 + (i % 7),
            )
        )
    return candles


def make_record(id: str, **kw) -> TradeRecord:
    defaults = dict(
        symbol="XAUUSD",
        side="buy",
        volume=1.0,
        open_price=2000.0,
        open_time=ORIGIN + timedelta(seconds=180 * M5_SECONDS),
        close_time=ORIGIN + timedelta(seconds=185 * M5_SECONDS),
        close_price=2005.0,
        sl=1990.0,
        tp=2030.0,
        spread_points_at_entry=25,
        profit=250.0,
        reason="DL Buy (tp=0.85, bull=0.58, bear=0.28)",
    )
    return TradeRecord(id=id, **{**defaults, **kw})


class FakeMarketContext:
    async def capture(self, symbol):
        return MarketSnapshot()


class FakeSymbolSpecRepository:
    """Stands in for the real `SymbolSpecRepository` — same sync `.get`
    signature, no DB needed for this test's single symbol."""

    def __init__(self, spec: SymbolSpec | None) -> None:
        self._spec = spec

    def get(self, symbol: str, account_id: str = "default") -> SymbolSpec | None:
        return self._spec


class FakeOrderBookCapture:
    """Stands in for `OrderBookCaptureService` — same async
    `get_for_signal` the real service exposes over
    `OrderBookSnapshotRepository.get_for_signal`."""

    def __init__(self, snapshots: dict[str, OrderBookSnapshot]) -> None:
        self._snapshots = snapshots
        self.calls: list[str] = []

    async def get_for_signal(self, signal_id: str) -> OrderBookSnapshot | None:
        self.calls.append(signal_id)
        return self._snapshots.get(signal_id)


@pytest.fixture
def candle_repository(tmp_path) -> CandleRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/candles.db")
    Base.metadata.create_all(engine)
    repo = CandleRepository(sessionmaker(bind=engine, expire_on_commit=False))
    repo.upsert_many(build_m5_candles())
    repo.enrich_missing("XAUUSD", Timeframe.M5)
    return repo


@pytest.fixture
def journal_repository(tmp_path) -> JournalRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/journal.db")
    Base.metadata.create_all(engine)
    return JournalRepository(sessionmaker(bind=engine, expire_on_commit=False))


SNAPSHOT_SIG_1 = OrderBookSnapshot(
    symbol="XAUUSD",
    time=ORIGIN + timedelta(seconds=180 * M5_SECONDS),
    levels=(BookLevel(side=BookSide.BID, price=1999.5, volume=10.0),),
)


@pytest.fixture
async def api(candle_repository, journal_repository):
    trade_journal = TradeJournalService(
        repository=journal_repository, market_context=FakeMarketContext(), event_bus=EventBus()
    )
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        accounts={
            "default": SimpleNamespace(
                id="default",
                trade_journal=trade_journal,
                candle_repository=candle_repository,
                symbol_spec_repository=FakeSymbolSpecRepository(
                    SymbolSpec(
                        point=0.01,
                        digits=2,
                        stops_level=100,
                        contract_size=100.0,
                        volume_min=0.01,
                        volume_max=100.0,
                        volume_step=0.01,
                    )
                ),
                order_book_capture=FakeOrderBookCapture({"sig-1": SNAPSHOT_SIG_1}),
            )
        }
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client


async def _get_json(client, **params):
    params.setdefault("symbol", "XAUUSD")
    params.setdefault("format", "json")
    response = await client.get("/accounts/default/journal/export/dataset", params=params)
    assert response.status_code == 200, response.text
    return response.json()


async def test_returns_404_when_no_closed_trades(api):
    response = await api.get(
        "/accounts/default/journal/export/dataset", params={"symbol": "XAUUSD"}
    )
    assert response.status_code == 404


async def test_v2_features_present_in_json_output(api, journal_repository):
    journal_repository.save(make_record("t-full", signal_id="sig-1"))

    rows = await _get_json(api)
    assert len(rows) == 1
    features = rows[0]["features"]
    # Time-of-day features are pure trig — always defined, regardless of the
    # synthetic price series — so they're a robust spot-check that the v2
    # feature module (63 features, smc_dl_features_v2.FEATURE_NAMES) ran.
    for name in ("sin_hour", "cos_hour", "sess_asian", "close_dir"):
        assert name in features, f"expected v2 feature {name!r} in output"
    # Most of the 63 features should have resolved to a real (non-NaN) value
    # this far past the warmup window.
    assert len(features) >= 50


async def test_profit_r_matches_hand_computed_formula(api, journal_repository):
    # profit=250, |open_price - sl|=10, volume=1.0, contract_size=100
    # -> initial_risk = 10 * 1.0 * 100 = 1000 -> profit_r = 250 / 1000 = 0.25
    journal_repository.save(make_record("t-full", signal_id="sig-1"))

    rows = await _get_json(api)
    assert rows[0]["profit_r"] == pytest.approx(0.25)


async def test_enrichment_fields_present_when_available(api, journal_repository):
    journal_repository.save(
        make_record(
            "t-full",
            signal_id="sig-1",
            regime_volatility="normal",
            regime_volatility_percentile=42.0,
            regime_trend="trending",
            regime_adx=28.5,
            regime_session="london",
            transaction_cost=1.23,
        )
    )

    row = (await _get_json(api))[0]
    assert row["regime_volatility"] == "normal"
    assert row["regime_volatility_percentile"] == 42.0
    assert row["regime_trend"] == "trending"
    assert row["regime_adx"] == 28.5
    assert row["regime_session"] == "london"
    assert row["transaction_cost"] == 1.23
    assert row["signal_id"] == "sig-1"
    # Candle-level enrichment: the entry M5 candle's own stored columns.
    assert row["real_volume"] is not None
    assert row["atr_14"] is not None
    assert row["day_of_week"] is not None


async def test_enrichment_fields_null_when_not_captured(api, journal_repository):
    journal_repository.save(make_record("t-minimal", signal_id=None, sl=None))

    row = (await _get_json(api))[0]
    assert row["signal_id"] is None
    assert row["regime_volatility"] is None
    assert row["regime_trend"] is None
    assert row["regime_session"] is None
    assert row["transaction_cost"] is None
    # No `sl` -> risk undefined -> profit_r must be null, not a guessed 0.
    assert row["profit_r"] is None
    # No signal_id at all -> the order-book join is never even attempted.
    assert row["order_book_levels"] is None


async def test_order_book_join_present_when_snapshot_exists(api, journal_repository):
    journal_repository.save(make_record("t-full", signal_id="sig-1"))

    row = (await _get_json(api))[0]
    assert row["order_book_levels"] == [{"side": "bid", "price": 1999.5, "volume": 10.0}]


async def test_order_book_join_absent_when_no_snapshot_captured(api, journal_repository):
    # Has a signal_id, but no snapshot was ever saved for it (the normal
    # case for a symbol/broker that reports no depth) — must stay null, not
    # an empty-but-present list.
    journal_repository.save(make_record("t-no-depth", signal_id="sig-unknown"))

    row = (await _get_json(api))[0]
    assert row["order_book_levels"] is None


async def test_csv_format_works_end_to_end(api, journal_repository):
    journal_repository.save(make_record("t-full", signal_id="sig-1"))

    response = await api.get(
        "/accounts/default/journal/export/dataset",
        params={"symbol": "XAUUSD", "format": "csv"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t-full"
    assert "sin_hour" in reader.fieldnames
    assert float(rows[0]["profit_r"]) == pytest.approx(0.25)


async def test_json_format_works_end_to_end(api, journal_repository):
    journal_repository.save(make_record("t-full", signal_id="sig-1"))

    rows = await _get_json(api)
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "t-full"
    assert rows[0]["symbol"] == "XAUUSD"
