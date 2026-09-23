import dataclasses
import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backtest.api import routes as routes_module
from src.backtest.api.routes import _JobStatus, _resolve_strategy_name, router
from src.backtest.application.run_backtest import NoHistoryError
from src.backtest.domain.models import (
    ActivityLogEntry,
    BacktestReport,
    BacktestTrade,
    BacktestZone,
    EquityPoint,
)
from src.backtest.reports.writer import write_report
from src.market_data.domain.models import MarketDataUnavailable, Timeframe
from src.shared.db.base import Base
from src.strategies.adapters.repository import StrategyVersionRepository
from src.strategies.application.versioning import StrategyVersionService
from src.strategies.domain.versioning import CodeSource
from src.strategies.registry import StrategyRegistry

T0 = datetime(2025, 1, 1, tzinfo=UTC)
T1 = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)

# Two families that internally hardcode the SAME `spec.name` — the exact
# collision `id`-based bot discovery/selection must stay immune to (see
# `strategies/registry.py`'s module docstring).
_STRATEGY_CODE = """
from src.strategies.domain.models import Direction, MarketContext, Signal, StrategySpec


class Sample:
    def __init__(self):
        self.spec = StrategySpec(
            name="pob_price_action_snd", version=1, symbols=("XAUUSD",),
            entry_timeframe="M5", confirmation_timeframes=(), params={},
        )

    def evaluate(self, ctx: MarketContext):
        return None
"""


def make_report(profit: float = 96.8) -> BacktestReport:
    trade = BacktestTrade(
        side="buy",
        volume=0.04,
        open_time=T0,
        open_price=2410.125,
        sl=2399.125,
        tp=2434.325,
        close_time=T1,
        close_price=2434.325,
        profit=profit,
        r_multiple=2.2,
    )
    return BacktestReport(
        strategy="breakout_v1",
        symbol="XAUUSD",
        period="2025-01:2025-01",
        starting_balance=10_000.0,
        ending_balance=10_000.0 + profit,
        trades=(trade,),
        equity_curve=(
            EquityPoint(time=T0, balance=10_000.0),
            EquityPoint(time=T1, balance=10_000.0 + profit),
        ),
        win_rate=1.0,
        profit_factor=float("inf"),
        max_drawdown_pct=0.0,
        avg_r=2.2,
        worst_losing_streak=0,
    )


@pytest.fixture
async def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_module, "REPORTS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client, tmp_path


@pytest.fixture
async def api_with_strategies(tmp_path, monkeypatch):
    """Seeds two strategy families that share the exact same internal
    `spec.name` ("pob_price_action_snd") — the real-world collision that
    motivated keying bot discovery/selection off `StrategyVersion.id`
    instead of that self-declared name."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("TB_DATABASE_URL", db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    strategy_versions = StrategyVersionService(
        StrategyVersionRepository(session_factory), StrategyRegistry(), generated_dir
    )
    original = strategy_versions.save_generated_code(
        name="pob_price_action_snd", code=_STRATEGY_CODE, source=CodeSource.AI_GENERATED
    )
    strategy_versions.activate_version(original.id)
    duplicate = strategy_versions.duplicate_version(
        original.id, new_name="pob_price_action_snd  for boom 1000"
    )
    strategy_versions.activate_version(duplicate.id)

    monkeypatch.setattr(routes_module, "_STRATEGIES_GENERATED_DIR", generated_dir)
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client, db_url, original, duplicate


async def test_list_reports_empty_dir(api):
    client, _ = api
    response = await client.get("/backtest/reports")
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_list_reports_returns_summary(api):
    client, reports_dir = api
    write_report(make_report(), reports_dir)

    response = await client.get("/backtest/reports")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    (summary,) = body["items"]
    assert summary["strategy"] == "breakout_v1"
    assert summary["symbol"] == "XAUUSD"
    assert summary["trade_count"] == 1
    assert summary["profit_factor"] is None  # inf serialized as null


async def test_list_reports_paginates(api):
    client, reports_dir = api
    for symbol in ("XAUUSD", "BTCUSD", "EURUSD"):
        write_report(dataclasses.replace(make_report(), symbol=symbol), reports_dir)

    response = await client.get("/backtest/reports", params={"limit": 2, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2

    response = await client.get("/backtest/reports", params={"limit": 2, "offset": 2})
    body = response.json()
    assert len(body["items"]) == 1


async def test_get_report_returns_full_detail(api):
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    detail = response.json()
    assert len(detail["trades"]) == 1
    assert detail["trades"][0]["r_multiple"] == pytest.approx(2.2)
    assert detail["trades"][0]["open_time"] == int(T0.timestamp())
    assert len(detail["equity_curve"]) == 2


async def test_get_report_includes_zone_pattern_structure(api):
    client, reports_dir = api
    zone = BacktestZone(
        kind="demand",
        price_low=99.7,
        price_high=100.3,
        time_start=T0,
        time_end=T1,
        pattern="RBR",
    )
    trade = dataclasses.replace(
        make_report().trades[0],
        zone=zone,
        pattern="bullish_engulfing",
        structure=(("HH", 105.0, T0), ("LL", 95.0, T1)),
    )
    report = dataclasses.replace(make_report(), trades=(trade,))
    path = write_report(report, reports_dir)

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    out_trade = response.json()["trades"][0]

    assert out_trade["zone"] == {
        "kind": "demand",
        "price_low": 99.7,
        "price_high": 100.3,
        "time_start": int(T0.timestamp()),
        "time_end": int(T1.timestamp()),
        "pattern": "RBR",
    }
    assert out_trade["pattern"] == "bullish_engulfing"
    assert out_trade["structure"] == [
        {"label": "HH", "price": 105.0, "time": int(T0.timestamp())},
        {"label": "LL", "price": 95.0, "time": int(T1.timestamp())},
    ]


async def test_get_report_defaults_zone_fields_for_legacy_reports(api):
    """A report file written before this feature existed has no zone/pattern/
    structure keys at all — must still parse instead of 500ing."""
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    out_trade = response.json()["trades"][0]
    assert out_trade["zone"] is None
    assert out_trade["pattern"] is None
    assert out_trade["structure"] == []


async def test_get_report_includes_activity_log(api):
    client, reports_dir = api
    report = dataclasses.replace(
        make_report(),
        activity_log=(
            ActivityLogEntry(
                time=T0,
                level="INFO",
                logger="src.engine.application.trade_loop",
                message="ENTRY BLOCKED (HTF veto): XAUUSD buy — M15 trend (down) opposes buy",
            ),
        ),
    )
    path = write_report(report, reports_dir)

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    activity_log = response.json()["activity_log"]
    assert activity_log == [
        {
            "time": int(T0.timestamp()),
            "level": "INFO",
            "logger": "src.engine.application.trade_loop",
            "message": "ENTRY BLOCKED (HTF veto): XAUUSD buy — M15 trend (down) opposes buy",
        }
    ]


async def test_get_report_defaults_activity_log_for_legacy_reports(api):
    """A report file written before this feature existed has no activity_log
    key at all — must still parse instead of 500ing."""
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    assert response.json()["activity_log"] == []


async def test_get_report_defaults_min_rr_for_legacy_reports(api):
    """A report file written before min_rr was tracked has no such key at
    all (not even null) — must still parse and default to 1.0
    (SpreadGate.DEFAULT_MIN_RR), not 500."""
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)
    raw = json.loads(path.read_text())
    del raw["min_rr"]
    path.write_text(json.dumps(raw))

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    assert response.json()["min_rr"] == 1.0


async def test_get_report_defaults_risk_caps_for_legacy_reports(api):
    """A report file written before risk caps were tracked has none of those
    keys at all — must still parse with the RiskCaps dataclass's own
    defaults, not 500."""
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)
    raw = json.loads(path.read_text())
    for key in (
        "risk_per_trade_pct",
        "daily_loss_limit_pct",
        "max_open_positions",
        "max_trades_per_day_enabled",
        "consecutive_loss_pause",
        "min_lot_fallback_enabled",
        "max_risk_per_trade_pct",
    ):
        del raw[key]
    path.write_text(json.dumps(raw))

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 200
    body = response.json()
    assert body["risk_per_trade_pct"] == 0.5
    assert body["daily_loss_limit_pct"] == 2.0
    assert body["max_open_positions"] == 100
    assert body["max_trades_per_day_enabled"] is False
    assert body["consecutive_loss_pause"] == 10
    assert body["min_lot_fallback_enabled"] is False
    assert body["max_risk_per_trade_pct"] is None


async def test_get_report_unknown_id_404s(api):
    client, _ = api
    response = await client.get("/backtest/reports/does_not_exist")
    assert response.status_code == 404


async def test_get_report_rejects_path_traversal(api):
    client, _ = api
    response = await client.get("/backtest/reports/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


async def test_delete_report_removes_file(api):
    client, reports_dir = api
    path = write_report(make_report(), reports_dir)
    assert path.is_file()

    response = await client.delete(f"/backtest/reports/{path.stem}")
    assert response.status_code == 204
    assert not path.is_file()

    response = await client.get(f"/backtest/reports/{path.stem}")
    assert response.status_code == 404


async def test_delete_report_unknown_id_404s(api):
    client, _ = api
    response = await client.delete("/backtest/reports/does_not_exist")
    assert response.status_code == 404


async def test_delete_report_rejects_path_traversal(api):
    client, _ = api
    response = await client.delete("/backtest/reports/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


# ── POST /backtest/reports/import ─────────────────────────────────────────────


async def test_import_report_round_trips_a_downloaded_report(api):
    client, reports_dir = api
    zone = BacktestZone(kind="demand", price_low=99.7, price_high=100.3, time_start=T0, time_end=T1)
    trade = dataclasses.replace(
        make_report().trades[0],
        zone=zone,
        pattern="bullish_engulfing",
        structure=(("HH", 105.0, T0), ("LL", 95.0, T1)),
    )
    report = dataclasses.replace(make_report(), trades=(trade,))
    original_path = write_report(report, reports_dir)

    downloaded = await client.get(f"/backtest/reports/{original_path.stem}")
    assert downloaded.status_code == 200
    downloaded_json = downloaded.json()

    response = await client.post("/backtest/reports/import", json=downloaded_json)
    assert response.status_code == 201
    summary = response.json()
    assert summary["id"] != original_path.stem  # a fresh id, never overwrites
    assert summary["strategy"] == "breakout_v1"
    assert summary["symbol"] == "XAUUSD"
    assert summary["trade_count"] == 1
    assert summary["profit_factor"] is None  # inf round-trips as null both ways

    reimported = await client.get(f"/backtest/reports/{summary['id']}")
    assert reimported.status_code == 200
    assert reimported.json()["trades"] == downloaded_json["trades"]
    assert reimported.json()["equity_curve"] == downloaded_json["equity_curve"]

    listing = await client.get("/backtest/reports")
    assert listing.json()["total"] == 2


async def test_import_report_rejects_malformed_json(api):
    client, _reports_dir = api
    response = await client.post("/backtest/reports/import", json={"strategy": "x"})
    assert response.status_code == 422


async def test_list_bots_ids_stay_distinct_despite_shared_spec_name(api_with_strategies):
    client, _db_url, original, duplicate = api_with_strategies

    response = await client.get("/backtest/bots")
    assert response.status_code == 200
    bots = {b["id"]: b for b in response.json()}

    assert "breakout_v1" in bots
    assert bots["breakout_v1"]["name"] == "breakout_v1"

    # Two distinct families, two distinct ids — even though both versions'
    # generated code hardcodes the same `spec.name`.
    assert original.id in bots
    assert duplicate.id in bots
    assert bots[original.id]["name"] == "pob_price_action_snd"
    assert bots[duplicate.id]["name"] == "pob_price_action_snd  for boom 1000"


async def test_resolve_strategy_name_baseline_sentinel(api_with_strategies):
    _client, db_url, _original, _duplicate = api_with_strategies
    assert _resolve_strategy_name("breakout_v1", db_url) == "breakout_v1"


async def test_resolve_strategy_name_by_id(api_with_strategies):
    _client, db_url, original, duplicate = api_with_strategies
    assert _resolve_strategy_name(original.id, db_url) == "pob_price_action_snd"
    assert _resolve_strategy_name(duplicate.id, db_url) == "pob_price_action_snd  for boom 1000"


async def test_resolve_strategy_name_unknown_id_raises(api_with_strategies):
    _client, db_url, _original, _duplicate = api_with_strategies
    with pytest.raises(ValueError, match="unknown strategy id"):
        _resolve_strategy_name("does-not-exist", db_url)


# ── _run_job auto-backfill-and-retry ──────────────────────────────────────────


class _FakeCandleHistory:
    """Records the calls `_auto_backfill` makes. `fail_timeframes` lists which
    `backfill()` calls raise `MarketDataUnavailable` instead of succeeding
    (simulating the gateway's terminal rejecting specific timeframes, as seen
    for H4 on a synthetic index in practice); `fail_sync=True` does the same
    for `sync_symbol_spec`."""

    def __init__(
        self,
        fail_timeframes: set[Timeframe] | None = None,
        fail_sync: bool = False,
    ) -> None:
        self.synced_symbols: list[str] = []
        self.backfill_calls: list[tuple[str, Timeframe]] = []
        self._fail_timeframes = fail_timeframes or set()
        self._fail_sync = fail_sync

    async def sync_symbol_spec(self, symbol: str) -> None:
        if self._fail_sync:
            raise MarketDataUnavailable("gateway unreachable")
        self.synced_symbols.append(symbol)

    async def backfill(self, symbol, timeframe, count, start=None) -> int:
        if timeframe in self._fail_timeframes:
            raise MarketDataUnavailable(f"copy_rates_from_pos({symbol},{timeframe}) failed")
        self.backfill_calls.append((symbol, timeframe))
        return count


@pytest.fixture
def run_job_env(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_module, "REPORTS_DIR", tmp_path)
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("TB_DATABASE_URL", db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(routes_module, "_build_full_registry", lambda *a, **k: None)
    return db_url


async def test_run_job_auto_backfills_and_retries_after_no_history(run_job_env, monkeypatch):
    calls = {"n": 0}

    async def fake_run_backtest(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NoHistoryError("no M5 candle history for XAUUSD in 2025-01..2025-02")
        return make_report()

    monkeypatch.setattr(routes_module, "run_backtest", fake_run_backtest)
    candle_history = _FakeCandleHistory()
    job_id = "job-1"
    routes_module._jobs[job_id] = {"status": _JobStatus.PENDING, "report_id": None, "error": None}

    await routes_module._run_job(job_id, "breakout_v1", "XAUUSD", "2025-01:2025-01", candle_history)

    assert calls["n"] == 2
    assert candle_history.synced_symbols == ["XAUUSD"]
    assert {tf for _sym, tf in candle_history.backfill_calls} == set(Timeframe)
    job = routes_module._jobs[job_id]
    assert job["status"] == _JobStatus.DONE
    assert job["error"] is None


async def test_run_job_reports_error_when_second_no_history_after_backfill(
    run_job_env, monkeypatch
):
    async def always_no_history(*args, **kwargs):
        raise NoHistoryError("still nothing that far back")

    monkeypatch.setattr(routes_module, "run_backtest", always_no_history)
    candle_history = _FakeCandleHistory()
    job_id = "job-2"
    routes_module._jobs[job_id] = {"status": _JobStatus.PENDING, "report_id": None, "error": None}

    await routes_module._run_job(job_id, "breakout_v1", "XAUUSD", "2025-01:2025-01", candle_history)

    job = routes_module._jobs[job_id]
    assert job["status"] == _JobStatus.ERROR
    assert "still nothing" in job["error"]
    # Only one retry — the auto-backfill isn't attempted a second time.
    assert len(candle_history.backfill_calls) == len(Timeframe)


async def test_run_job_reports_gateway_unreachable_instead_of_retrying(run_job_env, monkeypatch):
    async def raise_no_history(*args, **kwargs):
        raise NoHistoryError("no history")

    monkeypatch.setattr(routes_module, "run_backtest", raise_no_history)
    # Every single gateway call fails (sync + all timeframes) — the gateway
    # itself is down, not just one flaky symbol/timeframe combination.
    candle_history = _FakeCandleHistory(fail_timeframes=set(Timeframe), fail_sync=True)
    job_id = "job-3"
    routes_module._jobs[job_id] = {"status": _JobStatus.PENDING, "report_id": None, "error": None}

    await routes_module._run_job(job_id, "breakout_v1", "XAUUSD", "2025-01:2025-01", candle_history)

    job = routes_module._jobs[job_id]
    assert job["status"] == _JobStatus.ERROR
    assert "gateway unreachable" in job["error"]


async def test_run_job_tolerates_one_bad_timeframe_during_auto_backfill(run_job_env, monkeypatch):
    """Regression test for a real failure: the broker's terminal rejected
    `copy_rates_from_pos` for H4 on a synthetic index ('Volatility 75 Index')
    while every other timeframe backfilled fine. That single flaky timeframe
    must not abort the whole auto-backfill — M5 (the only timeframe the
    retried backtest actually needs) succeeded, so the job should still
    complete rather than fail with a misleading "gateway unreachable"."""
    calls = {"n": 0}

    async def fake_run_backtest(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NoHistoryError("no M5 candle history")
        return make_report()

    monkeypatch.setattr(routes_module, "run_backtest", fake_run_backtest)
    candle_history = _FakeCandleHistory(fail_timeframes={Timeframe.H4})
    job_id = "job-5"
    routes_module._jobs[job_id] = {"status": _JobStatus.PENDING, "report_id": None, "error": None}

    await routes_module._run_job(
        job_id, "breakout_v1", "Volatility 75 Index", "2025-01:2025-01", candle_history
    )

    job = routes_module._jobs[job_id]
    assert job["status"] == _JobStatus.DONE
    assert job["error"] is None
    assert calls["n"] == 2
    assert Timeframe.M5 in {tf for _sym, tf in candle_history.backfill_calls}
    assert Timeframe.H4 not in {tf for _sym, tf in candle_history.backfill_calls}


async def test_run_job_skips_auto_backfill_when_history_is_already_complete(
    run_job_env, monkeypatch
):
    async def fake_run_backtest(*args, **kwargs):
        return make_report()

    monkeypatch.setattr(routes_module, "run_backtest", fake_run_backtest)
    candle_history = _FakeCandleHistory()
    job_id = "job-4"
    routes_module._jobs[job_id] = {"status": _JobStatus.PENDING, "report_id": None, "error": None}

    await routes_module._run_job(job_id, "breakout_v1", "XAUUSD", "2025-01:2025-01", candle_history)

    assert candle_history.backfill_calls == []
    assert candle_history.synced_symbols == []
    assert routes_module._jobs[job_id]["status"] == _JobStatus.DONE
