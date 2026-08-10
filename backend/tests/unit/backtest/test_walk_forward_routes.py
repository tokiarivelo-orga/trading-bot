"""API tests for the `/backtest/walk-forward/*` endpoints (OBSERVABILITY_PLAN.md
Phase 6 Pass B). Mirrors `test_api_routes.py`'s pattern exactly: mock the
harness's own top-level function (`run_walk_forward`, not `run_backtest`) so
these run in milliseconds regardless of fold count — never a real backtest."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine

from src.backtest.api import routes as routes_module
from src.backtest.api.routes import _JobStatus, router
from src.backtest.application.walk_forward import FoldResult, WalkForwardReport
from src.backtest.reports.walk_forward_writer import write_walk_forward_report
from src.market_data.domain.models import MarketDataUnavailable, Timeframe
from src.shared.db.base import Base


def make_wf_report(
    strategy: str = "breakout_v1",
    symbol: str = "XAUUSD",
    period: str = "2025-01:2025-02",
    fold_count: int = 2,
) -> WalkForwardReport:
    folds = tuple(
        FoldResult(
            period=f"2025-{i + 1:02d}:2025-{i + 1:02d}",
            trade_count=3,
            win_rate=0.66,
            profit_factor=1.8,
            expectancy=4.0,
            broker_acceptance_rate=0.95,
        )
        for i in range(fold_count)
    )
    return WalkForwardReport(
        strategy=strategy,
        symbol=symbol,
        period=period,
        fold_months=1,
        starting_balance=10_000.0,
        folds=folds,
        fold_count=fold_count,
        folds_with_zero_trades=0,
        losing_fold_count=0,
        mean_profit_factor=1.8,
        stddev_profit_factor=0.0,
        worst_fold_profit_factor=1.8,
        mean_expectancy=4.0,
    )


# ── Report list/get/delete ────────────────────────────────────────────────────


@pytest.fixture
async def api(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_module, "WALK_FORWARD_REPORTS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client, tmp_path


async def test_list_wf_reports_empty_dir(api):
    client, _ = api
    response = await client.get("/backtest/walk-forward/reports")
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_list_wf_reports_returns_summary(api):
    client, reports_dir = api
    write_walk_forward_report(make_wf_report(), reports_dir)

    response = await client.get("/backtest/walk-forward/reports")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    (summary,) = body["items"]
    assert summary["strategy"] == "breakout_v1"
    assert summary["symbol"] == "XAUUSD"
    assert summary["fold_count"] == 2
    assert summary["mean_profit_factor"] == 1.8
    assert "folds" not in summary  # list view is the aggregate summary only


async def test_list_wf_reports_paginates(api):
    client, reports_dir = api
    for symbol in ("XAUUSD", "BTCUSD", "EURUSD"):
        write_walk_forward_report(make_wf_report(symbol=symbol), reports_dir)

    response = await client.get("/backtest/walk-forward/reports", params={"limit": 2, "offset": 0})
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2

    response = await client.get("/backtest/walk-forward/reports", params={"limit": 2, "offset": 2})
    assert len(response.json()["items"]) == 1


async def test_get_wf_report_returns_full_detail_with_folds(api):
    client, reports_dir = api
    path = write_walk_forward_report(make_wf_report(), reports_dir)

    response = await client.get(f"/backtest/walk-forward/reports/{path.stem}")
    assert response.status_code == 200
    detail = response.json()
    assert len(detail["folds"]) == 2
    assert detail["folds"][0]["period"] == "2025-01:2025-01"
    assert detail["folds"][0]["profit_factor"] == 1.8


async def test_get_wf_report_with_null_profit_factor_folds(api):
    client, reports_dir = api
    report = make_wf_report(fold_count=1)
    zero_trade_fold = dataclasses.replace(
        report.folds[0], trade_count=0, profit_factor=None, expectancy=0.0
    )
    report = dataclasses.replace(
        report, folds=(zero_trade_fold,), mean_profit_factor=None, worst_fold_profit_factor=None
    )
    path = write_walk_forward_report(report, reports_dir)

    response = await client.get(f"/backtest/walk-forward/reports/{path.stem}")
    assert response.status_code == 200
    body = response.json()
    assert body["folds"][0]["profit_factor"] is None
    assert body["mean_profit_factor"] is None


async def test_get_wf_report_unknown_id_404s(api):
    client, _ = api
    response = await client.get("/backtest/walk-forward/reports/does_not_exist")
    assert response.status_code == 404


async def test_get_wf_report_rejects_path_traversal(api):
    client, _ = api
    response = await client.get("/backtest/walk-forward/reports/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


async def test_delete_wf_report_removes_file(api):
    client, reports_dir = api
    path = write_walk_forward_report(make_wf_report(), reports_dir)
    assert path.is_file()

    response = await client.delete(f"/backtest/walk-forward/reports/{path.stem}")
    assert response.status_code == 204
    assert not path.is_file()

    response = await client.get(f"/backtest/walk-forward/reports/{path.stem}")
    assert response.status_code == 404


async def test_delete_wf_report_unknown_id_404s(api):
    client, _ = api
    response = await client.delete("/backtest/walk-forward/reports/does_not_exist")
    assert response.status_code == 404


async def test_delete_wf_report_rejects_path_traversal(api):
    client, _ = api
    response = await client.delete("/backtest/walk-forward/reports/..%2F..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404


# ── POST /backtest/walk-forward/run + GET .../run/{job_id} ───────────────────


@pytest.fixture
async def api_with_container(tmp_path, monkeypatch):
    """Same as `api`, but with `app.state.container.candle_history` set —
    `POST /backtest/walk-forward/run` reads it (mirrors how
    `POST /backtest/run` does), unlike the plain report list/get/delete
    endpoints, which don't need a container at all."""
    monkeypatch.setattr(routes_module, "WALK_FORWARD_REPORTS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(router)

    class _Container:
        candle_history = object()  # never actually called: run_walk_forward is mocked

    app.state.container = _Container()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client


async def test_start_walk_forward_returns_pending_job(api_with_container, monkeypatch):
    client = api_with_container

    async def fake_run_walk_forward(*args, **kwargs):
        return make_wf_report()

    monkeypatch.setattr(routes_module, "run_walk_forward", fake_run_walk_forward)
    monkeypatch.setattr(routes_module, "_build_full_registry", lambda *a, **k: None)

    response = await client.post(
        "/backtest/walk-forward/run",
        json={"strategy_id": "breakout_v1", "symbol": "XAUUSD", "period": "2025-01:2025-02"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["job_id"]


async def test_start_walk_forward_rejects_fold_months_below_one(api_with_container):
    client = api_with_container
    response = await client.post(
        "/backtest/walk-forward/run",
        json={
            "strategy_id": "breakout_v1",
            "symbol": "XAUUSD",
            "period": "2025-01:2025-02",
            "fold_months": 0,
        },
    )
    assert response.status_code == 422


async def test_get_wf_job_status_unknown_id_404s(api):
    client, _ = api
    response = await client.get("/backtest/walk-forward/run/does-not-exist")
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("status", "extra", "expected"),
    [
        (_JobStatus.PENDING, {}, {"status": "pending", "report_id": None, "error": None}),
        (_JobStatus.RUNNING, {}, {"status": "running", "report_id": None, "error": None}),
        (
            _JobStatus.DONE,
            {"report_id": "breakout_v1_XAUUSD_2025-01_2025-02"},
            {
                "status": "done",
                "report_id": "breakout_v1_XAUUSD_2025-01_2025-02",
                "error": None,
            },
        ),
        (
            _JobStatus.ERROR,
            {"error": "no candle history"},
            {"status": "error", "report_id": None, "error": "no candle history"},
        ),
    ],
)
async def test_get_wf_job_status_reflects_job_store(api, status, extra, expected):
    client, _ = api
    job_id = "wf-lifecycle-job"
    routes_module._wf_jobs[job_id] = {
        "status": status,
        "report_id": extra.get("report_id"),
        "error": extra.get("error"),
    }

    response = await client.get(f"/backtest/walk-forward/run/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == expected["status"]
    assert body["report_id"] == expected["report_id"]
    assert body["error"] == expected["error"]


# ── _run_wf_job directly (mirrors test_api_routes.py's _run_job tests) ───────


class _FakeCandleHistory:
    def __init__(self, fail_timeframes: set[Timeframe] | None = None, fail_sync: bool = False):
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


def _seed_pending_wf_job(job_id: str) -> None:
    routes_module._wf_jobs[job_id] = {
        "status": _JobStatus.PENDING,
        "report_id": None,
        "error": None,
    }


async def _run_wf_job_default(job_id: str, strategy_id: str, candle_history) -> None:
    """`_run_wf_job` call with the fixed symbol/period every job test below uses."""
    await routes_module._run_wf_job(
        job_id, strategy_id, "XAUUSD", "2025-01:2025-02", candle_history
    )


@pytest.fixture
def wf_job_env(tmp_path, monkeypatch):
    """DB plumbing `_resolve_strategy_name`/`_build_full_registry` touch, plus
    a tmp_path-redirected writer so job tests never write into the real
    `backend/src/backtest/reports/walk_forward/` directory."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    monkeypatch.setenv("TB_DATABASE_URL", db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(routes_module, "_build_full_registry", lambda *a, **k: None)

    def redirected_write(report, reports_dir: Path = tmp_path):
        return write_walk_forward_report(report, reports_dir)

    monkeypatch.setattr(routes_module, "write_walk_forward_report", redirected_write)
    return db_url


async def test_run_wf_job_writes_report_and_marks_done(wf_job_env, monkeypatch):
    async def fake_run_walk_forward(*args, **kwargs):
        return make_wf_report()

    monkeypatch.setattr(routes_module, "run_walk_forward", fake_run_walk_forward)
    candle_history = _FakeCandleHistory()
    job_id = "wf-job-1"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "breakout_v1", candle_history)

    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.DONE
    assert job["report_id"] is not None
    assert job["error"] is None
    # No auto-backfill needed — the fake report already has folds.
    assert candle_history.backfill_calls == []


async def test_run_wf_job_auto_backfills_when_every_fold_skipped(wf_job_env, monkeypatch):
    calls = {"n": 0}

    async def fake_run_walk_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return make_wf_report(fold_count=0)  # every fold skipped: no history at all
        return make_wf_report(fold_count=2)

    monkeypatch.setattr(routes_module, "run_walk_forward", fake_run_walk_forward)
    candle_history = _FakeCandleHistory()
    job_id = "wf-job-2"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "breakout_v1", candle_history)

    assert calls["n"] == 2
    assert candle_history.synced_symbols == ["XAUUSD"]
    assert {tf for _sym, tf in candle_history.backfill_calls} == set(Timeframe)
    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.DONE
    assert job["error"] is None


async def test_run_wf_job_reports_error_when_still_zero_after_backfill(wf_job_env, monkeypatch):
    async def always_zero_folds(*args, **kwargs):
        return make_wf_report(fold_count=0)

    monkeypatch.setattr(routes_module, "run_walk_forward", always_zero_folds)
    candle_history = _FakeCandleHistory()
    job_id = "wf-job-3"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "breakout_v1", candle_history)

    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.ERROR
    assert "no candle history" in job["error"]
    # Only one retry — the auto-backfill isn't attempted a second time.
    assert len(candle_history.backfill_calls) == len(Timeframe)


async def test_run_wf_job_reports_gateway_unreachable_instead_of_retrying(wf_job_env, monkeypatch):
    async def always_zero_folds(*args, **kwargs):
        return make_wf_report(fold_count=0)

    monkeypatch.setattr(routes_module, "run_walk_forward", always_zero_folds)
    candle_history = _FakeCandleHistory(fail_timeframes=set(Timeframe), fail_sync=True)
    job_id = "wf-job-4"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "breakout_v1", candle_history)

    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.ERROR
    assert "gateway unreachable" in job["error"]


async def test_run_wf_job_partial_folds_skipped_does_not_trigger_backfill(wf_job_env, monkeypatch):
    """Some (not all) folds had history — the harness's own per-fold skip
    already produced a usable report, so the job accepts it as-is rather
    than forcing a whole-run retry over one bad month."""

    async def fake_run_walk_forward(*args, **kwargs):
        return make_wf_report(fold_count=1)  # 1 of e.g. 2 requested folds ran

    monkeypatch.setattr(routes_module, "run_walk_forward", fake_run_walk_forward)
    candle_history = _FakeCandleHistory()
    job_id = "wf-job-5"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "breakout_v1", candle_history)

    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.DONE
    assert candle_history.backfill_calls == []  # no retry triggered


async def test_run_wf_job_surfaces_unknown_strategy_error(wf_job_env, monkeypatch):
    async def raises_value_error(*args, **kwargs):
        raise ValueError("unknown strategy id: 'nope'")

    monkeypatch.setattr(routes_module, "run_walk_forward", raises_value_error)
    candle_history = _FakeCandleHistory()
    job_id = "wf-job-6"
    _seed_pending_wf_job(job_id)

    await _run_wf_job_default(job_id, "nope", candle_history)

    job = routes_module._wf_jobs[job_id]
    assert job["status"] == _JobStatus.ERROR
    assert "unknown strategy" in job["error"]
