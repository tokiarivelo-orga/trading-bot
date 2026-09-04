"""Strategy version listing and activation endpoints (§6.5, §8.1).

Activation is the only user-triggered action here, and it doubles as
rollback: activating an older version id reactivates that exact file and
archives whatever was active — nothing is ever edited in place.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from src.shared.api.dependencies import AccountRuntimeDep
from src.strategies.api.schemas import (
    DuplicateVersionRequest,
    EditVersionCodeRequest,
    RenameVersionRequest,
    StrategyVersionDetailOut,
    StrategyVersionOut,
    UpdateVersionSpecRequest,
)
from src.strategies.application.versioning import (
    StrategyNameConflictError,
    StrategyValidationError,
    StrategyVersionService,
    VersionActiveError,
    VersionAlreadyArchivedError,
    VersionNotActiveError,
)
from src.strategies.domain.versioning import VersionStatus

router = APIRouter(prefix="/accounts/{account_id}/strategies", tags=["strategies"])

# `evaluate-custom` is a sandbox scratchpad over raw candle history, not tied
# to any account's own strategy registry — stays global/unprefixed, unlike
# every other route in this module (MULTI_ACCOUNT_PLAN.md Phase 6).
sandbox_router = APIRouter(prefix="/strategies", tags=["strategies"])

_VERSION_NOT_FOUND = {404: {"description": "No strategy version with that id."}}

# Module-level singleton, not a call in the argument default — ruff's B008
# check doesn't special-case `Query()` for non-primitive annotations
# (VersionStatus is a StrEnum) the way it does for str/int/Literal.
_STATUS_QUERY = Query(
    default=None, description="Restrict to this lifecycle stage: validated, active, archived."
)


def _service(account: AccountRuntimeDep) -> StrategyVersionService:
    return account.strategy_versions


@router.get(
    "/versions",
    response_model=list[StrategyVersionOut],
    summary="List strategy versions",
    description="Every recorded strategy version, newest first per name. Filter to one "
    "strategy family with `name`, or to one lifecycle stage with `status` (e.g. `status=active` "
    "to fetch only what's currently live, without pulling every historical version).",
)
async def list_versions(
    account: AccountRuntimeDep,
    name: str | None = Query(default=None, description="Restrict to this strategy family name."),
    status: VersionStatus | None = _STATUS_QUERY,
) -> list[StrategyVersionOut]:
    versions = _service(account).list_versions(name, status)
    return [StrategyVersionOut.from_domain(v) for v in versions]


@router.get(
    "/versions/{version_id}",
    response_model=StrategyVersionDetailOut,
    summary="Get a single strategy version, including its source code",
    description="Full version detail for the version diff/detail screen — same fields as "
    "GET /strategies/versions plus the generated Python source.",
    responses=_VERSION_NOT_FOUND,
)
async def get_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id, as returned by GET /strategies/versions."),
) -> StrategyVersionDetailOut:
    service = _service(account)
    version = service.get_version(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="strategy version not found")
    code = service.get_code(version)
    return StrategyVersionDetailOut.from_domain_with_code(version, code)


@router.post(
    "/versions/{version_id}/activate",
    response_model=StrategyVersionOut,
    summary="Activate a strategy version",
    description=(
        "Re-validates the file on disk in the sandbox, registers it live in the "
        "StrategyRegistry, and archives whichever version of the same strategy name was "
        "previously active. This is also how rollback works: activate an older version id "
        "to bring it back. Does not change configs/app.yaml's paper/live mode — this only "
        "selects which strategy code the engine runs, never real-money risk settings."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        422: {"description": "The version's file no longer passes sandbox validation."},
    },
)
async def activate_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id to activate (or reactivate, for rollback)."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        version = await asyncio.to_thread(service.activate_version, version_id)
    except StrategyValidationError as exc:
        raise HTTPException(status_code=422, detail="; ".join(exc.errors)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(version)


@router.post(
    "/versions/{version_id}/duplicate",
    response_model=StrategyVersionOut,
    summary="Duplicate a strategy version into a new strategy family",
    description=(
        "Clones this version's generated code and spec snapshot into a brand-new "
        "strategy family — new name, version 1, no parent — a fork for retargeting "
        "(e.g. same logic, different symbol), not a supersession of the original. "
        "Optionally pass `symbols` to also retarget the clone to different symbols; "
        "this rewrites the `StrategySpec(symbols=...)` literal in the generated source "
        "and re-validates it in the sandbox before saving. Never edits configs/app.yaml — "
        "the engine won't trade the new symbol live until a human adds it there."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {
            "description": "The requested new name is already in use by another strategy family."
        },
        422: {
            "description": "The clone (with the symbols override applied, if any) failed "
            "sandbox validation, or no `symbols=(...)` literal could be found to rewrite."
        },
    },
)
async def duplicate_version(
    account: AccountRuntimeDep,
    body: DuplicateVersionRequest,
    version_id: str = Path(description="Version id to duplicate."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        duplicated = await asyncio.to_thread(
            service.duplicate_version,
            version_id,
            new_name=body.name,
            symbols=tuple(body.symbols) if body.symbols else None,
        )
    except StrategyNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StrategyValidationError as exc:
        raise HTTPException(status_code=422, detail="; ".join(exc.errors)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(duplicated)


@router.patch(
    "/versions/{version_id}/rename",
    response_model=StrategyVersionOut,
    summary="Rename a strategy family",
    description=(
        "Renames the display name shared by every version of this strategy family "
        "(not just this one) — updates the stored records only, never the generated "
        "file's on-disk name or contents."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {
            "description": "The requested new name is already in use by another strategy family."
        },
    },
)
async def rename_version(
    account: AccountRuntimeDep,
    body: RenameVersionRequest,
    version_id: str = Path(description="Any version id belonging to the family to rename."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        renamed = await asyncio.to_thread(service.rename_family, version_id, body.name)
    except StrategyNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(renamed)


@router.post(
    "/versions/{version_id}/edit",
    response_model=StrategyVersionDetailOut,
    summary="Save a manual code edit as a new strategy version",
    description=(
        "Re-validates the given source in the sandbox and, if it passes, saves it — by "
        "default as a new version of this version's strategy family, parented on "
        "`version_id` itself (not necessarily the active version), so editing an old or "
        "archived version doesn't silently rebase onto whatever is currently live. Pass "
        "`new_name` to fork the edit into a brand-new strategy family at version 1 instead "
        "('duplicate' destination), for trying a change without touching the original. The "
        "new version's status is always 'validated', never 'active' — activating it is a "
        "separate call (POST .../activate). This is the manual counterpart to AI "
        "regeneration (POST /ai/strategies/versions/{version_id}/regenerate)."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {"description": "`new_name` is already in use by another strategy family."},
        422: {
            "description": "The edited code failed sandbox validation (import whitelist, "
            "AST scan, or the smoke-test evaluate() call)."
        },
    },
)
async def edit_version_code(
    account: AccountRuntimeDep,
    body: EditVersionCodeRequest,
    version_id: str = Path(description="Version id whose code is being edited."),
) -> StrategyVersionDetailOut:
    service = _service(account)
    try:
        edited = await asyncio.to_thread(
            service.edit_code, version_id, body.code, new_name=body.new_name
        )
    except StrategyNameConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StrategyValidationError as exc:
        raise HTTPException(status_code=422, detail="; ".join(exc.errors)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionDetailOut.from_domain_with_code(edited, body.code)


@router.patch(
    "/versions/{version_id}/spec",
    response_model=StrategyVersionOut,
    summary="Edit a strategy version's spec snapshot",
    description=(
        "Overwrites the descriptive spec snapshot shown on the version detail page — "
        "symbols, timeframes, indicators, entry/exit rules, risk notes, params, price "
        "levels, chart notes — with the given values. This is annotation only: unlike "
        "POST .../edit, it never touches the generated Python source or re-runs sandbox "
        "validation, and it updates this exact version's record in place rather than "
        "creating a new version — the same way PATCH .../rename does. To change what the "
        "code actually does, edit the code (POST .../edit) or regenerate it with AI."
    ),
    responses=_VERSION_NOT_FOUND,
)
async def update_version_spec(
    account: AccountRuntimeDep,
    body: UpdateVersionSpecRequest,
    version_id: str = Path(description="Version id whose spec snapshot is being edited."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        updated = await asyncio.to_thread(service.update_spec, version_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(updated)


@router.post(
    "/versions/{version_id}/archive",
    response_model=StrategyVersionOut,
    summary="Archive a strategy version",
    description=(
        "Retires this version without deleting it: marks it 'archived' and, if it was the "
        "live active version, unregisters it from the StrategyRegistry so the engine stops "
        "evaluating it on the next candle close. Unlike activation's implicit archive-on-"
        "supersede, this is a direct action with no replacement version — the strategy "
        "family can end up with no active version at all."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {"description": "The version is already archived."},
    },
)
async def archive_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id to archive."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        archived = await asyncio.to_thread(service.archive_version, version_id)
    except VersionAlreadyArchivedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(archived)


@router.delete(
    "/versions/{version_id}",
    status_code=204,
    summary="Delete a strategy version",
    description=(
        "Hard-deletes this version's database record and its generated Python file. "
        "Refuses to delete the currently active version — archive it (POST .../archive) or "
        "activate a replacement first, so the engine is never left pointing at a file "
        "that's about to disappear. This cannot be undone."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {"description": "The version is currently active; archive it before deleting."},
    },
)
async def delete_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id to delete."),
) -> None:
    service = _service(account)
    try:
        await asyncio.to_thread(service.delete_version, version_id)
    except VersionActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/versions/{version_id}/pause",
    response_model=StrategyVersionOut,
    summary="Pause a strategy version",
    description=(
        "Suspends live trading for this active version without deactivating or archiving "
        "it: the StrategyRegistry stops returning it to the engine, so no new entries are "
        "evaluated for it, but it stays 'active' and POST .../resume brings it straight "
        "back. Distinct from the engine-wide kill switch (POST /engine/kill), which pauses "
        "every strategy at once."
    ),
    responses={
        **_VERSION_NOT_FOUND,
        409: {"description": "The version isn't the active one for its strategy family."},
    },
)
async def pause_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id to pause — must be the active version."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        paused = await asyncio.to_thread(service.pause_version, version_id)
    except VersionNotActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(paused)


@router.post(
    "/versions/{version_id}/resume",
    response_model=StrategyVersionOut,
    summary="Resume a paused strategy version",
    description="Reverses POST .../pause: the StrategyRegistry resumes returning this "
    "version to the engine so it evaluates entries again.",
    responses={
        **_VERSION_NOT_FOUND,
        409: {"description": "The version isn't the active one for its strategy family."},
    },
)
async def resume_version(
    account: AccountRuntimeDep,
    version_id: str = Path(description="Version id to resume — must be the active version."),
) -> StrategyVersionOut:
    service = _service(account)
    try:
        resumed = await asyncio.to_thread(service.resume_version, version_id)
    except VersionNotActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StrategyVersionOut.from_domain(resumed)


class CustomSignalOut(BaseModel):
    """One signal `evaluate()` produced while stepping through history."""

    time: int = Field(description="Candle open time, epoch seconds UTC.")
    direction: str = Field(description="'buy' or 'sell'.")
    sl_points: float = Field(description="Stop loss distance, in points.")
    tp_points: float = Field(description="Take profit distance, in points.")
    confidence: float = Field(description="Strategy-reported confidence, 0-1.")
    reason: str = Field(description="Free-text reason the strategy gave for this signal.")


class EvaluateCustomCodeRequest(BaseModel):
    """Body for `POST /strategies/evaluate-custom`."""

    code: str = Field(
        description="Candidate strategy source, validated in the sandbox before running."
    )
    symbol: str = Field(description="Trading symbol to evaluate against, e.g. 'XAUUSD'.")
    timeframe: str = Field(default="M5", description="Entry timeframe, e.g. 'M5'.")
    period: str = Field(
        default="2026-07-01:2026-07-13",
        description="Date range to evaluate over, 'YYYY-MM-DD:YYYY-MM-DD'.",
    )


class EvaluateCustomCodeResponse(BaseModel):
    """Result of running `code` over `symbol`'s history for `period`."""

    signals: list[CustomSignalOut] = Field(
        description="Every signal evaluate() produced, in order."
    )
    indicators: dict[str, list[float | None]] = Field(
        description="Named indicator series from the strategy's optional indicators() hook, "
        "aligned to `candles`."
    )
    candles: list[dict[str, Any]] = Field(description="The candle history evaluated against.")
    error: str | None = Field(
        default=None,
        description="Set instead of results if the code failed to load, parse the "
        "period, or find any candles.",
    )


@sandbox_router.post(
    "/evaluate-custom",
    response_model=EvaluateCustomCodeResponse,
    summary="Evaluate custom strategy code against symbol history",
    description="Loads arbitrary strategy code, compiles it in the sandbox, runs evaluate() "
    "over the historical candles, and returns signals/indicators.",
)
async def evaluate_custom_code(
    request: Request,
    body: EvaluateCustomCodeRequest,
) -> EvaluateCustomCodeResponse:
    import logging
    from datetime import timedelta

    import pandas as pd

    from src.backtest.application.period import parse_period
    from src.engine.application.context import candles_to_dataframe
    from src.market_data.adapters.candle_repository import CandleRepository
    from src.market_data.domain.models import Timeframe
    from src.shared.db.base import make_session_factory
    from src.strategies.domain.models import MarketContext
    from src.strategies.sandbox import validate_and_load

    logger = logging.getLogger(__name__)

    # 1. Validate and load strategy from code
    strategy, errors = validate_and_load(body.code)
    if not strategy:
        return EvaluateCustomCodeResponse(
            signals=[], indicators={}, candles=[], error="; ".join(errors)
        )

    # 2. Parse period and set up dates
    try:
        start, end = parse_period(body.period)
    except Exception as exc:
        return EvaluateCustomCodeResponse(
            signals=[], indicators={}, candles=[], error=f"Invalid period format: {exc}"
        )

    # Warmup buffer of 30 days
    history_start = start - timedelta(days=30)

    # 3. Load candles
    session_factory = make_session_factory(request.app.state.container.settings.database_url)
    candle_repo = CandleRepository(session_factory)

    try:
        tf_enum = Timeframe(body.timeframe)
    except ValueError:
        tf_enum = Timeframe.M5

    raw_candles = candle_repo.get_range(body.symbol, tf_enum, history_start, end)
    if not raw_candles:
        return EvaluateCustomCodeResponse(
            signals=[],
            indicators={},
            candles=[],
            error=f"No candles found for {body.symbol} in the requested range.",
        )

    candles_by_tf = {body.timeframe: raw_candles}
    for tf_str in getattr(strategy.spec, "confirmation_timeframes", []):
        if tf_str != body.timeframe:
            try:
                c_enum = Timeframe(tf_str)
                candles_by_tf[tf_str] = candle_repo.get_range(
                    body.symbol, c_enum, history_start, end
                )
            except Exception:
                pass

    # Convert to pandas DataFrames
    dfs = {tf: candles_to_dataframe(candles) for tf, candles in candles_by_tf.items()}
    entry_df = dfs[body.timeframe]

    if entry_df.empty:
        return EvaluateCustomCodeResponse(
            signals=[], indicators={}, candles=[], error="Empty candle data set."
        )

    # 4. Step-by-step evaluation
    signals = []
    eval_indices = entry_df[entry_df["time"] >= start].index
    if len(eval_indices) > 1000:
        eval_indices = eval_indices[-1000:]

    for idx in eval_indices:
        current_time = entry_df.loc[idx, "time"]

        # Slice DataFrames up to current_time
        sliced_candles = {}
        for tf, df in dfs.items():
            sliced_df = df[df["time"] <= current_time]
            sliced_candles[tf] = sliced_df

        ctx = MarketContext(symbol=body.symbol, candles=sliced_candles, spread_points=20.0)

        try:
            sig_res = strategy.evaluate(ctx)
            if sig_res:
                sig_list = sig_res if isinstance(sig_res, (list, tuple)) else [sig_res]
                for sig in sig_list:
                    signals.append(
                        CustomSignalOut(
                            time=int(current_time.timestamp()),
                            direction=sig.direction.value,
                            sl_points=sig.sl_points,
                            tp_points=sig.tp_points,
                            confidence=sig.confidence,
                            reason=sig.reason,
                        )
                    )
        except Exception:
            # Silently ignore evaluation errors on specific candles
            pass

    # 5. Extract indicators
    indicators_data = {}
    if hasattr(strategy, "indicators"):
        try:
            raw_inds = strategy.indicators(dfs)
            valid_indices = entry_df[entry_df["time"] >= start].index
            for name, val_list in raw_inds.items():
                if hasattr(val_list, "iloc"):
                    vals = [val_list.iloc[i] for i in valid_indices]
                else:
                    vals = [val_list[i] for i in valid_indices]

                cleaned = []
                for val in vals:
                    if pd.isna(val) or val is None:
                        cleaned.append(None)
                    else:
                        cleaned.append(float(val))
                indicators_data[name] = cleaned
        except Exception as e:
            logger.warning("Failed to compute custom indicators: %s", e)

    # 6. Format candles to return to frontend
    candles_out = [
        {
            "time": int(c.time.timestamp()),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
        }
        for c in raw_candles
        if c.time >= start
    ]

    return EvaluateCustomCodeResponse(
        signals=signals, indicators=indicators_data, candles=candles_out
    )
