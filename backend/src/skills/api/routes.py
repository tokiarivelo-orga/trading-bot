"""Symbol -> bots routing endpoints (§6.6) — the real "activate a bot on a
symbol" action, distinct from the `strategies` tag's per-family version
activation. A symbol may have several bots active at once; see
`SkillAssignmentService` for what changes on disk and live.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Request

from src.skills.api.schemas import (
    AddBotIn,
    NormalSkillOut,
    RenameBotIn,
    UpdateBotConfigIn,
    UpdateBotIn,
)
from src.skills.application.skill_assignment import (
    DuplicateBotError,
    InvalidBotNameError,
    InvalidParamValueError,
    InvalidSessionError,
    SkillAssignmentService,
    UnknownBotError,
    UnknownParamError,
    UnknownStrategyError,
    UnknownSymbolError,
)

router = APIRouter(prefix="/skills", tags=["skills"])


def _service(request: Request) -> SkillAssignmentService:
    return request.app.state.container.skill_assignment


@router.get(
    "/normal",
    response_model=list[NormalSkillOut],
    summary="List every bot currently active on every symbol",
    description="One entry per bot currently routed for live trading — every "
    "skills/normal/<symbol>/<bot_name>.yaml file on disk, which may include bots activated at "
    "runtime via POST .../normal/{symbol}/bots since the backend last started, not just "
    "configs/app.yaml's startup list. A symbol may appear multiple times, once per active bot. "
    "This is the source TradeEngine._try_enter reads to resolve which bots to evaluate for a "
    "symbol.",
)
async def list_normal_skills(request: Request) -> list[NormalSkillOut]:
    skills = await _service(request).list_assignments()
    return [NormalSkillOut.from_domain(s, strategy=strategy) for s, strategy in skills]


@router.post(
    "/normal/{symbol}/bots",
    response_model=NormalSkillOut,
    summary="Activate a new bot on a symbol",
    description=(
        "Adds a new, independent bot trading `symbol` live, alongside any bots already routed "
        "there — never replaces one (see PUT to reassign an existing bot instead). Writes "
        "skills/normal/<symbol>/<bot_name>.yaml and hot-swaps the running SkillSelector so the "
        "new bot starts evaluating on the very next candle close, no restart needed. If "
        "`symbol` isn't yet part of the automated-trading universe, this also activates it — "
        "permanently: persists it into configs/app.yaml, hot-adds it to candle streaming, and "
        "loads its spread-gate config, all immediately, no restart (see `newly_activated` on "
        "the response). It does not activate or change any StrategyVersion; the target family "
        "must already have an active, non-paused version or the engine will simply find "
        "nothing to evaluate for this bot."
    ),
    responses={
        404: {"description": "No configs/symbols/<symbol>.yaml for this symbol."},
        409: {"description": "This symbol already has a bot with that bot_name."},
        422: {
            "description": "strategy_name has no currently active, non-paused StrategyVersion, "
            "or bot_name has no valid slug characters."
        },
    },
)
async def add_bot(
    request: Request,
    body: AddBotIn,
    symbol: str = Path(description="Broker symbol, e.g. XAUUSD."),
) -> NormalSkillOut:
    try:
        result = await _service(request).add_bot(
            symbol, body.strategy_name, body.bot_name, body.risk_multiplier
        )
    except UnknownSymbolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnknownStrategyError, InvalidBotNameError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DuplicateBotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return NormalSkillOut.from_domain(
        result.skill, newly_activated=result.newly_activated, strategy=result.strategy
    )


@router.put(
    "/normal/{symbol}/bots/{bot_name}",
    response_model=NormalSkillOut,
    summary="Reassign one bot's strategy",
    description=(
        "Rewrites which strategy family `bot_name` trades on `symbol`, keeping that bot's "
        "existing sessions/risk_multiplier and leaving every other bot on the symbol untouched "
        "— writes the YAML and hot-swaps the running SkillSelector immediately, no restart "
        "needed."
    ),
    responses={
        404: {"description": "This symbol has no bot with that bot_name."},
        422: {"description": "strategy_name has no currently active, non-paused StrategyVersion."},
    },
)
async def update_bot(
    request: Request,
    body: UpdateBotIn,
    symbol: str = Path(description="Broker symbol, e.g. XAUUSD."),
    bot_name: str = Path(description="This bot's short id on the symbol, e.g. 'breakout'."),
) -> NormalSkillOut:
    try:
        skill, strategy = await _service(request).update_bot(symbol, bot_name, body.strategy_name)
    except UnknownBotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnknownStrategyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NormalSkillOut.from_domain(skill, strategy=strategy)


@router.put(
    "/normal/{symbol}/bots/{bot_name}/name",
    response_model=NormalSkillOut,
    summary="Rename one bot",
    description=(
        "Renames `bot_name` on `symbol` in place, keeping its strategy, risk_multiplier, "
        "sessions, and param/htf_veto overrides — writes the YAML under the new slug, deletes "
        "the old file, and hot-swaps the running SkillSelector immediately, no restart needed. "
        "Renaming changes this bot's MT5 magic number (derived from the full skill name), so "
        "any position the broker already has open under the old name will no longer be "
        "recognized as this bot's — avoid renaming a bot that currently holds an open position."
    ),
    responses={
        404: {"description": "This symbol has no bot with that bot_name."},
        409: {"description": "This symbol already has a bot with that new_bot_name."},
        422: {"description": "new_bot_name has no valid slug characters."},
    },
)
async def rename_bot(
    request: Request,
    body: RenameBotIn,
    symbol: str = Path(description="Broker symbol, e.g. XAUUSD."),
    bot_name: str = Path(description="This bot's current short id on the symbol."),
) -> NormalSkillOut:
    try:
        skill, strategy = await _service(request).rename_bot(symbol, bot_name, body.new_bot_name)
    except UnknownBotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidBotNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DuplicateBotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return NormalSkillOut.from_domain(skill, strategy=strategy)


@router.put(
    "/normal/{symbol}/bots/{bot_name}/config",
    response_model=NormalSkillOut,
    summary="Update one bot's full configuration",
    description=(
        "Replaces this bot's risk_multiplier, sessions, and per-bot strategy-param/htf_veto "
        "overrides in one call — every field is a full replacement, not a partial patch (send "
        "the bot's current values, from GET /skills/normal, for anything you don't want to "
        "change). `param_overrides` lets a single bot tune its strategy's own params (e.g. "
        "`counter_trend_penalty`) or its HTF veto without forking the strategy's generated "
        "code — see `strategy_default_params`/`strategy_default_htf_veto` on GET "
        "/skills/normal for the base values these overrides layer onto. Writes the YAML and "
        "hot-swaps the running SkillSelector immediately, no restart needed. Does not change "
        "which strategy family the bot trades — see PUT .../bots/{bot_name} for that (which "
        "also resets any overrides set here, since they may not apply to the new strategy)."
    ),
    responses={
        404: {"description": "This symbol has no bot with that bot_name."},
        422: {
            "description": "A session's start/end isn't valid HH:MM, or param_overrides has a "
            "key that isn't in this bot's strategy's own params, or a value whose type doesn't "
            "match that param's default."
        },
    },
)
async def update_bot_config(
    request: Request,
    body: UpdateBotConfigIn,
    symbol: str = Path(description="Broker symbol, e.g. XAUUSD."),
    bot_name: str = Path(description="This bot's short id on the symbol, e.g. 'breakout'."),
) -> NormalSkillOut:
    try:
        skill, strategy = await _service(request).update_config(
            symbol,
            bot_name,
            risk_multiplier=body.risk_multiplier,
            sessions=[(s.start, s.end) for s in body.sessions],
            param_overrides=body.param_overrides,
            htf_veto_override=body.htf_veto_override,
        )
    except UnknownBotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (UnknownParamError, InvalidParamValueError, InvalidSessionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NormalSkillOut.from_domain(skill, strategy=strategy)


@router.delete(
    "/normal/{symbol}/bots/{bot_name}",
    status_code=204,
    summary="Deactivate a bot on a symbol",
    description=(
        "Stops `bot_name` from trading `symbol`: deletes its YAML file and removes it from the "
        "running SkillSelector immediately, no restart needed. Every other bot on the symbol "
        "keeps trading unaffected. If this was the symbol's last bot, the symbol also leaves "
        "the automated-trading universe: it is removed from `configs/app.yaml`'s `symbols` "
        "and its candle streaming stops (adding a bot again re-activates it)."
    ),
    responses={404: {"description": "This symbol has no bot with that bot_name."}},
)
async def remove_bot(
    request: Request,
    symbol: str = Path(description="Broker symbol, e.g. XAUUSD."),
    bot_name: str = Path(description="This bot's short id on the symbol, e.g. 'breakout'."),
) -> None:
    try:
        await _service(request).remove_bot(symbol, bot_name)
    except UnknownBotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
