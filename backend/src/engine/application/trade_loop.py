"""Trade loop (§6.4, §7.1): the engine's entry point, driven by `CandleClosed`.

On every candle close: each active bot whose strategy's own
`spec.entry_timeframe` matches the closed timeframe runs skill selection ->
strategy evaluation -> HTF confirmation -> risk gate & sizing -> order
placement. This is what lets M1 scalp and M15 swing strategies fire live —
the engine-level `entry_timeframe` config is only the default cadence for
position management and for reporting bots that can't be resolved to a
strategy. Position management (breakeven, time-stop) still runs every
engine-entry-TF (M5) close regardless of the filtering below.

HTF confirmation is per-bot, not a fixed engine-wide list: the veto
timeframe is always the single timeframe immediately above the bot's own
`spec.entry_timeframe` (`Timeframe.next_up`) — an M1 bot is vetoed by M5,
an M5 bot by M15, and so on. There is no separate global config for this;
two bots entering on different timeframes get different veto timeframes.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from src.activity.domain.models import DecisionCheck
from src.activity.ports.signal_decisions import SignalDecisionSinkPort
from src.broker.application.account_service import AccountService
from src.broker.application.order_service import OrderService
from src.broker.domain.trading import OrderRejected, Position, Side
from src.engine.application.context import build_market_context
from src.engine.application.mtf_confirm import confirm
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager
from src.engine.domain.models import EngineStatus
from src.engine.domain.regime import EntryRegime, RegimeConfig, compute_entry_regime
from src.engine.domain.volatility import (
    VolatilityConfig,
    VolatilityRegime,
    latest_volatility_regime,
)
from src.engine.ports.strategy_source import StrategySourcePort
from src.market_data.domain.models import Candle, MarketDataUnavailable, SymbolInfo, Timeframe
from src.market_data.ports.market_data import MarketDataPort
from src.order_book.ports.capture import OrderBookCapturePort
from src.shared.events.bus import EventBus
from src.shared.events.definitions import (
    CandleClosed,
    CircuitBreakerTripped,
    NewsWindowEntered,
    PositionClosed,
)
from src.shared.logging.account_context import bind_signal_id, current_account_id
from src.shared.metrics.registry import (
    ENGINE_LOOP_DURATION,
    observe_signal_fired,
    record_signal_outcome,
)
from src.skills.ports.skill_selector import SkillDecision, SkillSelectorPort
from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
    MarketContext,
    PositionSnapshot,
    Signal,
    Strategy,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_BARS = 200

# `RiskDecision.code` from the pre-trade gate -> the decision-trail outcome it
# maps to (OBSERVABILITY_PLAN.md Phase 2). "paused" covers every circuit
# breaker — daily loss, consecutive losses, manual kill — all of which surface
# as the `daily_loss_breaker` bucket; the pause reason itself is carried in the
# decision's `reason` text. Anything unmapped falls back to the pre-Phase-2
# catch-all `risk_rejected`.
_PRETRADE_OUTCOMES: dict[str, str] = {
    "paused": "daily_loss_breaker",
    "max_positions": "max_positions",
    "daily_trading_disabled": "risk_rejected",
}


def _htf_check(*, passed: bool) -> DecisionCheck:
    """The HTF-confirmation gate as a `DecisionCheck`. It's boolean, not
    numeric, so it's encoded the way `TradeRecord.indicators` encodes boolean
    readings: 1.0 for confirmed, compared `==` against the required 1.0."""
    return DecisionCheck(
        name="htf_confirm",
        value=1.0 if passed else 0.0,
        threshold=1.0,
        comparison="==",
        passed=passed,
    )


def _volatility_check(
    percentile: float, config: VolatilityConfig, *, passed: bool
) -> DecisionCheck:
    """The ATR-percentile volatility guard as a `DecisionCheck`. `percentile`
    is NaN when the entry timeframe had no candles to classify — recorded as
    0.0 (the guard didn't block) rather than letting NaN reach the JSON
    column, which can't represent it."""
    value = 0.0 if math.isnan(percentile) else percentile
    return DecisionCheck(
        name="volatility_percentile",
        value=value,
        threshold=float(config.extreme_percentile),
        comparison="<",
        passed=passed,
    )


def _regime_kwargs(regime: EntryRegime | None) -> dict[str, str | float | None]:
    """Flattens an `EntryRegime` (or `None`) into the five kwargs both
    `_record_decision` and the `order_service.open_position` call thread
    through — defined once so the two call sites can never drift apart, and
    so `nan` (the "insufficient history" sentinel `latest_volatility_regime`/
    `latest_trend_regime` return, not `None`) is normalized to `None` at this
    one boundary. Everything downstream of this point — `SignalDecision`,
    `TradeRecord`, `PositionOpened`, and their JSON serialization — is typed
    `float | None` and follows this codebase's "unmeasured is None, never
    NaN" convention throughout; `nan` would otherwise reach a DB column and
    an API response as an invalid-JSON literal."""
    if regime is None:
        return {
            "regime_volatility": None,
            "regime_volatility_percentile": None,
            "regime_trend": None,
            "regime_adx": None,
            "regime_session": None,
        }
    return {
        "regime_volatility": regime.volatility.value,
        "regime_volatility_percentile": (
            None if math.isnan(regime.volatility_percentile) else regime.volatility_percentile
        ),
        "regime_trend": regime.trend.value,
        "regime_adx": None if math.isnan(regime.adx) else regime.adx,
        "regime_session": regime.session.value,
    }


def _veto_timeframe(strategy: Strategy) -> str | None:
    """The bot's own HTF veto timeframe: the one immediately above its
    `entry_timeframe`. `None` for a bot already entering on `MN` (nothing to
    veto against above it) or for a bot with `spec.htf_veto=False` (opted
    out — see `StrategySpec.htf_veto`)."""
    if not strategy.spec.htf_veto:
        return None
    above = Timeframe(strategy.spec.entry_timeframe).next_up()
    return above.value if above is not None else None


def _effective_strategy(strategy: Strategy, decision: SkillDecision) -> Strategy:
    """Per-bot view of `strategy` with this decision's param/htf_veto
    overrides merged in (see `NormalSkill.param_overrides`/`htf_veto_override`)
    — a fresh shallow copy, since `StrategyRegistry` stores one `Strategy`
    instance shared by every bot on this strategy family; mutating it in
    place would leak one bot's overrides onto every other bot trading the
    same strategy."""
    if not decision.param_overrides and decision.htf_veto_override is None:
        return strategy
    merged_params = {**strategy.spec.params, **decision.param_overrides}
    htf_veto = (
        decision.htf_veto_override
        if decision.htf_veto_override is not None
        else strategy.spec.htf_veto
    )
    effective = copy.copy(strategy)
    effective.spec = replace(strategy.spec, params=merged_params, htf_veto=htf_veto)
    return effective


class TradeEngine:
    def __init__(
        self,
        *,
        market_data: MarketDataPort,
        order_service: OrderService,
        account: AccountService,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        skill_selector: SkillSelectorPort,
        strategy_source: StrategySourcePort,
        entry_timeframe: str,
        volatility_config: VolatilityConfig,
        regime_config: RegimeConfig | None = None,
        signal_decisions: SignalDecisionSinkPort | None = None,
        order_book_capture: OrderBookCapturePort | None = None,
        event_bus: EventBus | None = None,
        enabled: bool = True,
        context_bars: int = DEFAULT_CONTEXT_BARS,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        context_builder: Callable[
            [str, dict[str, list[Candle]], float], MarketContext
        ] = build_market_context,
    ) -> None:
        self._market_data = market_data
        self._order_service = order_service
        self._account = account
        self._risk_manager = risk_manager
        self._position_manager = position_manager
        self._skill_selector = skill_selector
        self._strategy_source = strategy_source
        self._entry_timeframe = entry_timeframe
        self._volatility_config = volatility_config
        # Regime tagging (OBSERVABILITY_PLAN.md Phase 6) — always-on and
        # unconditional, unlike `volatility_config`'s guard: every signal
        # gets a regime snapshot for analytics regardless of whether the
        # volatility guard itself is enabled. Defaulted (unlike
        # `volatility_config`, which is required) so the many existing tests
        # constructing a bare `TradeEngine(...)` don't all need updating —
        # same reasoning `context_bars`/`clock`/`context_builder` above are
        # defaulted for. `None` (rather than a mutable-default-style
        # `RegimeConfig()` literal in the signature, which ruff's B008 rule
        # rejects for a plain function parameter) resolved to a fresh
        # default instance here.
        self._regime_config = regime_config if regime_config is not None else RegimeConfig()
        self._volatility_guard_enabled = True
        # Typed decision trail (OBSERVABILITY_PLAN.md Phase 1). Optional so a
        # backtest engine / unit test can run without a database; when absent
        # the human-readable log lines below are all that's produced.
        self._signal_decisions = signal_decisions
        # Order-book capture (order_book/ Phase 5) — optional, like
        # `signal_decisions`, so the many bare `TradeEngine(...)` tests and
        # the backtest runner (which must never hit a live gateway) don't
        # need it. Fired-and-forgotten per evaluated signal in
        # `_enter_for_bot` via `_capture_order_book_safely`, never awaited on
        # the entry path itself.
        self._order_book_capture = order_book_capture
        self._event_bus = event_bus
        self._enabled = enabled
        self._context_bars = context_bars
        self._clock = clock
        # Swappable only so the backtest runner can serve cached DataFrame
        # slices over replay history instead of rebuilding frames on every
        # bar; the frames it produces are value-identical to
        # `build_market_context`'s (see backtest/adapters/context_builder.py).
        self._context_builder = context_builder

    @property
    def status(self) -> EngineStatus:
        return replace(self._risk_manager.status, enabled=self._enabled)

    @property
    def volatility_config(self) -> VolatilityConfig:
        return self._volatility_config

    @property
    def volatility_guard_enabled(self) -> bool:
        return self._volatility_guard_enabled

    def set_volatility_guard_enabled(self, enabled: bool) -> None:
        """Live-updates whether the volatility guard (EXTREME-regime entry
        block + SL/TP regime scaling here in `_enter_for_bot`, plus the
        mirrored EXTREME/HIGH position-management rules in
        `PositionManager`) is active — takes effect on the very next entry
        decision. When `False`, entries behave exactly as if
        `volatility_config` didn't exist (`sl_mult`/`tp_mult` both 1.0, no
        EXTREME check). Not persisted: a backend restart reverts to
        `configs/volatility.yaml` (enabled by default)."""
        self._volatility_guard_enabled = enabled
        logger.info("trade engine: volatility guard enabled=%s", enabled)

    async def on_candle_closed(self, event: CandleClosed) -> None:
        # Engine-loop-duration metric (OBSERVABILITY_PLAN.md Phase 5): the
        # whole per-candle pass, position management included, since that's
        # what "is the engine keeping up with candles" actually needs to
        # measure. Labeled from the ContextVar rather than a constructor
        # param — this method always runs inside the account's own
        # `CandleStreamService._run` task, which sets it (see
        # `shared/logging/account_context.py`).
        with ENGINE_LOOP_DURATION.labels(account_id=current_account_id.get()).time():
            if event.timeframe == self._entry_timeframe:
                await self._position_manager.on_candle_closed(event.symbol)
            if not self._enabled:
                return
            await self._try_enter(event.symbol, event.timeframe)

    async def on_position_closed(self, event: PositionClosed) -> None:
        balance = await self._current_balance()
        was_paused = self._risk_manager.paused
        self._risk_manager.record_trade_closed(event.profit, balance=balance, now=self._clock())
        if not was_paused and self._risk_manager.paused:
            await self._publish_pause_alert()

    async def kill_switch(self) -> None:
        """Close every open position and pause the engine (F kill switch)."""
        self._risk_manager.kill()
        await self._publish_pause_alert()
        for position in await self._order_service.get_positions():
            try:
                await self._order_service.close_position(position.ticket)
            except OrderRejected:
                logger.exception("kill switch: failed to close ticket=%d", position.ticket)

    async def _publish_pause_alert(self) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            CircuitBreakerTripped(reason=self._risk_manager.status.pause_reason)
        )

    async def on_news_window_entered(self, event: NewsWindowEntered) -> None:
        """Pre-news flatten (§6.6, §8): closes open positions in the news
        skill's `symbols` when `pre_event.close_all` requested it. Unlike
        `kill_switch`, this never pauses the engine — new-entry blocking for
        the window is handled entirely by `NewsSkillSelector` returning
        `allowed=False`."""
        if not event.close_all:
            return
        for symbol in event.symbols:
            for position in await self._order_service.get_positions(symbol):
                try:
                    await self._order_service.close_position(position.ticket)
                except OrderRejected:
                    logger.exception(
                        "news flatten: failed to close ticket=%d ahead of %s",
                        position.ticket,
                        event.event_name,
                    )
            logger.info("news flatten: closed %s positions ahead of %s", symbol, event.event_name)

    def resume(self) -> None:
        self._risk_manager.resume()

    async def _current_balance(self) -> float | None:
        try:
            status = await self._account.status()
        except Exception:
            logger.exception("could not fetch account status")
            return None
        account = status.get("account")
        return account["balance"] if account else None

    async def _try_enter(self, symbol: str, timeframe: str) -> None:
        now = self._clock()
        decisions = self._skill_selector.select_all(symbol, now)
        if symbol == "Step Index 200":
            logger.info(
                "TRACE _try_enter symbol=%s timeframe=%s entry_tf=%s decisions=%r",
                symbol, timeframe, self._entry_timeframe, decisions,
            )
        if not decisions:
            if timeframe == self._entry_timeframe:
                # DEBUG, not INFO (OBSERVABILITY_PLAN.md Phase 5 log hygiene):
                # this is structural ("nothing is configured to trade this
                # symbol at all"), not a decision — it repeats every single
                # entry-timeframe candle close for as long as the symbol has
                # no bot, which for an idle symbol is forever. Contrast the
                # `decision.allowed is False` branch below, which is a real
                # per-attempt veto (news window, paused skill) and stays INFO.
                logger.debug("ENTRY SKIPPED (skill routing): %s — no active bots", symbol)
            return

        candidates: list[tuple[SkillDecision, Strategy]] = []
        for decision in decisions:
            strategy = (
                self._strategy_source.get(decision.strategy_name)
                if decision.strategy_name
                else None
            )
            if strategy is not None:
                if strategy.spec.symbols and symbol not in strategy.spec.symbols:
                    continue
                if strategy.spec.entry_timeframe != timeframe:
                    continue  # this bot enters on a different timeframe's closes
            elif timeframe != self._entry_timeframe:
                # Bots we can't resolve to a strategy (blocked with no
                # strategy_name, or unregistered) get reported once per
                # engine-entry-TF close, not on every finer close.
                continue
            if not decision.allowed:
                logger.info(
                    "ENTRY BLOCKED (skill routing): %s [%s] — %s",
                    symbol,
                    decision.skill_name,
                    decision.reason,
                )
                continue
            if strategy is None:
                logger.warning(
                    "ENTRY BLOCKED (no strategy registered): %s [%s] wants strategy=%s",
                    symbol,
                    decision.skill_name,
                    decision.strategy_name,
                )
                continue
            # Applied here, before the veto-timeframe candle prefetch below,
            # so a bot whose override changes htf_veto has the right
            # timeframe's candles already fetched by the time _enter_for_bot
            # runs confirm() against it.
            candidates.append((decision, _effective_strategy(strategy, decision)))
        if symbol == "Step Index 200":
            logger.info(
                "TRACE _try_enter candidates symbol=%s timeframe=%s n_candidates=%d",
                symbol, timeframe, len(candidates),
            )
        if not candidates:
            return

        # Fetched once per symbol per candle close — every bot on this
        # symbol evaluates against the same bars/spread, so N bots cost the
        # same one round trip a single bot would. The context carries the
        # closed entry timeframe, each candidate strategy's own confirmation
        # timeframes, and each candidate's own HTF-veto timeframe (the one
        # immediately above its entry_timeframe).
        timeframes = dict.fromkeys(
            (
                timeframe,
                *(
                    tf
                    for _, strategy in candidates
                    for tf in strategy.spec.confirmation_timeframes
                ),
                *(
                    veto_tf
                    for _, strategy in candidates
                    if (veto_tf := _veto_timeframe(strategy)) is not None
                ),
            )
        )
        try:
            candles_by_tf = {
                tf: await self._market_data.get_candles(symbol, Timeframe(tf), self._context_bars)
                for tf in timeframes
            }
            info = await self._market_data.get_symbol_info(symbol)
        except MarketDataUnavailable as exc:
            logger.warning("ENTRY SKIPPED (no market data): %s — %s", symbol, exc)
            return

        # Fetched once per symbol per candle close, alongside `info` —
        # balance only changes on a realized close (a `close_on_opposite_
        # signal` exit), never from opening a position, so N candidate bots
        # reuse this one value instead of each round-tripping the gateway
        # (+ keyring read) for it. `_enter_for_bot` re-fetches it, but only
        # when `_close_opposite_position` actually closed something.
        balance = await self._current_balance()

        ctx = self._context_builder(symbol, candles_by_tf, info.spread_points)
        for decision, strategy in candidates:
            # Minted per candidate, before `_enter_for_bot` knows whether it
            # will actually emit a signal (OBSERVABILITY_PLAN.md Phase 5):
            # `bind_signal_id` has to be active before the `SIGNAL:` log line
            # inside `_enter_for_bot` runs, or that line wouldn't carry it.
            # Harmless when the candidate emits nothing — the id is simply
            # never referenced or persisted. Scoped to exactly this call (not
            # the whole `_try_enter` pass) so it can never leak onto the next
            # candidate or, via `on_candle_closed`'s outer span, the next
            # candle's position-management pass.
            signal_id = uuid.uuid4().hex
            with bind_signal_id(signal_id):
                balance = await self._enter_for_bot(
                    symbol, decision, strategy, ctx, info, now, balance, signal_id
                )

    async def _record_decision(
        self,
        *,
        signal_id: str,
        decision: SkillDecision,
        strategy: Strategy,
        symbol: str,
        signal: Signal,
        price: float,
        created_at: datetime,
        regime: EntryRegime | None,
    ) -> None:
        if self._signal_decisions is None:
            return
        await self._signal_decisions.record(
            signal_id=signal_id,
            bot=decision.skill_name,
            strategy=strategy.spec.name,
            symbol=symbol,
            timeframe=strategy.spec.entry_timeframe,
            direction=signal.direction.value,
            price=price,
            created_at=created_at,
            reason=signal.reason,
            confidence=signal.confidence,
            **_regime_kwargs(regime),
        )

    async def _record_outcome(
        self,
        signal_id: str,
        outcome: str,
        *,
        base_reason: str,
        explanation: str,
        checks: tuple[DecisionCheck, ...] = (),
    ) -> None:
        """Stamps a decision's terminal outcome, appending the gate's own
        explanation to the strategy's reason the same way the legacy
        log-scraper did (` — <explanation>`), so the chart's tooltip text is
        unchanged by the move off log parsing. `checks` are the structured
        numbers the failing gate saw (Phase 2), appended to whatever earlier
        gates already recorded.

        Also increments the `tradingbot_signal_outcomes_total` metric
        (OBSERVABILITY_PLAN.md Phase 5) in the same closed vocabulary the
        veto funnel uses (`SIGNAL_OUTCOMES`), independent of whether a
        decision-trail sink is wired — a unit test building a bare
        `TradeEngine` still gets accurate metrics even with
        `signal_decisions=None`."""
        record_signal_outcome(outcome)
        if self._signal_decisions is None:
            return
        reason = base_reason if explanation in base_reason else f"{base_reason} — {explanation}"
        await self._signal_decisions.record_outcome(
            signal_id, outcome, reason=reason, checks=checks
        )

    async def _record_checks(self, signal_id: str, *checks: DecisionCheck) -> None:
        """Records gates this signal **passed**, leaving the outcome alone —
        so a filled (or later-vetoed) signal's trail shows every gate it
        cleared and by how much, not only the one that stopped it."""
        if self._signal_decisions is None or not checks:
            return
        await self._signal_decisions.record_checks(signal_id, tuple(checks))

    async def _capture_order_book_safely(self, signal_id: str, symbol: str) -> None:
        """Runs as a fire-and-forget task (see the `asyncio.create_task` call
        site above) — this is what makes "order-book capture never crashes
        the trade loop" true even if `OrderBookCapturePort.capture`'s own
        never-raises contract is ever violated by a future change. Belt and
        suspenders: `OrderBookCaptureService.capture` already swallows
        `OrderBookUnavailable` and empty-book cases itself. Only ever
        scheduled when `self._order_book_capture is not None` (checked at
        the `asyncio.create_task` call site)."""
        try:
            await self._order_book_capture.capture(signal_id=signal_id, symbol=symbol)  # type: ignore[union-attr]
        except Exception:
            logger.exception(
                "order-book capture task failed for signal_id=%s symbol=%s", signal_id, symbol
            )

    async def _enter_for_bot(
        self,
        symbol: str,
        decision: SkillDecision,
        strategy: Strategy,
        ctx: MarketContext,
        info: SymbolInfo,
        now: datetime,
        balance: float | None,
        signal_id: str,
    ) -> float | None:
        # Fetched fresh per bot (not hoisted above the candidates loop) so a
        # bot later in this same candle sees the position(s) an earlier bot
        # on the same symbol just opened.
        open_positions = await self._order_service.get_positions()

        # This bot's own open position on `symbol` (matched by magic, same
        # rule `_close_opposite_position` uses), handed to the strategy via
        # `ctx.own_position` so it can reason about an in-flight trade —
        # e.g. to emit an `ExitDecision`. `None`/unchanged `ctx` when there
        # isn't one, so a strategy that never reads this field sees exactly
        # what it always has.
        own_position = next(
            (p for p in open_positions if p.symbol == symbol and p.magic == decision.magic),
            None,
        )
        bot_ctx = (
            replace(
                ctx,
                own_position=PositionSnapshot(
                    direction=Direction.BUY if own_position.side is Side.BUY else Direction.SELL,
                    entry_price=own_position.open_price,
                    sl=own_position.sl,
                    tp=own_position.tp,
                    opened_at=own_position.open_time,
                ),
            )
            if own_position is not None
            else ctx
        )

        # Evaluated ahead of the pretrade gate (unlike previously) so a
        # `close_on_opposite_signal` strategy can free up its own slot below
        # before the max-open-positions cap is checked against the count.
        if symbol == "Step Index 200":
            logger.info(
                "TRACE _enter_for_bot pre-evaluate symbol=%s strategy=%s m5_bars=%s own_position=%s",
                symbol,
                strategy.spec.name,
                len(bot_ctx.candles.get("M5", [])) if bot_ctx.candles.get("M5") is not None else None,
                bot_ctx.own_position,
            )
        try:
            signal_res = strategy.evaluate(bot_ctx)
        except Exception:
            logger.exception("TRACE _enter_for_bot evaluate raised symbol=%s strategy=%s", symbol, strategy.spec.name)
            raise
        if symbol == "Step Index 200":
            logger.info("TRACE _enter_for_bot post-evaluate symbol=%s signal_res=%r", symbol, signal_res)
        if signal_res is None:
            return balance
        raw = tuple(signal_res) if isinstance(signal_res, (list, tuple)) else (signal_res,)
        exit_decisions = tuple(item for item in raw if isinstance(item, ExitDecision))
        signals: tuple[Signal, ...] = tuple(item for item in raw if isinstance(item, Signal))

        if exit_decisions:
            open_positions, closed = await self._apply_strategy_exit_decisions(
                symbol, decision, exit_decisions, open_positions
            )
            if closed:
                # Same reasoning as the `close_on_opposite_signal` refetch
                # below: a realized close is the only thing that changes
                # account balance mid-candle, so downstream sizing (this bot's
                # or a later one's) must see the post-close value.
                balance = await self._current_balance()

        if not signals:
            return balance

        first_signal = signals[0]
        # Reference price for the signal as a whole: the side's tradable price
        # (ask to buy, bid to sell) off the SymbolInfo already fetched for this
        # candle. Logged so the decision trail — and the chart overlay built
        # from it — can place every signal, not only the ones that filled.
        signal_price = (
            info.ask if Side(first_signal.direction.value) is Side.BUY else info.bid
        )
        logger.info(
            "SIGNAL: %s %s @ %.5f (%d target position(s)) via strategy=%s skill=%s — %s",
            symbol,
            first_signal.direction.value,
            signal_price,
            len(signals),
            strategy.spec.name,
            decision.skill_name,
            first_signal.reason,
        )
        observe_signal_fired(bot=decision.skill_name, symbol=symbol)

        # Regime tagging (OBSERVABILITY_PLAN.md Phase 6) — a new, always-on
        # snapshot independent of the gated volatility-guard block further
        # down (which only runs when the guard is enabled and after earlier
        # gates already passed): every recorded signal needs a regime tag,
        # not only the ones that reach that gate. Some redundant computation
        # with the guard below is expected and fine — this is a tagging
        # concern, not a live risk-management decision, so it's computed
        # unconditionally here rather than reusing the guard's result. Named
        # `entry_regime` (not `regime`) to avoid shadowing the volatility
        # guard's own `regime: VolatilityRegime` local further down.
        tag_entry_frame = ctx.candles.get(strategy.spec.entry_timeframe)
        entry_regime = (
            compute_entry_regime(
                tag_entry_frame,
                now=now,
                volatility_config=self._volatility_config,
                regime_config=self._regime_config,
            )
            if tag_entry_frame is not None and not tag_entry_frame.empty
            else None
        )

        # Recorded before any gate runs, so a signal that is immediately
        # vetoed still exists in the trail with outcome="skipped" until its
        # terminal outcome lands below. `signal_id` was minted by the caller
        # (`_try_enter`), before this method ran, so `bind_signal_id` there
        # is already active and this log line above carries it too.
        await self._record_decision(
            signal_id=signal_id,
            decision=decision,
            strategy=strategy,
            symbol=symbol,
            signal=first_signal,
            price=signal_price,
            created_at=now,
            regime=entry_regime,
        )
        # Order-book capture (order_book/ Phase 5) — fire-and-forget, same
        # moment `entry_regime` is stamped and recorded above: every
        # evaluated signal gets a capture attempt, fired or vetoed, so the
        # snapshot exists regardless of how this signal's gates resolve.
        # `asyncio.create_task`, never `await`, is load-bearing here — a live
        # gateway round-trip must never sit between signal detection and
        # order placement.
        if self._order_book_capture is not None:
            asyncio.create_task(self._capture_order_book_safely(signal_id, symbol))

        if strategy.spec.close_on_opposite_signal:
            open_positions, closed = await self._close_opposite_position(
                symbol, decision, strategy, first_signal, open_positions
            )
            if closed:
                # The only thing in this loop that changes account balance
                # mid-candle — re-fetch so this bot's sizing below (and any
                # later bot's, since `balance` is threaded back to the
                # caller) sees the post-close balance instead of the value
                # hoisted once at the top of `_try_enter`.
                balance = await self._current_balance()

        pretrade = self._risk_manager.check_pretrade(len(open_positions), now)
        if not pretrade.approved:
            logger.info(
                "ENTRY BLOCKED (risk gate): %s [%s] — %s",
                symbol,
                decision.skill_name,
                pretrade.reason,
            )
            await self._record_outcome(
                signal_id,
                _PRETRADE_OUTCOMES.get(pretrade.code, "risk_rejected"),
                base_reason=first_signal.reason,
                explanation=pretrade.reason,
                checks=(
                    DecisionCheck(
                        name="open_positions",
                        value=float(len(open_positions)),
                        threshold=float(self._risk_manager.caps.max_open_positions),
                        comparison="<",
                        passed=pretrade.code != "max_positions",
                    ),
                ),
            )
            return balance
        await self._record_checks(
            signal_id,
            DecisionCheck(
                name="open_positions",
                value=float(len(open_positions)),
                threshold=float(self._risk_manager.caps.max_open_positions),
                comparison="<",
                passed=True,
            ),
        )

        veto_tf = _veto_timeframe(strategy)
        veto_timeframes = (veto_tf,) if veto_tf is not None else ()
        confirmed, veto_reason = confirm(first_signal.direction, ctx, veto_timeframes)
        if not confirmed:
            logger.info(
                "ENTRY BLOCKED (HTF veto): %s %s [%s] — %s",
                symbol,
                first_signal.direction.value,
                decision.skill_name,
                veto_reason,
            )
            await self._record_outcome(
                signal_id,
                "htf_veto",
                base_reason=first_signal.reason,
                explanation=veto_reason,
                checks=(_htf_check(passed=False),),
            )
            return balance
        await self._record_checks(signal_id, _htf_check(passed=True))

        if balance is None:
            # Carries the skill token and a " — <reason>" tail like every other
            # outcome line, so the per-bot signal-trail parsers can attribute it.
            logger.info(
                "ENTRY SKIPPED (no account connected): %s %s [%s] — no account balance "
                "available, cannot size the entry",
                symbol,
                first_signal.direction.value,
                decision.skill_name,
            )
            await self._record_outcome(
                signal_id,
                "skipped",
                base_reason=first_signal.reason,
                explanation="no account balance available, cannot size the entry",
            )
            return balance

        # Volatility guard (bot-agnostic, engine-level): classified off this
        # bot's own entry timeframe so an M1 scalp bot and an M15 swing bot
        # on the same symbol each react to their own candle's regime, not a
        # shared engine-wide one. Computed once for the whole `signals`
        # tuple (not per-signal) since an EXTREME regime blocks the entire
        # entry, not individual tiered TPs. Skipped entirely when the live
        # on/off switch (`set_volatility_guard_enabled`) is off — entries
        # then behave exactly as if `volatility_config` didn't exist.
        if self._volatility_guard_enabled:
            entry_frame = ctx.candles.get(strategy.spec.entry_timeframe)
            if entry_frame is not None and not entry_frame.empty:
                regime, percentile, _atr_value = latest_volatility_regime(
                    entry_frame["high"].to_numpy(),
                    entry_frame["low"].to_numpy(),
                    entry_frame["close"].to_numpy(),
                    atr_period=self._volatility_config.atr_period,
                    regime_lookback_bars=self._volatility_config.regime_lookback_bars,
                    low_percentile=self._volatility_config.low_percentile,
                    high_percentile=self._volatility_config.high_percentile,
                    extreme_percentile=self._volatility_config.extreme_percentile,
                )
            else:
                regime, percentile = VolatilityRegime.NORMAL, float("nan")

            if regime is VolatilityRegime.EXTREME:
                logger.info(
                    "ENTRY BLOCKED (volatility guard): %s %s [%s] — regime=EXTREME percentile=%.1f",
                    symbol,
                    first_signal.direction.value,
                    decision.skill_name,
                    percentile,
                )
                await self._record_outcome(
                    signal_id,
                    "volatility_guard",
                    base_reason=first_signal.reason,
                    explanation=f"regime=EXTREME percentile={percentile:.1f}",
                    checks=(_volatility_check(percentile, self._volatility_config, passed=False),),
                )
                return balance
            await self._record_checks(
                signal_id, _volatility_check(percentile, self._volatility_config, passed=True)
            )

            sl_mult, tp_mult = {
                VolatilityRegime.LOW: (
                    self._volatility_config.sl_multiplier_low,
                    self._volatility_config.tp_multiplier_low,
                ),
                VolatilityRegime.NORMAL: (
                    self._volatility_config.sl_multiplier_normal,
                    self._volatility_config.tp_multiplier_normal,
                ),
                VolatilityRegime.HIGH: (
                    self._volatility_config.sl_multiplier_high,
                    self._volatility_config.tp_multiplier_high,
                ),
            }[regime]
        else:
            sl_mult, tp_mult = 1.0, 1.0

        # Split risk across multiple targets so total risk per trade setup
        # remains aligned with user config

        pos_risk_multiplier = decision.risk_multiplier / len(signals)

        for idx, signal in enumerate(signals):
            if len(open_positions) + idx >= self._risk_manager._caps.max_open_positions:
                logger.info(
                    "ENTRY BLOCKED (max open positions cap reached): %s %s [%s] — TP%d of "
                    "%d skipped, %d open position(s) at cap %d",
                    symbol,
                    signal.direction.value,
                    decision.skill_name,
                    idx + 1,
                    len(signals),
                    len(open_positions) + idx,
                    self._risk_manager._caps.max_open_positions,
                )
                await self._record_outcome(
                    signal_id,
                    "max_positions",
                    base_reason=first_signal.reason,
                    explanation=(
                        f"TP{idx + 1} of {len(signals)} skipped, "
                        f"{len(open_positions) + idx} open position(s) at cap "
                        f"{self._risk_manager._caps.max_open_positions}"
                    ),
                    checks=(
                        DecisionCheck(
                            name="open_positions",
                            value=float(len(open_positions) + idx),
                            threshold=float(self._risk_manager._caps.max_open_positions),
                            comparison="<",
                            passed=False,
                        ),
                    ),
                )
                break

            side = Side(signal.direction.value)
            reference_price = info.ask if side is Side.BUY else info.bid
            sign = 1 if side is Side.BUY else -1
            sl_price = reference_price - sign * signal.sl_points * sl_mult
            tp_price = reference_price + sign * signal.tp_points * tp_mult

            sizing = self._risk_manager.size_position(
                balance=balance,
                sl_distance_price=abs(reference_price - sl_price),
                contract_size=info.contract_size,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                volume_step=info.volume_step,
                risk_multiplier=pos_risk_multiplier,
            )
            if not sizing.approved:
                logger.info(
                    # The TP index lives in the message body, not the prefix:
                    # both signal-trail parsers match the literal
                    # "ENTRY REJECTED (risk sizing):" prefix.
                    "ENTRY REJECTED (risk sizing): %s %s [%s] — TP%d: %s (balance=%.2f, "
                    "sl_distance=%.5f, risk_multiplier=%.2f)",
                    symbol,
                    side.value,
                    decision.skill_name,
                    idx + 1,
                    sizing.reason,
                    balance,
                    abs(reference_price - sl_price),
                    pos_risk_multiplier,
                )
                await self._record_outcome(
                    signal_id,
                    "risk_sizing",
                    base_reason=first_signal.reason,
                    explanation=f"TP{idx + 1}: {sizing.reason}",
                    checks=(
                        DecisionCheck(
                            name="position_volume",
                            value=sizing.volume,
                            threshold=info.volume_min,
                            comparison=">=",
                            passed=False,
                        ),
                    ),
                )
                continue
            logger.info(
                "SIZING OK (TP%d/%d): %s %s %.2f lots [%s] (balance=%.2f, risk_multiplier=%.2f)",
                idx + 1,
                len(signals),
                symbol,
                side.value,
                sizing.volume,
                decision.skill_name,
                balance,
                pos_risk_multiplier,
            )
            await self._record_checks(
                signal_id,
                DecisionCheck(
                    name="position_volume",
                    value=sizing.volume,
                    threshold=info.volume_min,
                    comparison=">=",
                    passed=True,
                ),
            )

            zone = signal.zone
            comment_text = (
                f"TP{idx+1}:{signal.reason}"[:29] if len(signals) > 1 else signal.reason[:29]
            )
            try:
                await self._order_service.open_position(
                    symbol,
                    side,
                    sizing.volume,
                    sl=sl_price,
                    tp=tp_price,
                    comment=comment_text,
                    strategy_version=f"{strategy.spec.name}:v{strategy.spec.version}",
                    skill=decision.skill_name,
                    magic=decision.magic,
                    max_spread_points=decision.max_spread_points,
                    reason=signal.reason,
                    confidence=signal.confidence,
                    # Lets the order service stamp this decision's fill /
                    # spread-veto / broker-reject outcome without the engine
                    # having to reach into the broker's own gates.
                    signal_id=signal_id,
                    # Emit end of the signal→fill latency span: the exact
                    # timestamp this signal's `SignalDecision` was recorded
                    # with above (OBSERVABILITY_PLAN.md Phase 3).
                    signal_emitted_at=now,
                    zone_kind=zone.kind.value if zone is not None else None,
                    zone_price_low=zone.price_low if zone is not None else None,
                    zone_price_high=zone.price_high if zone is not None else None,
                    zone_time_start=zone.time_start if zone is not None else None,
                    zone_time_end=zone.time_end if zone is not None else None,
                    zone_pattern=zone.pattern if zone is not None else None,
                    pattern=signal.pattern,
                    structure=tuple((p.label.value, p.price, p.time) for p in signal.structure),
                    indicators=tuple(
                        (r.name, r.value, r.threshold, r.comparison, r.passed)
                        for r in signal.indicators
                    ),
                    **_regime_kwargs(entry_regime),
                )
            except OrderRejected:
                continue  # spread/RR gate already logged the veto inside order_service
            self._risk_manager.record_trade_opened(now)
        return balance

    async def _apply_strategy_exit_decisions(
        self,
        symbol: str,
        decision: SkillDecision,
        exit_decisions: tuple[ExitDecision, ...],
        open_positions: list[Position],
    ) -> tuple[list[Position], bool]:
        """Routes every `ExitDecision` this bot's `evaluate()` just returned
        to `PositionManager.apply_strategy_action`, against every one of
        this bot's own open positions on `symbol` (matched by `magic`, same
        rule `_close_opposite_position` uses — usually a single position,
        but a multi-TP-leg bot can have more than one open at once).
        Mirrors `_close_opposite_position`'s return contract: `open_positions`
        has any closed ticket removed so the pretrade gate below sees the
        freed slot immediately, and `closed` tells the caller whether
        account balance needs re-fetching."""
        own_positions = [
            p for p in open_positions if p.symbol == symbol and p.magic == decision.magic
        ]
        if not own_positions:
            return open_positions, False
        closed_tickets: set[int] = set()
        for action in exit_decisions:
            for position in own_positions:
                if position.ticket in closed_tickets:
                    continue
                await self._position_manager.apply_strategy_action(position, action)
                if action.action is ExitActionKind.CLOSE:
                    closed_tickets.add(position.ticket)
        if not closed_tickets:
            return open_positions, False
        return [p for p in open_positions if p.ticket not in closed_tickets], True

    async def _close_opposite_position(
        self,
        symbol: str,
        decision: SkillDecision,
        strategy: Strategy,
        signal: Signal,
        open_positions: list[Position],
    ) -> tuple[list[Position], bool]:
        """For a `close_on_opposite_signal` strategy: closes this bot's own
        open position on `symbol` (matched by `magic`, so other bots' or
        manually-opened positions are untouched) when its side opposes the
        fresh `signal` — a signal-flip exit instead of waiting for
        SL/TP/time-stop. Returns `(open_positions, closed)`: `open_positions`
        has the closed ticket removed so the caller's pretrade gate sees the
        freed slot right away, in the same pass that opens the new position;
        `closed` is `True` only when a position was actually closed here —
        the caller uses it to decide whether account balance needs
        re-fetching, since a realized close is the only thing in this loop
        that changes it."""
        opposite_side = Side.SELL if signal.direction == Direction.BUY else Side.BUY
        position = next(
            (
                p
                for p in open_positions
                if p.symbol == symbol and p.magic == decision.magic and p.side is opposite_side
            ),
            None,
        )
        if position is None:
            return open_positions, False
        try:
            await self._order_service.close_position(position.ticket)
        except OrderRejected:
            logger.exception(
                "SIGNAL FLIP: failed to close ticket=%d %s ahead of new %s signal",
                position.ticket,
                symbol,
                signal.direction.value,
            )
            return open_positions, False
        logger.info(
            "SIGNAL FLIP: %s ticket=%d %s closed [%s] — new %s signal via strategy=%s",
            symbol,
            position.ticket,
            position.side.value,
            decision.skill_name,
            signal.direction.value,
            strategy.spec.name,
        )
        return [p for p in open_positions if p.ticket != position.ticket], True
