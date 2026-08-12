"""Candle streaming: poll the gateway on bar boundaries, emit CandleClosed.

This is the engine's clock (M5 entries, H1/H4/D1 confirmations) plus any
finer timeframe configured for charting (e.g. M1). Every configured
timeframe closes on the finest one's boundaries, so one wake-up shortly
after each such close covers everything — see `_seconds_until_next_poll`.
Every closed bar is also persisted, so history accumulates for backtests and
AI snapshots as a side effect of streaming.

Polling always covers the engine's traded universe (`configs/app.yaml:
symbols` at startup, plus anything `add_symbol()`-activated live since —
see `SkillAssignmentService.assign()`); `watch`/`unwatch` extend that set
for symbols a chart is browsing on demand (see `market_data.api.ws`), so
any symbol currently open on a chart gets live `candle_closed` updates
too, not just the ones the bot actually trades — that ref-counted set
decays to zero and stops once no chart tab has it open, unlike the
permanent set `add_symbol()` grows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.application.history import CandleHistoryService
from src.market_data.domain.models import Candle, MarketDataUnavailable, Timeframe
from src.market_data.ports.broadcast import MarketBroadcastPort
from src.market_data.ports.market_data import MarketDataPort
from src.shared.events.bus import EventBus
from src.shared.events.definitions import CandleClosed
from src.shared.logging.account_context import current_account_id

logger = logging.getLogger(__name__)

# `poll_lookback_for` scales to roughly this many seconds of buffer on every
# timeframe, floored/capped below — a single flat bar count across all 9
# timeframes (the old `POLL_LOOKBACK = 20`) was undersized for M1/M5 (a
# 20-minute buffer on a 1-minute timeframe — the direct mechanism behind the
# gap in OPTIMIZATION_CHECKLIST.md §1) and needlessly oversized for D1/W1/MN
# (20 D1 bars = 20 days fetched every poll).
_LOOKBACK_BUFFER_S = 2 * 3600
_MIN_POLL_LOOKBACK = 6
# Matches the engine's known context_bars hard cap (strategy lookback params
# past ~200 bars silently never fire) — no reason for a poll fetch to ever
# ask for more than that.
_MAX_POLL_LOOKBACK = 200
_BOUNDARY_GRACE_S = 2.0
# Bounds how many symbol/timeframe gateway fetches `poll_once` runs at once.
# At a shared bar boundary (e.g. top of the hour) every configured
# symbol/timeframe needs fetching in the same tick — sequential awaits turned
# that into 25-45 back-to-back round trips, so a degraded gateway (each call
# up to the 8s read timeout) could stretch one poll tick long enough to delay
# the *next* poll for everything, including the engine's own entry clock.
_MAX_CONCURRENT_FETCHES = 8


def poll_lookback_for(timeframe: Timeframe) -> int:
    """Bars fetched per poll for `timeframe` — enough to bridge a short
    gateway outage without re-downloading history (a longer one is healed
    by the explicit backfill in `poll_once` below, or at startup by
    `CandleHistoryService.reconcile_gaps`, which also uses this to size its
    own gap threshold per timeframe)."""
    return max(
        _MIN_POLL_LOOKBACK,
        min(_MAX_POLL_LOOKBACK, round(_LOOKBACK_BUFFER_S / timeframe.seconds)),
    )


class CandleStreamService:
    def __init__(
        self,
        market_data: MarketDataPort,
        repository: CandleRepository,
        event_bus: EventBus,
        broadcaster: MarketBroadcastPort,
        symbols: list[str],
        timeframes: list[Timeframe],
        candle_history: CandleHistoryService | None = None,
        recent_candle_cache: dict[tuple[str, Timeframe], tuple[float, Candle]] | None = None,
        account_id: str = "default",
    ) -> None:
        self._market_data = market_data
        self._repository = repository
        self._event_bus = event_bus
        self._broadcaster = broadcaster
        self._symbols = symbols
        self._timeframes = timeframes
        self._account_id = account_id
        # Used by `poll_once` to heal a gap left by a gateway/broker outage
        # that happens *without* a process restart (see its `previous`
        # gap check below) — `CandleHistoryService.reconcile_gaps` only ever
        # runs once, at startup, and can't see those. Optional so lightweight
        # test fixtures that don't care about gap-healing can omit it.
        self._candle_history = candle_history
        # Shared with `LiveCandleService` (wired in `container.py`) so its
        # own ~1.5s poll can reuse the bar this service just fetched instead
        # of hitting the gateway again for the same symbol/timeframe right
        # at a bar close, when both would otherwise want the same latest bar
        # within a second or two of each other (OPTIMIZATION_CHECKLIST.md
        # §2). Keyed by (symbol, timeframe) -> (`time.monotonic()` fetched
        # at, the most recent bar from that fetch, closed or still forming).
        # Optional so tests that don't care about this can omit it.
        self._recent_candle_cache = recent_candle_cache
        self._last_emitted: dict[tuple[str, Timeframe], datetime] = {}
        self._task: asyncio.Task[None] | None = None
        self._gateway_ok = True
        # Ref-counted: several chart subscribers can watch the same ad-hoc
        # symbol at once (e.g. two browser tabs), so it stays active until
        # all of them unwatch it. Configured symbols are always active and
        # never go through this map.
        self._extra_refcounts: dict[str, int] = {}

    @property
    def active_symbols(self) -> list[str]:
        """Configured symbols plus any ad-hoc symbols currently watched."""
        return [*self._symbols, *self._extra_refcounts]

    def watch(self, symbol: str) -> None:
        """Start streaming `symbol` on demand (chart browsing a symbol not
        yet permanently active). No-op for symbols already in the permanent
        set (`_symbols`, see `add_symbol()`) — those are always active."""
        if symbol in self._symbols:
            return
        if self._extra_refcounts.get(symbol, 0) == 0:
            logger.info("candle stream: now watching ad-hoc symbol %s", symbol)
        self._extra_refcounts[symbol] = self._extra_refcounts.get(symbol, 0) + 1

    def add_symbol(self, symbol: str) -> bool:
        """Permanently activates `symbol` for automated trading — used by
        `SkillAssignmentService.assign()` when a symbol is newly routed to a
        strategy, as opposed to `watch()`'s ref-counted, chart-only
        membership that decays to zero once no chart tab has it open.

        Appends to `_symbols` in place rather than rebinding: `Container.
        symbols` (e.g. the candle-backfill endpoint's default set) is the
        same list object, and relies on the alias holding.

        Idempotent — returns whether this call actually added it, so a
        retried `assign()` for an already-active symbol is a safe no-op.
        Stays fully synchronous (no `await` inside) so a single call can't
        be interleaved by another coroutine mid-mutation."""
        if symbol in self._symbols:
            return False
        self._symbols.append(symbol)
        # Was ad-hoc chart-watched — now folded into permanent status, not
        # double-tracked. A later unwatch() from that chart tab no-ops
        self._extra_refcounts.pop(symbol, None)
        logger.info("candle stream: symbol %s permanently activated for automated trading", symbol)
        return True

    def remove_symbol(self, symbol: str) -> bool:
        """Removes `symbol` from permanent automated trading status. 
        If it's not in the list, returns False."""
        if symbol not in self._symbols:
            return False
        self._symbols.remove(symbol)
        logger.info("candle stream: symbol %s permanently deactivated", symbol)
        return True

    def unwatch(self, symbol: str) -> None:
        """Stop streaming `symbol` once nothing is watching it anymore."""
        if symbol not in self._extra_refcounts:
            return
        self._extra_refcounts[symbol] -= 1
        if self._extra_refcounts[symbol] > 0:
            return
        del self._extra_refcounts[symbol]
        logger.info("candle stream: no longer watching ad-hoc symbol %s", symbol)
        for timeframe in self._timeframes:
            self._last_emitted.pop((symbol, timeframe), None)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="candle-stream")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        current_account_id.set(self._account_id)
        while True:
            try:
                await self.poll_once()
                if not self._gateway_ok:
                    self._gateway_ok = True
                    logger.info("gateway reachable again — candle stream resumed")
            except MarketDataUnavailable as exc:
                if self._gateway_ok:
                    self._gateway_ok = False
                    logger.warning("candle stream paused, gateway unavailable: %s", exc)
            except Exception:
                logger.exception("candle stream poll failed")
            await asyncio.sleep(self._seconds_until_next_poll())

    async def poll_once(self, now: datetime | None = None) -> list[Candle]:
        """Fetch, persist, and emit newly closed bars. Returns what was emitted."""
        now = now or datetime.now(UTC)
        emitted: list[Candle] = []

        pending: list[tuple[str, Timeframe]] = [
            (symbol, timeframe)
            for symbol in self.active_symbols
            for timeframe in self._timeframes
            # already saw the latest closed bar of this timeframe
            if self._last_emitted.get((symbol, timeframe)) != timeframe.last_closed_open(now)
        ]
        if not pending:
            return emitted

        semaphore = asyncio.Semaphore(min(_MAX_CONCURRENT_FETCHES, len(pending)))

        async def fetch(symbol: str, timeframe: Timeframe) -> list[Candle]:
            async with semaphore:
                return await self._market_data.get_candles(
                    symbol, timeframe, poll_lookback_for(timeframe)
                )

        # Fetches run concurrently (bounded) since they're independent I/O —
        # only the persistence/emit side below, which mutates shared state
        # (`_last_emitted`, event/broadcast ordering), stays sequential, in
        # the same symbol/timeframe order as before this change.
        results = await asyncio.gather(
            *(fetch(symbol, timeframe) for symbol, timeframe in pending),
            return_exceptions=True,
        )

        # A single symbol/timeframe hitting the gateway's read timeout used
        # to abort the whole tick (the old sequential loop just propagated
        # the exception). With concurrent fetches that's needlessly fragile —
        # every *other* pair's already-fetched candles still get processed;
        # the first `MarketDataUnavailable` is re-raised at the end so
        # `_run()` still flags the gateway as down and logs same as before.
        first_error: MarketDataUnavailable | None = None
        for (symbol, timeframe), result in zip(pending, results, strict=True):
            if isinstance(result, MarketDataUnavailable):
                if first_error is None:
                    first_error = result
                continue
            if isinstance(result, BaseException):
                raise result
            candles = result
            key = (symbol, timeframe)
            if self._recent_candle_cache is not None and candles:
                self._recent_candle_cache[key] = (time.monotonic(), candles[-1])
            closed = [c for c in candles if c.is_closed(now)]
            if not closed:
                continue
            await asyncio.to_thread(self._repository.upsert_many, closed, self._account_id)
            previous = self._last_emitted.get(key)
            self._last_emitted[key] = closed[-1].time
            if previous is None:
                continue  # startup baseline
            # `closed` only reaches back `poll_lookback_for(timeframe)` bars — a gateway
            # outage (network blip, Wine/terminal hiccup) longer than that
            # leaves a hole between `previous` and `closed[0]` that no
            # amount of normal polling will ever backfill on its own, even
            # though the stream itself never stopped running (so the
            # startup-only `CandleHistoryService.reconcile_gaps` never
            # gets a chance to see it either). Heal it here, right at the
            # moment it's detected, before emitting the tail as usual.
            if (
                self._candle_history is not None
                and (closed[0].time - previous).total_seconds() > timeframe.seconds
            ):
                logger.warning(
                    "candle gap detected for %s %s: last seen %s, resumed at %s — backfilling",
                    symbol,
                    timeframe.value,
                    previous.isoformat(),
                    closed[0].time.isoformat(),
                )
                try:
                    await self._candle_history.backfill(symbol, timeframe, 1000, start=previous)
                except MarketDataUnavailable:
                    logger.warning(
                        "candle gap backfill failed for %s %s: gateway unavailable",
                        symbol,
                        timeframe.value,
                    )
            # Emit every bar that closed since the last poll, not just the
            # newest — with a finer timeframe like M1 in the mix, more than
            # one bar can close between two polls and none should be lost.
            new_bars = [c for c in closed if c.time > previous]
            for candle in new_bars:
                logger.info(
                    "candle closed %s %s @ %s close=%s spread=%s",
                    symbol,
                    timeframe.value,
                    candle.time.isoformat(),
                    candle.close,
                    candle.spread_points,
                )
                await self._event_bus.publish(
                    CandleClosed(symbol=symbol, timeframe=timeframe.value)
                )
                await self._broadcaster.broadcast(
                    {"type": "candle_closed", "candle": candle_message(candle)}
                )
                emitted.append(candle)

        if first_error is not None:
            raise first_error
        return emitted

    def _seconds_until_next_poll(self, now: datetime | None = None) -> float:
        now = now or datetime.now(UTC)
        finest = min(tf.seconds for tf in self._timeframes)
        until_boundary = finest - (now.timestamp() % finest)
        return max(until_boundary + _BOUNDARY_GRACE_S, 5.0)


def candle_message(candle: Candle) -> dict[str, float | int | str]:
    """Wire shape for WS/REST — `time` in epoch seconds (lightweight-charts native)."""
    return {
        "symbol": candle.symbol,
        "timeframe": candle.timeframe.value,
        "time": int(candle.time.timestamp()),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "tick_volume": candle.tick_volume,
        "spread_points": candle.spread_points,
    }
