"""History use cases: serve chart/backtest reads, backfill the DB from the gateway."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.replay import SymbolSpec
from src.market_data.adapters.symbol_spec_repository import SymbolSpecRepository
from src.market_data.domain.gaps import (
    CandleGap,
    GapRepairReport,
    bars_between,
    find_gaps,
    is_weekend_closure,
)
from src.market_data.domain.models import Candle, MarketDataUnavailable, Timeframe
from src.market_data.ports.market_data import MarketDataPort

logger = logging.getLogger(__name__)

# Widest gap between consecutive stored bars that's explainable by a normal
# weekend/holiday session closure (Friday close to Sunday/Monday reopen,
# plus slack for holidays) rather than a hole left by a gateway/DB outage.
_MAX_SESSION_GAP = timedelta(days=3)

# Bounds `repair_gaps`'s per-gap backward paging the same way `backfill`'s
# 2000-page cap bounds its own loop: a hole wider than this many pages of
# `count` bars isn't a stream outage, it's history that was never downloaded
# at all — `POST /market-data/backfill` is the tool for that.
_MAX_REPAIR_PAGES = 20

# `scan_gaps` only reports a hole at the very start/end of the scanned range
# once it's at least this many bars wide: the stored history legitimately
# trails the live stream by a bar or two (the forming bar is persisted on the
# stream's own ~1/minute cadence), and flagging that as damage would leave a
# permanent "gaps found" badge on every healthy chart.
_EDGE_GAP_TOLERANCE_BARS = 2


def _gap(timeframe: Timeframe, start: datetime, end: datetime) -> CandleGap:
    """A range-edge gap — `find_gaps` builds the ones *between* stored bars."""
    return CandleGap(
        start=start,
        end=end,
        missing_bars=bars_between(timeframe, start, end),
        weekend=is_weekend_closure(timeframe, start, end),
    )


def _has_internal_gap(candles: list[Candle], timeframe: Timeframe) -> bool:
    """True if `candles` (oldest first, as `get_latest`/`get_before` return
    them) is missing bars somewhere in the middle — `get_latest`'s plain
    `ORDER BY time DESC LIMIT` and `get_before`'s equivalent can't see this on
    their own, since both only look at the newest end of the window. Skipped
    for W1/MN, whose bar spacing already spans calendar gaps by design."""
    if timeframe in (Timeframe.W1, Timeframe.MN) or len(candles) < 2:
        return False
    return any(
        later.time - earlier.time > _MAX_SESSION_GAP
        for earlier, later in zip(candles, candles[1:], strict=False)
    )


class CandleHistoryService:
    def __init__(
        self,
        market_data: MarketDataPort,
        repository: CandleRepository,
        symbol_spec_repository: SymbolSpecRepository | None = None,
        account_id: str = "default",
    ) -> None:
        self._market_data = market_data
        self._repository = repository
        self._symbol_spec_repository = symbol_spec_repository
        self._account_id = account_id

    async def get_candles(
        self, symbol: str, timeframe: Timeframe, count: int, before: datetime | None = None
    ) -> list[Candle]:
        """Serves straight from the local DB when it's already caught up —
        `CandleStreamService` keeps every symbol a chart currently has open
        polled and persisted roughly once a minute (see its `watch()`), so a
        chart that's been sitting on a symbol for a bit can switch timeframe
        as a local read instead of paying for a live Wine/MT5 round trip on
        every click. Only a cold open (a symbol/timeframe never streamed
        yet, or the DB genuinely behind) falls through to the gateway;
        gateway failures still fall back to whatever's in the DB either way.
        `before` pages further back than the most recent `count` bars —
        see `MarketDataPort.get_candles`. Either branch also checks the
        cached window for an internal gap (`_has_internal_gap`) before
        trusting it — a hole from a mid-session outage wouldn't otherwise be
        visible to `get_latest`/`get_before`'s plain `ORDER BY ... LIMIT`."""
        cached: list[Candle] | None = None
        if self._repository is not None:
            if before is None:
                cached = await asyncio.to_thread(
                    self._repository.get_latest, symbol, timeframe, count, self._account_id
                )
                if (
                    cached
                    and cached[-1].time == timeframe.last_closed_open(datetime.now(UTC))
                    and not _has_internal_gap(cached, timeframe)
                ):
                    return cached
            else:
                cached = await asyncio.to_thread(
                    self._repository.get_before,
                    symbol,
                    timeframe,
                    before,
                    count,
                    self._account_id,
                )
                if len(cached) >= count and not _has_internal_gap(cached, timeframe):
                    return cached
        try:
            return await self._market_data.get_candles(symbol, timeframe, count, before)
        except MarketDataUnavailable:
            logger.info("gateway unavailable — serving %s %s from DB", symbol, timeframe.value)
            if cached is not None:
                return cached
            raise

    async def backfill(
        self, symbol: str, timeframe: Timeframe, count: int, start: datetime | None = None
    ) -> int:
        """Download bars and persist them. Without `start`, fetches the most
        recent `count` bars (forming bar included; it gets overwritten on the
        next poll) — this is what seeds `GET /candles`'s DB fallback. With
        `start`, pages backward from now in `count`-sized chunks via the
        `before` cursor (see `GatewayMarketData.get_candles`) until the oldest
        bar fetched reaches `start` or the broker's history runs out — a
        single gateway call is capped at `count` (<=5000) bars, so this is
        what lets `POST /backtest/run` replay a multi-month/year range.
        Returns total bars stored."""
        if start is None:
            candles = await self._market_data.get_candles(symbol, timeframe, count)
            stored = await asyncio.to_thread(
                self._repository.upsert_many, candles, self._account_id
            )
            logger.info("backfilled %s bars for %s %s", stored, symbol, timeframe.value)
            return stored

        total = 0
        before: datetime | None = None
        # Bounds the loop even if the gateway ever misbehaves (e.g. returns a
        # full page without moving `before` backward) — 2000 pages covers
        # decades of M1 history, far beyond any real request.
        for _ in range(2000):
            page = await self._market_data.get_candles(symbol, timeframe, count, before)
            if not page:
                break
            total += await asyncio.to_thread(self._repository.upsert_many, page, self._account_id)
            oldest = page[0].time
            if oldest <= start or len(page) < count:
                break
            before = oldest
        logger.info(
            "backfilled %s bars for %s %s back to %s", total, symbol, timeframe.value, start
        )
        return total

    async def scan_gaps(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> list[CandleGap]:
        """Holes in *stored* history for bar opens in `[start, end]` — the
        chart's currently loaded window, or any range a backtest is about to
        replay. See `domain/gaps.py` for why these matter: a hole is invisible
        to `get_latest`/`get_before` (and therefore to the chart, which draws
        straight across it), yet every indicator, zone detector and backtest
        computed over that window silently treats bars that are hours or days
        apart as adjacent.

        Reports the range's leading/trailing edges too, not just holes between
        stored bars, so a window with no local history at all comes back as
        one repairable gap instead of a misleading "no gaps". Both edges get a
        2-bar tolerance, since the DB legitimately trails the live stream by a
        bar or so. W1/MN are skipped — see `find_gaps`."""
        if self._repository is None or timeframe in (Timeframe.W1, Timeframe.MN):
            return []
        stored = await asyncio.to_thread(
            self._repository.get_range,
            symbol,
            timeframe,
            start,
            end + timedelta(seconds=1),  # get_range's `end` is exclusive
            self._account_id,
        )
        if not stored:
            missing = bars_between(timeframe, start, end)
            if missing < _EDGE_GAP_TOLERANCE_BARS:
                return []
            return [_gap(timeframe, start, end)]
        gaps = find_gaps(stored, timeframe)
        if bars_between(timeframe, start, stored[0].time) >= _EDGE_GAP_TOLERANCE_BARS:
            gaps.insert(0, _gap(timeframe, start, stored[0].time))
        trailing_start = timeframe.close_of(stored[-1].time)
        if bars_between(timeframe, trailing_start, end) >= _EDGE_GAP_TOLERANCE_BARS:
            gaps.append(_gap(timeframe, trailing_start, end))
        return gaps

    async def repair_gaps(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        count: int = 1000,
        include_weekend: bool = False,
    ) -> GapRepairReport:
        """Re-download every hole `scan_gaps` finds in `[start, end]` and
        persist it, then rescan so the caller learns which holes actually
        closed. This is the "the chart has a hole in it" button's backend: the
        gateway is asked only for the missing stretches (one backward-paged
        request per hole via the `before` cursor, `_MAX_REPAIR_PAGES` deep),
        not for the whole window, so repairing a 40-bar hole inside a
        60k-bar session-replay range costs one call rather than a dozen.

        Weekend closures are skipped by default — the broker was never going
        to have those bars, and asking wastes a Wine/MT5 round trip per hole.
        Pass `include_weekend` to ask anyway (some synthetic/crypto symbols do
        trade through the weekend, and their "weekend" holes are real).

        Whatever `scan_gaps` still reports afterwards is in
        `GapRepairReport.remaining`: the broker itself has no bars there, so
        it's a genuine closure (holiday, halt, symbol listed later), not a
        hole in our copy — callers should say so rather than invite a retry.
        Raises `MarketDataUnavailable` if the gateway is unreachable."""
        found = await self.scan_gaps(symbol, timeframe, start, end)
        targets = [gap for gap in found if include_weekend or not gap.weekend]
        downloaded = 0
        for gap in targets:
            downloaded += await self._download_between(symbol, timeframe, gap.start, gap.end, count)
        rescanned = await self.scan_gaps(symbol, timeframe, start, end) if targets else found
        remaining = [gap for gap in rescanned if include_weekend or not gap.weekend]
        repaired = [
            gap
            for gap in targets
            if not any(left.start < gap.end and gap.start < left.end for left in remaining)
        ]
        # Counted in bars, not holes, and against the gaps actually targeted
        # (weekend closures are in `found` but were never asked for). A hole
        # that merely shrank still recovered real bars — an outage running
        # into the broker's nightly break gives back its trading hours and
        # keeps the closure, which `repaired` alone would report as nothing
        # gained.
        recovered = max(
            sum(gap.missing_bars for gap in targets) - sum(gap.missing_bars for gap in remaining),
            0,
        )
        logger.info(
            "gap repair %s %s [%s..%s]: %s found, %s closed, %s remaining, "
            "%s bars recovered from %s downloaded",
            symbol,
            timeframe.value,
            start,
            end,
            len(found),
            len(repaired),
            len(remaining),
            recovered,
            downloaded,
        )
        return GapRepairReport(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            found=found,
            repaired=repaired,
            remaining=remaining,
            bars_downloaded=downloaded,
            bars_recovered=recovered,
        )

    async def _download_between(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime, count: int
    ) -> int:
        """Page backward from `end` until the fetched bars reach `start` (or
        the broker's history runs out), persisting each page. Same `before`
        cursor contract as `backfill`; the overlap either side of the hole is
        deliberate and harmless — `upsert_many` overwrites in place."""
        total = 0
        before: datetime | None = end
        for _ in range(_MAX_REPAIR_PAGES):
            page = await self._market_data.get_candles(symbol, timeframe, count, before)
            if not page:
                break
            total += await asyncio.to_thread(self._repository.upsert_many, page, self._account_id)
            oldest = page[0].time
            if oldest <= start or len(page) < count:
                break
            before = oldest
        return total

    async def reconcile_gaps(
        self,
        symbols: list[str],
        timeframes: list[Timeframe],
        poll_lookback: Callable[[Timeframe], int],
        count: int = 1000,
        now: datetime | None = None,
    ) -> dict[str, int]:
        """Startup gap reconciliation (OPTIMIZATION_CHECKLIST.md §1):
        `CandleStreamService.poll_once` only re-fetches `poll_lookback(timeframe)`
        bars per tick, so a downtime longer than that leaves a permanent hole
        between the last bar stored before shutdown and `now -
        poll_lookback(timeframe) bars` once streaming resumes — `poll_once`
        never looks further back than that on its own. Call this once at
        startup, before `candle_stream.start()`, so any such hole gets paged
        in from the gateway first. No-ops per symbol/timeframe when there's
        no stored history yet (a genuinely cold symbol seeds itself via the
        normal `get_candles`/manual-backfill paths instead) or when the gap
        is within `poll_lookback(timeframe)` bars (nothing missing). Swallows
        `MarketDataUnavailable` per pair so one unreachable symbol/timeframe
        (or a gateway that's down at startup) doesn't block the rest or
        startup itself — streaming will just retry on its own schedule.
        Returns bars backfilled per '<symbol>:<timeframe>' key, mirroring
        `POST /market-data/backfill`'s response shape."""
        if self._repository is None:
            return {}
        now = now or datetime.now(UTC)
        stored: dict[str, int] = {}
        for symbol in symbols:
            for timeframe in timeframes:
                latest = await asyncio.to_thread(
                    self._repository.get_latest, symbol, timeframe, 1, self._account_id
                )
                if not latest:
                    continue
                last_stored = latest[-1].time
                gap_seconds = (now - last_stored).total_seconds()
                if gap_seconds <= poll_lookback(timeframe) * timeframe.seconds:
                    continue
                try:
                    bars = await self.backfill(symbol, timeframe, count, start=last_stored)
                except MarketDataUnavailable:
                    logger.warning(
                        "gap reconciliation skipped for %s %s: gateway unavailable",
                        symbol,
                        timeframe.value,
                    )
                    continue
                stored[f"{symbol}:{timeframe.value}"] = bars
        return stored

    async def sync_symbol_spec(self, symbol: str) -> None:
        """Fetches `symbol`'s static broker facts (point, digits, stops_level,
        contract_size, volume min/max/step) from the live gateway and persists
        them, so `run_backtest` can replay this symbol offline afterward
        without a hand-authored `configs/symbols/<symbol>.yaml` — see
        `SymbolSpecRepository`. No-ops with a warning if this service wasn't
        wired with a `symbol_spec_repository` (e.g. a lightweight test
        container that never calls it)."""
        if self._symbol_spec_repository is None:
            logger.warning(
                "sync_symbol_spec(%s): no symbol_spec_repository wired — skipping", symbol
            )
            return
        info = await self._market_data.get_symbol_info(symbol)
        spec = SymbolSpec(
            point=info.point,
            digits=info.digits,
            stops_level=info.stops_level,
            contract_size=info.contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
        )
        await asyncio.to_thread(self._symbol_spec_repository.upsert, symbol, spec, self._account_id)
        logger.info("synced symbol spec for %s", symbol)
