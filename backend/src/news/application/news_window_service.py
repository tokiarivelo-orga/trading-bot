"""Economic calendar refresh + active-window detection (§6.7, §8, Phase 8).

Two cadences, deliberately separate:
  - `refresh()` hits the calendar source (network I/O) every
    `configs/news.yaml: calendar.refresh_minutes` — calendars change rarely,
    so there is no reason to hammer the source more often.
  - `_check_transitions()` re-evaluates the (already-cached) events every
    `_TRANSITION_CHECK_INTERVAL_S` to publish `NewsWindowEntered`/
    `NewsWindowExited` close to real time, with zero extra network calls.

`active_window_for()` is synchronous and reads only the last-fetched cache —
it's called from `NewsSkillSelector.select_all()` on the engine's hot path
(every M5 close) and must never block on I/O.

Phase 6 Parts B/C: `refresh()` additionally upserts every fetched batch into
the durable `news_events` table via `NewsEventRepository` (`self._events`
itself is untouched — still the in-memory, zero-I/O cache
`active_window_for()` reads), and `_check_transitions()` stamps
`balance_before`/`balance_after` on each event's row at the moment it
publishes `NewsWindowEntered`/`NewsWindowExited`. Both are optional
(`repository`/`account_service` default to `None`) so this service stays
constructible without a database in unit tests that only exercise window
math, exactly as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.news.domain.models import (
    NewsCalendarUnavailable,
    NewsConfig,
    NewsEvent,
    NewsWindow,
    WindowSpec,
)
from src.news.ports.calendar import NewsCalendarPort
from src.shared.events.bus import EventBus
from src.shared.events.definitions import NewsWindowEntered, NewsWindowExited

if TYPE_CHECKING:
    from src.broker.application.account_service import AccountService
    from src.news.adapters.repository import NewsEventRecord, NewsEventRepository
    from src.news.domain.models import ImpactLevel

logger = logging.getLogger(__name__)

_DAYS_AHEAD = 7
_TRANSITION_CHECK_INTERVAL_S = 30.0


class NewsWindowService:
    def __init__(
        self,
        calendar: NewsCalendarPort,
        config: NewsConfig,
        window_specs: dict[str, WindowSpec],
        event_bus: EventBus,
        refresh_interval_s: float | None = None,
        repository: NewsEventRepository | None = None,
    ) -> None:
        self._calendar = calendar
        self._config = config
        self._window_specs = window_specs
        self._event_bus = event_bus
        self._refresh_interval_s = refresh_interval_s or config.refresh_minutes * 60
        self._repository = repository
        self._events: list[NewsEvent] = []
        self._active_keys: set[tuple[str, str]] = set()  # (event_name, skill_name)
        # Event time for each currently-active (event_name, skill_name) key —
        # populated on entry, consumed on exit, so `_check_transitions` can
        # stamp `balance_after` on the same `(name, time)` row it stamped
        # `balance_before` on, without `NewsWindowExited` needing to carry
        # the event's scheduled time itself (Phase 6 Part C).
        self._entered_event_times: dict[tuple[str, str], datetime] = {}
        # Balance source for Part C — process-wide `NewsWindowService` is
        # constructed before any per-account `AccountService` exists
        # (container.py builds account runtimes afterward), so this is
        # injected post-construction via `set_account_service` rather than
        # taken as a constructor arg.
        self._account_service: AccountService | None = None
        self._task: asyncio.Task[None] | None = None

    def set_account_service(self, account_service: AccountService) -> None:
        """Injects the balance source for Part C's `balance_before`/
        `balance_after` stamping. Optional: if never called (e.g. in unit
        tests), stamping is silently skipped — this is enrichment, not core
        function (see module docstring)."""
        self._account_service = account_service

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="news-calendar-refresh")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def refresh(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        self._events = await self._calendar.fetch_upcoming(_DAYS_AHEAD)
        logger.info("news calendar refreshed: %d events", len(self._events))
        if self._repository is not None:
            # Persistence is a pure addition — `self._events` (the
            # zero-I/O cache `active_window_for()` reads) is already set
            # above regardless of whether this succeeds.
            try:
                await asyncio.to_thread(self._repository.save_many, self._events)
            except Exception:
                logger.exception("failed to persist news calendar events")

    async def list_events(
        self, start: datetime, end: datetime, impact: ImpactLevel | None = None
    ) -> list[NewsEventRecord]:
        """Persisted event history in `[start, end]`, each with its row id
        and `balance_before`/`balance_after` — backs `GET
        /accounts/{account_id}/news/events`. `[]` (not an error) when
        persistence isn't wired (`repository` is `None`, e.g. a unit test
        constructing this service directly) or nothing matches."""
        if self._repository is None:
            return []
        return await asyncio.to_thread(self._repository.list_records_between, start, end, impact)

    def upcoming(
        self, days_ahead: int = _DAYS_AHEAD, now: datetime | None = None
    ) -> list[NewsEvent]:
        """Cached events in [now, now + days_ahead], for the UI panel."""
        now = now or datetime.now(UTC)
        cutoff = now + timedelta(days=days_ahead)
        return [e for e in self._events if now <= e.time <= cutoff]

    def skill_for(self, event: NewsEvent) -> str | None:
        """The news skill that would activate around `event`, if any — for
        showing which upcoming releases actually affect trading (§8 UI)."""
        spec = self._resolve_window_spec(event)
        return spec.skill_name if spec else None

    def symbols_for(self, skill_name: str) -> tuple[str, ...]:
        spec = self._window_specs.get(skill_name)
        return spec.symbols if spec else ()

    def active_windows(self, now: datetime | None = None) -> list[NewsWindow]:
        now = now or datetime.now(UTC)
        windows = []
        for event in self._events:
            spec = self._resolve_window_spec(event)
            if spec is None:
                continue
            window = NewsWindow(
                event=event,
                skill=spec.skill_name,
                window_start=event.time - timedelta(minutes=spec.before_min),
                window_end=event.time + timedelta(minutes=spec.after_min),
            )
            if window.contains(now):
                windows.append(window)
        return windows

    def active_window_for(self, symbol: str, now: datetime | None = None) -> NewsWindow | None:
        now = now or datetime.now(UTC)
        for window in self.active_windows(now):
            spec = self._window_specs.get(window.skill)
            if spec is not None and (not spec.symbols or symbol in spec.symbols):
                return window
        return None

    async def _run(self) -> None:
        last_refresh: datetime | None = None
        while True:
            now = datetime.now(UTC)
            due = (
                last_refresh is None
                or (now - last_refresh).total_seconds() >= self._refresh_interval_s
            )
            if due:
                try:
                    await self.refresh(now)
                    last_refresh = now
                except NewsCalendarUnavailable as exc:
                    logger.warning("news calendar refresh failed: %s", exc)
                except Exception:
                    logger.exception("news calendar refresh failed")
            await self._check_transitions(now)
            await asyncio.sleep(_TRANSITION_CHECK_INTERVAL_S)

    async def _check_transitions(self, now: datetime) -> None:
        windows = self.active_windows(now)
        current = {(w.event.name, w.skill): w for w in windows}
        current_keys = set(current)

        for key in current_keys - self._active_keys:
            window = current[key]
            spec = self._window_specs.get(window.skill)
            symbols = spec.symbols if spec else ()
            # close_all only matters at the moment the window opens (the
            # pre-event phase, since windows always start at window_start);
            # NewsSkillSelector re-derives block/override behavior itself
            # for the rest of the window's lifetime.
            close_all = spec.close_all if spec else False
            await self._event_bus.publish(
                NewsWindowEntered(
                    event_name=window.event.name, symbols=symbols, close_all=close_all
                )
            )
            logger.info(
                "news window entered: %s skill=%s symbols=%s close_all=%s",
                window.event.name,
                window.skill,
                symbols,
                close_all,
            )
            self._entered_event_times[key] = window.event.time
            await self._stamp_balance(window.event.name, window.event.time, entering=True)

        for event_name, skill_name in self._active_keys - current_keys:
            spec = self._window_specs.get(skill_name)
            symbols = spec.symbols if spec else ()
            await self._event_bus.publish(NewsWindowExited(event_name=event_name, symbols=symbols))
            logger.info("news window exited: %s skill=%s", event_name, skill_name)
            event_time = self._entered_event_times.pop((event_name, skill_name), None)
            if event_time is not None:
                await self._stamp_balance(event_name, event_time, entering=False)

        self._active_keys = current_keys

    async def _stamp_balance(self, name: str, time: datetime, *, entering: bool) -> None:
        """Writes `balance_before` (on entry) or `balance_after` (on exit)
        for the `(name, time)` news_events row — Phase 6 Part C. Never
        raises: a missing repository/account service, an unreachable
        gateway, or a not-yet-persisted row are all silently skipped
        (logged at most) rather than blocking the 30s transition-check loop
        or losing a live news window. `balance_after` staying null forever
        (window never cleanly exits — e.g. a process restart mid-window) is
        an accepted, expected outcome of that same "never block" contract."""
        if self._repository is None or self._account_service is None:
            return
        try:
            status = await self._account_service.status()
        except Exception:
            logger.exception("could not read account balance for news balance stamp")
            return
        account = status.get("account")
        balance = account["balance"] if account else None
        if balance is None:
            return
        write = (
            self._repository.set_balance_before if entering else self._repository.set_balance_after
        )
        try:
            await asyncio.to_thread(write, name, time, balance)
        except Exception:
            logger.exception("failed to persist news event balance stamp")

    def _resolve_window_spec(self, event: NewsEvent) -> WindowSpec | None:
        matched_skill: str | None = None
        for tracked in self._config.tracked_events:
            if tracked.name != "*" and tracked.name.lower() == event.name.lower():
                matched_skill = tracked.skill
                break
        if matched_skill is None:
            for tracked in self._config.tracked_events:
                if tracked.name == "*" and tracked.impact == event.impact:
                    matched_skill = tracked.skill
                    break
        if matched_skill is None:
            return None
        return self._window_specs.get(matched_skill)
