from datetime import UTC, datetime

from src.activity.application.activity_log_service import ActivityLogService
from src.activity.domain.models import LogEntry, SignalDecision


class FakeRepository:
    def __init__(self, entries, total, deleted=0):
        self.entries = entries
        self.total = total
        self.deleted = deleted
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return self.entries, self.total

    def delete_by_ids(self, ids, account_id="default"):
        self.calls.append(("delete_by_ids", ids))
        return self.deleted

    def delete_by_filter(self, **kwargs):
        self.calls.append(("delete_by_filter", kwargs))
        return self.deleted


async def test_search_delegates_to_repository_and_returns_result():
    entry = LogEntry(id=1, created_at=datetime.now(UTC), level="INFO", logger="src.x", message="m")
    repository = FakeRepository([entry], 1)
    service = ActivityLogService(repository)

    entries, total = await service.search(level="INFO", q="m", limit=10, offset=0)

    assert entries == [entry]
    assert total == 1
    assert repository.calls[0] == ("search", repository.calls[0][1])
    assert repository.calls[0][1]["level"] == "INFO"
    assert repository.calls[0][1]["q"] == "m"


async def test_delete_by_ids_delegates_to_repository():
    repository = FakeRepository([], 0, deleted=2)
    service = ActivityLogService(repository)

    deleted = await service.delete_by_ids([1, 2])

    assert deleted == 2
    assert repository.calls == [("delete_by_ids", [1, 2])]


async def test_delete_by_filter_delegates_to_repository():
    repository = FakeRepository([], 0, deleted=3)
    service = ActivityLogService(repository)

    deleted = await service.delete_by_filter(level="WARNING", q="veto")

    assert deleted == 3
    assert repository.calls[0][0] == "delete_by_filter"
    assert repository.calls[0][1]["level"] == "WARNING"
    assert repository.calls[0][1]["q"] == "veto"


class FakeLoggerFilteringRepository:
    """Unlike `FakeRepository` above, filters by `logger_contains` like the
    real repository does — needed to test `get_bot_signals`, which queries
    the trade_loop and order_service loggers separately and merges."""

    def __init__(self, entries: list[LogEntry]) -> None:
        self.entries = entries
        self.calls = []

    def search(self, *, logger_contains=None, **kwargs):
        self.calls.append(("search", logger_contains, kwargs))
        rows = [e for e in self.entries if logger_contains in e.logger]
        return rows, len(rows)


def _log(seconds: int, logger: str, message: str) -> LogEntry:
    return LogEntry(
        id=seconds,
        created_at=datetime(2026, 7, 17, 12, 0, seconds, tzinfo=UTC),
        level="INFO",
        logger=logger,
        message=message,
    )


async def test_get_bot_signals_merges_both_loggers_in_time_order():
    skill = "normal/xauusd/breakout_v1"
    entries = [
        _log(
            0,
            "src.engine.application.trade_loop",
            f"SIGNAL: XAUUSD buy via strategy=breakout_v1 skill={skill} — retest",
        ),
        _log(
            1,
            "src.broker.application.order_service",
            f"ENTRY OPENED: ticket=1 buy XAUUSD 0.01 lots @ 4000.00 sl=None tp=None spread=1pts "
            f"strategy=breakout_v1:v1 skill={skill} magic=1 reason=retest",
        ),
    ]
    repository = FakeLoggerFilteringRepository(entries)
    service = ActivityLogService(repository)

    signals = await service.get_bot_signals(skill=skill)

    assert len(signals) == 1
    assert signals[0].outcome == "opened"
    queried_loggers = {call[1] for call in repository.calls}
    assert queried_loggers == {
        "src.engine.application.trade_loop",
        "src.broker.application.order_service",
    }


async def test_get_bot_signals_defaults_a_bounded_time_window():
    repository = FakeLoggerFilteringRepository([])
    service = ActivityLogService(repository)

    await service.get_bot_signals(skill="normal/xauusd/breakout_v1")

    for _, _, kwargs in repository.calls:
        assert kwargs["created_from"] is not None


class FakeSignalDecisionRepository:
    """Enough of `SignalDecisionRepository` to exercise the table-vs-legacy
    boundary in `get_bot_signals`."""

    def __init__(self, decisions: list[SignalDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict] = []

    def earliest_created_at(self, *, account_id="default"):
        times = [int(d.created_at.timestamp()) for d in self.decisions]
        return min(times) if times else None

    def list_between(
        self, *, account_id="default", created_from=None, created_to=None, bot=None, limit=20000
    ):
        rows = self.decisions if bot is None else [d for d in self.decisions if d.bot == bot]
        if created_from is not None:
            rows = [d for d in rows if int(d.created_at.timestamp()) >= created_from]
        if created_to is not None:
            rows = [d for d in rows if int(d.created_at.timestamp()) <= created_to]
        return sorted(rows, key=lambda d: d.created_at)

    def list_for_bot(
        self, *, bot, account_id="default", created_from=None, created_to=None, limit=5000
    ):
        self.calls.append(dict(bot=bot, created_from=created_from, created_to=created_to))
        rows = [d for d in self.decisions if d.bot == bot]
        if created_from is not None:
            rows = [d for d in rows if int(d.created_at.timestamp()) >= created_from]
        if created_to is not None:
            rows = [d for d in rows if int(d.created_at.timestamp()) <= created_to]
        return sorted(rows, key=lambda d: d.created_at)


def _decision(at: int, *, bot: str, outcome: str = "opened") -> SignalDecision:
    return SignalDecision(
        signal_id=f"sig-{at}",
        account_id="default",
        bot=bot,
        strategy="breakout_v1",
        symbol="XAUUSD",
        timeframe="M5",
        direction="buy",
        price=4000.0,
        created_at=datetime.fromtimestamp(at, tz=UTC),
        outcome=outcome,
        reason=f"typed reason {at}",
        confidence=0.7,
    )


async def test_get_bot_signals_reads_the_typed_table_when_it_covers_the_window():
    skill = "normal/xauusd/breakout_v1"
    logs = FakeLoggerFilteringRepository(
        [
            _log(
                0,
                "src.engine.application.trade_loop",
                f"SIGNAL: XAUUSD buy via strategy=breakout_v1 skill={skill} — scraped",
            )
        ]
    )
    decisions = FakeSignalDecisionRepository([_decision(2_000, bot=skill)])
    service = ActivityLogService(logs, signal_decisions=decisions)

    signals = await service.get_bot_signals(skill=skill, created_from=2_000)

    assert [s.reason for s in signals] == ["typed reason 2000"]
    assert logs.calls == []  # legacy scrape not consulted at all


async def test_get_bot_signals_falls_back_to_the_log_scrape_below_the_table_start():
    skill = "normal/xauusd/breakout_v1"
    logs = FakeLoggerFilteringRepository(
        [
            _log(
                0,
                "src.engine.application.trade_loop",
                f"SIGNAL: XAUUSD buy via strategy=breakout_v1 skill={skill} — scraped",
            ),
            _log(
                1,
                "src.broker.application.order_service",
                f"ENTRY OPENED: ticket=1 buy XAUUSD 0.01 lots @ 4000.00 sl=None tp=None "
                f"spread=1pts strategy=breakout_v1:v1 skill={skill} magic=1 reason=scraped",
            ),
        ]
    )
    table_start = int(datetime(2026, 7, 17, 13, 0, tzinfo=UTC).timestamp())
    decisions = FakeSignalDecisionRepository([_decision(table_start, bot=skill)])
    service = ActivityLogService(logs, signal_decisions=decisions)

    signals = await service.get_bot_signals(skill=skill, created_from=0)

    # Legacy half first (it is older), typed half second, no overlap.
    assert [s.reason for s in signals] == ["scraped", f"typed reason {table_start}"]
    # The legacy query is capped at the row before the table's first decision.
    for _, _, kwargs in logs.calls:
        assert kwargs["created_to"] == table_start - 1


async def test_get_bot_signals_uses_only_the_log_scrape_while_the_table_is_empty():
    skill = "normal/xauusd/breakout_v1"
    logs = FakeLoggerFilteringRepository(
        [
            _log(
                0,
                "src.engine.application.trade_loop",
                f"SIGNAL: XAUUSD buy via strategy=breakout_v1 skill={skill} — scraped",
            )
        ]
    )
    service = ActivityLogService(logs, signal_decisions=FakeSignalDecisionRepository([]))

    signals = await service.get_bot_signals(skill=skill, created_from=0)

    assert [s.reason for s in signals] == ["scraped"]
    assert logs.calls


# --- Phase 2: the veto funnel -------------------------------------------------


async def test_get_signal_funnel_aggregates_the_typed_table_per_bot():
    skill = "normal/xauusd/breakout_v1"
    decisions = FakeSignalDecisionRepository(
        [
            _decision(1000, bot=skill, outcome="opened"),
            _decision(1100, bot=skill, outcome="htf_veto"),
            _decision(1200, bot=skill, outcome="spread_veto"),
            _decision(1300, bot="normal/xauusd/other", outcome="opened"),
        ]
    )
    service = ActivityLogService(FakeLoggerFilteringRepository([]), signal_decisions=decisions)

    funnels = await service.get_signal_funnel(created_from=0, created_to=9999)

    by_bot = {f.bot: f for f in funnels}
    assert (by_bot[skill].fired, by_bot[skill].passed_htf, by_bot[skill].filled) == (3, 2, 1)
    assert by_bot["normal/xauusd/other"].filled == 1


async def test_get_signal_funnel_can_be_narrowed_to_one_bot():
    skill = "normal/xauusd/breakout_v1"
    decisions = FakeSignalDecisionRepository(
        [
            _decision(1000, bot=skill),
            _decision(1100, bot="normal/xauusd/other"),
        ]
    )
    service = ActivityLogService(FakeLoggerFilteringRepository([]), signal_decisions=decisions)

    funnels = await service.get_signal_funnel(skill=skill, created_from=0)

    assert [f.bot for f in funnels] == [skill]


async def test_get_signal_funnel_never_falls_back_to_the_log_scrape():
    """The legacy vocabulary collapsed every risk block into one bucket, so
    scraping it would produce a misleading funnel — an empty table means an
    empty funnel, not a reconstructed one."""
    logs = FakeLoggerFilteringRepository(
        [
            _log(
                10,
                "src.engine.application.trade_loop",
                "SIGNAL: XAUUSD buy @ 4000.00000 (1 target position(s)) via "
                "strategy=breakout_v1 skill=normal/xauusd/breakout_v1 — legacy reason",
            ),
        ]
    )
    service = ActivityLogService(logs, signal_decisions=FakeSignalDecisionRepository([]))

    assert await service.get_signal_funnel(created_from=0) == []


async def test_get_signal_funnel_without_a_repository_is_empty():
    service = ActivityLogService(FakeLoggerFilteringRepository([]))

    assert await service.get_signal_funnel() == []
