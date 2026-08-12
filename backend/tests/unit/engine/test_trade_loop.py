from datetime import UTC, datetime

from src.broker.domain.trading import ExecutionResult, OrderRejected, Position, Side
from src.engine.application.risk_manager import RiskManager
from src.engine.application.trade_loop import TradeEngine, _veto_timeframe
from src.engine.domain.models import RiskCaps
from src.engine.domain.regime import RegimeConfig
from src.engine.domain.volatility import VolatilityConfig
from src.market_data.domain.models import Candle, SymbolInfo, Timeframe
from src.shared.events.bus import EventBus
from src.shared.events.definitions import (
    CandleClosed,
    CircuitBreakerTripped,
    NewsWindowEntered,
    PositionClosed,
)
from src.shared.logging.account_context import current_signal_id
from src.skills.ports.skill_selector import SkillDecision
from src.strategies.domain.models import (
    Direction,
    ExitActionKind,
    ExitDecision,
    MarketContext,
    PositionSnapshot,
    Signal,
    StrategySpec,
)

XAUUSD_INFO = SymbolInfo(
    symbol="XAUUSD",
    bid=2400.00,
    ask=2400.30,
    spread_points=30,
    point=0.01,
    digits=2,
    stops_level=10,
    contract_size=100.0,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
)

CAPS = RiskCaps(
    risk_per_trade_pct=1.0,
    daily_loss_limit_pct=5.0,
    max_open_positions=5,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=5,
)

ALLOWED_DECISION = SkillDecision(
    allowed=True,
    skill_name="normal/xauusd/fake",
    strategy_name="fake",
    risk_multiplier=1.0,
    magic=999,
)
BUY_SIGNAL = Signal(direction=Direction.BUY, sl_points=10.0, tp_points=15.0, reason="test buy")


def _uptrend_candles(symbol: str, timeframe: Timeframe, count: int) -> list[Candle]:
    base = datetime(2026, 7, 11, 8, 0, tzinfo=UTC)
    # high/low track the close (constant +-1.0 band) rather than sitting at a
    # fixed absolute level, so true range/ATR stays flat across the series —
    # otherwise a fixed high/low against a steadily drifting close inflates
    # the true-range "gap" component bar over bar, which the volatility
    # guard (added in Phase B) would misread as escalating volatility purely
    # from this fixture's shape, unrelated to whatever the test is checking.
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            time=base,
            open=2400.0,
            high=2400.0 + i * 0.5 + 1.0,
            low=2400.0 + i * 0.5 - 1.0,
            close=2400.0 + i * 0.5,
            tick_volume=100,
            spread_points=30,
        )
        for i in range(count)
    ]


def _downtrend_candles(symbol: str, timeframe: Timeframe, count: int) -> list[Candle]:
    base = datetime(2026, 7, 11, 8, 0, tzinfo=UTC)
    # Same flat-true-range rationale as _uptrend_candles above.
    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            time=base,
            open=2450.0,
            high=2450.0 - i * 0.5 + 1.0,
            low=2450.0 - i * 0.5 - 1.0,
            close=2450.0 - i * 0.5,
            tick_volume=100,
            spread_points=30,
        )
        for i in range(count)
    ]


def _volatility_ramp_candles(
    symbol: str, timeframe: Timeframe, count: int, *, last_frac: float | None = None
) -> list[Candle]:
    """Widening-true-range candles (bar `i`'s range is `1 + i`), so
    `latest_volatility_regime` ranks the most recent bar against its own
    trailing ATR history without needing hundreds of bars of real market
    data. `last_frac=None` keeps the ramp increasing through the final bar,
    driving its ATR to the very top of its own history (EXTREME);
    `last_frac=0.75` caps the final bar's range at 75% of the ramp's peak,
    landing it in HIGH territory instead (elevated but not the outlier)."""
    base = datetime(2026, 7, 11, 8, 0, tzinfo=UTC)
    candles = []
    for i in range(count):
        if last_frac is not None and i == count - 1:
            rng = float(count - 1) * last_frac
        else:
            rng = float(1 + i)
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                time=base,
                open=100.0,
                high=100.0 + rng / 2,
                low=100.0 - rng / 2,
                close=100.0,
                tick_volume=100,
                spread_points=30,
            )
        )
    return candles


class FakeMarketData:
    def __init__(
        self,
        info: SymbolInfo = XAUUSD_INFO,
        bar_count: int = 5,
        downtrend: bool = False,
        candles: list[Candle] | None = None,
    ):
        self.info = info
        self.bar_count = bar_count
        self._downtrend = downtrend
        # When set, every timeframe gets this exact series regardless of
        # `bar_count`/`downtrend` — used by the volatility-guard tests, which
        # need a specific ATR/percentile shape rather than a generic trend.
        self._fixed_candles = candles
        self.requested_timeframes: list[Timeframe] = []

    async def get_candles(self, symbol, timeframe, count):
        self.requested_timeframes.append(timeframe)
        if self._fixed_candles is not None:
            return self._fixed_candles
        builder = _downtrend_candles if self._downtrend else _uptrend_candles
        return builder(symbol, timeframe, self.bar_count)

    async def get_tick(self, symbol):
        raise NotImplementedError

    async def get_symbol_info(self, symbol):
        return self.info


class FakeOrderService:
    def __init__(self, positions: list[Position] | None = None, raise_on_open=None):
        self._positions = positions or []
        self.opened: list[dict] = []
        self.closed: list[int] = []
        self.signal_ids: list[str | None] = []
        self.signal_emit_times: list = []
        # What `current_signal_id.get()` reads *inside* this call — proves
        # the correlation id (OBSERVABILITY_PLAN.md Phase 5) is actually
        # bound by the time the engine reaches the order service, not just
        # passed as the `signal_id=` parameter alongside it.
        self.signal_id_in_context: list[str | None] = []
        self._raise_on_open = raise_on_open

    async def get_positions(self, symbol=None):
        return list(self._positions)

    async def open_position(
        self,
        symbol,
        side,
        volume,
        sl=None,
        tp=None,
        comment="",
        strategy_version=None,
        skill=None,
        magic=0,
        max_spread_points=None,
        reason="",
        confidence=None,
        zone_kind=None,
        zone_price_low=None,
        zone_price_high=None,
        zone_time_start=None,
        zone_time_end=None,
        zone_pattern=None,
        pattern=None,
        structure=(),
        indicators=(),
        signal_id=None,
        signal_emitted_at=None,
        regime_volatility=None,
        regime_volatility_percentile=None,
        regime_trend=None,
        regime_adx=None,
        regime_session=None,
    ):
        self.signal_ids.append(signal_id)
        self.signal_emit_times.append(signal_emitted_at)
        self.signal_id_in_context.append(current_signal_id.get())
        if self._raise_on_open:
            raise self._raise_on_open
        ticket = len(self.opened) + 1
        self.opened.append(
            dict(
                symbol=symbol,
                side=side,
                volume=volume,
                sl=sl,
                tp=tp,
                comment=comment,
                strategy_version=strategy_version,
                skill=skill,
                magic=magic,
                max_spread_points=max_spread_points,
                reason=reason,
                confidence=confidence,
                zone_kind=zone_kind,
                zone_price_low=zone_price_low,
                zone_price_high=zone_price_high,
                zone_time_start=zone_time_start,
                zone_time_end=zone_time_end,
                zone_pattern=zone_pattern,
                pattern=pattern,
                structure=structure,
                indicators=indicators,
                regime_volatility=regime_volatility,
                regime_volatility_percentile=regime_volatility_percentile,
                regime_trend=regime_trend,
                regime_adx=regime_adx,
                regime_session=regime_session,
            )
        )
        # Reflected in the next get_positions() call, same as a real broker
        # would — lets a later bot's pretrade risk check, in the same
        # candle close, see an earlier bot's just-opened position.
        self._positions.append(
            Position(
                ticket=ticket,
                symbol=symbol,
                side=side,
                volume=volume,
                open_price=2400.30 if side is Side.BUY else 2400.00,
                sl=sl,
                tp=tp,
                open_time=datetime.now(UTC),
                profit=0.0,
                comment=comment,
                magic=magic,
            )
        )
        return ExecutionResult(
            ticket=ticket,
            symbol=symbol,
            side=side,
            volume=volume,
            price=2400.30 if side is Side.BUY else 2400.00,
            sl=sl,
            tp=tp,
            time=datetime.now(UTC),
            spread_points=30,
            comment=comment,
            magic=magic,
        )

    async def close_position(self, ticket, volume=None):
        self.closed.append(ticket)
        return ExecutionResult(
            ticket=ticket,
            symbol="XAUUSD",
            side=Side.BUY,
            volume=volume or 0.1,
            price=2400.0,
            sl=None,
            tp=None,
            time=datetime.now(UTC),
            spread_points=30,
            profit=0.0,
        )

    async def modify_position(self, ticket, sl, tp):
        pass


class FakeAccountService:
    def __init__(self, balance: float | None = 10_000.0):
        self.balance = balance
        self.calls = 0

    async def status(self):
        self.calls += 1
        account = {"balance": self.balance} if self.balance is not None else None
        return {"account": account}


class FakePositionManager:
    def __init__(self):
        self.calls: list[str] = []
        self.applied_actions: list[tuple[int, ExitActionKind, str]] = []

    async def on_candle_closed(self, symbol):
        self.calls.append(symbol)

    async def apply_strategy_action(self, position, action):
        self.applied_actions.append((position.ticket, action.action, action.reason))


class FakeSkillSelector:
    def __init__(self, decisions: list[SkillDecision]):
        self.decisions = decisions

    def select_all(self, symbol, now):
        return self.decisions


class FakeStrategy:
    def __init__(
        self,
        signal: Signal | None,
        symbols: tuple[str, ...] = ("XAUUSD",),
        entry_timeframe: str = "M5",
        confirmation_timeframes: tuple[str, ...] = ("H1", "H4"),
        params: dict | None = None,
        htf_veto: bool = True,
        close_on_opposite_signal: bool = False,
    ):
        self.spec = StrategySpec(
            name="fake",
            version=1,
            symbols=symbols,
            entry_timeframe=entry_timeframe,
            confirmation_timeframes=confirmation_timeframes,
            params=params if params is not None else {},
            htf_veto=htf_veto,
            close_on_opposite_signal=close_on_opposite_signal,
        )
        self._signal = signal

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        return self._signal


class ThresholdFakeStrategy(FakeStrategy):
    """A strategy whose signal depends on `self.spec.params["threshold"]`,
    read fresh on every `evaluate()` call — mirrors how real generated
    strategies read `self.spec.params`, so per-bot param overrides can be
    exercised without a real generated strategy file."""

    def evaluate(self, ctx: MarketContext) -> Signal | None:
        if self.spec.params.get("threshold", 1) <= 0:
            return BUY_SIGNAL
        return None


class FakeExitStrategy(FakeStrategy):
    """Returns `ExitDecision`s (optionally alongside a `Signal`) instead of
    just a `Signal`, and records the `MarketContext.own_position` it was
    handed on each call — so tests can assert both what the engine did with
    the decisions and what it told the strategy about its own position."""

    def __init__(self, exit_decisions, signal: Signal | None = None, **kwargs):
        super().__init__(signal, **kwargs)
        self._exit_decisions = (
            exit_decisions if isinstance(exit_decisions, (list, tuple)) else (exit_decisions,)
        )
        self.seen_own_positions: list[PositionSnapshot | None] = []

    def evaluate(self, ctx: MarketContext):
        self.seen_own_positions.append(ctx.own_position)
        if self._signal is not None:
            return (*self._exit_decisions, self._signal)
        return self._exit_decisions if len(self._exit_decisions) != 1 else self._exit_decisions[0]


class FakeStrategySource:
    def __init__(self, strategies: dict[str, object]):
        self._strategies = strategies

    def get(self, name):
        return self._strategies.get(name)


def make_engine(
    *,
    market_data=None,
    order_service=None,
    account=None,
    position_manager=None,
    skill_selector=None,
    strategy=None,
    strategy_source=None,
    enabled=True,
    risk_manager=None,
    context_bars=5,
    event_bus=None,
    volatility_config=None,
    regime_config=None,
    signal_decisions=None,
    clock=None,
):
    market_data = market_data or FakeMarketData(bar_count=context_bars)
    order_service = order_service or FakeOrderService()
    account = account or FakeAccountService()
    position_manager = position_manager or FakePositionManager()
    skill_selector = skill_selector or FakeSkillSelector([ALLOWED_DECISION])
    strategy = strategy if strategy is not None else FakeStrategy(BUY_SIGNAL)
    strategy_source = strategy_source or FakeStrategySource({"fake": strategy})
    risk_manager = risk_manager or RiskManager(caps=CAPS, timezone="UTC")
    event_bus = event_bus if event_bus is not None else EventBus()
    # Default config's insufficient-history guard (needs atr_period=14 bars,
    # existing tests use far fewer) keeps every pre-existing test's regime at
    # NORMAL/nan — i.e. today's unscaled behavior — unless a test opts into a
    # tuned config via the parameter below.
    volatility_config = volatility_config or VolatilityConfig()
    regime_config = regime_config or RegimeConfig()

    engine = TradeEngine(
        market_data=market_data,
        order_service=order_service,
        account=account,
        risk_manager=risk_manager,
        position_manager=position_manager,
        skill_selector=skill_selector,
        strategy_source=strategy_source,
        entry_timeframe="M5",
        volatility_config=volatility_config,
        regime_config=regime_config,
        signal_decisions=signal_decisions,
        event_bus=event_bus,
        enabled=enabled,
        context_bars=context_bars,
        **({"clock": clock} if clock is not None else {}),
    )
    return engine, order_service, risk_manager, position_manager


async def test_successful_entry_opens_position_with_strategy_and_skill():
    engine, order_service, risk_manager, position_manager = make_engine()
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    order = order_service.opened[0]
    assert order["side"] is Side.BUY
    assert order["sl"] == 2400.30 - 10.0
    assert order["tp"] == 2400.30 + 15.0
    assert order["strategy_version"] == "fake:v1"
    assert order["skill"] == "normal/xauusd/fake"
    assert order["magic"] == 999
    assert order["reason"] == "test buy"
    assert order["confidence"] == 1.0
    assert risk_manager.status.trades_today == 1
    assert position_manager.calls == ["XAUUSD"]


# ── regime tagging (OBSERVABILITY_PLAN.md Phase 6) ──────────────────────────


async def test_order_carries_a_regime_tag_computed_from_the_entry_frame():
    """5 bars is far short of ADX's/ATR's warm-up (needs 14+), so the tag
    still lands on the classifiers' own "insufficient history" defaults
    (NORMAL/RANGING, nan normalized to None) — the point here is that the
    order call receives *a* regime tag at all, sourced from the entry
    timeframe's own candles, not that this particular fixture trends."""
    fixed_now = datetime(2026, 8, 7, 13, 0, tzinfo=UTC)  # OVERLAP under default RegimeConfig
    engine, order_service, *_ = make_engine(clock=lambda: fixed_now)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    order = order_service.opened[0]
    assert order["regime_volatility"] == "normal"
    assert order["regime_volatility_percentile"] is None
    assert order["regime_trend"] == "ranging"
    assert order["regime_adx"] is None
    assert order["regime_session"] == "overlap"


async def test_regime_is_computed_once_and_identical_on_the_decision_and_the_order():
    """The engine must not recompute the regime independently for the
    decision record and the order call — both have to read off the exact
    same `compute_entry_regime` result for one signal."""
    from tests.unit.engine.test_signal_decisions import FakeSignalDecisionSink

    fixed_now = datetime(2026, 8, 7, 3, 0, tzinfo=UTC)  # ASIAN (midnight-wrap)
    sink = FakeSignalDecisionSink()
    engine, order_service, *_ = make_engine(clock=lambda: fixed_now, signal_decisions=sink)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    recorded = sink.recorded[0]
    order = order_service.opened[0]
    regime_fields = (
        "regime_volatility",
        "regime_volatility_percentile",
        "regime_trend",
        "regime_adx",
        "regime_session",
    )
    for field in regime_fields:
        assert recorded[field] == order[field], field
    assert recorded["regime_session"] == "asian"


async def test_no_candles_for_the_entry_timeframe_tags_no_regime():
    """`compute_entry_regime` returns `None` on an empty/missing frame —
    this must reach the order call as all-`None` regime kwargs, not a
    fabricated tag."""
    engine, order_service, *_ = make_engine(market_data=FakeMarketData(bar_count=0))

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    order = order_service.opened[0]
    assert order["regime_volatility"] is None
    assert order["regime_volatility_percentile"] is None
    assert order["regime_trend"] is None
    assert order["regime_adx"] is None
    assert order["regime_session"] is None


async def test_non_entry_timeframe_is_ignored():
    engine, order_service, _, position_manager = make_engine()
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="H1"))

    assert order_service.opened == []
    assert position_manager.calls == []


async def test_disabled_engine_still_manages_positions_but_skips_entries():
    engine, order_service, _, position_manager = make_engine(enabled=False)
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert position_manager.calls == ["XAUUSD"]
    assert order_service.opened == []


async def test_skill_blocked_skips_entry():
    decision = SkillDecision(allowed=False, reason="outside trading session")
    engine, order_service, *_ = make_engine(skill_selector=FakeSkillSelector([decision]))
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_no_active_bots_skips_entry():
    engine, order_service, *_ = make_engine(skill_selector=FakeSkillSelector([]))
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_missing_strategy_skips_entry():
    decision = SkillDecision(allowed=True, skill_name="normal/xauusd/x", strategy_name="missing")
    engine, order_service, *_ = make_engine(skill_selector=FakeSkillSelector([decision]))
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_symbol_not_covered_by_strategy_skips_entry():
    strategy = FakeStrategy(BUY_SIGNAL, symbols=("BTCUSD",))
    engine, order_service, *_ = make_engine(strategy=strategy)
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_no_signal_skips_entry():
    engine, order_service, *_ = make_engine(strategy=FakeStrategy(None))
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_htf_veto_skips_entry():
    market_data = FakeMarketData(bar_count=60, downtrend=True)
    engine, order_service, *_ = make_engine(market_data=market_data, context_bars=60)
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_pretrade_risk_block_skips_entry():
    caps = RiskCaps(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_open_positions=1,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=5,
    )
    risk_manager = RiskManager(caps=caps, timezone="UTC")
    existing = Position(
        ticket=1,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=None,
        tp=None,
        open_time=datetime.now(UTC),
        profit=0.0,
    )
    order_service = FakeOrderService(positions=[existing])
    engine, order_service, risk_manager, _ = make_engine(
        order_service=order_service, risk_manager=risk_manager
    )
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_order_rejected_does_not_crash_or_record_trade():
    order_service = FakeOrderService(raise_on_open=OrderRejected("spread too wide"))
    engine, order_service, risk_manager, _ = make_engine(order_service=order_service)
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []
    assert risk_manager.status.trades_today == 0


async def test_no_account_connected_skips_entry():
    engine, order_service, *_ = make_engine(account=FakeAccountService(balance=None))
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_kill_switch_pauses_and_closes_all_positions():
    position = Position(
        ticket=1,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=None,
        tp=None,
        open_time=datetime.now(UTC),
        profit=0.0,
    )
    order_service = FakeOrderService(positions=[position])
    engine, order_service, risk_manager, _ = make_engine(order_service=order_service)
    await engine.kill_switch()

    assert risk_manager.paused
    assert order_service.closed == [1]


async def test_news_window_entered_flattens_positions_when_close_all():
    position = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=None,
        tp=None,
        open_time=datetime.now(UTC),
        profit=0.0,
    )
    order_service = FakeOrderService(positions=[position])
    engine, order_service, risk_manager, _ = make_engine(order_service=order_service)

    await engine.on_news_window_entered(
        NewsWindowEntered(event_name="Non-Farm Payrolls", symbols=("XAUUSD",), close_all=True)
    )

    assert order_service.closed == [7]
    assert not risk_manager.paused  # unlike kill_switch, this never pauses the engine


def _collector():
    published: list[CircuitBreakerTripped] = []

    async def handler(event: CircuitBreakerTripped) -> None:
        published.append(event)

    return published, handler


async def test_kill_switch_publishes_circuit_breaker_tripped_once():
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(CircuitBreakerTripped, handler)
    engine, *_ = make_engine(event_bus=event_bus)

    await engine.kill_switch()

    assert len(published) == 1
    assert published[0].reason == "manual kill switch"


async def test_consecutive_loss_pause_publishes_circuit_breaker_tripped_once():
    event_bus = EventBus()
    published, handler = _collector()
    event_bus.subscribe(CircuitBreakerTripped, handler)
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    engine, *_ = make_engine(risk_manager=risk_manager, event_bus=event_bus)

    for _ in range(CAPS.consecutive_loss_pause):
        await engine.on_position_closed(
            PositionClosed(symbol="XAUUSD", position_id="1", close_price=2400.0, profit=-10.0)
        )

    assert risk_manager.paused
    assert len(published) == 1
    assert "consecutive losses" in published[0].reason


async def test_news_window_entered_does_nothing_when_close_all_false():
    position = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=None,
        tp=None,
        open_time=datetime.now(UTC),
        profit=0.0,
    )
    order_service = FakeOrderService(positions=[position])
    engine, order_service, *_ = make_engine(order_service=order_service)

    await engine.on_news_window_entered(
        NewsWindowEntered(event_name="CPI", symbols=("XAUUSD",), close_all=False)
    )

    assert order_service.closed == []


def test_resume_clears_pause():
    engine, _, risk_manager, _ = make_engine()
    risk_manager.kill("test")
    engine.resume()

    assert not risk_manager.paused


async def test_on_position_closed_forwards_to_risk_manager():
    engine, _, risk_manager, _ = make_engine(account=FakeAccountService(balance=10_000.0))
    await engine.on_position_closed(
        PositionClosed(symbol="XAUUSD", position_id="1", close_price=2390.0, profit=-50.0)
    )

    assert risk_manager.status.consecutive_losses == 1


def test_status_reports_enabled_flag():
    engine, *_ = make_engine(enabled=False)
    assert engine.status.enabled is False


async def test_m1_entry_strategy_fires_on_m1_close_not_on_m5():
    # Regression: the engine used to evaluate only on its global entry-TF
    # (M5) closes with M5/H1/H4 context, so an M1-entry scalp strategy could
    # never fire live — its `ctx.candles.get("M1")` was always None.
    strategy = FakeStrategy(BUY_SIGNAL, entry_timeframe="M1", confirmation_timeframes=("M5",))
    engine, order_service, *_ = make_engine(strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))
    assert order_service.opened == []

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M1"))
    assert len(order_service.opened) == 1
    assert order_service.opened[0]["strategy_version"] == "fake:v1"


async def test_context_fetch_covers_strategy_confirmation_and_veto_timeframes():
    # own confirmation timeframe (H4) deliberately differs from the veto
    # timeframe (M5, next_up of M1) so the two are unambiguously exercised.
    strategy = FakeStrategy(BUY_SIGNAL, entry_timeframe="M1", confirmation_timeframes=("H4",))
    market_data = FakeMarketData()
    engine, *_ = make_engine(market_data=market_data, strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M1"))

    assert set(market_data.requested_timeframes) == {
        Timeframe.M1,  # the strategy's entry timeframe (the closed candle)
        Timeframe.H4,  # the strategy's own confirmation timeframe
        Timeframe.M5,  # this bot's HTF-veto timeframe (next_up of M1)
    }


def test_veto_timeframe_is_next_above_entry_timeframe():
    expected = {
        "M1": "M5",
        "M5": "M15",
        "M15": "M30",
        "M30": "H1",
        "H1": "H4",
        "H4": "D1",
        "D1": "W1",
        "W1": "MN",
        "MN": None,
    }
    for entry_tf, veto_tf in expected.items():
        assert _veto_timeframe(FakeStrategy(BUY_SIGNAL, entry_timeframe=entry_tf)) == veto_tf


async def test_mixed_timeframe_bots_each_fire_on_their_own_closes():
    decisions = [
        SkillDecision(allowed=True, skill_name="normal/xauusd/a", strategy_name="a", magic=111),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="b", magic=222),
    ]
    strategy_source = FakeStrategySource(
        {
            "a": FakeStrategy(BUY_SIGNAL),  # M5 entry
            "b": FakeStrategy(BUY_SIGNAL, entry_timeframe="M1", confirmation_timeframes=("M5",)),
        }
    )
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector(decisions), strategy_source=strategy_source
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))
    assert [o["magic"] for o in order_service.opened] == [111]

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M1"))
    assert [o["magic"] for o in order_service.opened] == [111, 222]


async def test_two_bots_on_one_symbol_each_place_their_own_order():
    decisions = [
        SkillDecision(allowed=True, skill_name="normal/xauusd/a", strategy_name="a", magic=111),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="b", magic=222),
    ]
    strategy_source = FakeStrategySource(
        {"a": FakeStrategy(BUY_SIGNAL), "b": FakeStrategy(BUY_SIGNAL)}
    )
    engine, order_service, risk_manager, _ = make_engine(
        skill_selector=FakeSkillSelector(decisions), strategy_source=strategy_source
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 2
    assert {o["skill"] for o in order_service.opened} == {"normal/xauusd/a", "normal/xauusd/b"}
    assert {o["magic"] for o in order_service.opened} == {111, 222}
    assert risk_manager.status.trades_today == 2


async def test_second_bot_sizing_sees_first_bots_fresh_position():
    # max_open_positions=1 means the second bot in the same candle close
    # must see the first bot's just-opened position and get blocked by the
    # risk gate — proving the pretrade check is re-fetched per bot, not
    # hoisted once for the whole candle.
    caps = RiskCaps(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_open_positions=1,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=5,
    )
    decisions = [
        SkillDecision(allowed=True, skill_name="normal/xauusd/a", strategy_name="a", magic=111),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="b", magic=222),
    ]
    strategy_source = FakeStrategySource(
        {"a": FakeStrategy(BUY_SIGNAL), "b": FakeStrategy(BUY_SIGNAL)}
    )
    engine, order_service, risk_manager, _ = make_engine(
        skill_selector=FakeSkillSelector(decisions),
        strategy_source=strategy_source,
        risk_manager=RiskManager(caps=caps, timezone="UTC"),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    assert order_service.opened[0]["magic"] == 111


async def test_param_override_reaches_strategy_evaluate():
    strategy = ThresholdFakeStrategy(None, params={"threshold": 100})
    decision = SkillDecision(
        allowed=True,
        skill_name="normal/xauusd/fake",
        strategy_name="fake",
        magic=999,
        param_overrides={"threshold": 0},
    )
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector([decision]),
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    # The shared StrategyRegistry instance's own spec is never mutated.
    assert strategy.spec.params == {"threshold": 100}


async def test_param_override_absent_keeps_strategy_default_behavior():
    strategy = ThresholdFakeStrategy(None, params={"threshold": 100})
    decision = SkillDecision(
        allowed=True, skill_name="normal/xauusd/fake", strategy_name="fake", magic=999
    )
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector([decision]),
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_two_bots_same_strategy_different_param_overrides_do_not_leak():
    # Both bots share one StrategyRegistry-registered Strategy instance —
    # only bot "a"'s override should affect its own evaluation.
    strategy = ThresholdFakeStrategy(None, params={"threshold": 100})
    decisions = [
        SkillDecision(
            allowed=True,
            skill_name="normal/xauusd/a",
            strategy_name="fake",
            magic=111,
            param_overrides={"threshold": 0},
        ),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="fake", magic=222),
    ]
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector(decisions),
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert [o["magic"] for o in order_service.opened] == [111]
    assert strategy.spec.params == {"threshold": 100}


async def test_htf_veto_override_forces_veto_on_despite_strategy_default_off():
    market_data = FakeMarketData(bar_count=60, downtrend=True)
    strategy = FakeStrategy(BUY_SIGNAL, htf_veto=False)
    decision = SkillDecision(
        allowed=True,
        skill_name="normal/xauusd/fake",
        strategy_name="fake",
        magic=999,
        htf_veto_override=True,
    )
    engine, order_service, *_ = make_engine(
        market_data=market_data,
        context_bars=60,
        skill_selector=FakeSkillSelector([decision]),
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []


async def test_htf_veto_override_forces_veto_off_despite_strategy_default_on():
    market_data = FakeMarketData(bar_count=60, downtrend=True)
    strategy = FakeStrategy(BUY_SIGNAL, htf_veto=True)
    decision = SkillDecision(
        allowed=True,
        skill_name="normal/xauusd/fake",
        strategy_name="fake",
        magic=999,
        htf_veto_override=False,
    )
    engine, order_service, *_ = make_engine(
        market_data=market_data,
        context_bars=60,
        skill_selector=FakeSkillSelector([decision]),
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1


async def test_close_on_opposite_signal_flips_this_bots_position():
    # An open SELL from this same bot (magic=999, matching ALLOWED_DECISION)
    # plus a fresh BUY signal from a close_on_opposite_signal strategy ->
    # the SELL is closed and the BUY opens in the same pass, instead of
    # waiting for SL/TP/time-stop.
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeStrategy(BUY_SIGNAL, close_on_opposite_signal=True)
    engine, order_service, risk_manager, _ = make_engine(
        order_service=order_service, strategy=strategy
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.closed == [7]
    assert len(order_service.opened) == 1
    assert order_service.opened[0]["side"] is Side.BUY
    assert order_service.opened[0]["magic"] == 999


async def test_close_on_opposite_signal_ignores_other_bots_and_manual_positions():
    # A SELL on the same symbol but a different magic (another bot, or a
    # manually-opened position) must never be touched by this bot's flip.
    other_bots_position = Position(
        ticket=8,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=111,
    )
    order_service = FakeOrderService(positions=[other_bots_position])
    strategy = FakeStrategy(BUY_SIGNAL, close_on_opposite_signal=True)
    engine, order_service, *_ = make_engine(order_service=order_service, strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.closed == []
    assert len(order_service.opened) == 1


async def test_close_on_opposite_signal_false_leaves_opposite_position_open():
    # Default behavior (unset on every existing strategy): an opposing
    # position from the same bot is left alone; SL/TP/time-stop still own
    # its exit.
    existing = Position(
        ticket=9,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    engine, order_service, *_ = make_engine(order_service=order_service)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.closed == []
    assert len(order_service.opened) == 1


async def test_account_status_fetched_once_per_symbol_not_per_bot():
    # Regression for the redundant-gateway-call bug: three candidate bots on
    # one symbol/candle must share a single AccountService.status() call,
    # not trigger three (one real gateway HTTP round trip + keyring read
    # each) — matching the candles/symbol_info hoisting pattern above it.
    decisions = [
        SkillDecision(allowed=True, skill_name="normal/xauusd/a", strategy_name="a", magic=111),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="b", magic=222),
        SkillDecision(allowed=True, skill_name="normal/xauusd/c", strategy_name="c", magic=333),
    ]
    strategy_source = FakeStrategySource(
        {
            "a": FakeStrategy(BUY_SIGNAL),
            "b": FakeStrategy(BUY_SIGNAL),
            "c": FakeStrategy(BUY_SIGNAL),
        }
    )
    account = FakeAccountService(balance=10_000.0)
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector(decisions),
        strategy_source=strategy_source,
        account=account,
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 3
    assert account.calls == 1


async def test_account_status_refetched_after_close_on_opposite_signal_closes():
    # `_close_opposite_position` is the only thing that can change balance
    # mid-loop (a realized close), so a bot that flips its own position
    # must trigger exactly one re-fetch of account status on top of the
    # one hoisted per-symbol fetch — total 2, not 1 and not 3 (one per bot).
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeStrategy(BUY_SIGNAL, close_on_opposite_signal=True)
    account = FakeAccountService(balance=10_000.0)
    engine, order_service, *_ = make_engine(
        order_service=order_service, strategy=strategy, account=account
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.closed == [7]
    assert len(order_service.opened) == 1
    assert account.calls == 2


async def test_account_status_not_refetched_when_no_close_on_opposite_signal_happens():
    # Two bots, neither of which closes anything mid-loop (default
    # close_on_opposite_signal=False) — still exactly one status() call for
    # the whole candle, confirming the second bot reuses the first's value
    # rather than independently re-fetching or re-triggering a refetch.
    decisions = [
        SkillDecision(allowed=True, skill_name="normal/xauusd/a", strategy_name="a", magic=111),
        SkillDecision(allowed=True, skill_name="normal/xauusd/b", strategy_name="b", magic=222),
    ]
    strategy_source = FakeStrategySource(
        {"a": FakeStrategy(BUY_SIGNAL), "b": FakeStrategy(BUY_SIGNAL)}
    )
    account = FakeAccountService(balance=10_000.0)
    engine, order_service, *_ = make_engine(
        skill_selector=FakeSkillSelector(decisions),
        strategy_source=strategy_source,
        account=account,
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 2
    assert account.calls == 1


async def test_multi_position_scaling_opens_tiered_tp_orders():
    # When strategy.evaluate returns a sequence of 3 signals (TP1, TP2, TP3),
    # trade loop should open 3 distinct positions with corresponding tiered TPs and shared SL.
    sig1 = Signal(direction=Direction.BUY, sl_points=10.0, tp_points=10.0, reason="TP1 scalp")
    sig2 = Signal(direction=Direction.BUY, sl_points=10.0, tp_points=25.0, reason="TP2 zone")
    sig3 = Signal(direction=Direction.BUY, sl_points=10.0, tp_points=40.0, reason="TP3 runner")
    strategy = FakeStrategy((sig1, sig2, sig3))
    engine, order_service, *_ = make_engine(strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 3
    # Verify tiered TPs on Ask (2400.3)
    assert order_service.opened[0]["tp"] == 2400.3 + 10.0
    assert order_service.opened[1]["tp"] == 2400.3 + 25.0
    assert order_service.opened[2]["tp"] == 2400.3 + 40.0
    # Verify shared SL
    assert order_service.opened[0]["sl"] == 2400.3 - 10.0
    assert order_service.opened[1]["sl"] == 2400.3 - 10.0
    assert order_service.opened[2]["sl"] == 2400.3 - 10.0


# ---- volatility guard (bot-agnostic, engine-level) --------------------------


async def test_extreme_volatility_regime_blocks_entry(caplog):
    volatility_config = VolatilityConfig(atr_period=3, regime_lookback_bars=10)
    candles = _volatility_ramp_candles("XAUUSD", Timeframe.M5, 16)
    market_data = FakeMarketData(candles=candles)
    engine, order_service, *_ = make_engine(
        market_data=market_data, context_bars=16, volatility_config=volatility_config
    )

    with caplog.at_level("INFO"):
        await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []
    assert "ENTRY BLOCKED (volatility guard)" in caplog.text
    assert "regime=EXTREME" in caplog.text


async def test_high_volatility_regime_scales_sl_and_tp():
    volatility_config = VolatilityConfig(atr_period=3, regime_lookback_bars=10)
    candles = _volatility_ramp_candles("XAUUSD", Timeframe.M5, 16, last_frac=0.75)
    market_data = FakeMarketData(candles=candles)
    engine, order_service, *_ = make_engine(
        market_data=market_data, context_bars=16, volatility_config=volatility_config
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    order = order_service.opened[0]
    # VolatilityConfig defaults: sl_multiplier_high = tp_multiplier_high = 1.3,
    # versus the unscaled NORMAL baseline in
    # test_successful_entry_opens_position_with_strategy_and_skill
    # (sl=2400.30-10.0, tp=2400.30+15.0).
    sl_mult = volatility_config.sl_multiplier_high
    tp_mult = volatility_config.tp_multiplier_high
    assert order["sl"] == 2400.30 - 1 * BUY_SIGNAL.sl_points * sl_mult
    assert order["tp"] == 2400.30 + 1 * BUY_SIGNAL.tp_points * tp_mult


async def test_disabled_volatility_guard_does_not_block_extreme_entry():
    # Same EXTREME fixture as test_extreme_volatility_regime_blocks_entry, but
    # with the live guard switched off first -- entry must go through
    # unblocked and SL/TP must be unscaled (mult=1.0), as if volatility_config
    # didn't exist at all.
    volatility_config = VolatilityConfig(atr_period=3, regime_lookback_bars=10)
    candles = _volatility_ramp_candles("XAUUSD", Timeframe.M5, 16)
    market_data = FakeMarketData(candles=candles)
    engine, order_service, *_ = make_engine(
        market_data=market_data, context_bars=16, volatility_config=volatility_config
    )
    engine.set_volatility_guard_enabled(False)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    order = order_service.opened[0]
    assert order["sl"] == 2400.30 - 1 * BUY_SIGNAL.sl_points
    assert order["tp"] == 2400.30 + 1 * BUY_SIGNAL.tp_points



# --- decision-trail log format (consumed by the signal-trail parsers) -------
#
# The `SIGNAL:`/`ENTRY ...` lines this engine emits are the *only* source the
# live bot signal trail (`activity/application/bot_signals.py`) and the
# backtest report (`backtest/application/signals.py`) have. These tests feed
# the real emitted lines through the real parser so a reword here fails loudly
# instead of silently emptying the chart overlay.


def _trail(caplog, skill: str = "normal/xauusd/fake"):
    from src.activity.application.bot_signals import extract_bot_signals
    from src.activity.domain.models import LogEntry

    entries = [
        LogEntry(
            id=None,
            created_at=datetime.now(UTC),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
        )
        for record in caplog.records
    ]
    return extract_bot_signals(entries, skill=skill)


async def test_signal_line_carries_the_side_reference_price(caplog):
    caplog.set_level("INFO")
    engine, *_ = make_engine()

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    signal_lines = [m for m in caplog.messages if m.startswith("SIGNAL:")]
    assert len(signal_lines) == 1
    # BUY -> ask, not bid.
    assert "buy @ 2400.30000" in signal_lines[0]

    # `ENTRY OPENED:` is emitted by the real OrderService (faked out here), so
    # the parsed outcome stays "skipped" — what matters is that the current
    # SIGNAL line parses at all and carries its price.
    signals = _trail(caplog)
    assert len(signals) == 1
    assert signals[0].price == 2400.30
    assert signals[0].direction == "buy"


async def test_sell_signal_line_uses_the_bid(caplog):
    caplog.set_level("INFO")
    sell_signal = Signal(
        direction=Direction.SELL, sl_points=10.0, tp_points=15.0, reason="test sell"
    )
    engine, *_ = make_engine(
        market_data=FakeMarketData(bar_count=60, downtrend=True),
        context_bars=60,
        strategy=FakeStrategy(sell_signal),
        strategy_source=FakeStrategySource({"fake": FakeStrategy(sell_signal)}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    signal_lines = [m for m in caplog.messages if m.startswith("SIGNAL:")]
    assert len(signal_lines) == 1
    assert "sell @ 2400.00000" in signal_lines[0]
    assert _trail(caplog)[0].price == 2400.00


async def test_no_account_connected_line_is_skill_scoped_and_parses(caplog):
    caplog.set_level("INFO")
    engine, order_service, *_ = make_engine(account=FakeAccountService(balance=None))

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []
    line = next(m for m in caplog.messages if m.startswith("ENTRY SKIPPED (no account connected)"))
    assert "[normal/xauusd/fake]" in line
    assert " — " in line

    signals = _trail(caplog)
    assert len(signals) == 1
    assert signals[0].outcome == "skipped"
    assert "no account balance available" in signals[0].reason


async def test_max_open_positions_line_is_skill_scoped_and_parses(caplog):
    caplog.set_level("INFO")
    caps = RiskCaps(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_open_positions=1,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=5,
    )
    two_targets = (
        Signal(direction=Direction.BUY, sl_points=10.0, tp_points=15.0, reason="tp1"),
        Signal(direction=Direction.BUY, sl_points=10.0, tp_points=30.0, reason="tp2"),
    )
    strategy = FakeStrategy(two_targets)
    engine, order_service, *_ = make_engine(
        risk_manager=RiskManager(caps=caps, timezone="UTC"),
        strategy=strategy,
        strategy_source=FakeStrategySource({"fake": strategy}),
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1  # second target hits the cap
    line = next(
        m for m in caplog.messages if m.startswith("ENTRY BLOCKED (max open positions cap reached)")
    )
    assert "[normal/xauusd/fake]" in line
    assert " — " in line


async def test_risk_sizing_rejection_prefix_has_no_tp_index(caplog):
    caplog.set_level("INFO")
    # A zero balance makes sizing fail for every target.
    engine, order_service, *_ = make_engine(account=FakeAccountService(balance=0.0))

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert order_service.opened == []
    line = next(m for m in caplog.messages if m.startswith("ENTRY REJECTED (risk sizing)"))
    assert line.startswith("ENTRY REJECTED (risk sizing): ")
    assert " — TP1: " in line

    signals = _trail(caplog)
    assert len(signals) == 1
    assert signals[0].outcome == "risk_rejected"


async def test_own_position_populated_in_context_when_bot_has_one_open():
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeExitStrategy((), signal=None)
    engine, order_service, *_ = make_engine(order_service=order_service, strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(strategy.seen_own_positions) == 1
    snapshot = strategy.seen_own_positions[0]
    assert snapshot is not None
    assert snapshot.direction is Direction.SELL
    assert snapshot.entry_price == 2450.0
    assert snapshot.sl == 2460.0
    assert snapshot.tp == 2430.0


async def test_own_position_none_in_context_when_bot_has_no_open_position():
    strategy = FakeExitStrategy((), signal=None)
    engine, *_ = make_engine(strategy=strategy)

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert strategy.seen_own_positions == [None]


async def test_exit_decision_close_routes_to_position_manager_and_frees_pretrade_slot():
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeExitStrategy(
        ExitDecision(action=ExitActionKind.CLOSE, reason="fvg violated"), signal=BUY_SIGNAL
    )
    engine, order_service, _, position_manager = make_engine(
        order_service=order_service, strategy=strategy
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert position_manager.applied_actions == [(7, ExitActionKind.CLOSE, "fvg violated")]
    # Freed the slot in the same pass a new entry opened, mirroring
    # close_on_opposite_signal's contract.
    assert len(order_service.opened) == 1


async def test_exit_decision_ignores_other_bots_and_manual_positions():
    other_bots_position = Position(
        ticket=8,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=111,
    )
    order_service = FakeOrderService(positions=[other_bots_position])
    strategy = FakeExitStrategy(
        ExitDecision(action=ExitActionKind.CLOSE, reason="fvg violated"), signal=BUY_SIGNAL
    )
    engine, order_service, _, position_manager = make_engine(
        order_service=order_service, strategy=strategy
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert position_manager.applied_actions == []


async def test_exit_decision_alone_with_no_fresh_signal_still_applies():
    # A strategy with an open position but no new entry opportunity this
    # candle (evaluate() returns only an ExitDecision, no Signal) must still
    # get routed — this is exactly the case `close_on_opposite_signal`
    # cannot cover, since that only fires alongside a fresh signal.
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=2390.0,
        tp=2420.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeExitStrategy(
        ExitDecision(action=ExitActionKind.BREAKEVEN, reason="continuation confirmed"),
        signal=None,
    )
    engine, order_service, _, position_manager = make_engine(
        order_service=order_service, strategy=strategy
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert position_manager.applied_actions == [
        (7, ExitActionKind.BREAKEVEN, "continuation confirmed")
    ]
    assert order_service.opened == []


async def test_account_status_refetched_after_exit_decision_close():
    existing = Position(
        ticket=7,
        symbol="XAUUSD",
        side=Side.SELL,
        volume=0.1,
        open_price=2450.0,
        sl=2460.0,
        tp=2430.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        magic=999,
    )
    order_service = FakeOrderService(positions=[existing])
    strategy = FakeExitStrategy(ExitDecision(action=ExitActionKind.CLOSE), signal=BUY_SIGNAL)
    account = FakeAccountService(balance=10_000.0)
    engine, order_service, *_ = make_engine(
        order_service=order_service, strategy=strategy, account=account
    )

    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1
    assert account.calls == 2
