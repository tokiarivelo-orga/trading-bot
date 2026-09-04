"""Engine-side of the typed signal-decision trail (OBSERVABILITY_PLAN.md
Phase 1, extended in Phase 2): one recorded decision per fired signal, the
right — now per-gate, no longer collapsed into `risk_rejected` — outcome
stamped on every path out of `_enter_for_bot`, and the structured
`DecisionCheck`s each gate saw."""

from src.activity.domain.models import DecisionCheck
from src.broker.domain.trading import OrderRejected
from src.engine.application.risk_manager import RiskManager
from src.engine.domain.models import RiskCaps
from src.engine.domain.volatility import VolatilityConfig
from src.market_data.domain.models import Timeframe
from src.shared.events.definitions import CandleClosed
from src.strategies.domain.models import Direction, Signal
from tests.unit.engine.test_trade_loop import (
    BUY_SIGNAL,
    CAPS,
    FakeAccountService,
    FakeMarketData,
    FakeOrderService,
    FakeStrategy,
    FakeStrategySource,
    _volatility_ramp_candles,
    make_engine,
)


class FakeSignalDecisionSink:
    """In-memory `SignalDecisionSinkPort`, with the same "opened is final"
    rule the real repository enforces in SQL."""

    def __init__(self) -> None:
        self.recorded: list[dict] = []
        self.outcomes: list[tuple[str, str, str | None]] = []
        self.checks: list[DecisionCheck] = []

    async def record(self, **kwargs) -> None:
        self.recorded.append(kwargs)

    async def record_outcome(self, signal_id, outcome, *, reason=None, checks=()) -> None:
        self.outcomes.append((signal_id, outcome, reason))
        self.checks.extend(checks)

    async def record_checks(self, signal_id, checks) -> None:
        self.checks.extend(checks)

    def check(self, name: str) -> DecisionCheck:
        """The last check recorded under `name` — gates that run per target
        stamp one per target."""
        return [c for c in self.checks if c.name == name][-1]

    @property
    def final_outcome(self) -> str:
        for _signal_id, outcome, _reason in self.outcomes:
            if outcome == "opened":
                return "opened"
        return self.outcomes[-1][1] if self.outcomes else "skipped"


async def _run(**kwargs):
    sink = FakeSignalDecisionSink()
    engine, order_service, *_ = make_engine(signal_decisions=sink, **kwargs)
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))
    return sink, order_service


async def test_a_fired_signal_is_recorded_with_its_full_context():
    sink, _ = await _run()

    assert len(sink.recorded) == 1
    recorded = sink.recorded[0]
    assert recorded["bot"] == "normal/xauusd/fake"
    assert recorded["strategy"] == "fake"
    assert recorded["symbol"] == "XAUUSD"
    assert recorded["timeframe"] == "M5"
    assert recorded["direction"] == "buy"
    assert recorded["price"] == 2400.30  # ask, since it's a buy
    assert recorded["reason"] == "test buy"
    assert recorded["confidence"] == 1.0
    assert recorded["signal_id"]


async def test_no_signal_records_nothing():
    sink, _ = await _run(strategy=FakeStrategy(None))

    assert sink.recorded == []
    assert sink.outcomes == []


async def test_the_outcome_is_stamped_on_the_same_signal_id_that_was_recorded():
    sink, order_service = await _run()

    signal_id = sink.recorded[0]["signal_id"]
    assert [o[0] for o in sink.outcomes] == []  # order service stamps 'opened', not the engine
    assert order_service.signal_ids == [signal_id]


async def test_htf_veto_outcome():
    # Engine-level HTF veto is intentionally bypassed (see trade_loop.py's
    # "BYPASSED PER USER REQUEST" logging) — the veto is still evaluated
    # (hence still checked/recorded below) but no longer blocks the entry
    # or stamps a final "htf_veto" outcome; the signal proceeds to "opened".
    #
    # Buy signal against a downtrend on the veto timeframe (M15, one above M5).
    sink, order_service = await _run(
        market_data=FakeMarketData(bar_count=60, downtrend=True), context_bars=60
    )

    assert len(order_service.opened) == 1
    # The engine itself never stamps "opened" (see
    # test_the_outcome_is_stamped_on_the_same_signal_id_that_was_recorded —
    # that's the real OrderService's job, and FakeOrderService here doesn't
    # do it either), and the bypassed veto no longer stamps "htf_veto"
    # either, so `sink.outcomes` stays empty and `final_outcome` falls back
    # to its "skipped" default — despite the trade actually opening, which
    # `order_service.opened` above is the real proof of.
    assert sink.final_outcome == "skipped"
    # The bypassed path also always records this check as passed=True now,
    # even when the veto actually failed (`_htf_check(passed=True)` is
    # called unconditionally) — the decision trail no longer reflects the
    # true HTF-confirm result on a vetoed signal, a side effect of the same
    # bypass this test documents.
    assert sink.check("htf_confirm").passed is True


async def test_risk_gate_outcome_when_the_circuit_breaker_is_paused():
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    risk_manager.kill()

    sink, order_service = await _run(risk_manager=risk_manager)

    assert order_service.opened == []
    assert sink.final_outcome == "daily_loss_breaker"


async def test_max_open_positions_cap_outcome():
    # The per-signal (multi-TP-leg) max-open-positions loop cap is
    # intentionally bypassed — both its logic and its "max_positions"
    # DecisionCheck/outcome recording are commented out (see trade_loop.py's
    # "BYPASSED PER USER REQUEST" logging elsewhere in this same method):
    # both TP targets now open even though the cap is 1, and no
    # "max_positions" outcome is ever recorded for this candle. The only
    # "open_positions" check still recorded is the top-level pretrade one
    # (unconditionally passed=True now too), taken before either TP opens
    # — value 0.0 (no existing positions), not the old per-TP "at cap"
    # snapshot.
    caps = RiskCaps(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_open_positions=1,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=5,
    )
    two_targets = (
        BUY_SIGNAL,
        Signal(direction=Direction.BUY, sl_points=10.0, tp_points=30.0, reason="test buy tp2"),
    )
    strategy = FakeStrategy(two_targets)
    sink, order_service = await _run(
        strategy=strategy,
        strategy_source=FakeStrategySource({"fake": strategy}),
        risk_manager=RiskManager(caps=caps, timezone="UTC"),
    )

    assert len(order_service.opened) == 2
    assert not any(o[1] == "max_positions" for o in sink.outcomes)
    cap_check = sink.check("open_positions")
    assert (cap_check.value, cap_check.threshold, cap_check.passed) == (0.0, 1.0, True)


async def test_risk_sizing_rejection_outcome():
    # Risk-sizing rejection is intentionally bypassed too — see
    # trade_loop.py's "BYPASSED PER USER REQUEST, USING MIN VOLUME"
    # logging: its "risk_sizing" outcome recording is commented out and the
    # entry opens anyway at the broker minimum volume, so the recorded
    # "position_volume" check (sizing.volume >= volume_min) now always
    # passes too, since sizing.volume was itself forced to volume_min.
    #
    # A balance too small to fund even the minimum lot at this SL distance.
    sink, order_service = await _run(account=FakeAccountService(balance=1.0))

    assert len(order_service.opened) == 1
    assert not any(o[1] == "risk_sizing" for o in sink.outcomes)
    assert sink.check("position_volume").passed is True


async def test_no_account_connected_is_recorded_as_skipped():
    sink, order_service = await _run(account=FakeAccountService(balance=None))

    assert order_service.opened == []
    assert sink.final_outcome == "skipped"


async def test_volatility_guard_block_outcome():
    # Engine-level EXTREME-regime volatility guard is intentionally bypassed
    # (see trade_loop.py's "BYPASSED PER USER REQUEST" logging) — the
    # EXTREME regime is still detected (percentile still reflects it below)
    # but no longer blocks the entry or stamps a final "volatility_guard"
    # outcome; the signal proceeds to "opened". The recorded check itself
    # is also always passed=True now, unconditionally — see
    # `_volatility_check(percentile, self._volatility_config, passed=True)`
    # in trade_loop.py — so it no longer reflects the true pass/fail result
    # either, the same side effect the HTF-veto test above documents.
    candles = _volatility_ramp_candles("XAUUSD", Timeframe.M5, 40)
    config = VolatilityConfig(atr_period=5, regime_lookback_bars=30)
    sink, order_service = await _run(
        market_data=FakeMarketData(candles=candles),
        volatility_config=config,
        context_bars=40,
        strategy=FakeStrategy(BUY_SIGNAL, htf_veto=False),
        strategy_source=FakeStrategySource({"fake": FakeStrategy(BUY_SIGNAL, htf_veto=False)}),
    )

    assert len(order_service.opened) == 1
    # Same "engine never stamps opened, bypass no longer stamps a
    # rejection either" reasoning as test_htf_veto_outcome above —
    # `final_outcome` falls back to "skipped" despite the trade opening.
    assert sink.final_outcome == "skipped"
    guard = sink.check("volatility_percentile")
    assert guard.passed is True
    assert guard.value >= guard.threshold  # percentile itself still reflects EXTREME


async def test_broker_rejection_leaves_the_outcome_to_the_order_service():
    sink, order_service = await _run(
        order_service=FakeOrderService(raise_on_open=OrderRejected("invalid stops"))
    )

    # The engine records the signal and passes its id down; stamping
    # 'broker_rejected' is `OrderService.open_position`'s job (it owns that
    # log line), so the engine itself stamps nothing here.
    assert len(sink.recorded) == 1
    assert order_service.signal_ids == [sink.recorded[0]["signal_id"]]
    assert sink.outcomes == []


async def test_no_sink_wired_still_trades():
    engine, order_service, *_ = make_engine()
    await engine.on_candle_closed(CandleClosed(symbol="XAUUSD", timeframe="M5"))

    assert len(order_service.opened) == 1


async def test_a_filled_signal_records_every_gate_it_cleared():
    """The funnel needs passing gates, not only the failing one: a signal that
    made it to the broker must carry its HTF, volatility, position-cap and
    sizing checks, all passed."""
    sink, order_service = await _run()

    assert len(order_service.opened) == 1
    passed = {c.name: c for c in sink.checks}
    assert set(passed) >= {
        "open_positions",
        "htf_confirm",
        "volatility_percentile",
        "position_volume",
    }
    assert all(c.passed for c in passed.values())


async def test_the_circuit_breaker_and_the_position_cap_are_no_longer_one_bucket():
    """Phase 2's whole point: a paused engine and a full position book used to
    both read `risk_rejected`."""
    paused = RiskManager(caps=CAPS, timezone="UTC")
    paused.kill()
    paused_sink, _ = await _run(risk_manager=paused)

    full_caps = RiskCaps(
        risk_per_trade_pct=1.0,
        daily_loss_limit_pct=5.0,
        max_open_positions=0,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=5,
    )
    full_sink, _ = await _run(risk_manager=RiskManager(caps=full_caps, timezone="UTC"))

    assert paused_sink.final_outcome == "daily_loss_breaker"
    assert full_sink.final_outcome == "max_positions"


async def test_the_order_gets_the_signals_emit_time_for_latency_measurement():
    """The signal→fill latency span's emit end is exactly the `created_at`
    the engine recorded this `signal_id`'s decision with — passed down
    explicitly so `OrderService` never has to read it back from the database
    on the order path (OBSERVABILITY_PLAN.md Phase 3)."""
    sink, order_service = await _run()

    assert order_service.signal_emit_times == [sink.recorded[0]["created_at"]]
