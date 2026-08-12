from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np

from src.broker.domain.trading import ExecutionResult, OrderType, PendingOrder, Position, Side
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager
from src.engine.domain.models import RiskCaps
from src.engine.domain.volatility import (
    VolatilityConfig,
    VolatilityRegime,
    latest_volatility_regime,
)
from src.engine.domain.zone_detection import Base, BaseKind
from src.market_data.domain.models import Candle, SymbolInfo, Timeframe
from src.strategies.domain.models import ExitActionKind, ExitDecision

CAPS = RiskCaps(
    risk_per_trade_pct=0.5,
    daily_loss_limit_pct=2.0,
    max_open_positions=5,
    max_trades_per_day_enabled=False,
    consecutive_loss_pause=3,
)

INFO = SymbolInfo(
    symbol="XAUUSD",
    bid=2410.00,
    ask=2410.20,
    spread_points=20,
    point=0.01,
    digits=2,
    stops_level=10,
    contract_size=100.0,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
)


def _position(**overrides) -> Position:
    defaults = dict(
        ticket=1,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=2400.0,
        sl=2390.0,
        tp=2420.0,
        open_time=datetime.now(UTC),
        profit=0.0,
        comment="",
    )
    defaults.update(overrides)
    return Position(**defaults)


class FakeOrderService:
    def __init__(
        self,
        positions: list[Position],
        pending: list[PendingOrder] | None = None,
        simulates_pending_fills: bool = True,
    ) -> None:
        self._positions = positions
        self._pending = pending or []
        self._simulates_pending_fills = simulates_pending_fills
        self.modified: list[tuple[int, float | None, float | None]] = []
        self.modify_reasons: list[str] = []
        self.closed: list[int] = []
        self.close_reasons: list[str] = []
        self.opened: list = []
        self.pending_cancelled: list[int] = []

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        return list(self._positions)

    async def modify_position(self, ticket: int, sl, tp, reason: str = "") -> None:
        self.modified.append((ticket, sl, tp))
        self.modify_reasons.append(reason)

    async def close_position(self, ticket: int, volume=None, reason: str = ""):
        self.closed.append(ticket)
        self.close_reasons.append(reason)
        self._positions = [p for p in self._positions if p.ticket != ticket]

    async def get_pending_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        return [p for p in self._pending if symbol is None or p.symbol == symbol]

    @property
    def simulates_pending_fills(self) -> bool:
        return self._simulates_pending_fills

    async def open_position(self, symbol, side, volume, sl=None, tp=None, comment=""):
        self.opened.append((symbol, side, volume, sl, tp, comment))
        ticket = 100 + len(self.opened)
        new_position = Position(
            ticket=ticket,
            symbol=symbol,
            side=side,
            volume=volume,
            open_price=INFO.ask if side is Side.BUY else INFO.bid,
            sl=sl,
            tp=tp,
            open_time=datetime.now(UTC),
            profit=0.0,
        )
        self._positions.append(new_position)
        return ExecutionResult(
            ticket=ticket,
            symbol=symbol,
            side=side,
            volume=volume,
            price=new_position.open_price,
            sl=sl,
            tp=tp,
            time=datetime.now(UTC),
            spread_points=20,
            comment=comment,
        )

    async def cancel_pending_order(self, ticket: int) -> None:
        self.pending_cancelled.append(ticket)
        self._pending = [p for p in self._pending if p.ticket != ticket]


class FakeReconciliation:
    def __init__(self) -> None:
        self.vanished_calls: list[tuple[str, set[int]]] = []
        self.pending_fill_calls: list[tuple[str, int, Side, float]] = []
        self.fill_result = True

    async def reconcile_vanished(self, symbol: str, tickets: set[int]) -> None:
        self.vanished_calls.append((symbol, tickets))

    async def reconcile_pending_fill(
        self, symbol: str, ticket: int, side: Side, volume: float
    ) -> bool:
        self.pending_fill_calls.append((symbol, ticket, side, volume))
        return self.fill_result


def _pending_order(**overrides) -> PendingOrder:
    defaults = dict(
        ticket=50,
        symbol="XAUUSD",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        volume=0.1,
        price=2405.0,
        sl=2395.0,
        tp=2420.0,
        placed_time=datetime.now(UTC),
        comment="",
    )
    defaults.update(overrides)
    return PendingOrder(**defaults)


class FakeMarketData:
    def __init__(self, info: SymbolInfo = INFO, candles: list[Candle] | None = None) -> None:
        self.info = info
        self.candles = candles or []
        self.symbol_info_calls: list[str] = []

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        self.symbol_info_calls.append(symbol)
        return self.info

    async def get_candles(
        self, symbol: str, timeframe: Timeframe, count: int, before: datetime | None = None
    ) -> list[Candle]:
        return self.candles[-count:]


def _candle(i: int, o: float, h: float, low: float, c: float) -> Candle:
    return Candle(
        symbol="XAUUSD",
        timeframe=Timeframe.M5,
        time=datetime(2026, 1, 1, tzinfo=UTC) + i * timedelta(minutes=5),
        open=o,
        high=h,
        low=low,
        close=c,
        tick_volume=1000,
        spread_points=20,
    )


def _flat_candles(n: int) -> list[Candle]:
    return [_candle(i, 100.0, 100.6, 99.4, 100.4) for i in range(n)]


def _demand_base_candles() -> list[Candle]:
    """35 flat warmup candles, then a clean RBR base [103.6, 104.4],
    unbroken — same geometry proven in test_rbr_dbd_zones_scalp_xauusd's
    `test_detect_zones_finds_rbr_with_retest`."""
    bars = _flat_candles(35)
    i = len(bars)
    bars.append(_candle(i, 100.4, 104.2, 100.0, 104.0))  # rally in
    bars.append(_candle(i + 1, 104.0, 104.4, 103.6, 104.1))  # base
    bars.append(_candle(i + 2, 104.1, 108.3, 104.0, 108.0))  # rally out
    return bars


def _supply_base_candles() -> list[Candle]:
    """Mirror of `_demand_base_candles`: a clean DBD base [95.6, 96.4]."""
    bars = _flat_candles(35)
    i = len(bars)
    bars.append(_candle(i, 100.0, 100.0, 95.8, 96.0))  # drop in
    bars.append(_candle(i + 1, 96.0, 96.4, 95.6, 95.9))  # base
    bars.append(_candle(i + 2, 95.9, 95.9, 91.7, 92.0))  # drop out
    return bars


# ---- volatility guard (bot-agnostic, engine-level) --------------------------

# Small atr_period/regime_lookback_bars so a short, hand-built candle series
# (rather than hundreds of bars of real market data) is enough to exercise a
# specific regime deterministically.
VOLATILITY_CFG = VolatilityConfig(atr_period=3, regime_lookback_bars=10)


def _volatility_ramp_candles(count: int, *, last_frac: float | None = None) -> list[Candle]:
    """Widening-true-range, zero-body candles (open == close == 100.0 on
    every bar, so `detect_bases` never finds a base here — see
    `test_no_secure_without_any_base`'s flat-candle precedent): bar `i`'s
    range is `0.1 + i*0.1`, so `latest_volatility_regime` ranks the most
    recent bar's ATR against its own trailing history. `last_frac=None`
    keeps the ramp increasing through the final bar (drives its ATR to the
    very top of its history -> EXTREME); `last_frac=0.85` caps the final
    bar's range at 85% of the ramp's peak, landing it in HIGH territory
    instead (elevated, but not the outlier)."""
    candles = []
    for i in range(count):
        rng = 0.1 + i * 0.1
        if last_frac is not None and i == count - 1:
            max_rng = 0.1 + (count - 2) * 0.1
            rng = max_rng * last_frac
        candles.append(_candle(i, 100.0, 100.0 + rng / 2, 100.0 - rng / 2, 100.0))
    return candles


def _regime_and_atr(candles: list[Candle], cfg: VolatilityConfig = VOLATILITY_CFG):
    """Computes `latest_volatility_regime` off `candles` the same way
    `PositionManager._detect_bases` does, so tests can assert against the
    exact regime/ATR the production code will see for a given fixture
    instead of hand-deriving (and risking a mismatched) expected float."""
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    return latest_volatility_regime(
        highs,
        lows,
        closes,
        atr_period=cfg.atr_period,
        regime_lookback_bars=cfg.regime_lookback_bars,
        low_percentile=cfg.low_percentile,
        high_percentile=cfg.high_percentile,
        extreme_percentile=cfg.extreme_percentile,
    )


async def test_apply_strategy_action_close_closes_position():
    position = _position(open_price=2400.0, sl=2390.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.apply_strategy_action(
        position, ExitDecision(action=ExitActionKind.CLOSE, reason="fvg violated")
    )

    assert order_service.closed == [1]
    assert order_service.close_reasons == ["fvg violated"]
    assert order_service.modified == []


async def test_apply_strategy_action_breakeven_moves_sl_to_entry():
    position = _position(open_price=2400.0, sl=2390.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.apply_strategy_action(
        position, ExitDecision(action=ExitActionKind.BREAKEVEN, reason="continuation confirmed")
    )

    assert order_service.modified == [(1, 2400.0, 2420.0)]
    assert order_service.modify_reasons == ["continuation confirmed"]
    assert order_service.closed == []


async def test_apply_strategy_action_breakeven_never_loosens_an_already_tighter_sl():
    # SL already tighter than entry (e.g. `_manage`'s give-back rule already
    # ratcheted it past breakeven) — a strategy-requested breakeven must
    # never loosen it back to entry, same invariant every other SL rule
    # obeys via `_improves`.
    position = _position(open_price=2400.0, sl=2405.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.apply_strategy_action(position, ExitDecision(action=ExitActionKind.BREAKEVEN))

    assert order_service.modified == []


async def test_apply_strategy_action_breakeven_sets_sl_when_position_has_none():
    position = _position(open_price=2400.0, sl=None)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.apply_strategy_action(position, ExitDecision(action=ExitActionKind.BREAKEVEN))

    assert order_service.modified == [(1, 2400.0, 2420.0)]


async def test_apply_strategy_action_breakeven_respects_sell_direction():
    # For a SELL, breakeven only improves when it moves SL *down* toward
    # entry; an SL already below entry must not be loosened back up to it.
    position = _position(side=Side.SELL, open_price=2400.0, sl=2395.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.apply_strategy_action(position, ExitDecision(action=ExitActionKind.BREAKEVEN))

    assert order_service.modified == []


async def test_moves_sl_to_breakeven_once_risk_is_covered():
    # risk = open(2400) - sl(2390) = 10; bid(2410) - open(2400) = 10 >= risk
    position = _position(open_price=2400.0, sl=2390.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 2400.0, 2420.0)]
    assert order_service.closed == []


async def test_does_not_move_sl_before_risk_is_covered():
    # risk = 2400-2380=20; progress = bid(2410)-2400=10 < risk
    position = _position(open_price=2400.0, sl=2380.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == []


async def test_time_stop_closes_position_without_progress():
    # sell position marked at ask(2410.20) > open(2400) -> losing, no progress
    position = _position(side=Side.SELL, open_price=2400.0, sl=2420.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData(), time_stop_candles=2)

    await manager.on_candle_closed("XAUUSD")
    await manager.on_candle_closed("XAUUSD")

    assert order_service.closed == [1]


async def test_get_symbol_info_fetched_once_per_symbol_with_multiple_positions():
    # Two open positions on the same symbol must share a single
    # get_symbol_info call per on_candle_closed cycle, same hoisting pattern
    # already used for _detect_bases -- not one fetch per position.
    positions = [
        _position(ticket=1, open_price=2400.0, sl=2390.0),
        _position(ticket=2, open_price=2400.0, sl=2390.0),
    ]
    order_service = FakeOrderService(positions)
    market_data = FakeMarketData()
    manager = PositionManager(order_service, market_data)

    await manager.on_candle_closed("XAUUSD")

    assert market_data.symbol_info_calls == ["XAUUSD"]
    # both positions were still managed off that single fetched info
    assert order_service.modified == [(1, 2400.0, 2420.0), (2, 2400.0, 2420.0)]


# ---- secure-on-base-clear (bot-agnostic profit protection) -------------------


async def test_secures_profit_when_fresh_base_is_cleared():
    # RBR base at [103.6, 104.4]; bid 108.0 clears it. Also satisfies +1R
    # breakeven (progress 8 >= risk 5), but structural continuation trailing
    # (Rule A) ratchets SL to base low minus buffer: 103.6 - (5 * 0.2) = 102.6,
    # winning over both plain breakeven (100.0) and standard secure (+0.2R = 101.0).
    position = _position(open_price=100.0, sl=95.0)
    info = replace(INFO, bid=108.0, ask=108.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_demand_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 102.6, 2420.0)]


async def test_secures_profit_independent_of_breakeven_rule():
    # risk = 10; progress = 4.5 < risk -> +1R breakeven does NOT trigger, but
    # the base [103.6, 104.4] is already cleared by bid 104.5 -> secure rule
    # fires on its own: 100 + 10*0.2 = 102.0.
    position = _position(open_price=100.0, sl=90.0)
    info = replace(INFO, bid=104.5, ask=104.7)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_demand_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 102.0, 2420.0)]


async def test_no_secure_when_base_not_yet_cleared():
    # bid 104.0 is inside the base [103.6, 104.4], not clear of it yet.
    position = _position(open_price=100.0, sl=90.0)
    info = replace(INFO, bid=104.0, ask=104.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_demand_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == []


async def test_no_secure_without_any_base():
    # Flat history has no RBR/DBD/RBD/DBR structure at all -> nothing to
    # secure against even though price (bid) has run up.
    position = _position(open_price=100.0, sl=90.0)
    info = replace(INFO, bid=108.0, ask=108.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData(info=info, candles=_flat_candles(60)))

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == []


async def test_secure_rule_never_loosens_an_already_better_sl():
    # sl already at 103.0 -- better than the secure candidate (102.0) for a
    # buy -- so the base-clear rule must not move it backward.
    position = _position(open_price=100.0, sl=103.0)
    info = replace(INFO, bid=104.5, ask=104.7)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_demand_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == []


async def test_secures_profit_for_sell_side():
    # DBD base at [95.6, 96.4]; ask 90.0 clears it below. risk = 10;
    # progress = 100 - 90 = 10 >= risk -> breakeven candidate is 100.0, but
    # secure candidate (100 - 10*0.2 = 98.0) is more protective and wins.
    position = _position(side=Side.SELL, open_price=100.0, sl=110.0, tp=80.0)
    info = replace(INFO, bid=89.8, ask=90.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_supply_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 98.0, 80.0)]


async def test_structural_continuation_trailing_for_sell_side():
    # DBD base at [95.6, 96.4]; ask 90.0 clears it below. risk = 5 (105 to 110);
    # buffer = 5 * 0.2 = 1.0. Standard secure breakeven is 104.0, but the cleared
    # base sits further in profit, so structural trailing sets SL to base high
    # plus buffer: 96.4 + 1.0 = 97.4.
    position = _position(side=Side.SELL, open_price=105.0, sl=110.0, tp=80.0)
    info = replace(INFO, bid=89.8, ask=90.0)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_supply_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 97.4, 80.0)]


def test_select_nearest_opposing_base():
    manager = PositionManager(FakeOrderService([]), FakeMarketData())
    bases = [
        Base(BaseKind.SUPPLY, price_low=105.0, price_high=107.0, base_start=1, leg_out_end=3, broken=False),
        Base(BaseKind.SUPPLY, price_low=110.0, price_high=112.0, base_start=5, leg_out_end=7, broken=False),
        Base(BaseKind.DEMAND, price_low=95.0, price_high=97.0, base_start=9, leg_out_end=11, broken=False),
    ]

    res_buy = manager._select_nearest_opposing_base(bases, Side.BUY, 104.0)
    assert res_buy is not None and res_buy.price_low == 105.0

    res_sell = manager._select_nearest_opposing_base(bases, Side.SELL, 99.0)
    assert res_sell is not None and res_sell.price_high == 97.0

    broken_supply = Base(BaseKind.SUPPLY, price_low=103.0, price_high=104.0, base_start=1, leg_out_end=3, broken=True)
    assert manager._select_nearest_opposing_base([broken_supply], Side.BUY, 102.0) is None


async def test_zone_contraire_defensive_breakeven_for_buy():
    # BUY position approaching opposing Supply zone at [95.6, 96.4].
    # open=85.0, sl=75.0 -> risk=10.0. 0.5 * risk = 5.0.
    # At bid=91.0, distance to proximal edge (95.6) is 4.6 <= 5.0.
    # Progress (6.0) < risk (10.0), so plain +1R breakeven does not trigger.
    # Zone Contraire rule triggers defensive breakeven lock-in: 85.0 + 10.0 * 0.2 = 87.0.
    position = _position(open_price=85.0, sl=75.0, tp=120.0)
    info = replace(INFO, bid=91.0, ask=91.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_supply_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 87.0, 120.0)]


async def test_zone_contraire_defensive_breakeven_for_sell():
    # SELL position approaching opposing Demand zone at [103.6, 104.4].
    # open=112.0, sl=118.0 -> risk=6.0. 0.5 * risk = 3.0.
    # At ask=106.5, distance to proximal edge (104.4) is 2.1 <= 3.0.
    # Progress is 5.5 < risk 6.0, so plain +1R breakeven does not trigger.
    # Zone Contraire rule triggers defensive breakeven lock-in: 112.0 - 6.0 * 0.2 = 110.8.
    position = _position(side=Side.SELL, open_price=112.0, sl=118.0, tp=90.0)
    info = replace(INFO, bid=106.3, ask=106.5)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service, FakeMarketData(info=info, candles=_demand_base_candles())
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 110.8, 90.0)]


async def test_no_sl_means_position_is_left_alone():
    position = _position(sl=None)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData(), time_stop_candles=1)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == []
    assert order_service.closed == []


async def test_paper_pending_order_fills_when_triggered_and_gate_approves():
    # ask=2410.20 <= price(2415.0) -> buy-limit triggers
    order_service = FakeOrderService([], pending=[_pending_order(price=2415.0)])
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    manager = PositionManager(order_service, FakeMarketData(), risk_manager=risk_manager)

    await manager.on_candle_closed("XAUUSD")

    assert len(order_service.opened) == 1
    assert order_service.pending_cancelled == [50]
    assert risk_manager.status.trades_today == 1


async def test_paper_pending_order_stays_resting_when_not_triggered():
    # ask=2410.20 > price(2405.0) -> buy-limit not yet triggered
    order_service = FakeOrderService([], pending=[_pending_order(price=2405.0)])
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    manager = PositionManager(order_service, FakeMarketData(), risk_manager=risk_manager)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.opened == []
    assert order_service.pending_cancelled == []


async def test_paper_pending_order_left_pending_when_risk_cap_blocks():
    blocked_caps = RiskCaps(
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=2.0,
        max_open_positions=0,
        max_trades_per_day_enabled=False,
        consecutive_loss_pause=3,
    )
    order_service = FakeOrderService([], pending=[_pending_order(price=2415.0)])
    risk_manager = RiskManager(caps=blocked_caps, timezone="UTC")
    manager = PositionManager(order_service, FakeMarketData(), risk_manager=risk_manager)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.opened == []
    assert order_service.pending_cancelled == []


async def test_live_mode_reconciles_a_pending_order_that_vanished():
    pending_order = _pending_order(ticket=50)
    order_service = FakeOrderService([], pending=[pending_order], simulates_pending_fills=False)
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    reconciliation = FakeReconciliation()
    manager = PositionManager(
        order_service, FakeMarketData(), reconciliation=reconciliation, risk_manager=risk_manager
    )

    await manager.on_candle_closed("XAUUSD")  # seed: ticket 50 seen resting
    order_service._pending = []  # MT5 triggered it server-side
    await manager.on_candle_closed("XAUUSD")

    assert reconciliation.pending_fill_calls == [("XAUUSD", 50, Side.BUY, 0.1)]
    assert risk_manager.status.trades_today == 1


async def test_live_mode_does_not_record_trade_when_no_match_found():
    pending_order = _pending_order(ticket=50)
    order_service = FakeOrderService([], pending=[pending_order], simulates_pending_fills=False)
    risk_manager = RiskManager(caps=CAPS, timezone="UTC")
    reconciliation = FakeReconciliation()
    reconciliation.fill_result = False
    manager = PositionManager(
        order_service, FakeMarketData(), reconciliation=reconciliation, risk_manager=risk_manager
    )

    await manager.on_candle_closed("XAUUSD")
    order_service._pending = []
    await manager.on_candle_closed("XAUUSD")

    assert reconciliation.pending_fill_calls == [("XAUUSD", 50, Side.BUY, 0.1)]
    assert risk_manager.status.trades_today == 0


async def test_risk_manager_none_skips_pending_order_handling():
    order_service = FakeOrderService([], pending=[_pending_order(price=2415.0)])
    manager = PositionManager(order_service, FakeMarketData())

    await manager.on_candle_closed("XAUUSD")

    assert order_service.opened == []
    assert order_service.pending_cancelled == []


# ---- volatility guard (bot-agnostic, engine-level) --------------------------


async def test_extreme_regime_while_losing_closes_position_outright():
    candles = _volatility_ramp_candles(45)
    regime, _pct, _atr = _regime_and_atr(candles)
    assert regime is VolatilityRegime.EXTREME  # sanity on the fixture

    # risk = 10; bid(2395) - open(2400) = -5 <= 0 -> losing.
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2395.0, ask=2395.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service,
        FakeMarketData(info=info, candles=candles),
        volatility_config=VOLATILITY_CFG,
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.closed == [1]
    assert order_service.modified == []


async def test_extreme_regime_while_winning_locks_in_profit_fraction():
    candles = _volatility_ramp_candles(45)
    regime, _pct, _atr = _regime_and_atr(candles)
    assert regime is VolatilityRegime.EXTREME  # sanity on the fixture

    # risk = 10; progress = bid(2408) - open(2400) = 8 < risk -> plain +1R
    # breakeven does not trigger, so the profit-lock candidate wins outright:
    # 2400 + 8 * extreme_profit_lock_r_mult(0.5) = 2404.0.
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2408.0, ask=2408.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service,
        FakeMarketData(info=info, candles=candles),
        volatility_config=VOLATILITY_CFG,
    )

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 2404.0, 2420.0)]
    assert order_service.closed == []


async def test_high_regime_running_profit_applies_chandelier_trailing_stop():
    candles = _volatility_ramp_candles(45, last_frac=0.85)
    regime, _pct, atr_value = _regime_and_atr(candles)
    assert regime is VolatilityRegime.HIGH  # sanity on the fixture

    # risk = 10; progress = bid(2415) - open(2400) = 15 >=
    # chandelier_min_profit_r(1.0) * risk -> chandelier rule is eligible, and
    # (favorable=2415) - chandelier_atr_mult(2.0)*atr_value beats the plain
    # +1R breakeven candidate (2400.0).
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2415.0, ask=2415.2)
    order_service = FakeOrderService([position])
    market_data = FakeMarketData(info=info, candles=candles)
    manager = PositionManager(order_service, market_data, volatility_config=VOLATILITY_CFG)

    await manager.on_candle_closed("XAUUSD")

    expected_sl = 2415.0 - VOLATILITY_CFG.chandelier_atr_mult * atr_value
    assert order_service.modified == [(1, expected_sl, 2420.0)]

    # Simulate the SL having actually been tightened to expected_sl, then
    # price pulls back but stays above the trailed level -- the chandelier
    # rule must not loosen it back, and no further modify_position call
    # should happen at all.
    order_service._positions = [replace(position, sl=expected_sl)]
    market_data.info = replace(INFO, bid=2410.0, ask=2410.2)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, expected_sl, 2420.0)]


async def test_disabled_volatility_guard_skips_extreme_and_high_rules():
    # Same EXTREME-losing fixture as
    # test_extreme_regime_while_losing_closes_position_outright, but with the
    # live guard switched off -- volatility_config is still supplied, yet
    # _detect_bases must no longer classify a regime at all, so _manage sees
    # regime=None and reproduces pre-Phase-B behavior (no outright close).
    candles = _volatility_ramp_candles(45)
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2395.0, ask=2395.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(
        order_service,
        FakeMarketData(info=info, candles=candles),
        volatility_config=VOLATILITY_CFG,
    )
    manager.set_volatility_guard_enabled(False)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.closed == []
    assert order_service.modified == []


async def test_disabled_volatility_guard_skips_high_regime_chandelier_rule():
    # Same HIGH-regime chandelier fixture as
    # test_high_regime_running_profit_applies_chandelier_trailing_stop, but
    # with the live guard switched off -- no chandelier trailing, and since
    # progress(15) >= risk(10), only the plain +1R breakeven candidate
    # (2400.0) is available among the ordinary SL-tightening rules.
    candles = _volatility_ramp_candles(45, last_frac=0.85)
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2415.0, ask=2415.2)
    order_service = FakeOrderService([position])
    market_data = FakeMarketData(info=info, candles=candles)
    manager = PositionManager(order_service, market_data, volatility_config=VOLATILITY_CFG)
    manager.set_volatility_guard_enabled(False)

    await manager.on_candle_closed("XAUUSD")

    assert order_service.modified == [(1, 2400.0, 2420.0)]


async def test_volatility_config_none_disables_all_volatility_rules():
    # Regression guard: the exact EXTREME-losing fixture from
    # test_extreme_regime_while_losing_closes_position_outright, but with no
    # volatility_config -- must reproduce pre-Phase-B behavior exactly (no
    # outright close; ordinary SL-tightening rules decide, and none of them
    # fire here since progress < risk and there are no bases).
    candles = _volatility_ramp_candles(45)
    position = _position(open_price=2400.0, sl=2390.0)
    info = replace(INFO, bid=2395.0, ask=2395.2)
    order_service = FakeOrderService([position])
    manager = PositionManager(order_service, FakeMarketData(info=info, candles=candles))

    await manager.on_candle_closed("XAUUSD")

    assert order_service.closed == []
    assert order_service.modified == []
