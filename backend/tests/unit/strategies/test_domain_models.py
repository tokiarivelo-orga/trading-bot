from src.strategies.domain.models import Direction, Signal


def test_signal_size_multiplier_defaults_to_one():
    # Every strategy that doesn't set size_multiplier must keep today's
    # behavior unchanged — trade_loop.py multiplies risk by this field, so
    # a default other than 1.0 would silently resize every existing bot.
    signal = Signal(direction=Direction.BUY, sl_points=10.0, tp_points=15.0)
    assert signal.size_multiplier == 1.0


def test_signal_size_multiplier_round_trips_explicit_value():
    signal = Signal(direction=Direction.SELL, sl_points=12.0, tp_points=20.0, size_multiplier=2.0)
    assert signal.size_multiplier == 2.0
