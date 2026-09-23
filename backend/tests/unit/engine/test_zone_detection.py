"""Unit tests for the engine-side RBR/DBD/RBD/DBR base detector used by
`PositionManager`'s secure-on-base-clear rule."""

from __future__ import annotations

import numpy as np

from src.engine.domain.zone_detection import BaseKind, atr, detect_bases


def _flat(n: int, o: float = 100.0, h: float = 100.6, low: float = 99.4, c: float = 100.4):
    return [(o, h, low, c)] * n


def _arrays(bars: list[tuple[float, float, float, float]]):
    opens = np.array([b[0] for b in bars])
    highs = np.array([b[1] for b in bars])
    lows = np.array([b[2] for b in bars])
    closes = np.array([b[3] for b in bars])
    return opens, highs, lows, closes


def test_detects_unbroken_demand_base():
    bars = _flat(35) + [
        (100.4, 104.2, 100.0, 104.0),  # rally in
        (104.0, 104.4, 103.6, 104.1),  # base
        (104.1, 108.3, 104.0, 108.0),  # rally out
    ]
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    bases = detect_bases(opens, highs, lows, closes, atr_values)

    assert len(bases) == 1
    base = bases[0]
    assert base.kind == BaseKind.DEMAND
    assert base.price_low == 103.6
    assert base.price_high == 104.4
    assert base.broken is False


def test_detects_unbroken_supply_base():
    bars = _flat(35) + [
        (100.0, 100.0, 95.8, 96.0),  # drop in
        (96.0, 96.4, 95.6, 95.9),  # base
        (95.9, 95.9, 91.7, 92.0),  # drop out
    ]
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    bases = detect_bases(opens, highs, lows, closes, atr_values)

    assert len(bases) == 1
    base = bases[0]
    assert base.kind == BaseKind.SUPPLY
    assert base.price_low == 95.6
    assert base.price_high == 96.4
    assert base.broken is False


def test_flags_a_base_broken_by_a_later_close_through_it():
    bars = _flat(35) + [
        (100.4, 104.2, 100.0, 104.0),  # rally in
        (104.0, 104.4, 103.6, 104.1),  # base
        (104.1, 108.3, 104.0, 108.0),  # rally out
        (108.0, 108.1, 102.9, 103.1),  # closes back through price_low(103.6)
    ]
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    bases = detect_bases(opens, highs, lows, closes, atr_values)

    assert len(bases) == 1
    assert bases[0].broken is True


def test_no_bases_on_flat_history():
    bars = _flat(60)
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    assert detect_bases(opens, highs, lows, closes, atr_values) == []


def test_no_bases_when_atr_never_settles():
    bars = _flat(5)
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    assert detect_bases(opens, highs, lows, closes, atr_values) == []


def test_detects_multibar_range_base_with_internal_bounce():
    # 35 flat bars around 100.0, ATR ~ 1.2
    bars = _flat(35, o=100.0, h=100.6, low=99.4, c=100.0) + [
        (100.0, 105.0, 99.8, 104.8),  # leg in (rally +4.8 points)
        # 8-bar range consolidation between 103.5 and 105.2
        (104.8, 105.2, 104.2, 104.5),
        (104.5, 104.8, 103.8, 104.0),
        (104.0, 105.1, 103.9, 105.0),  # internal bounce (+1.0 point, >= 0.7*ATR)
        (105.0, 105.1, 103.7, 103.8),  # internal drop (-1.2 point)
        (103.8, 104.5, 103.5, 104.2),
        (104.2, 104.9, 104.0, 104.5),
        (104.5, 104.8, 104.1, 104.4),
        (104.4, 104.7, 103.9, 104.6),
        (104.6, 110.0, 104.5, 109.8),  # true breakout leg out (closes 109.8 > 105.2)
    ]
    opens, highs, lows, closes = _arrays(bars)
    atr_values = atr(highs, lows, closes, 14)

    bases = detect_bases(opens, highs, lows, closes, atr_values)

    assert len(bases) == 1
    base = bases[0]
    assert base.kind == BaseKind.DEMAND
    assert base.price_low == 103.5
    assert base.price_high == 105.1
    assert base.broken is False
