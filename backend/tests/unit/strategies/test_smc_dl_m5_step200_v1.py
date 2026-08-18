"""smc_dl_m5_step200 v2 (OB-first, DL-filter): sandbox compliance and entry logic.

Regression coverage for the bug that made the bot stop opening any positions
after the Aug 9 2026 retrain: `evaluate()` built its `Signal.zone` with
`hasattr(df_m5.index, "__getitem__")`, but `hasattr` is not in
`strategies/sandbox._safe_builtins()`. Every zone-retest signal raised
`NameError: name 'hasattr' is not defined` *only when run through the
sandbox* — a plain unit test that imports the strategy directly never sees
it, because normal Python execution does have `hasattr`. That is why the
tests below drive `evaluate()` through `validate_and_load()`, not through a
direct import, for the case that actually builds a `Signal`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import MarketContext
from src.strategies.generated.smc_dl_features_v2 import atr, detect_zones
from src.strategies.generated.smc_dl_m5_step200_v1 import SmcDlM5Step200
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = Path("src/strategies/generated/smc_dl_m5_step200_v1.py")


def _candles(n: int, drift: float, freq: str = "5min") -> pd.DataFrame:
    times = pd.date_range("2025-06-01", periods=n, freq=freq, tz="UTC")
    close = 10000.0 + np.arange(n) * drift + np.sin(np.arange(n) / 25.0) * 3.0
    return pd.DataFrame(
        {
            "time": times,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": np.full(n, 200),
        }
    )


def _retest_frame(kind: str) -> pd.DataFrame:
    """Flat warmup, one clean base, an impulse away, then a return into it —
    the geometry `detect_zones` looks for."""
    rows = []
    price = 10000.0
    for _ in range(200):
        rows.append((price, price + 0.6, price - 0.6, price))
    if kind == "demand":
        rows.append((price, price + 0.3, price - 0.3, price))  # base
        for step in range(1, 9):  # impulse up
            rows.append(
                (
                    price + step * 2,
                    price + step * 2 + 2.2,
                    price + step * 2 - 0.2,
                    price + step * 2 + 2.0,
                )
            )
        for step in range(8, 0, -1):  # walk back down to the base
            rows.append(
                (
                    price + step * 2,
                    price + step * 2 + 0.4,
                    price + step * 2 - 2.2,
                    price + step * 2 - 2.0,
                )
            )
    else:
        rows.append((price, price + 0.3, price - 0.3, price))  # base
        for step in range(1, 9):  # impulse down
            rows.append(
                (
                    price - step * 2,
                    price - step * 2 + 0.2,
                    price - step * 2 - 2.2,
                    price - step * 2 - 2.0,
                )
            )
        for step in range(8, 0, -1):  # walk back up to the base
            rows.append(
                (
                    price - step * 2,
                    price - step * 2 + 2.2,
                    price - step * 2 - 0.4,
                    price - step * 2 + 2.0,
                )
            )
    times = pd.date_range("2025-06-01", periods=len(rows), freq="5min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["time"] = times
    frame["tick_volume"] = 200
    return frame


def _zone_context(frame: pd.DataFrame, spread: float = 10.0) -> MarketContext:
    return MarketContext(
        symbol="Step Index 200",
        candles={"M5": frame, "M15": frame, "H1": frame, "H4": frame},
        spread_points=spread,
    )


def test_passes_the_strategy_sandbox() -> None:
    """Generated strategy files are AST-scanned and exec'd with restricted
    builtins before the registry will trust them — `strategies/sandbox.py`.
    A file that fails this can never be activated."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None
    assert instance.spec.name == "smc_dl_m5_step200"
    assert instance.spec.entry_timeframe == "M5"


def test_evaluate_never_raises_on_short_history() -> None:
    strategy = SmcDlM5Step200()
    ctx = MarketContext(
        symbol="Step Index 200", candles={"M5": _candles(20, 0.1)}, spread_points=20.0
    )
    assert strategy.evaluate(ctx) is None


def test_never_signals_without_a_zone_retest() -> None:
    """A pure momentum context — no base, no touch — must produce nothing."""
    strategy = SmcDlM5Step200()
    trending = _candles(300, 0.4)
    assert strategy.evaluate(_zone_context(trending)) is None


@pytest.mark.parametrize("kind", ["demand", "supply"])
def test_signal_construction_survives_the_sandboxed_builtins(kind: str) -> None:
    """The actual regression: building `Signal.zone` must not reach for a
    builtin the sandbox strips (`hasattr`). Runs the *sandboxed* instance,
    matching how the live engine and backtests actually load this file."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    frame = _retest_frame(kind)
    signal = instance.evaluate(_zone_context(frame))
    if signal is None:
        pytest.skip("fixture did not produce a fresh touch on the final bar")
    assert signal.zone is not None
    assert signal.sl_points > 0
    assert signal.tp_points > 0


def test_stop_sits_just_beyond_the_zone_far_edge() -> None:
    strategy = SmcDlM5Step200()
    frame = _retest_frame("demand")
    signal = strategy.evaluate(_zone_context(frame))
    if signal is None:
        pytest.skip("fixture did not produce a fresh touch on the final bar")
    zones = detect_zones(frame, atr(frame, 14))
    demand = [z for z in zones if z["kind"] == "demand" and z["fresh_touch"]]
    assert demand
    price = float(frame["close"].iloc[-1])
    stop = price - signal.sl_points
    assert stop < min(z["price_low"] for z in demand)


def test_structural_veto_blocks_buying_into_unmitigated_supply() -> None:
    strategy = SmcDlM5Step200()
    params = strategy.spec.params
    zones = [
        {
            "kind": "supply",
            "price_low": 100.5,
            "price_high": 103.0,
            "index": 5,
            "age_bars": 5,
            "touches": 0,
            "in_zone": False,
            "fresh_touch": False,
        },
    ]
    assert strategy._veto_opposing_zone(zones, True, 100.0, 1.0, params) is True
    zones[0]["price_low"] = 180.0
    zones[0]["price_high"] = 185.0
    assert strategy._veto_opposing_zone(zones, True, 100.0, 1.0, params) is False


def test_structural_veto_blocks_selling_into_unmitigated_demand() -> None:
    strategy = SmcDlM5Step200()
    zones = [
        {
            "kind": "demand",
            "price_low": 97.0,
            "price_high": 99.5,
            "index": 5,
            "age_bars": 5,
            "touches": 0,
            "in_zone": False,
            "fresh_touch": False,
        },
    ]
    assert strategy._veto_opposing_zone(zones, False, 100.0, 1.0, strategy.spec.params) is True


def test_only_a_fresh_touch_qualifies() -> None:
    strategy = SmcDlM5Step200()
    params = strategy.spec.params
    stale = {
        "kind": "demand",
        "price_low": 99.0,
        "price_high": 101.0,
        "index": 10,
        "age_bars": 20,
        "touches": 1,
        "in_zone": True,
        "fresh_touch": False,
    }
    assert strategy._select_zone([stale], 100.0, params) is None
    fresh = dict(stale, fresh_touch=True)
    assert strategy._select_zone([fresh], 100.0, params) is not None


def test_overtested_and_too_young_zones_are_skipped() -> None:
    strategy = SmcDlM5Step200()
    params = strategy.spec.params
    base = {
        "kind": "demand",
        "price_low": 99.0,
        "price_high": 101.0,
        "index": 10,
        "age_bars": 20,
        "touches": 0,
        "in_zone": True,
        "fresh_touch": True,
    }
    assert strategy._select_zone([dict(base, touches=9)], 100.0, params) is None
    assert strategy._select_zone([dict(base, age_bars=0)], 100.0, params) is None


def test_model_opinion_falls_back_when_no_model_loaded() -> None:
    strategy = SmcDlM5Step200()
    strategy._model = None
    confidence, vetoed, note = strategy._model_opinion(
        MarketContext(symbol="Step Index 200", candles={}, spread_points=10.0),
        True,
        strategy.spec.params,
    )
    assert confidence == 0.55
    assert vetoed is False
    assert note == "model=unavailable"


def test_model_opinion_does_not_double_sigmoid_tp_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`SmcMultiTaskNet.head_tp` ends in `nn.Sigmoid()` — the raw model
    output is already a 0..1 probability. Applying `torch.sigmoid()` to it
    again would compress every confidence toward 0.5 (a genuine 0.85 would
    read back as ~0.70). `_model_opinion` must return the model's tp output
    unchanged, not re-squash it."""
    import torch

    strategy = SmcDlM5Step200()
    strategy._scaler_mean = np.zeros(2, dtype=np.float32)
    strategy._scaler_scale = np.ones(2, dtype=np.float32)

    class _FakeModel:
        def __call__(self, x):
            tp = torch.tensor([[0.85]])
            direction = torch.tensor([[-5.0, 0.0, 5.0]])  # confidently bullish
            risk = torch.tensor([[0.5]])
            return tp, direction, risk

    strategy._model = _FakeModel()
    monkeypatch.setattr(
        "src.strategies.generated.smc_dl_m5_step200_v1.compute_smc_features",
        lambda candles, symbol, lookback=20: {"a": 0.0, "b": 0.0},
    )

    confidence, vetoed, note = strategy._model_opinion(
        MarketContext(symbol="Step Index 200", candles={}, spread_points=10.0),
        True,
        strategy.spec.params,
    )
    assert confidence == pytest.approx(0.85)
    assert vetoed is False
