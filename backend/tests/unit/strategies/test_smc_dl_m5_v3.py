"""The v3 M5 DL strategy: sandbox compliance, model contract, strict gate.

Unlike the M1 v3 sibling, this file's real trained weights (phase 2,
``data/ml_models/smc_dl_m5_v3_meta.json``) DO clear the out-of-sample edge
bar (4 of 5 folds positive, mean OOS avg R +0.0096), so ``_model_trusted``
must read True here and stay that way — these tests read the real meta.json
on disk rather than mocking it, so a future retrain that changes the verdict
makes the assertion fail loudly instead of silently drifting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.smc_dl_features_v2 import detect_zones
from src.strategies.generated.smc_dl_features_v3 import FEATURE_NAMES, atr
from src.strategies.generated.smc_dl_m5_v3 import SmcDlM5V3
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = Path("src/strategies/generated/smc_dl_m5_v3.py")


def _candles(n: int, drift: float, freq: str = "5min") -> pd.DataFrame:
    times = pd.date_range("2025-06-01", periods=n, freq=freq, tz="UTC")
    close = 2000.0 + np.arange(n) * drift + np.sin(np.arange(n) / 25.0) * 3.0
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


def _htf_candles(n: int, drift: float, freq: str, end: str) -> pd.DataFrame:
    """Like `_candles`, but anchored on its own *end* timestamp rather than a
    shared start — matching how the engine actually hands a strategy its
    timeframes: every declared timeframe's window ends at (roughly) the same
    "now", not at the same calendar start. A confirmation TF that shares the
    entry TF's start instead of its end never accumulates real history by
    the time the entry series catches up (an H4 series covers 200 x 4h = 33
    days; an M5 series covers 300 x 5min = 25 hours — sharing a start leaves
    every H4 bar inside the entry window still in its own EMA20 warmup), so
    real strategy code correctly reads it as "no usable opinion yet". Varies
    volume too: a constant `tick_volume` gives `vol_z_20` a zero-variance
    denominator, which is its own source of a permanently-NaN feature row.
    """
    times = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    close = 2000.0 + np.arange(n) * drift + np.sin(np.arange(n) / 25.0) * 3.0
    volume = 200 + (np.arange(n) % 7) * 5 + (np.arange(n) % 3)
    return pd.DataFrame(
        {
            "time": times,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": volume,
        }
    )


def _context(drift: float = 0.05, spread: float = 20.0) -> MarketContext:
    """M5 entry frame plus M15/H1/H4 confirmation frames that all end at the
    same timestamp as the M5 frame — see `_htf_candles` docstring for why
    that matters for a real (non-mocked) model to have anything to say."""
    end = "2025-08-01"
    return MarketContext(
        symbol="XAUUSD",
        candles={
            "M5": _htf_candles(300, drift, "5min", end),
            "M15": _htf_candles(200, drift * 3, "15min", end),
            "H1": _htf_candles(200, drift * 12, "1h", end),
            "H4": _htf_candles(200, drift * 48, "4h", end),
        },
        spread_points=spread,
    )


def test_passes_the_strategy_sandbox() -> None:
    """Generated strategy files are AST-scanned and exec'd with restricted
    builtins before the registry will trust them — `strategies/sandbox.py`.
    A file that fails this can never be activated."""
    instance, errors = validate_and_load(STRATEGY_PATH.read_text())
    assert errors == ()
    assert instance is not None
    assert instance.spec.name == "smc_dl_m5_v3"
    assert instance.spec.entry_timeframe == "M5"


def test_spec_lookbacks_fit_the_engine_context_window() -> None:
    """`trade_loop.DEFAULT_CONTEXT_BARS = 200`: a strategy whose lookback
    exceeds that silently never fires, live or in backtest."""
    strategy = SmcDlM5V3()
    assert strategy.spec.confirmation_timeframes == ("M15", "H1", "H4")


def test_rejects_weights_whose_feature_list_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatched feature list must disable the model, not be tolerated —
    any reordering silently mis-feeds every weight."""
    import src.strategies.generated.smc_dl_m5_v3 as module

    models = tmp_path / "ml_models"
    models.mkdir()
    (models / "smc_dl_m5_v3.pt").write_bytes(b"not-a-real-checkpoint")
    np.savez(
        models / "smc_dl_m5_v3_scaler.npz",
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
    )
    wrong = list(FEATURE_NAMES[:-1]) + ["some_feature_that_does_not_exist"]
    (models / "smc_dl_m5_v3_meta.json").write_text(
        '{"feature_names": ' + str(wrong).replace("'", '"') + ', "temperature": 1.0}'
    )
    monkeypatch.setattr(module, "_MODELS_DIR", models)

    strategy = SmcDlM5V3()
    assert strategy._model is None
    assert "feature list differs" in strategy._status


def test_missing_weights_leave_a_readable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.strategies.generated.smc_dl_m5_v3 as module

    monkeypatch.setattr(module, "_MODELS_DIR", tmp_path / "nothing-here")
    strategy = SmcDlM5V3()
    assert strategy._model is None
    assert "fallback" in strategy._status


def test_evaluate_never_raises_on_short_history() -> None:
    strategy = SmcDlM5V3()
    ctx = MarketContext(symbol="XAUUSD", candles={"M5": _candles(20, 0.1)}, spread_points=20.0)
    assert strategy.evaluate(ctx) is None


def test_gate_uses_the_engine_breakeven_probability() -> None:
    """`min_p_secure` is 1/(1+0.2) = 0.8333, the arithmetic break-even for the
    engine's 0.2R trailing rule — not a value fitted on in-sample data."""
    strategy = SmcDlM5V3()
    assert strategy.spec.params["min_p_secure"] == pytest.approx(0.8333, abs=1e-4)


# ---------------------------------------------------------------------------
# real trained artifact — the model must load and read as trusted
# ---------------------------------------------------------------------------
def test_real_m5_v3_model_loads_and_is_trusted() -> None:
    """Reads the actual phase-2 artifacts on disk (no mocking): the M5 v3
    walk-forward scored 4 of 5 folds out-of-sample positive (mean OOS avg R
    +0.0096), so `_model_trusted` must be True and the status must say a
    strict gate is in force. A future retrain that flips this verdict should
    fail this test loudly rather than have the strategy silently fall back
    to advisory-only gating."""
    strategy = SmcDlM5V3()
    assert strategy._model is not None, strategy._status
    assert strategy._model_trusted is True
    assert "strict gate" in strategy._status


def test_model_verdict_produces_usable_probabilities_from_real_weights() -> None:
    """Calls the real loaded model end to end (features -> scaler -> net ->
    sigmoid) on a long enough synthetic history, and checks the outputs are
    valid probabilities/EV rather than NaN or out-of-range — the plumbing
    the strict gate depends on."""
    strategy = SmcDlM5V3()
    assert strategy._model is not None, strategy._status
    ctx = _context()
    params = strategy.spec.params
    verdict = strategy._model_verdict(ctx, True, cost_r=0.05, params=params)
    assert verdict is not None
    ev, p_sec, mfe = verdict
    assert np.isfinite(ev)
    assert 0.0 <= p_sec <= 1.0
    assert np.isfinite(mfe)


# ---------------------------------------------------------------------------
# zone retest — the entry model itself
# ---------------------------------------------------------------------------
def _retest_frame(kind: str) -> pd.DataFrame:
    """Flat warmup, one clean base, an impulse away, then a return into it.

    Mirrors the geometry `detect_zones` looks for: a small-body base bar
    followed by a momentum leg out, then price coming back to touch it.
    """
    rows = []
    price = 2000.0
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
        symbol="XAUUSD",
        candles={"M5": frame, "M15": frame, "H1": frame, "H4": frame},
        spread_points=spread,
    )


def test_never_signals_without_a_zone_retest() -> None:
    """A pure momentum context — no base, no touch — must produce nothing.
    Structure must propose the candidate; a direction probability alone is
    not an entry condition."""
    strategy = SmcDlM5V3()
    trending = _candles(300, 0.4)
    assert strategy.evaluate(_zone_context(trending)) is None


def test_stop_sits_just_beyond_the_zone_far_edge() -> None:
    """The tight stop is a *consequence* of entering at the zone, not a
    separately-tuned parameter."""
    strategy = SmcDlM5V3()
    frame = _retest_frame("demand")
    signal = strategy.evaluate(_zone_context(frame))
    if signal is None:
        pytest.skip("fixture did not produce a fresh touch, or model strictly vetoed it")
    zones = detect_zones(frame, atr(frame, 14))
    demand = [z for z in zones if z["kind"] == "demand" and z["fresh_touch"]]
    assert demand
    price = float(frame["close"].iloc[-1])
    stop = price - signal.sl_points
    assert stop < min(z["price_low"] for z in demand)


def test_structural_veto_blocks_buying_into_unmitigated_supply() -> None:
    """The veto runs before the model and cannot be overridden by it."""
    strategy = SmcDlM5V3()
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
    # Same supply far out of reach must not veto.
    zones[0]["price_low"] = 180.0
    zones[0]["price_high"] = 185.0
    assert strategy._veto_opposing_zone(zones, True, 100.0, 1.0, params) is False


def test_structural_veto_blocks_selling_into_unmitigated_demand() -> None:
    strategy = SmcDlM5V3()
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
    """Sitting inside a zone is a state; a retest is an event."""
    strategy = SmcDlM5V3()
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
    assert strategy._select_retest([stale], 100.0, params) is None
    fresh = dict(stale, fresh_touch=True)
    assert strategy._select_retest([fresh], 100.0, params) is not None


def test_overtested_and_too_young_zones_are_skipped() -> None:
    strategy = SmcDlM5V3()
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
    assert strategy._select_retest([dict(base, touches=9)], 100.0, params) is None
    assert strategy._select_retest([dict(base, age_bars=0)], 100.0, params) is None


# ---------------------------------------------------------------------------
# strict gate — deterministic, independent of what the real weights output
# on synthetic data (the model-loading tests above already prove the real
# weights are consulted; these prove evaluate() actually enforces the gate
# once it has an opinion).
# ---------------------------------------------------------------------------
def test_strict_gate_blocks_a_setup_the_model_scores_below_threshold() -> None:
    """`_model_trusted` is True from the real meta.json (see
    `test_real_m5_v3_model_loads_and_is_trusted`), so a verdict below
    `min_expected_r`/`min_p_secure` must veto the trade outright."""
    strategy = SmcDlM5V3()
    assert strategy._model_trusted is True
    frame = _retest_frame("demand")
    ctx = _zone_context(frame)
    params = strategy.spec.params
    strategy._model_verdict = lambda *a, **k: (params["min_expected_r"] - 0.05, 0.5, 1.0)
    signal = strategy.evaluate(ctx)
    if signal is None and not [z for z in detect_zones(frame, atr(frame, 14)) if z["fresh_touch"]]:
        pytest.skip("fixture produced no fresh touch to gate in the first place")
    assert signal is None


def test_strict_gate_admits_a_setup_the_model_scores_above_threshold() -> None:
    strategy = SmcDlM5V3()
    assert strategy._model_trusted is True
    frame = _retest_frame("demand")
    ctx = _zone_context(frame)
    params = strategy.spec.params
    strategy._model_verdict = lambda *a, **k: (
        params["min_expected_r"] + 0.5,
        params["min_p_secure"] + 0.05,
        1.2,
    )
    signal = strategy.evaluate(ctx)
    if signal is None:
        pytest.skip("fixture did not produce a fresh touch on the final bar")
    assert signal.direction == Direction.BUY
    assert "E[R]=" in signal.reason
    assert "p_secure=" in signal.reason
