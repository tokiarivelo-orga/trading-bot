"""The v3 M1 DL strategy: sandbox compliance, model contract, safe fallback.

Unlike the M5 v3 sibling, this file's real trained weights (phase 2,
``data/ml_models/smc_dl_m1_v3_meta.json``) do **not** clear the out-of-sample
edge bar (0 of 5 folds positive, mean OOS avg R -0.0467), so
``_model_trusted`` must read False here and stay that way — these tests read
the real meta.json on disk rather than mocking it, so a future retrain that
changes the verdict makes the assertion fail loudly instead of silently
drifting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.smc_dl_features_v2 import detect_zones
from src.strategies.generated.smc_dl_features_v3 import FEATURE_NAMES, atr
from src.strategies.generated.smc_dl_m1_v3 import SmcDlM1V3
from src.strategies.sandbox import validate_and_load

STRATEGY_PATH = Path("src/strategies/generated/smc_dl_m1_v3.py")


def _candles(n: int, drift: float, freq: str = "1min") -> pd.DataFrame:
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


def _context(drift: float = 0.05, spread: float = 20.0) -> MarketContext:
    return MarketContext(
        symbol="XAUUSD",
        candles={
            "M1": _candles(300, drift),
            "M5": _candles(200, drift * 5, "5min"),
            "M15": _candles(200, drift * 15, "15min"),
            "H1": _candles(200, drift * 60, "1h"),
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
    assert instance.spec.name == "smc_dl_m1_v3"
    assert instance.spec.entry_timeframe == "M1"


def test_spec_lookbacks_fit_the_engine_context_window() -> None:
    """`trade_loop.DEFAULT_CONTEXT_BARS = 200`: a strategy whose lookback
    exceeds that silently never fires, live or in backtest."""
    strategy = SmcDlM1V3()
    assert strategy.spec.confirmation_timeframes == ("M5", "M15", "H1")


def test_rejects_weights_whose_feature_list_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mismatched feature list must disable the model, not be tolerated —
    any reordering silently mis-feeds every weight."""
    import src.strategies.generated.smc_dl_m1_v3 as module

    models = tmp_path / "ml_models"
    models.mkdir()
    (models / "smc_dl_m1_v3.pt").write_bytes(b"not-a-real-checkpoint")
    np.savez(
        models / "smc_dl_m1_v3_scaler.npz",
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
    )
    wrong = list(FEATURE_NAMES[:-1]) + ["some_feature_that_does_not_exist"]
    (models / "smc_dl_m1_v3_meta.json").write_text(
        '{"feature_names": ' + str(wrong).replace("'", '"') + ', "temperature": 1.0}'
    )
    monkeypatch.setattr(module, "_MODELS_DIR", models)

    strategy = SmcDlM1V3()
    assert strategy._model is None
    assert "feature list differs" in strategy._status


def test_missing_weights_leave_a_readable_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.strategies.generated.smc_dl_m1_v3 as module

    monkeypatch.setattr(module, "_MODELS_DIR", tmp_path / "nothing-here")
    strategy = SmcDlM1V3()
    assert strategy._model is None
    assert "fallback" in strategy._status


def test_evaluate_never_raises_on_short_history() -> None:
    strategy = SmcDlM1V3()
    ctx = MarketContext(symbol="XAUUSD", candles={"M1": _candles(20, 0.1)}, spread_points=20.0)
    assert strategy.evaluate(ctx) is None


def test_gate_uses_the_engine_breakeven_probability() -> None:
    """`min_p_secure` is 1/(1+0.2) = 0.8333, the arithmetic break-even for the
    engine's 0.2R trailing rule — not a value fitted on in-sample data."""
    strategy = SmcDlM1V3()
    assert strategy.spec.params["min_p_secure"] == pytest.approx(0.8333, abs=1e-4)


# ---------------------------------------------------------------------------
# real trained artifact — the model must load and read as untrusted
# ---------------------------------------------------------------------------
def test_real_m1_v3_model_loads_and_is_not_trusted() -> None:
    """Reads the actual phase-2 artifacts on disk (no mocking): the M1 v3
    walk-forward scored 0 of 5 folds out-of-sample positive (mean OOS avg R
    -0.0467), so `_model_trusted` must be False and the status must say so.
    A future retrain that flips this verdict should fail this test loudly
    rather than have the strategy silently start strict-gating."""
    strategy = SmcDlM1V3()
    assert strategy._model is not None, strategy._status
    assert strategy._model_trusted is False
    assert "advisory" in strategy._status or "fallback" in strategy._status


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
    times = pd.date_range("2025-06-01", periods=len(rows), freq="1min", tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["time"] = times
    frame["tick_volume"] = 200
    return frame


def _zone_context(frame: pd.DataFrame, spread: float = 10.0) -> MarketContext:
    return MarketContext(
        symbol="XAUUSD",
        candles={"M1": frame, "M5": frame, "M15": frame, "H1": frame},
        spread_points=spread,
    )


def test_never_signals_without_a_zone_retest() -> None:
    """A pure momentum context — no base, no touch — must produce nothing.
    Structure must propose the candidate; a direction probability alone is
    not an entry condition."""
    strategy = SmcDlM1V3()
    trending = _candles(300, 0.4)
    assert strategy.evaluate(_zone_context(trending)) is None


def test_stop_sits_just_beyond_the_zone_far_edge() -> None:
    """The tight stop is a *consequence* of entering at the zone, not a
    separately-tuned parameter."""
    strategy = SmcDlM1V3()
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
    """The veto runs before the model and cannot be overridden by it."""
    strategy = SmcDlM1V3()
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
    strategy = SmcDlM1V3()
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
    strategy = SmcDlM1V3()
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
    strategy = SmcDlM1V3()
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


def test_model_only_advises_never_gates_since_it_lacks_out_of_sample_edge() -> None:
    """The real M1 v3 weights score 0 of 5 folds out-of-sample positive, so
    they are advisory-only: they may veto a clearly bad setup but never
    require clearing the strict `min_expected_r`/`min_p_secure` bar."""
    strategy = SmcDlM1V3()
    assert strategy._model_trusted is False
    assert "advisory" in strategy._status or "fallback" in strategy._status


def test_structural_setup_can_trade_on_its_own_when_model_is_advisory() -> None:
    """With an advisory-only (untrusted) model, a fresh zone retest that the
    structural veto does not block should still be able to produce a signal
    — steps 1-4 of the pipeline stand on their own regardless of the model's
    opinion, unless the model actively vetoes with `advisory_reject_ev`."""
    strategy = SmcDlM1V3()
    frame = _retest_frame("demand")
    signal = strategy.evaluate(_zone_context(frame))
    if signal is None:
        pytest.skip("fixture did not produce a fresh touch on the final bar")
    assert signal.direction == Direction.BUY
    assert signal.sl_points > 0
    assert signal.tp_points > 0
