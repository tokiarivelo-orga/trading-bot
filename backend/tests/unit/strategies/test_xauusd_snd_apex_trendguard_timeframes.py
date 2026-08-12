"""Unit tests for the M5/M15/H1/D1 siblings of the Apex Trendguard family.

The behavioural gates are already covered once, against the M1 file, in
`test_xauusd_snd_apex_trendguard_m1.py` — the sandbox forbids these files
importing each other, so they are exact-substitution *copies* of it. What
matters here is therefore different:

  * the copies really are copies (identical algorithm, only the timeframe
    wiring differs) — otherwise the M1 test suite says nothing about them
  * each one's timeframe ladder is internally consistent: every timeframe
    it votes on or reads zones from is a timeframe it actually asked the
    engine for, or its own entry frame
  * every timeframe in a ladder can reach a live trend vote inside the
    context a *backtest* pre-loads, so a replay measures the same bot that
    trades (`run_backtest.HISTORY_BUFFER` is 60 days)
  * each file loads in the sandbox and stays silent on a flat market
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.strategies.domain.models import Direction, MarketContext
from src.strategies.generated.xauusd_snd_apex_trendguard_d1_v1 import (
    XauusdSndApexTrendguardD1,
)
from src.strategies.generated.xauusd_snd_apex_trendguard_h1_v1 import (
    XauusdSndApexTrendguardH1,
)
from src.strategies.generated.xauusd_snd_apex_trendguard_m1_v1 import (
    XauusdSndApexTrendguardM1,
)
from src.strategies.generated.xauusd_snd_apex_trendguard_m5_v1 import (
    XauusdSndApexTrendguardM5,
)
from src.strategies.generated.xauusd_snd_apex_trendguard_m15_v1 import (
    XauusdSndApexTrendguardM15,
)
from src.strategies.sandbox import validate_and_load

START = datetime(2026, 1, 1, tzinfo=UTC)
GENERATED = Path(__file__).resolve().parents[3] / "src/strategies/generated"

# Bars of context a backtest pre-loads per timeframe, from
# `run_backtest.HISTORY_BUFFER` (60 calendar days) — D1/W1 count trading
# days only (~5 per week), which is exactly why they are the tight ones.
BARS_IN_BACKTEST_BUFFER = {
    "M1": 86_400, "M5": 17_280, "M15": 5_760, "M30": 2_880,
    "H1": 1_440, "H4": 360, "D1": 43, "W1": 8,
}

FAMILY = [
    ("m1", XauusdSndApexTrendguardM1),
    ("m5", XauusdSndApexTrendguardM5),
    ("m15", XauusdSndApexTrendguardM15),
    ("h1", XauusdSndApexTrendguardH1),
    ("d1", XauusdSndApexTrendguardD1),
]
SIBLINGS = FAMILY[1:]


def _source(stem: str) -> str:
    return (GENERATED / f"xauusd_snd_apex_trendguard_{stem}_v1.py").read_text()


def _trending(n: int, start: float, drift: float, step: timedelta) -> pd.DataFrame:
    rows = []
    price = start
    for _ in range(n):
        close = price + drift
        rows.append((min(price, close) - 0.2, max(price, close) + 0.2, price, close))
        price = close
    return pd.DataFrame(
        {
            "time": [START + i * step for i in range(n)],
            "open": [r[2] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[0] for r in rows],
            "close": [r[3] for r in rows],
            "tick_volume": [1000] * n,
        }
    )


def _flat_context(strategy) -> MarketContext:
    """A featureless market on every timeframe the strategy asks about."""
    spec = strategy.spec
    steps = {
        "M1": timedelta(minutes=1), "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15), "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1), "H4": timedelta(hours=4),
        "D1": timedelta(days=1), "W1": timedelta(weeks=1),
    }
    timeframes = {spec.entry_timeframe, *spec.confirmation_timeframes}
    return MarketContext(
        symbol="XAUUSD",
        candles={tf: _trending(200, 4000.0, 0.0, steps[tf]) for tf in timeframes},
        spread_points=18.0,
    )


# ─────────────────────────────────────────────────────────────────────
# The copies really are copies
# ─────────────────────────────────────────────────────────────────────

def _algorithm_fingerprint(source: str) -> list[str]:
    """Every module-level function, unparsed with its docstring dropped —
    i.e. the executable algorithm, with prose and comments out of the way
    (the siblings' docstrings legitimately name their own timeframes).

    The class body is deliberately excluded too: it is the one place the
    siblings are *meant* to differ, since that is where `StrategySpec` and
    its params live."""
    tree = ast.parse(source)
    functions = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        functions.append(ast.unparse(ast.Module(body=body, type_ignores=[])))
    return functions


@pytest.mark.parametrize("stem", [stem for stem, _ in SIBLINGS])
def test_sibling_shares_the_m1_algorithm_exactly(stem: str) -> None:
    """If a sibling's helpers ever drift from the M1 file's, the M1 test
    suite stops covering it and the family quietly becomes five strategies
    instead of one on five ladders. Regenerate rather than hand-edit."""
    assert _algorithm_fingerprint(_source(stem)) == _algorithm_fingerprint(_source("m1"))


@pytest.mark.parametrize("stem", [stem for stem, _ in FAMILY])
def test_every_variant_loads_in_the_sandbox(stem: str) -> None:
    instance, errors = validate_and_load(_source(stem))
    assert errors == ()
    assert instance is not None


@pytest.mark.parametrize(("stem", "cls"), FAMILY)
def test_params_differ_from_m1_only_where_the_timeframe_forces_it(stem, cls) -> None:
    """Guards against a tuning change landing in one sibling and nowhere
    else. The listed keys are the ladder itself plus the four values that
    are measured *in bars* or *against a ladder*, so they have to move when
    the bar size does."""
    allowed = {
        "zone_timeframe", "htf_zone_timeframe", "trend_weights",
        "trend_ema_fast", "trend_ema_slow", "trend_min_score",
        "shock_cooldown_bars", "vol_lookback",
    }
    m1_params = XauusdSndApexTrendguardM1().spec.params
    params = cls().spec.params
    assert set(params) == set(m1_params)
    differing = {k for k in params if params[k] != m1_params[k]}
    assert differing <= allowed


# ─────────────────────────────────────────────────────────────────────
# Ladder consistency
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("stem", "cls"), FAMILY)
def test_every_timeframe_it_reads_is_a_timeframe_it_asked_for(stem, cls) -> None:
    """`ctx.candles` only ever holds the entry frame plus the declared
    confirmation frames. A zone or trend timeframe missing from that set is
    silently absent at runtime — zones vanish, or a trend vote is pinned at
    0 forever — with no error anywhere."""
    spec = cls().spec
    available = {spec.entry_timeframe, *spec.confirmation_timeframes}
    assert spec.params["zone_timeframe"] in available
    assert spec.params["htf_zone_timeframe"] in available
    assert set(spec.params["trend_weights"]) <= available


@pytest.mark.parametrize(("stem", "cls"), FAMILY)
def test_zone_timeframe_is_never_finer_than_the_entry_timeframe(stem, cls) -> None:
    """Zones are structure and the entry frame is the trigger inside it, so
    the zone frame must be the same size or larger. D1 is the only variant
    where they are equal — there is no coarser rung to trigger from."""
    order = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]
    spec = cls().spec
    entry = order.index(spec.entry_timeframe)
    zone = order.index(spec.params["zone_timeframe"])
    htf = order.index(spec.params["htf_zone_timeframe"])
    assert zone >= entry
    assert htf > zone


@pytest.mark.parametrize(("stem", "cls"), FAMILY)
def test_trend_votes_are_warm_within_a_backtest_history_buffer(stem, cls) -> None:
    """`_timeframe_trend` needs `slow + slope + 1` bars before it can return
    anything but 0. A backtest only pre-loads 60 days, so a ladder rung
    that needs more than that votes 0 for part of every replay while voting
    normally live — the replay would then be validating a different bot.
    This is why the coarse siblings run a 13/34 EMA pair rather than 21/55.
    """
    params = cls().spec.params
    needed = int(params["trend_ema_slow"]) + int(params["trend_slope_bars"]) + 1
    for timeframe in params["trend_weights"]:
        assert BARS_IN_BACKTEST_BUFFER[timeframe] >= needed, (
            f"{stem}: {timeframe} has ~{BARS_IN_BACKTEST_BUFFER[timeframe]} bars of "
            f"backtest pre-roll but needs {needed} to cast a vote"
        )


@pytest.mark.parametrize(("stem", "cls"), FAMILY)
def test_trend_min_score_sits_between_the_slowest_vote_and_the_whole_ladder(
    stem, cls
) -> None:
    """The floor the M1 flagship established and every sibling inherits: a
    score above the ladder's maximum would make every trade permanently
    'neutral', while one at or below a *faster* rung's weight would let an
    intraday wiggle count as the trend. At exactly the slowest weight (M1,
    M5, M15, H1) the slowest timeframe alone qualifies — which is the
    design; D1 goes above it because it only has two votes to spend."""
    params = cls().spec.params
    weights = params["trend_weights"]
    slowest = max(weights.values())
    assert float(params["trend_min_score"]) <= sum(weights.values())
    assert float(params["trend_min_score"]) >= slowest
    assert slowest > max(w for w in weights.values() if w != slowest)


# ─────────────────────────────────────────────────────────────────────
# Behaviour
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("stem", "cls"), SIBLINGS)
def test_sibling_is_silent_on_a_featureless_market(stem, cls) -> None:
    assert cls().evaluate(_flat_context(cls())) is None


@pytest.mark.parametrize(("stem", "cls"), SIBLINGS)
def test_sibling_never_buys_a_downtrend(stem, cls) -> None:
    """The 2026-08-05 lesson, re-checked per ladder: the direction gate has
    to survive being re-pointed at a different set of timeframes."""
    strategy = cls()
    spec = strategy.spec
    steps = {
        "M5": timedelta(minutes=5), "M15": timedelta(minutes=15),
        "H1": timedelta(hours=1), "H4": timedelta(hours=4),
        "D1": timedelta(days=1), "W1": timedelta(weeks=1),
    }
    drifts = {
        "M5": -0.40, "M15": -1.00, "H1": -3.00,
        "H4": -8.00, "D1": -20.0, "W1": -60.0,
    }
    timeframes = {spec.entry_timeframe, *spec.confirmation_timeframes}
    ctx = MarketContext(
        symbol="XAUUSD",
        candles={
            tf: _trending(200, 4200.0, drifts[tf], steps[tf]) for tf in timeframes
        },
        spread_points=18.0,
    )
    for signal in strategy.evaluate(ctx) or ():
        assert signal.direction is not Direction.BUY
