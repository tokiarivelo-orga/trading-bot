"""Structure-pivot profit-lock: ratchet SL to fresh swing structure once a
position has proven itself.

WHAT THIS RULE DOES
────────────────────────────────────────────────────────────────────────
Once a position's first take-profit leg (TP1) has closed in profit and price
keeps moving favourably, forming a new swing-structure point in the trade's
direction (a Higher-Low for a buy, a Lower-High for a sell), SL ratchets to
just beyond that pivot — below the HL for a buy, above the LH for a sell.
The same logic re-arms after TP2: because this module re-evaluates on every
M5 close and always selects the *most recent* qualifying pivot, a fresh,
later HL/LH found once the position is further in profit naturally produces
a tighter ratchet than the pivot used for the TP1 stage — there is no
separate "stage 2" code path, just the same selection re-run against a
longer price history.

This is distinct from `PositionManager`'s existing Rule 2 (secure-on-
base-clear / structural continuation trailing), which reacts to S&D supply/
demand *bases* (`zone_detection.Base`) — compression rectangles bounded by a
leg-in/leg-out move. This rule reacts to raw swing pivots (a single bar that
is locally the highest/lowest point over a window), the same HH/HL/LH/LL
vocabulary `strategies.domain.models.StructureLabel` already uses for
chart-annotation data — but computed fresh, engine-side, from live candles,
because that per-strategy `Signal.structure` is a one-shot snapshot taken at
entry time and never updates as the trade runs (see that dataclass's
docstring). A manually-opened position has no `Signal` at all, so an
engine-level detector is the only way this rule can be bot-agnostic.

THE TP1/TP2-CLOSED TRIGGER: WHAT WAS INVESTIGATED AND WHY A PROXY IS USED
────────────────────────────────────────────────────────────────────────
The intent is "arm once this position's sibling TP1 (or TP2) leg has
actually closed in profit." Three live-data correlation mechanisms were
checked and rejected as unreliable for this:

  * `signal_id` (the UUID that groups TP1/TP2/TP3 legs in the `trades` DB
    table and the activity/journal modules) is minted in
    `TradeLoop._try_enter` and threaded through `OrderService.open_position`
    only for internal bookkeeping (`SignalDecision` outcomes, the
    `PositionOpened` event, the journal). It is never sent to the broker —
    `OrderRequest` (`src/broker/domain/trading.py`) carries only
    `symbol/side/volume/sl/tp/comment/magic`. So live `Position` objects
    returned by `BrokerPort.get_positions()` (what `PositionManager` reads
    every candle) do not carry it at all. Querying the journal DB for it on
    every candle close, for every open position, would add a DB round trip
    to the hot position-management path and reach across a module boundary
    this engine code does not otherwise depend on.
  * MT5's `comment` field is capped at 29 characters
    (`bugfix-mt5-comment-length-limit`), and `TradeLoop` spends it on
    `f"TP{idx+1}:{signal.reason}"[:29]` (see `trade_loop.py`'s sizing loop).
    That reliably tells you *this ticket's own* leg index (TP1 vs TP2 vs
    TP3) but encodes no shared group id — the reason suffix is truncated
    identically for every leg of the same signal, so it is not even a safe
    fingerprint once two different signals happen to share a reason prefix.
  * `magic` identifies the *bot*, not the *signal* — every trade a strategy
    ever opens shares one magic number, so it cannot distinguish "this
    ticket's sibling" from "some unrelated earlier trade by the same bot."
  * Clustering on `(symbol, magic, open_time)` is the last option and was
    considered, but is a heuristic on two counts: it needs in-memory
    tracking that a backend restart loses outright, and it cannot tell "TP1
    closed by hitting its own take-profit" apart from "TP1 was stopped out
    at a loss" without also caching the sibling's last-seen SL/TP/profit —
    i.e. it degrades into needing this same module's own state tracking
    just to answer a question this module does not actually need answered
    (see below).

Given that, this module uses the proxy the task explicitly allows for this
case: **the position's own peak unrealised R has already reached where a
TP1 (or TP2) leg would typically trigger** (`StructurePivotConfig.arm_r`).
"Price already reached that level" is a fact derivable purely from the
position's own entry price, current SL-implied risk, and running high-water
mark — no sibling ticket, no journal query, no in-memory group tracking, and
it behaves identically for a genuine remaining leg of a multi-TP signal and
for a manually-opened single position (which has no siblings at all).

Trade-off, stated plainly: this arms on "this trade has independently
proven itself to about 1R (or 2R)," not on the literal fact "a sibling
ticket's take-profit order filled." The two are highly correlated in
practice (a signal's TP1 leg is normally set near where this proxy arms),
but they are not identical — e.g. a single-leg trade with no TP1 sibling at
all still arms under this proxy once it reaches `arm_r`, whereas the literal
trigger would never fire for it. This is judged acceptable, not just
tolerated, for two reasons: first, the intent behind the rule ("once a
trade has shown real, structural follow-through, start respecting fresh
price-action instead of a fixed R target") is preserved either way; second,
the engine-wide invariant this rule composes under (`PositionManager.
_improves` — SL only ever tightens) means an early or generous arming can
only ever produce a *tighter* stop than the position already had, never a
worse one. A false negative (never arming) is equally safe — the position
simply keeps whatever the other four rules already produced.

No I/O — pure functions/dataclasses over OHLC arrays and floats, matching
this module's hexagonal `domain/` placement (mirrors `exit_policy.py`,
which holds the give-back rule's decision logic for the same reason).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.strategies.domain.models import StructureLabel

DEFAULT_PIVOT_BARS = 2  # bars required strictly beyond the pivot on each side to confirm it
DEFAULT_ARM_R = 1.0  # proxy for "a TP1 leg would typically have triggered by here"
DEFAULT_MIN_PIVOT_R = 0.0  # a qualifying pivot must sit at least this far beyond entry, in R
DEFAULT_BUFFER_R_MULT = 0.1  # SL sits this many R beyond the pivot itself, not exactly on it


@dataclass(frozen=True)
class SwingPivot:
    """One labeled swing point, engine-detected from live candle data —
    the running counterpart to `strategies.domain.models.StructurePoint`,
    which is entry-time-only chart-annotation data a strategy captures once
    and never updates."""

    index: int
    time: datetime
    price: float
    label: StructureLabel


def detect_swing_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    times: Sequence[datetime],
    *,
    pivot_bars: int = DEFAULT_PIVOT_BARS,
) -> list[SwingPivot]:
    """Standard fractal swing-pivot detector.

    Bar `i` is a swing low if its low is *strictly* below every one of the
    `pivot_bars` bars on each side of it; a swing high analogously on highs
    (a tie against a neighbour disqualifies the bar — it is simply not a
    pivot, rather than picking a side). Confirmation lags `pivot_bars` bars
    by construction: the most recent `pivot_bars` bars can never yet be
    confirmed pivots, the same lag every fractal-based detector has.

    Consecutive swing points of the same kind are then labeled the same way
    `strategies.domain.models.StructureLabel` already names them: a swing
    low higher than the previous swing low is a Higher-Low (`HL`), otherwise
    a Lower-Low (`LL`) — ties count as `LL` (not strictly higher). A swing
    high higher than the previous swing high is a Higher-High (`HH`),
    otherwise a Lower-High (`LH`) — ties count as `LH`. The very first swing
    low/high found has no predecessor to compare against and is not labeled
    (there is nothing to call it higher or lower than).

    Returns pivots in chronological order (ascending `index`/`time`).
    """
    n = len(highs)
    if len(lows) != n or len(times) != n:
        raise ValueError("highs, lows, and times must be the same length")
    if pivot_bars < 1:
        raise ValueError("pivot_bars must be >= 1")

    pivots: list[SwingPivot] = []
    prev_low: float | None = None
    prev_high: float | None = None
    for i in range(pivot_bars, n - pivot_bars):
        left_low = lows[i - pivot_bars : i]
        right_low = lows[i + 1 : i + pivot_bars + 1]
        if lows[i] < left_low.min() and lows[i] < right_low.min():
            if prev_low is not None:
                label = StructureLabel.HL if lows[i] > prev_low else StructureLabel.LL
                pivots.append(
                    SwingPivot(index=i, time=times[i], price=float(lows[i]), label=label)
                )
            prev_low = float(lows[i])

        left_high = highs[i - pivot_bars : i]
        right_high = highs[i + 1 : i + pivot_bars + 1]
        if highs[i] > left_high.max() and highs[i] > right_high.max():
            if prev_high is not None:
                label = StructureLabel.HH if highs[i] > prev_high else StructureLabel.LH
                pivots.append(
                    SwingPivot(index=i, time=times[i], price=float(highs[i]), label=label)
                )
            prev_high = float(highs[i])

    pivots.sort(key=lambda p: p.index)
    return pivots


@dataclass(frozen=True, kw_only=True)
class StructurePivotConfig:
    """Every knob for the structure-pivot profit-lock rule.

    ``enabled=False`` (the class default's sibling — `PositionManager`
    treats a `None` config the same way) reproduces pre-existing behaviour
    exactly, mirroring `ExitPolicyConfig`/`VolatilityConfig`.
    """

    enabled: bool = True
    # Fractal half-window: a bar must be strictly the extreme of this many
    # bars on each side to confirm as a swing pivot. 2 is the standard
    # (Bill-Williams-style) fractal window.
    pivot_bars: int = DEFAULT_PIVOT_BARS
    # Peak unrealised R before this rule arms at all — the TP1/TP2-sibling-
    # closed proxy; see the module docstring for why this is a proxy and not
    # a literal sibling-ticket check.
    arm_r: float = DEFAULT_ARM_R
    # A qualifying pivot must sit at least this many R beyond entry in the
    # trade's favour. 0.0 (the default) accepts any HL/LH found after entry
    # regardless of how close to entry it sits — safe because `_improves`
    # in `PositionManager` still refuses to loosen the stop with it; raise
    # this to demand a more meaningfully-progressed pivot before ratcheting.
    min_pivot_r: float = DEFAULT_MIN_PIVOT_R
    # SL is placed this many R beyond the pivot's own price, not exactly on
    # it, so a shallow retest wick back to the pivot level does not itself
    # stop the trade out.
    buffer_r_mult: float = DEFAULT_BUFFER_R_MULT


def select_ratchet_pivot(
    *,
    is_buy: bool,
    entry_price: float,
    risk: float,
    pivots: Sequence[SwingPivot],
    entered_after: datetime,
    min_pivot_r: float = DEFAULT_MIN_PIVOT_R,
) -> SwingPivot | None:
    """The most recent qualifying pivot in the trade's favour: a Higher-Low
    for a buy, a Lower-High for a sell. Only pivots that formed strictly
    after `entered_after` (the position's own `open_time`) are eligible —
    "a NEW swing-structure point in the trade's direction," not a pivot
    that predates the trade. Iterates most-recent-first (mirrors
    `PositionManager._select_secure_base`) so an older, already-superseded
    pivot never wins over a fresher one."""
    if risk <= 0:
        return None
    wanted = StructureLabel.HL if is_buy else StructureLabel.LH
    direction = 1.0 if is_buy else -1.0
    for pivot in reversed(pivots):
        if pivot.label is not wanted:
            continue
        if pivot.time <= entered_after:
            continue
        progress_r = (pivot.price - entry_price) * direction / risk
        if progress_r < min_pivot_r:
            continue
        return pivot
    return None


def decide_structure_pivot_exit(
    *,
    is_buy: bool,
    entry_price: float,
    risk: float,
    peak_r_value: float,
    pivots: Sequence[SwingPivot],
    entered_after: datetime,
    config: StructurePivotConfig,
) -> float | None:
    """The candidate SL from this rule, or `None` when it has nothing to
    add. Pure decision function — `PositionManager` merges the result
    through the same `_improves` check every other rule uses, so this can
    never loosen a stop another rule already set.

    `peak_r_value` is the position's best-ever unrealised R (its running
    high-water mark, matching `exit_policy.peak_r`'s definition and
    `PositionManager._trade_extreme_favorable`) — arming looks at the peak
    ever reached, not just this candle's mark, for the same reason the
    give-back rule does: a position that touched 1.2R and pulled back to
    0.6R has still, as a fact, proven itself to 1.2R.
    """
    if not config.enabled or risk <= 0:
        return None
    if peak_r_value < config.arm_r:
        return None
    pivot = select_ratchet_pivot(
        is_buy=is_buy,
        entry_price=entry_price,
        risk=risk,
        pivots=pivots,
        entered_after=entered_after,
        min_pivot_r=config.min_pivot_r,
    )
    if pivot is None:
        return None
    direction = 1.0 if is_buy else -1.0
    buffer_price = config.buffer_r_mult * risk
    return pivot.price - direction * buffer_price
