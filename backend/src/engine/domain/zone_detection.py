"""Pure RBR/DBD/RBD/DBR base detector for engine-level position management.

A trusted, engine-side counterpart to the leg-in/base/leg-out zone detector
duplicated across sandboxed strategy files in `strategies/generated/`
(duplicated there rather than imported, because the strategy sandbox only
allows `math`/`statistics`/`numpy`/`pandas` imports — see
`strategies/sandbox.py`). This copy is simplified for its one job: telling
`PositionManager` whether a fresh base has formed and been left behind by
price, not full retest/zone-flip semantics a strategy's entry logic needs.

No I/O — pure functions over OHLC arrays, matching this module's hexagonal
`domain/` placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.shared.domain.indicators import atr  # noqa: F401 - re-exported for existing callers

DEFAULT_ATR_PERIOD = 14
DEFAULT_BASE_BODY_ATR_MULT = 0.5
DEFAULT_LEG_TRAVEL_ATR_MULT = 0.7
DEFAULT_MAX_BASE_CANDLES = 30


class BaseKind(StrEnum):
    DEMAND = "demand"
    SUPPLY = "supply"


@dataclass(frozen=True)
class Base:
    kind: BaseKind
    price_low: float
    price_high: float
    base_start: int
    leg_out_end: int
    broken: bool


def detect_bases(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr_values: pd.Series,
    *,
    base_body_atr_mult: float = DEFAULT_BASE_BODY_ATR_MULT,
    leg_travel_atr_mult: float = DEFAULT_LEG_TRAVEL_ATR_MULT,
    max_base_candles: int = DEFAULT_MAX_BASE_CANDLES,
) -> list[Base]:
    """Same leg-in/base/leg-out compression geometry as the strategy zone
    detectors: consecutive small-body ("base") candles between two momentum
    legs form a rectangle, confirmed once the leg-out candle closes clear of
    it. No zone-flip extension — position management only needs to know
    whether a base is intact and already cleared by price, not its
    post-invalidation polarity."""
    n = len(closes)
    valid_atr = atr_values.dropna()
    if valid_atr.empty:
        return []
    atr_filled = atr_values.fillna(valid_atr.iloc[0]).to_numpy()

    cls = np.where(
        np.abs(closes - opens) <= base_body_atr_mult * atr_filled,
        0,
        np.where(closes >= opens, 1, -1),
    )

    change = np.flatnonzero(cls[1:] != cls[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.append(change - 1, n - 1)
    runs: list[list[int]] = [
        [int(cls[s]), int(s), int(e)] for s, e in zip(starts, ends, strict=True)
    ]

    def is_leg(run: list[int]) -> bool:
        cls_, start, end = run
        travel = abs(closes[end] - opens[start])
        return cls_ != 0 and travel >= leg_travel_atr_mult * atr_filled[end]

    merged = True
    while merged:
        merged = False
        for k in range(len(runs) - 2):
            d1, pause, d2 = runs[k], runs[k + 1], runs[k + 2]
            if d1[0] == 0 or pause[0] != 0 or d2[0] != d1[0]:
                continue
            if pause[2] - pause[1] + 1 > max_base_candles:
                continue
            if is_leg(d1) and is_leg(d2):
                continue
            runs[k : k + 3] = [[d1[0], d1[1], d2[2]]]
            merged = True
            break

    legs = [r for r in runs if is_leg(r)]

    bases: list[Base] = []
    k = 0
    while k < len(legs) - 1:
        leg_in = legs[k]
        found_m = None
        for m in range(k + 1, len(legs)):
            leg_out = legs[m]
            base_start = leg_in[2] + 1
            base_end = leg_out[1] - 1
            base_count = base_end - base_start + 1
            if base_count > max_base_candles:
                break
            if base_count < 1:
                continue

            price_high = float(highs[base_start : base_end + 1].max())
            price_low = float(lows[base_start : base_end + 1].min())
            leg_out_up = leg_out[0] == 1
            demand = leg_out_up

            conf_idx = None
            for j in range(leg_out[1], leg_out[2] + 1):
                cleared = (closes[j] > price_high) if leg_out_up else (closes[j] < price_low)
                if cleared:
                    conf_idx = j
                    break
            if conf_idx is None:
                continue

            scan_start = leg_out[2] + 1
            broke = closes[scan_start:] < price_low if demand else closes[scan_start:] > price_high
            broken = bool(np.any(broke))

            bases.append(
                Base(
                    kind=BaseKind.DEMAND if demand else BaseKind.SUPPLY,
                    price_low=price_low,
                    price_high=price_high,
                    base_start=base_start,
                    leg_out_end=leg_out[2],
                    broken=broken,
                )
            )
            found_m = m
            break

        k = found_m if found_m is not None else k + 1
    return bases
