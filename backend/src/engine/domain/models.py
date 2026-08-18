"""Engine domain: risk caps, trade plans, and circuit-breaker state.

Pure values — no I/O. `RiskCaps` mirrors `configs/risk.yaml` (user-owned,
read-only from here — see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RiskCaps:
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    max_open_positions: int
    # Numeric daily trade-count cap (distinct from the manual kill switch
    # below): once RiskManager._trades_today reaches this, check_pretrade
    # blocks further entries until the next trading day. None means no
    # numeric cap is configured (unlimited) — kept optional so existing
    # callers that don't set it (tests, older configs) are unaffected.
    max_trades_per_day: int | None = None
    # Manual daily kill switch, not a count: true blocks every new trade for
    # the rest of the trading day; false leaves trade count today unlimited.
    max_trades_per_day_enabled: bool = False
    consecutive_loss_pause: int
    # When false, the consecutive-loss circuit breaker never pauses the
    # engine, regardless of consecutive_loss_pause's count.
    consecutive_loss_pause_enabled: bool = True
    # Broker-minimum-lot fallback (see RiskManager.size_position): when the
    # balance is too small for risk_per_trade_pct to reach volume_min,
    # sizing normally rejects outright. Setting min_lot_fallback_enabled
    # trades the broker minimum lot instead, as long as *that lot's*
    # effective risk stays under max_risk_per_trade_pct (None falls back to
    # risk_per_trade_pct itself as the ceiling).
    min_lot_fallback_enabled: bool = False
    max_risk_per_trade_pct: float | None = None


@dataclass(frozen=True, kw_only=True)
class RiskDecision:
    approved: bool
    volume: float = 0.0
    reason: str = ""
    code: str = ""
    """Machine-readable identity of *which* cap blocked, so callers don't have
    to pattern-match `reason` prose: "paused" (circuit breaker / kill switch),
    "max_positions", "daily_trading_disabled", "sl_distance", "balance",
    "min_lot". Empty on an approval (OBSERVABILITY_PLAN.md Phase 2)."""


@dataclass(frozen=True, kw_only=True)
class EngineStatus:
    enabled: bool
    paused: bool
    pause_reason: str = ""
    consecutive_losses: int = 0
    trades_today: int = 0
    daily_pnl: float = 0.0
