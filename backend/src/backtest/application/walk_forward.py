"""Walk-forward backtesting harness (OBSERVABILITY_PLAN.md Phase 6 Pass B).

Scope decision — read before touching this file:

This repo's generated strategies (`strategies/generated/*.py`) have FIXED
parameters. There is no parameter-sweep/optimization infrastructure anywhere
in the codebase, and a single real backtest run already takes ~7 minutes —
so classic walk-forward optimization (refit a strategy's parameters on an
in-sample fold, then validate the refit parameters on the next out-of-sample
fold) has nothing to refit and is explicitly NOT what this module does.

"Walk-forward" here instead means: split the requested `period` into
consecutive `fold_months`-sized windows (`period.split_into_folds`) and run
EACH one as an independent out-of-sample backtest through the existing,
unmodified `run_backtest()` — same strategy code, same parameters, every
fold. Nothing is fitted or carried over between folds; each starts fresh
from `starting_balance`, exactly as if you'd called `run_backtest()` by hand
once per sub-period. The point is measuring whether a bot's edge is
consistent across time instead of resting on one lucky window — the exact
concern OBSERVABILITY_PLAN.md raises about `pob_snd_zones_xauusd` reporting
PF 3.70 on a single 2026-06:2026-07 backtest, which a multi-fold view can
show is (or isn't) representative.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from src.backtest.application import metrics
from src.backtest.application.period import split_into_folds
from src.backtest.application.run_backtest import (
    DEFAULT_DATABASE_URL,
    DEFAULT_STARTING_BALANCE,
    NoHistoryError,
    run_backtest,
)
from src.backtest.domain.models import BacktestReport
from src.broker.domain.slippage import SlippageSampler
from src.engine.ports.strategy_source import StrategySourcePort
from src.shared.config.settings import CONFIGS_DIR

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class FoldResult:
    """One fold's headline stats — everything `WalkForwardReport`'s
    aggregate is built from."""

    period: str  # this fold's own "YYYY-MM:YYYY-MM" sub-range
    trade_count: int
    win_rate: float
    profit_factor: float | None
    """Gross profit / gross loss for this fold. `None` when undefined: no
    losing trades (`BacktestReport.profit_factor` reports `inf`) OR no
    trades at all (the pure `metrics.profit_factor()` reports `0.0` for an
    empty trade list, which reads as "this fold lost everything" if left
    as-is here — `None` correctly says "nothing to measure" instead, and
    keeps a quiet fold from dragging the aggregate below the folds that
    actually traded and lost.)"""
    expectancy: float  # avg profit per trade, account currency; 0.0 if no trades
    broker_acceptance_rate: float
    """`BacktestReport.broker_realism.acceptance_rate` — filled / (filled +
    refused) for this fold. 1.0 (or 0.0 for a zero-trade fold, per that
    property's own definition) when `simulate_broker_constraints=False`."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class WalkForwardReport:
    """Aggregate result of running `run_walk_forward()` — one JSON file per
    run (`reports/walk_forward_writer.py`), not one per fold."""

    strategy: str
    symbol: str
    period: str  # the full requested range, before splitting into folds
    fold_months: int
    starting_balance: float  # every fold starts fresh from this, uncompounded
    folds: tuple[FoldResult, ...]
    """Only the folds that actually ran — a fold skipped for missing history
    (see module docstring) is absent here, not a placeholder."""
    fold_count: int  # len(folds); may be less than split_into_folds(period) if any were skipped
    folds_with_zero_trades: int
    losing_fold_count: int  # folds with expectancy < 0
    mean_profit_factor: float | None
    """Mean of every fold's `profit_factor` that isn't `None` (see
    `FoldResult.profit_factor`'s docstring for why zero-trade/inf/nan folds
    are excluded rather than counted as 0). `None` if no fold had one."""
    stddev_profit_factor: float | None  # population stddev (statistics.pstdev) of the same set
    worst_fold_profit_factor: float | None  # min of the same set
    mean_expectancy: float  # 0.0 if every fold was skipped (no folds ran at all)


async def run_walk_forward(
    strategy_name: str,
    symbol: str,
    period: str,
    *,
    fold_months: int = 1,
    strategy_source: StrategySourcePort | None = None,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
    database_url: str = DEFAULT_DATABASE_URL,
    configs_dir: Path = CONFIGS_DIR,
    min_lot_fallback_enabled: bool | None = None,
    max_risk_per_trade_pct: float | None = None,
    min_rr: float | None = None,
    simulate_broker_constraints: bool = True,
    clamp_stops: bool = False,
    spread_widening_factor: float = 1.0,
    slippage_samples: Sequence[float] | None = None,
    slippage_seed: int = SlippageSampler.DEFAULT_SEED,
) -> WalkForwardReport:
    """Splits `period` into `fold_months`-sized consecutive windows and runs
    each as an INDEPENDENT out-of-sample backtest via `run_backtest()` — no
    parameter refitting between folds (see this module's own docstring for
    the full scope decision: this repo's strategies have fixed parameters
    and there is no optimization infrastructure to refit with, so "walk-
    forward" here means "many independent out-of-sample windows", not the
    textbook in-sample-refit definition).

    Every keyword argument other than `fold_months` is passed straight
    through to `run_backtest()` for every fold unchanged — same strategy
    source, same risk/broker-realism overrides, same starting balance each
    time (folds do not compound into one another).

    A fold whose candle history is missing (`NoHistoryError` — the DB simply
    doesn't have that month backfilled yet) is skipped, not fatal, so one
    unbackfilled month doesn't abort the whole run; `WalkForwardReport.
    fold_count` can therefore be less than `len(split_into_folds(period,
    fold_months))`. Any other exception `run_backtest()` raises (unknown
    strategy, no symbol spec, malformed period) is NOT caught — those are
    the same for every fold, so failing fast on the first one is more useful
    than silently producing an empty report.

    This can take roughly `fold_count` times as long as a single
    `run_backtest()` call — a 12-month period at the default monthly
    `fold_months` runs 12 full backtests, not one."""
    fold_periods = split_into_folds(period, fold_months)

    fold_results: list[FoldResult] = []
    for fold_period in fold_periods:
        try:
            report = await run_backtest(
                strategy_name,
                symbol,
                fold_period,
                strategy_source=strategy_source,
                starting_balance=starting_balance,
                database_url=database_url,
                configs_dir=configs_dir,
                min_lot_fallback_enabled=min_lot_fallback_enabled,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                min_rr=min_rr,
                simulate_broker_constraints=simulate_broker_constraints,
                clamp_stops=clamp_stops,
                spread_widening_factor=spread_widening_factor,
                slippage_samples=slippage_samples,
                slippage_seed=slippage_seed,
            )
        except NoHistoryError as exc:
            logger.warning(
                "walk-forward: fold %s skipped for %s %s — %s",
                fold_period,
                strategy_name,
                symbol,
                exc,
            )
            continue
        fold_results.append(_fold_result(fold_period, report))

    return _aggregate(
        strategy=strategy_name,
        symbol=symbol,
        period=period,
        fold_months=fold_months,
        starting_balance=starting_balance,
        folds=fold_results,
    )


def _fold_result(fold_period: str, report: BacktestReport) -> FoldResult:
    return FoldResult(
        period=fold_period,
        trade_count=len(report.trades),
        win_rate=report.win_rate,
        profit_factor=_normalized_profit_factor(report),
        expectancy=metrics.expectancy(report.trades),
        broker_acceptance_rate=report.broker_realism.acceptance_rate,
    )


def _normalized_profit_factor(report: BacktestReport) -> float | None:
    """`None` for a fold with no trades (profit factor is undefined, not
    zero) or whose raw value is `inf`/`nan` (no losing trades, or — in
    principle — no winning trades either); a finite value otherwise."""
    if not report.trades:
        return None
    pf = report.profit_factor
    return pf if isfinite(pf) else None


def _aggregate(
    *,
    strategy: str,
    symbol: str,
    period: str,
    fold_months: int,
    starting_balance: float,
    folds: list[FoldResult],
) -> WalkForwardReport:
    pf_values = [f.profit_factor for f in folds if f.profit_factor is not None]
    return WalkForwardReport(
        strategy=strategy,
        symbol=symbol,
        period=period,
        fold_months=fold_months,
        starting_balance=starting_balance,
        folds=tuple(folds),
        fold_count=len(folds),
        folds_with_zero_trades=sum(1 for f in folds if f.trade_count == 0),
        losing_fold_count=sum(1 for f in folds if f.expectancy < 0),
        mean_profit_factor=statistics.mean(pf_values) if pf_values else None,
        stddev_profit_factor=statistics.pstdev(pf_values) if pf_values else None,
        worst_fold_profit_factor=min(pf_values) if pf_values else None,
        mean_expectancy=statistics.mean(f.expectancy for f in folds) if folds else 0.0,
    )
