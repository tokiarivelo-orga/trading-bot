"""Train → Backtest → Adjust → Repeat loop for the SMC DL strategy.

Runs from ``backend/``:

    uv run python scripts/run_smc_dl_loop.py

Each iteration:
1. Trains the model with the current hyperparameters.
2. Runs a backtest over 2026-06:2026-08 on XAUUSD.
3. Parses win_rate and profit_factor from the backtest summary.
4. If targets are met (win_rate >= 0.5 AND profit_factor >= 1.3), stops.
5. Otherwise, adjusts hyperparameters and re-trains.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import textwrap

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_smc_dl import train as train_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smc_dl_loop")

# ---------------------------------------------------------------------------
# Hyperparameter grid to sweep through
# ---------------------------------------------------------------------------
HPARAM_GRID = [
    # Iter 1: Default
    {"hidden_dim": 128, "lr": 1e-3, "dropout": 0.2, "epochs": 15, "tp_atr_mult": 1.8, "sl_atr_mult": 1.0, "min_tp_prob": 0.55},
    # Iter 2: Deeper, more dropout
    {"hidden_dim": 256, "lr": 5e-4, "dropout": 0.3, "epochs": 25, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0, "min_tp_prob": 0.55},
    # Iter 3: Higher RR
    {"hidden_dim": 128, "lr": 1e-3, "dropout": 0.15, "epochs": 30, "tp_atr_mult": 2.5, "sl_atr_mult": 1.0, "min_tp_prob": 0.5},
    # Iter 4: Aggressive
    {"hidden_dim": 256, "lr": 3e-4, "dropout": 0.25, "epochs": 40, "tp_atr_mult": 1.8, "sl_atr_mult": 0.8, "min_tp_prob": 0.6},
    # Iter 5: Long train
    {"hidden_dim": 192, "lr": 1e-3, "dropout": 0.2, "epochs": 50, "tp_atr_mult": 2.5, "sl_atr_mult": 1.2, "min_tp_prob": 0.5},
]


def _run_backtest() -> tuple[float, float, int]:
    """Run the backtest CLI in-process and parse the summary.

    Returns (win_rate, profit_factor, n_trades).
    """
    import asyncio

    from src.backtest.application.run_backtest import run_backtest
    from src.backtest.reports.writer import render_summary, write_report
    from src.shared.config.settings import Settings

    settings = Settings()
    
    start_date = os.environ.get("START_DATE", "2026-06")
    end_date = os.environ.get("END_DATE", "2026-08")
    period = f"{start_date}:{end_date}"
    
    symbol = os.environ.get("SYMBOL", "XAUUSD")
    strategy_name = "smc_dl_m5" if symbol == "XAUUSD" else "smc_dl_m5_step200"

    try:
        report = asyncio.run(
            run_backtest(
                strategy_name,
                symbol,
                period,
                database_url=settings.database_url,
            )
        )
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        return 0.0, 0.0, 0

    write_report(report)

    summary = render_summary(report)
    print(summary)

    # Parse metrics from the rendered summary text
    win_rate = 0.0
    profit_factor = 0.0
    n_trades = 0

    for line in summary.splitlines():
        low = line.lower().strip()
        if "win rate" in low or "win_rate" in low:
            m = re.search(r"[\d.]+", low.split(":")[-1])
            if m:
                val = float(m.group())
                win_rate = val / 100.0 if val > 1 else val
        if "profit factor" in low or "profit_factor" in low:
            m = re.search(r"[\d.]+", low.split(":")[-1])
            if m:
                profit_factor = float(m.group())
        if "trades" in low and ("total" in low or "count" in low or "#" in low):
            m = re.search(r"\d+", low.split(":")[-1])
            if m:
                n_trades = int(m.group())

    # Fallback: read from report dataclass directly
    if hasattr(report, "trades"):
        n_trades = n_trades or len(report.trades)
    if hasattr(report, "win_rate"):
        win_rate = win_rate or report.win_rate
    if hasattr(report, "profit_factor"):
        profit_factor = profit_factor or report.profit_factor

    return win_rate, profit_factor, n_trades


def main() -> int:
    max_iterations = len(HPARAM_GRID)

    best_pf = 0.0
    best_iter = 0

    for i, hparams in enumerate(HPARAM_GRID, 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {i}/{max_iterations}")
        print(f"  Hyperparams: {hparams}")
        print(f"{'='*60}\n")

        # 1. Train
        try:
            train_model(
                hidden_dim=hparams["hidden_dim"],
                lr=hparams["lr"],
                dropout=hparams["dropout"],
                epochs=hparams["epochs"],
            )
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            continue

        # 2. Backtest
        win_rate, profit_factor, n_trades = _run_backtest()

        print(textwrap.dedent(f"""
        ┌──────────────────────────────┐
        │  Win Rate:      {win_rate:.1%}       │
        │  Profit Factor: {profit_factor:.2f}        │
        │  # Trades:      {n_trades:<12}│
        └──────────────────────────────┘
        """))

        if profit_factor > best_pf:
            best_pf = profit_factor
            best_iter = i

        # 3. Check targets
        if win_rate >= 0.5 and profit_factor >= 1.3:
            print("✅ TARGET ACHIEVED — stopping loop.")
            return 0

        print(f"❌ Target not met (need WR≥50% & PF≥1.3). Trying next config...")

    print(f"\nBest iteration was #{best_iter} with PF={best_pf:.2f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
