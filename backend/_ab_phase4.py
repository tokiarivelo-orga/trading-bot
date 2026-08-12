import asyncio, sys, logging
from src.backtest.application.run_backtest import run_backtest
from src.shared.config.settings import Settings
logging.basicConfig(level=logging.WARNING)
strategy, symbol, period = sys.argv[1], sys.argv[2], sys.argv[3]
async def main():
    for flag in (False, True):
        r = await run_backtest(strategy, symbol, period,
            database_url=Settings().database_url, simulate_broker_constraints=flag)
        br = r.broker_realism
        print(f"constraints={flag}: trades={len(r.trades)} PF={r.profit_factor:.2f} "
              f"win={r.win_rate:.3f} end={r.ending_balance:.2f} maxDD={r.max_drawdown_pct:.2f} "
              f"signals={len(r.signals)} accepted={br.accepted_count} rejected={br.rejected_count} "
              f"slip={br.slippage_source}/{br.slippage_mean:.5f}")
        for rej in br.rejections:
            print(f"   reject {rej.reason} x{rej.count} retcode={rej.retcode} :: {rej.example[:150]}")
        from collections import Counter
        print("   outcomes:", dict(Counter(s.outcome for s in r.signals)))
asyncio.run(main())
