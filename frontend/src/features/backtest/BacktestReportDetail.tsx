"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ActivityLogTable } from "@/features/logs/ActivityLogTable";
import { getBacktestReport, type BacktestReportDetail as ReportDetail } from "@/shared/api/client";
import { downloadJson } from "@/shared/utils/download";
import { BacktestStrategyEditor } from "./BacktestStrategyEditor";
import { BrokerRealismPanel } from "./BrokerRealismPanel";
import { DivergencePanel } from "./DivergencePanel";
import { EquityChart } from "./EquityChart";
import { SIGNAL_OUTCOME_META } from "./signalOutcome";
import { StatTile } from "./StatTile";

export function BacktestReportDetail({ reportId }: { reportId: string }) {
  const [report, setReport] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBacktestReport(reportId)
      .then(setReport)
      .catch(() => setError("report not found"));
  }, [reportId]);

  if (error) return <p className="p-4 text-sm text-err">{error}</p>;
  if (report === null) return <p className="p-4 text-sm text-ink-muted">Loading…</p>;

  const netProfit = report.ending_balance - report.starting_balance;

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex items-start justify-between">
        <div>
          <Link href="/backtest" className="text-xs text-ink-muted hover:text-accent">
            ← All reports
          </Link>
          <h2 className="mt-1 text-lg font-semibold">
            {report.strategy} on {report.symbol}
          </h2>
          <p className="text-sm text-ink-muted">{report.period}</p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => downloadJson(report, `backtest_${report.strategy}_${report.symbol}_${report.id}.json`)}
            className="shrink-0 cursor-pointer rounded border border-line px-3 py-1.5 text-sm text-ink hover:border-accent hover:text-accent transition-colors"
          >
            Download JSON
          </button>
          <Link
            href={`/?symbol=${encodeURIComponent(report.symbol)}&backtestReport=${encodeURIComponent(report.id)}`}
            className="shrink-0 rounded border border-accent px-3 py-1.5 text-sm text-accent hover:bg-accent/10"
            title="See this report's trades plotted against the actual candles they traded on"
          >
            View on chart →
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-7">
        <StatTile label="Trades" value={String(report.trade_count)} />
        <StatTile label="Win rate" value={`${(report.win_rate * 100).toFixed(1)}%`} />
        <StatTile
          label="Profit factor"
          value={report.profit_factor === null ? "∞" : report.profit_factor.toFixed(2)}
        />
        <StatTile label="Max drawdown" value={`${report.max_drawdown_pct.toFixed(2)}%`} tone="err" />
        <StatTile label="Avg R" value={report.avg_r.toFixed(2)} tone={report.avg_r >= 0 ? "ok" : "err"} />
        <StatTile label="Worst losing streak" value={String(report.worst_losing_streak)} />
        <StatTile
          label="Net P/L"
          value={`${netProfit >= 0 ? "+" : ""}${netProfit.toFixed(2)}`}
          tone={netProfit >= 0 ? "ok" : "err"}
        />
      </div>

      <section className="rounded-md border border-line bg-panel">
        <header className="border-b border-line px-3 py-2 text-sm text-ink-muted">
          Equity curve — {report.starting_balance.toFixed(2)} → {report.ending_balance.toFixed(2)}
        </header>
        <EquityChart points={report.equity_curve} />
      </section>

      <BrokerRealismPanel realism={report.broker_realism} />

      <DivergencePanel reportId={report.id} />

      <BacktestStrategyEditor
        strategyName={report.strategy}
        symbol={report.symbol}
        period={report.period}
      />

      <section className="overflow-x-auto rounded-md border border-line bg-panel">
        <header className="border-b border-line px-3 py-2 text-sm text-ink-muted">Trades</header>
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-muted">
              <th className="px-3 py-2 font-medium">Side</th>
              <th className="px-3 py-2 text-right font-medium">Volume</th>
              <th className="px-3 py-2 font-medium">Open</th>
              <th className="px-3 py-2 text-right font-medium">Open price</th>
              <th className="px-3 py-2 font-medium">Close</th>
              <th className="px-3 py-2 text-right font-medium">Close price</th>
              <th className="px-3 py-2 text-right font-medium">Profit</th>
              <th className="px-3 py-2 text-right font-medium">R</th>
            </tr>
          </thead>
          <tbody>
            {report.trades.map((t, i) => (
              <tr key={i} className="border-b border-line last:border-0">
                <td className={`px-3 py-2 ${t.side === "buy" ? "text-ok" : "text-err"}`}>
                  {t.side.toUpperCase()}
                </td>
                <td className="px-3 py-2 text-right">{t.volume}</td>
                <td className="px-3 py-2 text-ink-muted">{formatTime(t.open_time)}</td>
                <td className="px-3 py-2 text-right">{t.open_price}</td>
                <td className="px-3 py-2 text-ink-muted">{formatTime(t.close_time)}</td>
                <td className="px-3 py-2 text-right">{t.close_price}</td>
                <td className={`px-3 py-2 text-right ${t.profit >= 0 ? "text-ok" : "text-err"}`}>
                  {t.profit >= 0 ? "+" : ""}
                  {t.profit.toFixed(2)}
                </td>
                <td className="px-3 py-2 text-right">
                  {t.r_multiple === null ? "—" : t.r_multiple.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="overflow-x-auto rounded-md border border-line bg-panel">
        <header className="border-b border-line px-3 py-2 text-sm text-ink-muted">
          Signals — every valid setup the strategy emitted during the replay, including the ones
          the engine vetoed or rejected before they became trades.
        </header>
        {(report.signals ?? []).length === 0 ? (
          <p className="px-3 py-2 text-sm text-ink-muted">
            No signals recorded for this report (older reports predate this feature — re-run the
            backtest to get them).
          </p>
        ) : (
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-muted">
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 font-medium">Direction</th>
                <th className="px-3 py-2 font-medium">Outcome</th>
                <th className="px-3 py-2 font-medium">Details</th>
              </tr>
            </thead>
            <tbody>
              {report.signals.map((s, i) => (
                <tr key={i} className="border-b border-line last:border-0">
                  <td className="px-3 py-2 whitespace-nowrap text-ink-muted">
                    {formatTime(s.time)}
                  </td>
                  <td className={`px-3 py-2 ${s.direction === "buy" ? "text-ok" : "text-err"}`}>
                    {s.direction.toUpperCase()}
                  </td>
                  <td className={`px-3 py-2 whitespace-nowrap ${SIGNAL_OUTCOME_META[s.outcome].className}`}>
                    {SIGNAL_OUTCOME_META[s.outcome].label}
                  </td>
                  <td className="max-w-[480px] truncate px-3 py-2 text-ink-muted" title={s.reason}>
                    {s.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="rounded-md border border-line bg-panel">
        <header className="border-b border-line px-3 py-2 text-sm text-ink-muted">
          Activity log — the bot's decision trail during the replay (signals, HTF vetoes, sizing
          rejections, fills, circuit breakers). Explains a zero-trade report.
        </header>
        {report.activity_log.length === 0 ? (
          <p className="px-3 py-2 text-sm text-ink-muted">
            No activity captured for this report (older reports predate this feature — re-run the
            backtest to get one).
          </p>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <ActivityLogTable
              entries={report.activity_log.map((e, i) => ({ id: i, created_at: e.time, ...e }))}
            />
          </div>
        )}
      </section>
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
