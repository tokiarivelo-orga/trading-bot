"use client";

/** Daily realized P&L (OBSERVABILITY_PLAN.md Phase 7) — one bar per trading
 * day, green above zero / red below (sibling visual weight to
 * `BotEquityChart`/`TradePnLHistogram`/`BotDrawdownChart` above it, at the
 * coarser day granularity `journal/analytics/daily` reports rather than
 * per-trade), plus a table with the same per-day stats the bars summarize
 * visually. Each row is tagged with whether a HIGH-impact news event fell on
 * that date (client-side join in `useDailyPnl.ts` against the persisted news
 * calendar) — flagged with a newspaper icon rather than color, since
 * red/green is already spoken for by the day's P&L sign. */

import {
  createChart,
  HistogramSeries,
  type IChartApi,
  type MouseEventParams,
  type UTCTimestamp,
} from "lightweight-charts";
import { Newspaper } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { money, pct, plTone, profitFactor } from "./format";
import type { DailyPnlRow } from "./useDailyPnl";
import { getLocalTimeZoneOptions } from "../chart/chartFormat";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function dateToEpoch(date: string): UTCTimestamp {
  return Math.floor(Date.parse(`${date}T00:00:00Z`) / 1000) as UTCTimestamp;
}

interface HoverState {
  x: number;
  y: number;
  row: DailyPnlRow;
}

interface DailyPnlPanelProps {
  rows: DailyPnlRow[];
  loading: boolean;
  error: string | null;
}

export function DailyPnlPanel({ rows, loading, error }: DailyPnlPanelProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const rowsByEpoch = useMemo(() => new Map(rows.map((r) => [dateToEpoch(r.date), r])), [rows]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || rows.length === 0) return;

    const line = cssVar("--color-line");
    const ok = cssVar("--color-ok");
    const err = cssVar("--color-err");
    const timeZoneOpts = getLocalTimeZoneOptions();
    const chart = createChart(container, {
      layout: { background: { color: cssVar("--color-panel") }, textColor: cssVar("--color-ink") },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      timeScale: { borderColor: line , tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter },
      localization: timeZoneOpts.localization,
      rightPriceScale: { borderColor: line },
    });

    const series = chart.addSeries(HistogramSeries, {
      priceLineVisible: false,
      lastValueVisible: false,
      base: 0,
    });
    series.setData(
      rows.map((r) => ({ time: dateToEpoch(r.date), value: r.pnl, color: r.pnl >= 0 ? ok : err })),
    );
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const handler = (param: MouseEventParams) => {
      if (!param.point || param.time === undefined) {
        setHover(null);
        return;
      }
      const row = rowsByEpoch.get(param.time as UTCTimestamp);
      if (!row) {
        setHover(null);
        return;
      }
      setHover({ x: param.point.x, y: param.point.y, row });
    };
    chart.subscribeCrosshairMove(handler);

    const resize = () => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      chart.unsubscribeCrosshairMove(handler);
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      setHover(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows.length, rows.map((r) => r.date).join(",")]);

  if (error) {
    return <p className="p-4 text-sm text-err">{error}</p>;
  }
  if (loading && rows.length === 0) {
    return <p className="p-4 text-sm text-ink-muted">Loading daily P&amp;L…</p>;
  }
  if (rows.length === 0) {
    return <p className="p-4 text-sm text-ink-muted">No closed trades in this window yet.</p>;
  }

  // Table reads newest-first (most recent day on top); the chart above keeps
  // the natural oldest→newest left-to-right order the daily-P&L endpoint
  // already returns.
  const tableRows = [...rows].reverse();

  return (
    <div>
      <div className="relative border-b border-line">
        <div ref={containerRef} className="h-48 min-h-0 w-full" />
        {hover && (
          <div
            className="pointer-events-none absolute z-10 rounded border border-line bg-panel/95 px-2.5 py-1.5 text-xs shadow-lg backdrop-blur-sm"
            style={{
              left: Math.min(hover.x + 12, Math.max((containerRef.current?.clientWidth ?? 0) - 190, 0)),
              top: Math.max(hover.y - 12, 8),
            }}
          >
            <div className="mb-1 text-ink-muted">{hover.row.date}</div>
            <div className={plTone(hover.row.pnl)}>{money(hover.row.pnl, { sign: true })}</div>
            <div className="text-ink-muted">
              {hover.row.trade_count} trade{hover.row.trade_count === 1 ? "" : "s"} ·{" "}
              {pct(hover.row.win_rate)} win rate
            </div>
            {hover.row.hasHighImpactNews && (
              <div className="mt-1 flex items-center gap-1 text-ink">
                <Newspaper size={11} />
                {hover.row.highImpactEvents.join(", ")}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-muted">
              <th className="px-3 py-2 font-medium">Date</th>
              <th className="px-3 py-2 text-right font-medium">Trades</th>
              <th className="px-3 py-2 text-right font-medium">Win rate</th>
              <th className="px-3 py-2 text-right font-medium">Profit factor</th>
              <th className="px-3 py-2 text-right font-medium">Daily P/L</th>
              <th className="px-3 py-2 text-center font-medium" title="A HIGH-impact economic event released this day">
                News
              </th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r) => (
              <tr key={r.date} className="border-b border-line last:border-0 hover:bg-panel/40">
                <td className="px-3 py-2 font-medium text-ink">{r.date}</td>
                <td className="px-3 py-2 text-right">{r.trade_count}</td>
                <td className="px-3 py-2 text-right">{pct(r.win_rate)}</td>
                <td className="px-3 py-2 text-right">{profitFactor(r.profit_factor)}</td>
                <td className={`px-3 py-2 text-right ${plTone(r.pnl)}`}>{money(r.pnl, { sign: true })}</td>
                <td className="px-3 py-2 text-center">
                  {r.hasHighImpactNews && (
                    <span title={`High-impact news: ${r.highImpactEvents.join(", ")}`}>
                      <Newspaper size={14} className="inline-block text-accent" />
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
