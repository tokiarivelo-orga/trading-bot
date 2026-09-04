"use client";

/** Per-trade P/L, bars above zero for wins and below for losses — shows how
 * *consistent* a bot is (small steady bars) vs. streaky (a few huge bars
 * carrying the total), which the equity curve's running total can hide. */

import { createChart, HistogramSeries, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { collapseByTime } from "./chartData";
import { attachSeriesTooltip, type AnySeries, type ChartedBot, type TooltipState } from "./chartTooltip";
import { TooltipBox } from "./TooltipBox";
import { getLocalTimeZoneOptions } from "../chart/chartFormat";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function TradePnLHistogram({ bots }: { bots: ChartedBot[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const line = cssVar("--color-line");
    const timeZoneOpts = getLocalTimeZoneOptions();
    const chart = createChart(container, {
      layout: { background: { color: cssVar("--color-panel") }, textColor: cssVar("--color-ink") },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: line , tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter },
      localization: timeZoneOpts.localization,
      rightPriceScale: { borderColor: line },
    });

    const seriesBySkill = new Map<string, AnySeries>();
    for (const bot of bots) {
      const series = chart.addSeries(HistogramSeries, {
        color: bot.color,
        priceLineVisible: false,
        lastValueVisible: false,
        base: 0,
      });
      const points = bot.equity_curve.map((p) => ({ time: p.close_time as UTCTimestamp, value: p.profit }));
      // Same-second closes: sum their P/L into one bar at that timestamp
      // rather than dropping one (a single-time bar chart can't show both).
      series.setData(collapseByTime(points, (a, b) => ({ time: a.time, value: a.value + b.value })));
      seriesBySkill.set(bot.skill, series);
    }
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const detachTooltip = attachSeriesTooltip(chart, seriesBySkill, bots, setTooltip);

    const resize = () => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      detachTooltip();
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      setTooltip(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bots.map((b) => b.skill).join(","), bots.map((b) => b.equity_curve.length).join(",")]);

  if (bots.length === 0) {
    return <div className="flex h-56 items-center justify-center text-sm text-ink-muted">No bots selected.</div>;
  }

  return (
    <div className="relative">
      <div ref={containerRef} className="h-56 min-h-0 w-full" />
      {tooltip && <TooltipBox tooltip={tooltip} containerWidth={containerRef.current?.clientWidth ?? 0} />}
    </div>
  );
}
