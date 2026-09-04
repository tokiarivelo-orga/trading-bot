"use client";

/** Underwater chart — how far below its own running peak each bot's equity
 * curve sits, and for how long. Two bots can post the same total profit
 * while one grinds sideways-down for weeks and the other never dips; the
 * equity curve alone doesn't make that difference obvious. */

import { BaselineSeries, createChart, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { collapseByTime } from "./chartData";
import { attachSeriesTooltip, type AnySeries, type ChartedBot, type TooltipState } from "./chartTooltip";
import { TooltipBox } from "./TooltipBox";
import { getLocalTimeZoneOptions } from "../chart/chartFormat";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Drawdown at each point: how far cumulative_profit sits below the running
 * peak so far, always <= 0. */
function drawdownSeries(equityCurve: ChartedBot["equity_curve"]): { time: UTCTimestamp; value: number }[] {
  let peak = 0;
  const points = equityCurve.map((p) => {
    peak = Math.max(peak, p.cumulative_profit);
    return { time: p.close_time as UTCTimestamp, value: p.cumulative_profit - peak };
  });
  // Same-second closes: the last point already reflects the peak/drawdown
  // after both, so keep it and drop the earlier duplicate timestamp.
  return collapseByTime(points, (_a, b) => b);
}

export function BotDrawdownChart({ bots }: { bots: ChartedBot[] }) {
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
      const series = chart.addSeries(BaselineSeries, {
        baseValue: { type: "price", price: 0 },
        topLineColor: "transparent",
        topFillColor1: "transparent",
        topFillColor2: "transparent",
        bottomLineColor: bot.color,
        bottomFillColor1: hexToRgba(bot.color, 0.28),
        bottomFillColor2: hexToRgba(bot.color, 0.02),
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      series.setData(drawdownSeries(bot.equity_curve));
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
