"use client";

/** Overlaid cumulative-profit curves for the selected bots — the chart the
 * user reads to see which approach compounds cleanest (steady climb) vs.
 * which is a coin flip (choppy, deep drawdowns) even when totals are close. */

import { createChart, LineSeries, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { collapseByTime } from "./chartData";
import { attachSeriesTooltip, type AnySeries, type ChartedBot, type TooltipState } from "./chartTooltip";
import { TooltipBox } from "./TooltipBox";
import { getLocalTimeZoneOptions } from "../chart/chartFormat";

const SERIES_COLOR_VARS = [
  "--color-series-1",
  "--color-series-2",
  "--color-series-3",
  "--color-series-4",
  "--color-series-5",
  "--color-series-6",
];

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Fixed categorical order — assign by the bot's stable position in `bots`
 * (its rank in the table), never reassigned when the selection changes, so
 * a bot keeps its color as other checkboxes are toggled. */
export function seriesColor(index: number): string {
  return cssVar(SERIES_COLOR_VARS[index % SERIES_COLOR_VARS.length]);
}

export function BotEquityChart({ bots }: { bots: ChartedBot[] }) {
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
      const series = chart.addSeries(LineSeries, {
        color: bot.color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      const points = bot.equity_curve.map((p) => ({
        time: p.close_time as UTCTimestamp,
        value: p.cumulative_profit,
      }));
      // Same-second closes: the last one already reflects the running total
      // after both, so keep it and drop the earlier duplicate timestamp.
      series.setData(collapseByTime(points, (_a, b) => b));
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
    return (
      <div className="flex h-56 items-center justify-center text-sm text-ink-muted">
        Select one or more bots below to plot their equity curves.
      </div>
    );
  }

  return (
    <div className="relative">
      <div ref={containerRef} className="h-56 min-h-0 w-full" />
      {tooltip && <TooltipBox tooltip={tooltip} containerWidth={containerRef.current?.clientWidth ?? 0} />}
    </div>
  );
}
