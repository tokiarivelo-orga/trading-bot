"use client";

/** Equity curve: a single series over time, so no legend box — the section
 * heading above it already names what's plotted (see dataviz guidance: a
 * single series needs no legend). */

import { AreaSeries, createChart, type IChartApi, type UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { EquityPoint } from "@/shared/api/client";
import { getLocalTimeZoneOptions } from "../chart/chartFormat";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function EquityChart({ points }: { points: EquityPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const line = cssVar("--color-line");
    const accent = cssVar("--color-accent");
    const timeZoneOpts = getLocalTimeZoneOptions();
    const chart = createChart(container, {
      layout: { background: { color: cssVar("--color-panel") }, textColor: cssVar("--color-ink") },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: line , tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter },
      localization: timeZoneOpts.localization,
      rightPriceScale: { borderColor: line },
    });
    const series = chart.addSeries(AreaSeries, {
      lineColor: accent,
      lineWidth: 2,
      topColor: `${accent}1a`, // ~10% opacity wash, per mark spec
      bottomColor: `${accent}00`,
      priceLineVisible: false,
    });
    series.setData(points.map((p) => ({ time: p.time as UTCTimestamp, value: p.balance })));
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const resize = () => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [points]);

  return <div ref={containerRef} className="h-64 min-h-0 w-full" />;
}
