"use client";

/**
 * Small embedded candlestick chart showing the frozen entry-time price
 * action behind a trade's "Why" decision — the confirming zone, swing
 * structure, pattern, and the entry itself — inside `TradeDecisionModal`.
 *
 * Deliberately standalone: does NOT use the full chart's drawing-manager /
 * primitive system (`features/chart/useChartEngine.ts`, `useIndicators.ts`)
 * — that machinery is tightly coupled to the main full-page chart and is
 * overkill for a ~220px modal snippet. Uses only chart-engine-independent
 * APIs directly from `lightweight-charts`. Modeled on the createChart/
 * addSeries/cleanup structure of `features/analytics/BotEquityChart.tsx`.
 */

import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  LineStyle,
  type IChartApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";

import { useActiveAccount } from "@/shared/api/account-context";
import { getTradeDecisionContext, type DecisionContext } from "@/shared/api/client";
import { getLocalTimeZoneOptions } from "@/features/chart/chartFormat";

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function DecisionChartSnippet({ tradeId }: { tradeId: string }) {
  const accountId = useActiveAccount();
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [context, setContext] = useState<DecisionContext | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch the frozen decision-context snapshot for this trade.
  useEffect(() => {
    setContext(null);
    setError(null);
    if (!accountId) return;
    const controller = new AbortController();
    getTradeDecisionContext(accountId, tradeId, controller.signal)
      .then((ctx) => setContext(ctx))
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : "failed to load chart context");
      });
    return () => {
      controller.abort();
    };
  }, [accountId, tradeId]);

  // Build the chart once the snapshot has loaded and has candles to show.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || context === null || context.entry_candles.length === 0) return;

    const ok = cssVar("--color-ok");
    const err = cssVar("--color-err");
    const line = cssVar("--color-line");
    const timeZoneOpts = getLocalTimeZoneOptions();
    const chart = createChart(container, {
      layout: { background: { color: cssVar("--color-panel") }, textColor: cssVar("--color-ink") },
      grid: { vertLines: { color: line }, horzLines: { color: line } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: line , tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter },
      localization: timeZoneOpts.localization,
      rightPriceScale: { borderColor: line },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: ok,
      downColor: err,
      borderVisible: false,
      wickUpColor: ok,
      wickDownColor: err,
    });
    candleSeries.setData(
      context.entry_candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    const sideColor = context.side === "buy" ? ok : err;
    candleSeries.createPriceLine({
      price: context.open_price,
      color: sideColor,
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      title: "entry",
    });

    if (context.zone !== null) {
      const zoneColor = context.zone.kind === "demand" ? ok : err;
      candleSeries.createPriceLine({
        price: context.zone.price_high,
        color: zoneColor,
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        title: "zone high",
      });
      candleSeries.createPriceLine({
        price: context.zone.price_low,
        color: zoneColor,
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        title: "zone low",
      });
    }

    const markers: SeriesMarker<Time>[] = [];
    for (const point of context.structure) {
      const isHigh = point.label === "HH" || point.label === "LH";
      markers.push({
        time: point.time as UTCTimestamp,
        position: isHigh ? "aboveBar" : "belowBar",
        color: cssVar("--color-ink-muted"),
        shape: "circle",
        text: point.label,
      });
    }
    if (context.pattern !== null) {
      const lastCandle = context.entry_candles[context.entry_candles.length - 1];
      markers.push({
        time: lastCandle.time as UTCTimestamp,
        position: "aboveBar",
        color: cssVar("--color-accent"),
        shape: "circle",
        text: context.pattern.replace(/_/g, " "),
      });
    }
    markers.push({
      time: context.open_time as UTCTimestamp,
      position: context.side === "buy" ? "belowBar" : "aboveBar",
      color: sideColor,
      shape: context.side === "buy" ? "arrowUp" : "arrowDown",
      text: context.side.toUpperCase(),
    });
    // The markers plugin requires ascending time order.
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    createSeriesMarkers(candleSeries, markers);

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
  }, [context]);

  if (error !== null) {
    return <p className="text-xs text-err">{error}</p>;
  }
  if (context === null) {
    return <p className="text-xs text-ink-muted">Loading chart context…</p>;
  }
  if (context.entry_candles.length === 0) {
    return <p className="text-xs text-ink-muted">No chart context captured for this trade.</p>;
  }
  return <div ref={containerRef} className="h-[220px] w-full" />;
}
