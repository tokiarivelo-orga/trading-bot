/** Shared crosshair-tooltip wiring for the bot comparison charts (equity
 * curve, trade P/L histogram, drawdown) — same lookup-by-series logic, only
 * the value semantics differ per chart, so callers own formatting/state. */

import type { IChartApi, ISeriesApi, MouseEventParams } from "lightweight-charts";
import type { BotAnalytics } from "@/shared/api/client";

export type ChartedBot = BotAnalytics & { color: string };
export type AnySeries =
  | ISeriesApi<"Line">
  | ISeriesApi<"Histogram">
  | ISeriesApi<"Area">
  | ISeriesApi<"Baseline">;

export interface TooltipState {
  x: number;
  y: number;
  time: string;
  rows: { botName: string; color: string; value: number }[];
}

export function formatCrosshairTime(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function attachSeriesTooltip(
  chart: IChartApi,
  seriesBySkill: Map<string, AnySeries>,
  bots: ChartedBot[],
  onChange: (tooltip: TooltipState | null) => void,
): () => void {
  const handler = (param: MouseEventParams) => {
    if (!param.point || param.time === undefined) {
      onChange(null);
      return;
    }
    const rows: TooltipState["rows"] = [];
    for (const bot of bots) {
      const series = seriesBySkill.get(bot.skill);
      if (!series) continue;
      const point = param.seriesData.get(series) as { value?: number } | undefined;
      if (point?.value === undefined) continue;
      rows.push({ botName: bot.bot_name, color: bot.color, value: point.value });
    }
    if (rows.length === 0) {
      onChange(null);
      return;
    }
    onChange({ x: param.point.x, y: param.point.y, time: formatCrosshairTime(param.time as number), rows });
  };
  chart.subscribeCrosshairMove(handler);
  return () => chart.unsubscribeCrosshairMove(handler);
}
