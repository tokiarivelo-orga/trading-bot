"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDailyPnl, getNewsEvents, type DailyPnl, type NewsEventRecord } from "@/shared/api/client";
import { useActiveAccount } from "@/shared/api/account-context";
import { queryKeys } from "@/shared/api/queryKeys";

const POLL_MS = 3000;
const FALLBACK_LOOKBACK_DAYS = 180;

/** One day's realized P&L, tagged with whether a HIGH-impact news event fell
 * on that date — the join is done here, client-side, against two endpoints
 * that already exist (`journal/analytics/daily` + `news/events`) rather than
 * asking the backend for a combined one. */
export interface DailyPnlRow extends DailyPnl {
  hasHighImpactNews: boolean;
  /** Names of the HIGH-impact events that landed on this date, for a hover
   * title — empty when `hasHighImpactNews` is false. */
  highImpactEvents: string[];
}

/** Daily realized P&L history (OBSERVABILITY_PLAN.md Phase 7) plus a
 * client-side join against the persisted news-calendar endpoint to flag
 * high-impact-news days — the daily-granularity counterpart to
 * `useAnalytics.ts`'s per-bot/per-symbol totals.
 *
 * `openFrom`/`openTo` are epoch seconds and, when supplied, come from the
 * page's own date filters so this covers the same window as every other
 * analytics panel. When absent (no filter set), the news-events fetch — whose
 * `start`/`end` are required server-side, unlike the daily-P&L endpoint's
 * optional ones — falls back to the loaded daily rows' own min/max date, or
 * a fixed `FALLBACK_LOOKBACK_DAYS`-day trailing window once no rows have
 * loaded yet. */
export function useDailyPnl(openFrom?: number, openTo?: number) {
  const accountId = useActiveAccount();

  const dailyQuery = useQuery({
    queryKey: queryKeys.analytics.dailyPnl(accountId, openFrom, openTo),
    queryFn: ({ signal }) =>
      getDailyPnl(accountId as string, { open_from: openFrom, open_to: openTo }, signal),
    enabled: accountId !== null,
    refetchInterval: POLL_MS,
  });

  const daily: DailyPnl[] = useMemo(() => dailyQuery.data ?? [], [dailyQuery.data]);

  const newsRange = useMemo(() => {
    if (openFrom !== undefined && openTo !== undefined) return { start: openFrom, end: openTo };
    if (daily.length > 0) {
      return {
        start: Math.floor(Date.parse(`${daily[0].date}T00:00:00Z`) / 1000),
        end: Math.floor(Date.parse(`${daily[daily.length - 1].date}T23:59:59Z`) / 1000),
      };
    }
    const now = Math.floor(Date.now() / 1000);
    return { start: now - FALLBACK_LOOKBACK_DAYS * 86400, end: now };
  }, [openFrom, openTo, daily]);

  const newsQuery = useQuery({
    queryKey: queryKeys.news.eventsForRange(accountId, newsRange.start, newsRange.end),
    queryFn: ({ signal }) =>
      getNewsEvents(accountId as string, { ...newsRange, impact: "high" }, signal),
    enabled: accountId !== null,
    refetchInterval: POLL_MS,
  });

  const newsEvents: NewsEventRecord[] = useMemo(() => newsQuery.data ?? [], [newsQuery.data]);

  const highImpactByDate = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const event of newsEvents) {
      const date = new Date(event.time * 1000).toISOString().slice(0, 10);
      const names = map.get(date);
      if (names) {
        names.push(event.name);
      } else {
        map.set(date, [event.name]);
      }
    }
    return map;
  }, [newsEvents]);

  const rows: DailyPnlRow[] = useMemo(
    () =>
      daily.map((d) => {
        const events = highImpactByDate.get(d.date) ?? [];
        return { ...d, hasHighImpactNews: events.length > 0, highImpactEvents: events };
      }),
    [daily, highImpactByDate],
  );

  return {
    rows,
    loading: dailyQuery.isFetching || newsQuery.isFetching,
    error: dailyQuery.isError ? "Failed to load daily P&L." : null,
  };
}

export type UseDailyPnl = ReturnType<typeof useDailyPnl>;
