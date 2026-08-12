"use client";

/**
 * Fetches one page of filtered trade history — re-fetches page 0 whenever
 * the filters change (a new filter combination makes a stale page number
 * from the previous one meaningless), and re-fetches the same page when
 * only `page` changes (Prev/Next).
 *
 * Backed by TanStack Query with a 3000ms refetch interval so that when open
 * positions close (either autonomously via SL/TP/bot, or manually), both the
 * /history page and the docked History tab update automatically without needing
 * a manual reload or tab switch.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useActiveAccount } from "@/shared/api/account-context";
import { queryKeys } from "@/shared/api/queryKeys";
import {
  ApiError,
  getTradeHistory,
  type TradeHistoryFilters,
  type TradeHistoryItem,
} from "@/shared/api/client";

export const PAGE_SIZE = 50;
const POLL_MS = 3000;

export function useTradeHistory(filters: Omit<TradeHistoryFilters, "limit" | "offset">) {
  const accountId = useActiveAccount();
  const [page, setPage] = useState(0);
  const filtersKey = JSON.stringify(filters);

  useEffect(() => {
    setPage(0);
  }, [filtersKey]);

  const query = useQuery({
    queryKey: queryKeys.history.trades(accountId, filtersKey, page),
    queryFn: () =>
      getTradeHistory(accountId as string, {
        ...filters,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    enabled: accountId !== null,
    refetchInterval: POLL_MS,
  });

  const items: TradeHistoryItem[] | null = query.isError ? [] : (query.data?.items ?? null);
  const total = query.data?.total ?? 0;
  const totalProfit = query.data?.total_profit ?? 0;
  const error = query.isError
    ? query.error instanceof ApiError
      ? query.error.message
      : "failed to load trade history"
    : null;

  return { items, total, totalProfit, error, page, setPage, pageSize: PAGE_SIZE };
}
