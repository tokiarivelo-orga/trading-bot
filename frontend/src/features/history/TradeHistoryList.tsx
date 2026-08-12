"use client";

import { useMemo, useState, useEffect, Suspense } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import type { TradeHistoryFilters as ApiFilters, OrderSide, TradeOutcome, TradeHistoryOrderBy } from "@/shared/api/client";
import type { GroupBy } from "./groupTrades";
import {
  getDefaultFilters,
  TradeHistoryFilters,
  type TradeHistoryFilterState,
} from "./TradeHistoryFilters";
import { TradeHistoryTable } from "./TradeHistoryTable";
import { PAGE_SIZE, useTradeHistory } from "./useTradeHistory";
import { useHistoryExport } from "./useHistoryExport";

function toApiFilters(f: TradeHistoryFilterState): Omit<ApiFilters, "limit" | "offset"> {
  return {
    symbol: f.symbol || undefined,
    side: f.side || undefined,
    strategy_version: f.strategyVersion || undefined,
    skill: f.skill || undefined,
    outcome: f.outcome || undefined,
    open_from: f.openFrom ? Math.floor(Date.parse(`${f.openFrom}T00:00:00Z`) / 1000) : undefined,
    open_to: f.openTo ? Math.floor(Date.parse(`${f.openTo}T23:59:59Z`) / 1000) : undefined,
    order_by: f.orderBy,
    order_dir: f.orderDir,
  };
}

/** Trade history page: journaled trades across any symbol, filterable by
 * symbol/side/strategy/skill/outcome/date-range and categorizable (grouped,
 * with per-group win-rate + net P/L) by symbol, date, side, strategy
 * version, skill, or outcome. */
export function TradeHistoryList(props: {
  /** Ticket currently highlighted on the chart (see page.tsx's
   * `selectedOrderTicket`) — forwarded to TradeHistoryTable so a row stays
   * marked in sync with the chart however the selection changed. */
  selectedTicket?: string | number | null;
  /** Called with a row's ticket + symbol when clicked — forwarded straight
   * through from TradeHistoryTable. */
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
} = {}) {
  return (
    <Suspense fallback={<div className="p-4 text-sm text-ink-muted">Loading history...</div>}>
      <TradeHistoryListInner {...props} />
    </Suspense>
  );
}

function TradeHistoryListInner({
  selectedTicket = null,
  onSelectTicket,
}: {
  selectedTicket?: string | number | null;
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [filters, setFilters] = useState<TradeHistoryFilterState>(() => {
    const defaults = getDefaultFilters();
    if (!searchParams) return defaults;

    return {
      today: searchParams.has("today") ? searchParams.get("today") === "true" : defaults.today,
      symbol: searchParams.get("symbol") ?? defaults.symbol,
      side: (searchParams.get("side") as OrderSide | "") ?? defaults.side,
      strategyVersion: searchParams.get("strategyVersion") ?? defaults.strategyVersion,
      skill: searchParams.get("skill") ?? defaults.skill,
      outcome: (searchParams.get("outcome") as TradeOutcome | "") ?? defaults.outcome,
      openFrom: searchParams.get("openFrom") ?? defaults.openFrom,
      openTo: searchParams.get("openTo") ?? defaults.openTo,
      orderBy: (searchParams.get("orderBy") as TradeHistoryOrderBy) ?? defaults.orderBy,
      orderDir: (searchParams.get("orderDir") as "asc" | "desc") ?? defaults.orderDir,
    };
  });
  
  const [groupBy, setGroupBy] = useState<GroupBy>(() => {
    return (searchParams?.get("groupBy") as GroupBy) ?? "none";
  });

  const apiFilters = useMemo(() => toApiFilters(filters), [filters]);
  const { items, total, totalProfit, error, page, setPage } = useTradeHistory(apiFilters);
  const { exportJson, exportCsv, exporting, progress: exportProgress } = useHistoryExport(filters);

  useEffect(() => {
    // get current params to preserve any non-history params
    const params = new URLSearchParams(window.location.search);
    
    const update = (key: string, val: string | boolean | undefined) => {
      if (!val || val === "none") {
        params.delete(key);
      } else {
        params.set(key, String(val));
      }
    };

    update("today", filters.today ? "true" : undefined);
    update("symbol", filters.symbol);
    update("side", filters.side);
    update("strategyVersion", filters.strategyVersion);
    update("skill", filters.skill);
    update("outcome", filters.outcome);
    update("openFrom", filters.openFrom);
    update("openTo", filters.openTo);
    
    // For orderBy and orderDir, only persist if they differ from defaults
    update("orderBy", filters.orderBy === "open_time" ? undefined : filters.orderBy);
    update("orderDir", filters.orderDir === "desc" ? undefined : filters.orderDir);
    update("groupBy", groupBy === "none" ? undefined : groupBy);

    const query = params.toString();
    const newUrl = query ? `${pathname}?${query}` : pathname;
    
    // Only replace if it actually changed to prevent infinite loops and spamming history
    if (window.location.search !== (query ? `?${query}` : "")) {
      router.replace(newUrl, { scroll: false });
    }
  }, [filters, groupBy, pathname, router]);

  const hasNextPage = (page + 1) * PAGE_SIZE < total;

  return (
    <div className="flex flex-col">
      <TradeHistoryFilters
        filters={filters}
        onChange={setFilters}
        groupBy={groupBy}
        onGroupByChange={setGroupBy}
        totalCount={!error && items !== null ? total : undefined}
        totalProfit={!error && items !== null ? totalProfit : undefined}
        onExportJson={exportJson}
        onExportCsv={exportCsv}
        exporting={exporting}
        exportProgress={exportProgress}
      />
      {error && <p className="p-4 text-sm text-err">{error}</p>}
      {!error && items === null && <p className="p-4 text-sm text-ink-muted">Loading…</p>}
      {!error && items !== null && items.length === 0 && (
        <p className="p-4 text-sm text-ink-muted">No trades match these filters.</p>
      )}
      {!error && items !== null && items.length > 0 && (
        <TradeHistoryTable
          trades={items}
          groupBy={groupBy}
          selectedTicket={selectedTicket}
          onSelectTicket={onSelectTicket}
        />
      )}
      {!error && total > 0 && (
        <div className="flex items-center justify-between border-t border-line px-4 py-2 text-xs text-ink-muted">
          <button
            type="button"
            className="cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            ← Prev
          </button>
          <span>
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}
          </span>
          <button
            type="button"
            className="cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
            onClick={() => setPage((p) => (hasNextPage ? p + 1 : p))}
            disabled={!hasNextPage}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
