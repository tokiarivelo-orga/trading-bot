"use client";

import { useCallback, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import type { TradeHistoryFilters as ApiFilters } from "@/shared/api/client";
import type { TradeHistoryFilterState } from "./TradeHistoryFilters";
import { exportHistoryCsv, exportHistoryJson } from "./exportHistory";

function toApiFilters(f: TradeHistoryFilterState): Omit<ApiFilters, "limit" | "offset"> {
  return {
    symbol: f.symbol || undefined,
    side: f.side || undefined,
    strategy_version: f.strategyVersion || undefined,
    skill: f.skill || undefined,
    outcome: f.outcome || undefined,
    open_from: f.openFrom ? Math.floor(Date.parse(`${f.openFrom}T00:00:00`) / 1000) : undefined,
    open_to: f.openTo ? Math.floor(Date.parse(`${f.openTo}T23:59:59`) / 1000) : undefined,
    order_by: f.orderBy,
    order_dir: f.orderDir,
  };
}

function toFiltersLabel(f: TradeHistoryFilterState): Record<string, string | number | undefined> {
  const label: Record<string, string | number | undefined> = {};
  if (f.symbol) label.symbol = f.symbol;
  if (f.side) label.side = f.side;
  if (f.strategyVersion) label.strategy_version = f.strategyVersion;
  if (f.skill) label.skill = f.skill;
  if (f.outcome) label.outcome = f.outcome;
  if (f.openFrom) label.open_from = f.openFrom;
  if (f.openTo) label.open_to = f.openTo;
  if (f.today) label.today_only = "true";
  label.order_by = f.orderBy;
  label.order_dir = f.orderDir;
  return label;
}

export function useHistoryExport(filters: TradeHistoryFilterState) {
  const accountId = useActiveAccount();
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (format: "json" | "csv") => {
      if (!accountId) return;
      setExporting(true);
      setError(null);
      setProgress(null);
      try {
        const apiFilters = toApiFilters(filters);
        const onProgress = (loaded: number, total: number) => setProgress({ loaded, total });
        if (format === "json") {
          await exportHistoryJson(accountId, apiFilters, toFiltersLabel(filters), onProgress);
        } else {
          await exportHistoryCsv(accountId, apiFilters, onProgress);
        }
      } catch {
        setError("Export failed — please try again.");
      } finally {
        setExporting(false);
        setProgress(null);
      }
    },
    [accountId, filters],
  );

  return {
    exportJson: () => run("json"),
    exportCsv: () => run("csv"),
    exporting,
    progress,
    error,
  };
}
