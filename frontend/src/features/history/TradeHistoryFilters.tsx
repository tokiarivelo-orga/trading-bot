"use client";

import type { TradeHistoryOrderBy, TradeOutcome } from "@/shared/api/client";
import type { OrderSide } from "@/shared/api/client";
import type { GroupBy } from "./groupTrades";

export function getTodayString(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

export interface TradeHistoryFilterState {
  today: boolean;
  symbol: string;
  side: OrderSide | "";
  strategyVersion: string;
  skill: string;
  outcome: TradeOutcome | "";
  openFrom: string; // yyyy-mm-dd, local date input value
  openTo: string;
  orderBy: TradeHistoryOrderBy;
  orderDir: "asc" | "desc";
}

export const EMPTY_FILTERS: TradeHistoryFilterState = {
  today: false,
  symbol: "",
  side: "",
  strategyVersion: "",
  skill: "",
  outcome: "",
  openFrom: "",
  openTo: "",
  orderBy: "open_time",
  orderDir: "desc",
};

export function getDefaultFilters(): TradeHistoryFilterState {
  const today = getTodayString();
  return {
    ...EMPTY_FILTERS,
    today: true,
    openFrom: today,
    openTo: today,
  };
}

const inputCls =
  "rounded border border-line bg-bg px-2 py-1 text-sm text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none";

export function TradeHistoryFilters({
  filters,
  onChange,
  groupBy,
  onGroupByChange,
  totalCount,
  totalProfit,
  onExportJson,
  onExportCsv,
  exporting = false,
  exportProgress = null,
}: {
  filters: TradeHistoryFilterState;
  onChange: (next: TradeHistoryFilterState) => void;
  groupBy: GroupBy;
  onGroupByChange: (next: GroupBy) => void;
  totalCount?: number;
  totalProfit?: number;
  onExportJson?: () => void;
  onExportCsv?: () => void;
  exporting?: boolean;
  exportProgress?: { loaded: number; total: number } | null;
}) {
  function set<K extends keyof TradeHistoryFilterState>(key: K, value: TradeHistoryFilterState[K]) {
    onChange({ ...filters, [key]: value });
  }

  return (
    <div className="flex flex-wrap items-end gap-2 border-b border-line p-3">
      <Field label="Symbol">
        <input
          className={`${inputCls} w-28`}
          placeholder="e.g. XAUUSD"
          value={filters.symbol}
          onChange={(e) => set("symbol", e.target.value.toUpperCase())}
        />
      </Field>
      <Field label="Side">
        <select
          className={inputCls}
          value={filters.side}
          onChange={(e) => set("side", e.target.value as OrderSide | "")}
        >
          <option value="">Any</option>
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </Field>
      <Field label="Outcome">
        <select
          className={inputCls}
          value={filters.outcome}
          onChange={(e) => set("outcome", e.target.value as TradeOutcome | "")}
        >
          <option value="">Any</option>
          <option value="win">Win</option>
          <option value="loss">Loss</option>
          <option value="breakeven">Breakeven</option>
          <option value="open">Open</option>
        </select>
      </Field>
      <Field label="Strategy version">
        <input
          className={`${inputCls} w-36`}
          placeholder="e.g. breakout or v1"
          value={filters.strategyVersion}
          onChange={(e) => set("strategyVersion", e.target.value)}
        />
      </Field>
      <Field label="Skill">
        <input
          className={`${inputCls} w-32`}
          placeholder="e.g. normal or xauusd"
          value={filters.skill}
          onChange={(e) => set("skill", e.target.value)}
        />
      </Field>
      <label className="flex cursor-pointer items-center gap-1.5 self-end pb-2 text-xs font-medium text-ink hover:text-accent">
        <input
          type="checkbox"
          className="cursor-pointer accent-accent"
          checked={filters.today}
          onChange={(e) => {
            if (e.target.checked) {
              const today = getTodayString();
              onChange({ ...filters, today: true, openFrom: today, openTo: today });
            } else {
              onChange({ ...filters, today: false, openFrom: "", openTo: "" });
            }
          }}
        />
        Today
      </label>
      <Field label="Opened from">
        <input
          type="date"
          className={inputCls}
          value={filters.openFrom}
          onChange={(e) => {
            const val = e.target.value;
            const today = getTodayString();
            onChange({
              ...filters,
              openFrom: val,
              today: val === today && filters.openTo === today,
            });
          }}
        />
      </Field>
      <Field label="Opened to">
        <input
          type="date"
          className={inputCls}
          value={filters.openTo}
          onChange={(e) => {
            const val = e.target.value;
            const today = getTodayString();
            onChange({
              ...filters,
              openTo: val,
              today: filters.openFrom === today && val === today,
            });
          }}
        />
      </Field>
      <Field label="Sort by">
        <select
          className={inputCls}
          value={filters.orderBy}
          onChange={(e) => set("orderBy", e.target.value as TradeHistoryOrderBy)}
        >
          <option value="open_time">Open time</option>
          <option value="close_time">Close time</option>
          <option value="profit">Profit</option>
        </select>
      </Field>
      <Field label="Direction">
        <select
          className={inputCls}
          value={filters.orderDir}
          onChange={(e) => set("orderDir", e.target.value as "asc" | "desc")}
        >
          <option value="desc">Newest / highest first</option>
          <option value="asc">Oldest / lowest first</option>
        </select>
      </Field>
      <Field label="Group by">
        <select
          className={inputCls}
          value={groupBy}
          onChange={(e) => onGroupByChange(e.target.value as GroupBy)}
        >
          <option value="none">None</option>
          <option value="symbol">Symbol</option>
          <option value="date">Date opened</option>
          <option value="side">Side</option>
          <option value="strategy_version">Strategy version</option>
          <option value="skill">Skill</option>
          <option value="outcome">Outcome</option>
        </select>
      </Field>
      {hasActiveFilters(filters) && (
        <button
          type="button"
          className="cursor-pointer rounded border border-line px-2 py-1 text-xs text-ink-muted hover:border-accent hover:text-accent"
          onClick={() => onChange(EMPTY_FILTERS)}
        >
          Clear filters
        </button>
      )}
      {totalCount !== undefined && totalProfit !== undefined && (
        <div className="ml-auto flex items-center gap-2 self-center pt-1 md:pt-0">
          <span className="text-xs font-medium text-ink-muted">
            Period net P/L ({totalCount} trade{totalCount === 1 ? "" : "s"}):
          </span>
          <span
            className={`rounded px-2 py-0.5 text-xs font-bold ${
              totalProfit >= 0 ? "bg-ok text-white" : "bg-err text-white"
            }`}
            title={`Total realized profit/loss across ${totalCount} trade${totalCount === 1 ? "" : "s"} in selected period`}
          >
            {totalProfit >= 0 ? "+" : ""}
            {totalProfit.toFixed(2)}
          </span>
        </div>
      )}
      {(onExportJson || onExportCsv) && (
        <div className="flex items-center gap-1 self-center">
          {exporting && exportProgress ? (
            <span className="text-xs text-ink-muted">
              Fetching {exportProgress.loaded} / {exportProgress.total}…
            </span>
          ) : exporting ? (
            <span className="text-xs text-ink-muted">Preparing export…</span>
          ) : null}
          {onExportJson && (
            <button
              type="button"
              disabled={exporting}
              onClick={onExportJson}
              title="Export all matching trades as JSON (training-ready, deeply nested)"
              className="cursor-pointer rounded border border-line px-2 py-1 text-xs text-ink-muted hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              ↓ JSON
            </button>
          )}
          {onExportCsv && (
            <button
              type="button"
              disabled={exporting}
              onClick={onExportCsv}
              title="Export all matching trades as CSV (flat, one row per trade)"
              className="cursor-pointer rounded border border-line px-2 py-1 text-xs text-ink-muted hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              ↓ CSV
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function hasActiveFilters(filters: TradeHistoryFilterState): boolean {
  return (
    filters.today ||
    filters.symbol !== "" ||
    filters.side !== "" ||
    filters.strategyVersion !== "" ||
    filters.skill !== "" ||
    filters.outcome !== "" ||
    filters.openFrom !== "" ||
    filters.openTo !== ""
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-ink-muted">
      {label}
      {children}
    </label>
  );
}
