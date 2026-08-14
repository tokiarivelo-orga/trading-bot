/**
 * Trade History Export
 *
 * Fetches ALL trades matching the current filters (paginated in the background,
 * not just the current visible page) and builds a complete, training-ready export.
 *
 * JSON export: deeply nested payload with every field including indicators,
 *              zone/structure, derived metrics (R-multiple, duration, risk/reward),
 *              and contextual metadata — ideal for LLM-based review or ML training.
 *
 * CSV export: one flat row per trade, with every scalar expanded for spreadsheet
 *             or pandas/polars ingestion; arrays (indicators, structure) are
 *             serialised as JSON strings in their own columns.
 */

import { getTradeHistory, type TradeHistoryFilters, type TradeHistoryItem } from "@/shared/api/client";
import { EXPORT_PAGE_SIZE } from "@/shared/api/export";
import { downloadCsv, downloadJson } from "@/shared/utils/download";

// ─── Helpers ────────────────────────────────────────────────────────────────

function epochToIso(epoch: number | null): string {
  if (epoch === null) return "";
  return new Date(epoch * 1000).toISOString();
}

function durationSeconds(openTime: number, closeTime: number | null): number | null {
  if (closeTime === null) return null;
  return closeTime - openTime;
}

function durationHuman(seconds: number | null): string {
  if (seconds === null) return "";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h && `${h}h`, m && `${m}m`, `${s}s`].filter(Boolean).join(" ");
}

function rMultiple(
  profit: number | null,
  sl: number | null,
  openPrice: number,
  volume: number,
): number | null {
  if (profit === null || sl === null || volume === 0) return null;
  const riskPrice = Math.abs(openPrice - sl);
  if (riskPrice === 0) return null;
  // Approximate: profit per lot · (1 / pip_size) is broker-specific —
  // we report a dimensionless ratio profit / (riskPrice * volume) as a
  // consistent relative measure across instruments.
  return profit / (riskPrice * volume);
}

function riskRewardRatio(
  sl: number | null,
  tp: number | null,
  openPrice: number,
): number | null {
  if (sl === null || tp === null) return null;
  const risk = Math.abs(openPrice - sl);
  const reward = Math.abs(tp - openPrice);
  if (risk === 0) return null;
  return reward / risk;
}

function outcome(t: TradeHistoryItem): "win" | "loss" | "breakeven" | "open" {
  if (t.close_time === null) return "open";
  if (t.profit === null || t.profit === 0) return "breakeven";
  return t.profit > 0 ? "win" : "loss";
}

function indicatorsSummary(t: TradeHistoryItem): { passed: number; total: number } {
  const passed = t.indicators.filter((i) => i.passed).length;
  return { passed, total: t.indicators.length };
}

// ─── Fetch all pages ─────────────────────────────────────────────────────────

export async function fetchAllHistoryTrades(
  accountId: string,
  filters: Omit<TradeHistoryFilters, "limit" | "offset">,
  onProgress?: (loaded: number, total: number) => void,
): Promise<TradeHistoryItem[]> {
  const items: TradeHistoryItem[] = [];
  let offset = 0;
  for (;;) {
    const page = await getTradeHistory(accountId, {
      ...filters,
      order_by: filters.order_by ?? "open_time",
      order_dir: filters.order_dir ?? "asc",
      limit: EXPORT_PAGE_SIZE,
      offset,
    });
    items.push(...page.items);
    offset += page.items.length;
    onProgress?.(offset, page.total);
    if (page.items.length === 0 || offset >= page.total) break;
  }
  return items;
}

// ─── JSON payload ─────────────────────────────────────────────────────────────

export interface TradeExportRecord {
  // ── Identity ──────────────────────────────────────────────────────────────
  id: string;
  account_id: string;

  // ── Instrument ────────────────────────────────────────────────────────────
  symbol: string;
  side: "buy" | "sell";
  volume: number;
  outcome: "win" | "loss" | "breakeven" | "open";

  // ── Entry ─────────────────────────────────────────────────────────────────
  open_price: number;
  open_time_epoch: number;
  open_time_iso: string;
  sl: number | null;
  tp: number | null;

  // ── Exit ──────────────────────────────────────────────────────────────────
  close_price: number | null;
  close_time_epoch: number | null;
  close_time_iso: string;
  close_reason: string | null;
  comment: string;

  // ── P&L ───────────────────────────────────────────────────────────────────
  profit: number | null;
  duration_seconds: number | null;
  duration_human: string;
  r_multiple: number | null;       // profit / (|open - sl| * volume)  — relative measure
  risk_reward_ratio: number | null; // |tp - open| / |open - sl|

  // ── Strategy metadata ─────────────────────────────────────────────────────
  strategy_version: string | null;
  skill: string | null;

  // ── Signal quality ────────────────────────────────────────────────────────
  reason: string;                  // full free-text signal rationale
  confidence: number | null;       // 0..1 signal confidence score
  pattern: string | null;          // e.g. "RBR", "DBD", "engulfing"

  // ── Zone context ──────────────────────────────────────────────────────────
  zone: {
    kind: "demand" | "supply";
    price_low: number;
    price_high: number;
    time_start_epoch: number;
    time_start_iso: string;
    time_end_epoch: number;
    time_end_iso: string;
    zone_pattern: string | null;
  } | null;

  // ── Market structure ──────────────────────────────────────────────────────
  structure: Array<{
    label: "HH" | "HL" | "LH" | "LL";
    price: number;
    time_epoch: number;
    time_iso: string;
  }>;

  // ── Confluence indicators ─────────────────────────────────────────────────
  indicators: Array<{
    name: string;
    value: number;
    threshold: number;
    comparison: string;
    passed: boolean;
  }>;
  indicators_passed_count: number;
  indicators_total_count: number;
  indicators_pass_rate: number | null; // 0..1

  // ── Regime & execution context (OBSERVABILITY_PLAN.md Phase 6) ────────────
  regime_volatility: string | null;            // 'low'/'normal'/'high'/'extreme'
  regime_volatility_percentile: number | null; // 0..100, ATR percentile rank
  regime_trend: string | null;                 // 'trending'/'ranging'
  regime_adx: number | null;                   // raw ADX reading, 0..100
  regime_session: string | null;                // 'asian'/'london'/'overlap'/'new_york'/'off_session'
  transaction_cost: number | null;              // spread + slippage, account currency
  signal_id: string | null;                     // joinable against /activity/... and order-book snapshots
}

export interface HistoryExportPayload {
  schema_version: "1.1";
  generated_at: string;
  account_id: string;
  filters_applied: Record<string, string | number | undefined>;

  // ── Aggregate stats (across ALL exported trades) ──────────────────────────
  summary: {
    total_trades: number;
    closed_trades: number;
    open_trades: number;
    win_count: number;
    loss_count: number;
    breakeven_count: number;
    win_rate: number | null;
    total_profit: number;
    gross_profit: number;
    gross_loss: number;
    profit_factor: number | null;
    avg_profit_per_trade: number | null;
    avg_win: number | null;
    avg_loss: number | null;   // magnitude (positive)
    largest_win: number | null;
    largest_loss: number | null;
    avg_duration_seconds: number | null;
    avg_confidence: number | null;
    avg_r_multiple: number | null;
  };

  trades: TradeExportRecord[];
}

function buildTradeRecord(t: TradeHistoryItem, accountId: string): TradeExportRecord {
  const dur = durationSeconds(t.open_time, t.close_time);
  const rm = rMultiple(t.profit, t.sl, t.open_price, t.volume);
  const rr = riskRewardRatio(t.sl, t.tp, t.open_price);
  const { passed, total } = indicatorsSummary(t);
  return {
    id: t.id,
    account_id: accountId,
    symbol: t.symbol,
    side: t.side,
    volume: t.volume,
    outcome: outcome(t),
    open_price: t.open_price,
    open_time_epoch: t.open_time,
    open_time_iso: epochToIso(t.open_time),
    sl: t.sl,
    tp: t.tp,
    close_price: t.close_price,
    close_time_epoch: t.close_time,
    close_time_iso: epochToIso(t.close_time),
    close_reason: t.close_reason,
    comment: t.comment,
    profit: t.profit,
    duration_seconds: dur,
    duration_human: durationHuman(dur),
    r_multiple: rm !== null ? Math.round(rm * 100) / 100 : null,
    risk_reward_ratio: rr !== null ? Math.round(rr * 100) / 100 : null,
    strategy_version: t.strategy_version,
    skill: t.skill,
    reason: t.reason,
    confidence: t.confidence,
    pattern: t.pattern,
    zone: t.zone
      ? {
          kind: t.zone.kind,
          price_low: t.zone.price_low,
          price_high: t.zone.price_high,
          time_start_epoch: t.zone.time_start,
          time_start_iso: epochToIso(t.zone.time_start),
          time_end_epoch: t.zone.time_end,
          time_end_iso: epochToIso(t.zone.time_end),
          zone_pattern: t.zone.pattern,
        }
      : null,
    structure: t.structure.map((s) => ({
      label: s.label,
      price: s.price,
      time_epoch: s.time,
      time_iso: epochToIso(s.time),
    })),
    indicators: t.indicators,
    indicators_passed_count: passed,
    indicators_total_count: total,
    indicators_pass_rate: total > 0 ? Math.round((passed / total) * 1000) / 1000 : null,
    regime_volatility: t.regime_volatility,
    regime_volatility_percentile: t.regime_volatility_percentile,
    regime_trend: t.regime_trend,
    regime_adx: t.regime_adx,
    regime_session: t.regime_session,
    transaction_cost: t.transaction_cost,
    signal_id: t.signal_id,
  };
}

function buildSummary(trades: TradeExportRecord[]): HistoryExportPayload["summary"] {
  const closed = trades.filter((t) => t.outcome !== "open");
  const wins = closed.filter((t) => t.outcome === "win");
  const losses = closed.filter((t) => t.outcome === "loss");

  const profits = closed.map((t) => t.profit ?? 0);
  const totalProfit = profits.reduce((a, b) => a + b, 0);
  const grossProfit = profits.filter((p) => p > 0).reduce((a, b) => a + b, 0);
  const grossLoss = Math.abs(profits.filter((p) => p < 0).reduce((a, b) => a + b, 0));

  const avg = (arr: number[]) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : null);

  const durations = closed.map((t) => t.duration_seconds).filter((d): d is number => d !== null);
  const confidences = trades.map((t) => t.confidence).filter((c): c is number => c !== null);
  const rMultiples = closed.map((t) => t.r_multiple).filter((r): r is number => r !== null);

  return {
    total_trades: trades.length,
    closed_trades: closed.length,
    open_trades: trades.filter((t) => t.outcome === "open").length,
    win_count: wins.length,
    loss_count: losses.length,
    breakeven_count: closed.filter((t) => t.outcome === "breakeven").length,
    win_rate: closed.length > 0 ? Math.round((wins.length / closed.length) * 1000) / 1000 : null,
    total_profit: Math.round(totalProfit * 100) / 100,
    gross_profit: Math.round(grossProfit * 100) / 100,
    gross_loss: Math.round(grossLoss * 100) / 100,
    profit_factor: grossLoss > 0 ? Math.round((grossProfit / grossLoss) * 100) / 100 : null,
    avg_profit_per_trade: closed.length > 0 ? Math.round((totalProfit / closed.length) * 100) / 100 : null,
    avg_win: wins.length > 0 ? Math.round(avg(wins.map((t) => t.profit ?? 0))! * 100) / 100 : null,
    avg_loss: losses.length > 0 ? Math.round(Math.abs(avg(losses.map((t) => t.profit ?? 0))!) * 100) / 100 : null,
    largest_win: wins.length > 0 ? Math.max(...wins.map((t) => t.profit ?? 0)) : null,
    largest_loss: losses.length > 0 ? Math.min(...losses.map((t) => t.profit ?? 0)) : null,
    avg_duration_seconds: durations.length > 0 ? Math.round(avg(durations)!) : null,
    avg_confidence: confidences.length > 0 ? Math.round(avg(confidences)! * 1000) / 1000 : null,
    avg_r_multiple: rMultiples.length > 0 ? Math.round(avg(rMultiples)! * 100) / 100 : null,
  };
}

export function buildHistoryExportPayload(
  trades: TradeExportRecord[],
  accountId: string,
  filtersApplied: Record<string, string | number | undefined>,
): HistoryExportPayload {
  return {
    schema_version: "1.1",
    generated_at: new Date().toISOString(),
    account_id: accountId,
    filters_applied: filtersApplied,
    summary: buildSummary(trades),
    trades,
  };
}

// ─── CSV flat row ─────────────────────────────────────────────────────────────

export function flattenTradeForCsv(t: TradeExportRecord): Record<string, unknown> {
  return {
    // Identity
    id: t.id,
    account_id: t.account_id,

    // Instrument
    symbol: t.symbol,
    side: t.side,
    volume: t.volume,
    outcome: t.outcome,

    // Entry
    open_price: t.open_price,
    open_time_epoch: t.open_time_epoch,
    open_time_iso: t.open_time_iso,
    sl: t.sl ?? "",
    tp: t.tp ?? "",

    // Exit
    close_price: t.close_price ?? "",
    close_time_epoch: t.close_time_epoch ?? "",
    close_time_iso: t.close_time_iso,
    close_reason: t.close_reason ?? "",
    comment: t.comment,

    // P&L
    profit: t.profit ?? "",
    duration_seconds: t.duration_seconds ?? "",
    duration_human: t.duration_human,
    r_multiple: t.r_multiple ?? "",
    risk_reward_ratio: t.risk_reward_ratio ?? "",

    // Strategy
    strategy_version: t.strategy_version ?? "",
    skill: t.skill ?? "",

    // Signal quality
    reason: t.reason,
    confidence: t.confidence ?? "",
    pattern: t.pattern ?? "",

    // Zone (expanded columns — easy to filter in pandas/Excel)
    zone_kind: t.zone?.kind ?? "",
    zone_price_low: t.zone?.price_low ?? "",
    zone_price_high: t.zone?.price_high ?? "",
    zone_time_start_epoch: t.zone?.time_start_epoch ?? "",
    zone_time_start_iso: t.zone?.time_start_iso ?? "",
    zone_time_end_epoch: t.zone?.time_end_epoch ?? "",
    zone_time_end_iso: t.zone?.time_end_iso ?? "",
    zone_pattern: t.zone?.zone_pattern ?? "",

    // Market structure (JSON array — keeps CSV one-row-per-trade)
    structure_json:
      t.structure.length > 0
        ? JSON.stringify(t.structure.map((s) => ({ label: s.label, price: s.price, time_iso: s.time_iso })))
        : "",

    // Confluence indicators
    indicators_passed_count: t.indicators_passed_count,
    indicators_total_count: t.indicators_total_count,
    indicators_pass_rate: t.indicators_pass_rate ?? "",
    indicators_json:
      t.indicators.length > 0
        ? JSON.stringify(t.indicators.map((i) => ({ name: i.name, value: i.value, threshold: i.threshold, comparison: i.comparison, passed: i.passed })))
        : "",

    // Regime & execution context (OBSERVABILITY_PLAN.md Phase 6)
    regime_volatility: t.regime_volatility ?? "",
    regime_volatility_percentile: t.regime_volatility_percentile ?? "",
    regime_trend: t.regime_trend ?? "",
    regime_adx: t.regime_adx ?? "",
    regime_session: t.regime_session ?? "",
    transaction_cost: t.transaction_cost ?? "",
    signal_id: t.signal_id ?? "",
  };
}

// ─── Download orchestration ───────────────────────────────────────────────────

export async function exportHistoryJson(
  accountId: string,
  filters: Omit<TradeHistoryFilters, "limit" | "offset">,
  filtersLabel: Record<string, string | number | undefined>,
  onProgress?: (loaded: number, total: number) => void,
): Promise<void> {
  const raw = await fetchAllHistoryTrades(accountId, filters, onProgress);
  const records = raw.map((t) => buildTradeRecord(t, accountId));
  const payload = buildHistoryExportPayload(records, accountId, filtersLabel);
  const stamp = new Date().toISOString().slice(0, 10);
  downloadJson(payload, `trade_history_${stamp}.json`);
}

export async function exportHistoryCsv(
  accountId: string,
  filters: Omit<TradeHistoryFilters, "limit" | "offset">,
  onProgress?: (loaded: number, total: number) => void,
): Promise<void> {
  const raw = await fetchAllHistoryTrades(accountId, filters, onProgress);
  const records = raw.map((t) => buildTradeRecord(t, accountId));
  const rows = records.map(flattenTradeForCsv);
  const stamp = new Date().toISOString().slice(0, 10);
  downloadCsv(rows, `trade_history_${stamp}.csv`);
}
