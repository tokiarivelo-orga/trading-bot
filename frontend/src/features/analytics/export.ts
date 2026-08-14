import {
  getTradeHistory,
  type BotAnalytics,
  type SymbolAnalytics,
  type TradeHistoryItem,
} from "@/shared/api/client";
import { EXPORT_PAGE_SIZE } from "@/shared/api/export";

async function fetchAllTradesForSkill(
  accountId: string,
  skill: string,
  openFrom?: number,
  openTo?: number,
): Promise<TradeHistoryItem[]> {
  const items: TradeHistoryItem[] = [];
  let offset = 0;
  for (;;) {
    const page = await getTradeHistory(accountId, {
      skill,
      open_from: openFrom,
      open_to: openTo,
      order_by: "open_time",
      order_dir: "asc",
      limit: EXPORT_PAGE_SIZE,
      offset,
    });
    items.push(...page.items);
    offset += page.items.length;
    if (page.items.length === 0 || offset >= page.total) break;
  }
  return items;
}

export interface AnalyticsExportBot extends BotAnalytics {
  /** Every journaled trade for this bot — entry/exit, SL/TP, the strategy's
   * stated reason/confidence/pattern/zone/structure for each signal — the
   * per-trade detail an AI review needs to judge or refine the bot, beyond
   * the aggregate stats above. */
  trades: TradeHistoryItem[];
}

export interface AnalyticsExportPayload {
  generated_at: string;
  account_id: string;
  filters: { symbols: string[]; bots: string[]; date_from?: string; date_to?: string };
  symbols: SymbolAnalytics[];
  bots: AnalyticsExportBot[];
}

/** Builds the full export payload for the currently filtered bots/symbols —
 * aggregate stats plus every underlying trade, fetched fresh (paginated)
 * rather than reused from the analytics summary, since `BotAnalytics` alone
 * only carries the equity curve, not entry/exit/zone/reason detail. */
export async function buildAnalyticsExport(
  accountId: string,
  filteredSymbols: SymbolAnalytics[],
  filteredBots: BotAnalytics[],
  activeSymbolFilter: string[],
  activeBotFilter: string[],
  dateFrom: string = "",
  dateTo: string = "",
  openFrom?: number,
  openTo?: number,
): Promise<AnalyticsExportPayload> {
  const bots = await Promise.all(
    filteredBots.map(async (bot) => ({
      ...bot,
      trades: await fetchAllTradesForSkill(accountId, bot.skill, openFrom, openTo),
    })),
  );
  const filters: AnalyticsExportPayload["filters"] = {
    symbols: activeSymbolFilter,
    bots: activeBotFilter,
  };
  if (dateFrom) filters.date_from = dateFrom;
  if (dateTo) filters.date_to = dateTo;

  return {
    generated_at: new Date().toISOString(),
    account_id: accountId,
    filters,
    symbols: filteredSymbols,
    bots,
  };
}

/** Flattens every bot's trades into one row set for CSV — the tabular,
 * per-trade view suited to feeding an AI reviewer/trainer, as opposed to
 * the nested JSON export which also carries the aggregate bot/symbol stats. */
export function flattenTradesForCsv(bots: AnalyticsExportBot[]): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (const bot of bots) {
    for (const t of bot.trades) {
      rows.push({
        bot_name: bot.bot_name,
        skill: bot.skill,
        strategy_version: bot.strategy_version,
        symbol: t.symbol,
        side: t.side,
        volume: t.volume,
        open_price: t.open_price,
        open_time: t.open_time,
        close_price: t.close_price,
        close_time: t.close_time,
        sl: t.sl,
        tp: t.tp,
        profit: t.profit,
        confidence: t.confidence,
        reason: t.reason,
        pattern: t.pattern,
        zone_kind: t.zone?.kind ?? "",
        zone_price_low: t.zone?.price_low ?? "",
        zone_price_high: t.zone?.price_high ?? "",
        zone_time_start: t.zone?.time_start ?? "",
        zone_time_end: t.zone?.time_end ?? "",
        structure: t.structure.length > 0 ? JSON.stringify(t.structure) : "",
        comment: t.comment,
        regime_volatility: t.regime_volatility,
        regime_volatility_percentile: t.regime_volatility_percentile,
        regime_trend: t.regime_trend,
        regime_adx: t.regime_adx,
        regime_session: t.regime_session,
        transaction_cost: t.transaction_cost,
        signal_id: t.signal_id,
      });
    }
  }
  return rows;
}
