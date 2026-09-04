"use client";

/**
 * SignalsDock — TradingView-style panel listing every signal and trade (order taken)
 * a backtest report's strategy emitted, so the trader can audit each setup.
 * Clicking navigation buttons scrolls/centers the chart on that bar.
 *
 * Rendered on the right side of the chart inside ChartPanel.
 */

import { useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Clock, Search, MapPin, HelpCircle } from "lucide-react";
import { SIGNAL_OUTCOME_META } from "@/features/backtest/signalOutcome";
import { TradeDecisionModal } from "@/shared/ui/TradeDecisionModal";
import { BotSessionReplayTab } from "./BotSessionReplayTab";
import type { BotReplayControls } from "./types";
import type {
  BacktestSignal,
  BacktestTrade,
  IndicatorSpec,
  TradeHistoryItem,
} from "@/shared/api/client";

/** Builds the `TradeHistoryItem` shape `TradeDecisionModal` expects (shared
 * with the live Active Orders/History tables) out of a backtest report's
 * `BacktestTrade` + the report's own strategy/symbol — the "Why" button's
 * only consumer, so the synthesized object never leaves this module. */
function toTradeHistoryItem(
  trade: BacktestTrade,
  index: number,
  meta: { strategy: string; symbol: string } | null,
): TradeHistoryItem {
  return {
    id: String(index),
    symbol: meta?.symbol ?? "",
    side: trade.side as "buy" | "sell",
    volume: trade.volume,
    open_price: trade.open_price,
    open_time: trade.open_time,
    sl: trade.sl,
    tp: trade.tp,
    close_price: trade.close_price,
    close_time: trade.close_time,
    profit: trade.profit,
    // BacktestTrade doesn't report a close reason (backtests don't run the
    // live position manager's volatility guard/time-stop rules).
    close_reason: null,
    comment: "",
    strategy_version: meta?.strategy ?? null,
    skill: null,
    reason: trade.reason,
    confidence: trade.confidence,
    zone: trade.zone,
    pattern: trade.pattern,
    structure: trade.structure,
    // BacktestTrade doesn't report indicator readings; only live/journaled
    // trades (TradeHistoryItem from the backend) carry that field.
    indicators: [],
  } as unknown as TradeHistoryItem;
}

export function SignalsDock({
  signals,
  trades,
  indicators,
  backtestMeta = null,
  selectedTradeIndex = null,
  onSelectTrade,
  onNavigateTrade,
  selectedSignalIndex = null,
  onSelectSignal,
  replayCursorTime = null,
  currentPrice = null,
  replay,
}: {
  signals: BacktestSignal[];
  trades: BacktestTrade[];
  /** The bot/strategy version's own indicator specs (from its `StrategySpec`)
   * — undefined hides the tab entirely (e.g. no version resolved yet), an
   * empty array shows the tab with an empty state. */
  indicators?: IndicatorSpec[];
  /** The report's own strategy/symbol — fed to the "Why" button's
   * `TradeDecisionModal`. Null hides nothing (the modal just shows "—" for
   * strategy), only used for display. */
  backtestMeta?: { strategy: string; symbol: string } | null;
  /** Index (into the original, unsorted `trades` array) of the trade
   * currently highlighted on the chart with entry/SL/TP/close lines — drives
   * this row's selected style. Null when nothing's selected. */
  selectedTradeIndex?: number | null;
  /** Card click: toggles the trade's chart highlight off if it's already
   * selected, otherwise selects it and jumps the chart to its entry. */
  onSelectTrade?: (index: number) => void;
  /** Entry/Exit buttons: always selects (never toggles off) and jumps to
   * the given time — an explicit "look at this", not a selection toggle. */
  onNavigateTrade?: (index: number, time: number) => void;
  /** Index (into the original, unsorted `signals` array) of the signal
   * currently highlighted on the chart with a vertical dashed line — drives
   * this row's selected style. Null when nothing's selected. */
  selectedSignalIndex?: number | null;
  /** Row click: toggles the signal's chart highlight off if it's already
   * selected, otherwise selects it and jumps the chart to its time. */
  onSelectSignal?: (index: number) => void;
  /** The replay cursor bar's time while replaying a backtest report — null
   * outside replay (or the live-bot eye view). When set, the Trades tab
   * splits into "Active orders" (opened, not yet closed as of the cursor)
   * and "History" (closed as of the cursor), same labels as the live
   * Active Orders panel, and hides trades not yet opened — the same
   * "no lookahead" contract the chart's own markers already enforce. */
  replayCursorTime?: number | null;
  /** The replay cursor bar's own close price — null outside replay. Lets
   * each "Active orders" card mark an open trade to market the same way a
   * live broker position's `profit` field does, instead of a static "OPEN"
   * placeholder that never moves until the trade's real close is revealed. */
  currentPrice?: number | null;
  /** Session-replay wiring for the bot currently under the eye — undefined
   * hides the Replay tab entirely (saved-backtest views, which have the
   * chart toolbar's own replay controls instead). See `BotReplayControls`. */
  replay?: BotReplayControls;
}) {
  const [whyTrade, setWhyTrade] = useState<TradeHistoryItem | null>(null);
  const [activeTab, setActiveTab] = useState<
    "signals" | "trades" | "indicators" | "replay"
  >("signals");
  
  // Search & Filter state
  const [searchText, setSearchText] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState("");
  const [sideFilter, setSideFilter] = useState("");
  const [profitFilter, setProfitFilter] = useState("");

  const searchLower = searchText.toLowerCase();

  // Process signals (newest first)
  const filteredSignals = signals
    .map((s, idx) => ({ ...s, originalIndex: idx }))
    .filter((s) => {
      if (outcomeFilter && s.outcome !== outcomeFilter) return false;
      if (searchText && !s.reason.toLowerCase().includes(searchLower)) return false;
      return true;
    })
    .sort((a, b) => b.time - a.time);

  // Process trades (newest first)
  const filteredTrades = trades
    .map((t, idx) => ({ ...t, originalIndex: idx }))
    .filter((t) => {
      if (sideFilter && t.side !== sideFilter) return false;
      if (profitFilter === "win" && t.profit <= 0) return false;
      if (profitFilter === "loss" && t.profit >= 0) return false;
      const patternText = (t.pattern || "").toLowerCase();
      if (searchText && !patternText.includes(searchLower)) return false;
      return true;
    })
    .sort((a, b) => b.open_time - a.open_time);

  // While replaying, hide trades the strategy hasn't opened yet as of the
  // cursor (same "no lookahead" contract the chart's own markers enforce —
  // see useBacktestData.ts) and split the rest into "Active orders"
  // (opened, not yet closed) vs "History" (closed) — same two labels the
  // live Active Orders panel uses. Outside replay, `replayCursorTime` is
  // null and everything renders as one flat list, unchanged from before.
  const visibleTrades =
    replayCursorTime === null
      ? filteredTrades
      : filteredTrades.filter((t) => t.open_time <= replayCursorTime);
  const activeTrades =
    replayCursorTime === null
      ? []
      : visibleTrades.filter((t) => t.close_time > replayCursorTime);
  const historyTrades =
    replayCursorTime === null
      ? visibleTrades
      : visibleTrades.filter((t) => t.close_time <= replayCursorTime);

  return (
    <div className="w-[340px] border-l border-line bg-panel flex flex-col h-full shrink-0 min-w-0">
      {/* Tabs */}
      <div className="flex border-b border-line bg-panel-dark/50 shrink-0">
        <button
          type="button"
          onClick={() => {
            setActiveTab("signals");
            setSearchText("");
          }}
          className={`flex-1 py-2 px-3 text-xs font-semibold border-b-2 text-center transition-colors cursor-pointer ${
            activeTab === "signals"
              ? "border-accent text-accent bg-accent/5"
              : "border-transparent text-ink-muted hover:text-ink hover:bg-panel-dark/20"
          }`}
        >
          Signals ({signals.length})
        </button>
        <button
          type="button"
          onClick={() => {
            setActiveTab("trades");
            setSearchText("");
          }}
          className={`flex-1 py-2 px-3 text-xs font-semibold border-b-2 text-center transition-colors cursor-pointer ${
            activeTab === "trades"
              ? "border-accent text-accent bg-accent/5"
              : "border-transparent text-ink-muted hover:text-ink hover:bg-panel-dark/20"
          }`}
        >
          Trades ({trades.length})
        </button>
        {indicators !== undefined && (
          <button
            type="button"
            onClick={() => {
              setActiveTab("indicators");
              setSearchText("");
            }}
            className={`flex-1 py-2 px-3 text-xs font-semibold border-b-2 text-center transition-colors cursor-pointer ${
              activeTab === "indicators"
                ? "border-accent text-accent bg-accent/5"
                : "border-transparent text-ink-muted hover:text-ink hover:bg-panel-dark/20"
            }`}
          >
            Indicators ({indicators.length})
          </button>
        )}
        {replay && (
          <button
            type="button"
            onClick={() => {
              setActiveTab("replay");
              setSearchText("");
            }}
            className={`flex-1 py-2 px-3 text-xs font-semibold border-b-2 text-center transition-colors cursor-pointer ${
              activeTab === "replay"
                ? "border-accent text-accent bg-accent/5"
                : "border-transparent text-ink-muted hover:text-ink hover:bg-panel-dark/20"
            }`}
          >
            Replay
          </button>
        )}
      </div>

      {/* Filters Area — the indicators and replay tabs are forms/lists with
          too few rows to need search/filter */}
      {activeTab !== "indicators" && activeTab !== "replay" && (
      <div className="p-2 border-b border-line flex flex-col gap-1.5 bg-panel-dark/20 shrink-0">
        {/* Search */}
        <div className="relative">
          <Search size={12} className="absolute left-2 top-2 text-ink-muted" />
          <input
            type="text"
            placeholder={activeTab === "signals" ? "Search signals..." : "Search trades..."}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="w-full pl-7 pr-2 py-1 bg-panel border border-line rounded text-xs text-ink placeholder-ink-muted focus:border-accent focus:outline-none"
          />
        </div>

        {/* Dropdowns */}
        <div className="flex gap-1.5">
          {activeTab === "signals" ? (
            <select
              value={outcomeFilter}
              onChange={(e) => setOutcomeFilter(e.target.value)}
              className="flex-1 cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
            >
              <option value="">All outcomes</option>
              {Object.entries(SIGNAL_OUTCOME_META).map(([value, meta]) => (
                <option key={value} value={value}>
                  {meta.label}
                </option>
              ))}
            </select>
          ) : (
            <>
              <select
                value={sideFilter}
                onChange={(e) => setSideFilter(e.target.value)}
                className="flex-1 cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
              >
                <option value="">All sides</option>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
              <select
                value={profitFilter}
                onChange={(e) => setProfitFilter(e.target.value)}
                className="flex-1 cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
              >
                <option value="">All results</option>
                <option value="win">Win</option>
                <option value="loss">Loss</option>
              </select>
            </>
          )}
        </div>
      </div>
      )}

      {/* List Container — each tab's active list owns its own dedicated
          scroll region (rather than sharing one ambient scrollable
          ancestor) because the virtualized Signals/History lists below need
          `getScrollElement` to point at their own real scrolling element. */}
      <div className="flex-1 min-h-0 flex flex-col">
        {activeTab === "replay" ? (
          replay ? (
            <div className="flex-1 min-h-0 overflow-y-auto">
              <BotSessionReplayTab
                signals={signals}
                trades={trades}
                replay={replay}
              />
            </div>
          ) : null
        ) : activeTab === "indicators" ? (
          <div className="flex-1 min-h-0 overflow-y-auto">
            {!indicators || indicators.length === 0 ? (
              <p className="px-3 py-4 text-xs text-ink-muted text-center">
                No indicators recorded for this bot&apos;s strategy spec.
              </p>
            ) : (
              <ul className="divide-y divide-line">
                {indicators.map((ind, idx) => (
                  <li key={`${ind.type}-${ind.source}-${idx}`} className="p-2.5 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-bold uppercase px-1 rounded bg-accent/10 text-accent border border-accent/20">
                        {ind.type}
                      </span>
                      <span className="text-xs font-medium text-ink truncate">{ind.label}</span>
                    </div>
                    <div className="text-[10px] text-ink-muted">
                      period {ind.period} · source {ind.source}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : activeTab === "signals" ? (
          filteredSignals.length === 0 ? (
            <p className="px-3 py-4 text-xs text-ink-muted text-center">
              {signals.length === 0
                ? "No signals recorded for this report."
                : "No matching signals found."}
            </p>
          ) : (
            <VirtualSignalList
              signals={filteredSignals}
              selectedSignalIndex={selectedSignalIndex}
              onSelectSignal={onSelectSignal}
            />
          )
        ) : filteredTrades.length === 0 ? (
          <p className="px-3 py-4 text-xs text-ink-muted text-center">
            {trades.length === 0
              ? "No trades recorded for this report."
              : "No matching trades found."}
          </p>
        ) : replayCursorTime === null ? (
          <VirtualTradeList
            trades={historyTrades}
            selectedTradeIndex={selectedTradeIndex}
            onSelectTrade={onSelectTrade}
            onNavigateTrade={onNavigateTrade}
            onWhy={(t) => setWhyTrade(toTradeHistoryItem(t, t.originalIndex, backtestMeta))}
          />
        ) : (
          <>
            {/* Active orders — realistically never more than a handful of
                concurrently open positions for one strategy, so this stays
                a plain unvirtualized list; capped with its own scroll region
                as a defensive fallback so an unusually large count can't
                starve the History section below of space. */}
            <div className="shrink-0 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wider text-ink-muted bg-panel-dark/30">
              Active orders ({activeTrades.length})
            </div>
            {activeTrades.length === 0 ? (
              <p className="shrink-0 px-3 py-3 text-xs text-ink-muted text-center">Nothing open yet.</p>
            ) : (
              <ul className="shrink-0 max-h-48 overflow-y-auto divide-y divide-line">
                {activeTrades.map((t) => (
                  <li key={`${t.open_time}-${t.originalIndex}`}>
                    <TradeCard
                      trade={t}
                      isOpen
                      currentPrice={currentPrice}
                      selected={selectedTradeIndex === t.originalIndex}
                      onSelect={() => onSelectTrade?.(t.originalIndex)}
                      onNavigate={(time) => onNavigateTrade?.(t.originalIndex, time)}
                      onWhy={() => setWhyTrade(toTradeHistoryItem(t, t.originalIndex, backtestMeta))}
                    />
                  </li>
                ))}
              </ul>
            )}
            {/* History — grows unbounded as replay progresses, so this is
                the section that needs virtualization. Its header sits
                outside (above) the dedicated scroll region as a plain flex
                sibling rather than a CSS `sticky` element — that pins it
                just as reliably without fighting the virtualizer's own
                `getScrollElement` container. */}
            <div className="shrink-0 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-wider text-ink-muted bg-panel-dark/30 border-t border-line">
              History ({historyTrades.length})
            </div>
            {historyTrades.length === 0 ? (
              <p className="shrink-0 px-3 py-3 text-xs text-ink-muted text-center">Nothing closed yet.</p>
            ) : (
              <VirtualTradeList
                trades={historyTrades}
                selectedTradeIndex={selectedTradeIndex}
                onSelectTrade={onSelectTrade}
                onNavigateTrade={onNavigateTrade}
                onWhy={(t) => setWhyTrade(toTradeHistoryItem(t, t.originalIndex, backtestMeta))}
              />
            )}
          </>
        )}
      </div>
      {whyTrade && <TradeDecisionModal trade={whyTrade} onClose={() => setWhyTrade(null)} />}
    </div>
  );
}

function TradeCard({
  trade: t,
  isOpen,
  currentPrice = null,
  selected,
  onSelect,
  onNavigate,
  onWhy,
}: {
  trade: BacktestTrade & { originalIndex: number };
  /** True while replaying and this trade hasn't closed as of the cursor yet
   * — shows a mark-to-market profit (or an "OPEN" badge, lacking a current
   * price) instead of the trade's final, not-yet-revealed profit. */
  isOpen: boolean;
  /** The replay cursor bar's close — used to mark an open trade to market.
   * A backtest trade's profit is exactly linear in price (see
   * `broker/adapters/paper.py`: `(price - open_price) * direction *
   * contract_size * volume`, no commission/swap folded in), so the trade's
   * own known open/close/profit triple gives the per-price-unit rate
   * directly — no need to duplicate the broker's contract-size/point-value
   * lookup on the frontend. */
  currentPrice?: number | null;
  selected: boolean;
  onSelect: () => void;
  onNavigate: (time: number) => void;
  onWhy: () => void;
}) {
  // Same linear relationship the backtest engine used to produce `t.profit`
  // from `t.close_price`, solved backwards for the rate (direction ×
  // contract size × volume) and reapplied at the cursor's current price.
  // `priceDelta === 0` only when the trade closed at its own open price
  // (flat), in which case `t.profit` is already 0 and mark-to-market has
  // nothing to interpolate.
  const priceDelta = t.close_price - t.open_price;
  const liveProfit =
    isOpen && currentPrice !== null && priceDelta !== 0
      ? (t.profit / priceDelta) * (currentPrice - t.open_price)
      : null;

  return (
    <div
      onClick={onSelect}
      title="Highlight this trade's entry/SL/TP/close on the chart"
      className={`p-2.5 hover:bg-accent/5 transition-colors flex flex-col gap-1.5 relative border-l-2 cursor-pointer ${
        selected ? "bg-accent/10 border-accent" : "border-transparent"
      }`}
    >
      {/* Header info */}
        <div className="flex items-center gap-1.5 text-[10px] text-ink-muted">
          <span className="font-semibold text-ink">Trade #{t.originalIndex + 1}</span>
          <span>•</span>
          <span>{formatTime(t.open_time)}</span>
          {isOpen && liveProfit !== null ? (
            <span
              className={`ml-auto inline-flex items-center gap-1 font-mono font-bold ${liveProfit >= 0 ? "text-ok" : "text-err"}`}
              title="Mark-to-market profit at the replay cursor's price"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
              {liveProfit >= 0 ? "+" : ""}
              {liveProfit.toFixed(2)} USD
            </span>
          ) : isOpen ? (
            <span className="ml-auto font-mono font-bold text-accent">OPEN</span>
          ) : (
            <span className={`ml-auto font-mono font-bold ${t.profit >= 0 ? "text-ok" : "text-err"}`}>
              {t.profit >= 0 ? "+" : ""}
              {t.profit.toFixed(2)} USD
            </span>
          )}
        </div>

        {/* Volume and price */}
        <div className="flex items-center gap-1.5">
          <span
            className={`text-[10px] font-bold px-1 rounded ${
              t.side === "buy"
                ? "bg-ok/10 text-ok border border-ok/20"
                : "bg-err/10 text-err border border-err/20"
            }`}
          >
            {t.side.toUpperCase()}
          </span>
          <span className="text-[11px] text-ink-muted">
            {t.volume.toFixed(2)} lots @ {t.open_price.toFixed(2)}
          </span>
          {!isOpen && t.r_multiple !== null && (
            <span className="text-[10px] font-medium px-1 rounded bg-panel-dark/50 border border-line">
              {t.r_multiple.toFixed(1)} R
            </span>
          )}
        </div>

        {/* Pattern/Reason if present */}
        {t.pattern && (
          <div className="text-[10px] text-ink-muted truncate font-mono bg-panel-dark/30 p-1.5 rounded border border-line/30">
            {t.pattern}
          </div>
        )}

        {/* Navigation buttons */}
        <div className="flex items-center gap-1.5 mt-1 border-t border-line/25 pt-1.5">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate(t.open_time);
            }}
            className="flex-1 py-1 px-2 rounded bg-panel border border-line text-[10px] text-ink hover:text-accent hover:border-accent transition-colors flex items-center justify-center gap-1 cursor-pointer"
          >
            <MapPin size={10} /> Entry
          </button>
          {!isOpen && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onNavigate(t.close_time);
              }}
              className="flex-1 py-1 px-2 rounded bg-panel border border-line text-[10px] text-ink hover:text-accent hover:border-accent transition-colors flex items-center justify-center gap-1 cursor-pointer"
            >
              <MapPin size={10} /> Exit
            </button>
          )}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onWhy();
            }}
            title="Why did the bot take this trade?"
            className="flex-1 py-1 px-2 rounded bg-panel border border-line text-[10px] text-ink hover:text-accent hover:border-accent transition-colors flex items-center justify-center gap-1 cursor-pointer"
          >
            <HelpCircle size={10} /> Why
          </button>
        </div>
    </div>
  );
}

/** Virtualized trade list — backs both the flat (non-replay) Trades tab and
 * the replay "History" section, which are the same `historyTrades` shape
 * (closed trades, `isOpen` always false) rendered in two different spots in
 * the tree. A backtest report can carry thousands of trades; without
 * windowing, scrubbing deep into replay renders every closed `TradeCard` on
 * every cursor tick and freezes the tab. `TradeCard`'s rendered height
 * varies (conditional pattern row, R-multiple chip, Exit button), so this
 * uses TanStack Virtual's dynamic-size pattern (`measureElement` +
 * `data-index`) rather than a fixed `estimateSize` that would cause
 * overlap/gaps. */
function VirtualTradeList({
  trades,
  selectedTradeIndex,
  onSelectTrade,
  onNavigateTrade,
  onWhy,
}: {
  trades: (BacktestTrade & { originalIndex: number })[];
  selectedTradeIndex: number | null;
  onSelectTrade?: (index: number) => void;
  onNavigateTrade?: (index: number, time: number) => void;
  onWhy: (trade: BacktestTrade & { originalIndex: number }) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: trades.length,
    getScrollElement: () => scrollRef.current,
    // Rough average TradeCard height; corrected per-row by `measureElement`
    // below once each row actually mounts and reports its real size.
    estimateSize: () => 132,
    overscan: 8,
    getItemKey: (index) => `${trades[index].open_time}-${trades[index].originalIndex}`,
  });

  return (
    <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto">
      <div
        className="relative w-full divide-y divide-line"
        style={{ height: virtualizer.getTotalSize() }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const t = trades[virtualRow.index];
          return (
            <div
              key={virtualRow.key}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              className="absolute top-0 left-0 w-full"
              style={{ transform: `translateY(${virtualRow.start}px)` }}
            >
              <TradeCard
                trade={t}
                isOpen={false}
                selected={selectedTradeIndex === t.originalIndex}
                onSelect={() => onSelectTrade?.(t.originalIndex)}
                onNavigate={(time) => onNavigateTrade?.(t.originalIndex, time)}
                onWhy={() => onWhy(t)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Virtualized signal list backing the Signals tab — same rationale as
 * `VirtualTradeList` above (a report can carry thousands of signals) and
 * the same dynamic-size measurement pattern, since a rejected signal's
 * reason text wraps to a variable number of lines (`whitespace-pre-wrap`)
 * while an opened one truncates to a single line. */
function VirtualSignalList({
  signals,
  selectedSignalIndex,
  onSelectSignal,
}: {
  signals: (BacktestSignal & { originalIndex: number })[];
  selectedSignalIndex: number | null;
  onSelectSignal?: (index: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: signals.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 56,
    overscan: 10,
    getItemKey: (index) => `${signals[index].time}-${signals[index].originalIndex}`,
  });

  return (
    <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto">
      <div
        className="relative w-full divide-y divide-line"
        style={{ height: virtualizer.getTotalSize() }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const s = signals[virtualRow.index];
          // Anything that didn't become a trade and wasn't merely
          // unresolved (`skipped`) was actively vetoed/rejected — matches
          // BotSelector's "Rej" chip count logic.
          const isRejected = s.outcome !== "opened" && s.outcome !== "skipped";
          return (
            <div
              key={virtualRow.key}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              className="absolute top-0 left-0 w-full"
              style={{ transform: `translateY(${virtualRow.start}px)` }}
            >
              <button
                type="button"
                onClick={() => onSelectSignal?.(s.originalIndex)}
                title="Highlight this signal on the chart"
                className={`w-full text-left p-2.5 hover:bg-accent/5 transition-colors flex flex-col gap-1 cursor-pointer ${
                  selectedSignalIndex === s.originalIndex ? "bg-accent/10 border-l-2 border-accent" : ""
                }`}
              >
                <div className="flex items-center gap-1.5 text-[10px] text-ink-muted">
                  <Clock size={10} />
                  <span>{formatTime(s.time)}</span>
                  <span
                    className={`ml-auto font-mono text-[9px] uppercase tracking-wider px-1 rounded border ${
                      isRejected
                        ? "bg-err/10 text-err border-err/30"
                        : "bg-panel-dark/50 border-line"
                    }`}
                  >
                    {SIGNAL_OUTCOME_META[s.outcome]?.label || s.outcome}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`text-[10px] font-bold px-1 rounded ${
                    s.direction === "buy"
                      ? "bg-ok/10 text-ok border border-ok/20"
                      : "bg-err/10 text-err border border-err/20"
                  }`}>
                    {s.direction.toUpperCase()}
                  </span>
                  <span
                    className={`text-xs font-medium text-ink flex-1 min-w-0 ${
                      isRejected ? "whitespace-pre-wrap break-words" : "truncate"
                    }`}
                    title={s.reason}
                  >
                    {s.reason}
                  </span>
                </div>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  // Returns "MM-DD HH:MM" format
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
