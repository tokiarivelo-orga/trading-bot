"use client";

/**
 * Backtest-view / live-bot "eye" view data: a backtest report's trades,
 * activity log and signals (or, for the live-bot overlay, the same slots
 * fed from `getLiveBotSignals`/`getTradeMarkers` — see the poll effect
 * below), plus the SignalsDock selection indices into `backtestTrades`/
 * `backtestSignals`, plus (phase 10) the effect that paints those trades/
 * signals onto the chart as series markers and zone/SL-TP/exit-line
 * drawings.
 *
 * The report fetch itself (`getBacktestReport`) is NOT here — it lives
 * inside ChartPanel.tsx's giant symbol/timeframe/report history-loading
 * effect, which anchors the initial candle window to the report's last
 * trade/signal and needs `candlesRef`/`chartRef`/replay state to do it; that
 * effect calls this hook's `setBacktestTrades`/`setBacktestMeta`/
 * `setBacktestActivityLog`/`setBacktestSignals`/`setBacktestError` rather
 * than owning separate state.
 *
 * The marker-application effect below moved here from ChartPanel.tsx in
 * phase 10, once `chartController` (the paint target it needs) existed.
 * It takes `chartController`/`candlesRef`/`replayActive`/`replayCursorIndex`/
 * `customCodeResult`/`orderLineStyle`/`showTradeLabels` as inputs rather than
 * owning them, since they're either genuinely owned elsewhere in
 * ChartPanel.tsx (replay cursor state, the custom-code drawer, order-line
 * style) or produced by a hook (`useChartEngine`) this hook has no reason to
 * depend on directly. `lastRevealedSignatureRef` (tracks whether the
 * revealed trade set actually changed since the last run, so a no-op replay
 * tick skips the drawings rebuild) is the one piece of state this effect
 * needs that has no other owner — it's created fresh here, since this
 * effect is now its only consumer.
 */

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { type UTCTimestamp } from "lightweight-charts";
import { useActiveAccount } from "@/shared/api/account-context";
import {
  getLiveBotSignals,
  getTradeMarkers,
  type ActivityLogEntry,
  type BacktestSignal,
  type BacktestTrade,
  type Candle,
  type EvaluateCustomCodeResponse,
  type TradeMarker,
} from "@/shared/api/client";
import { cssVar, pickZoneColor } from "./chartFormat";
import {
  BACKTEST_DRAWING_PREFIX,
  LIVE_TRADE_DRAWING_PREFIX,
} from "./chartStorage";
import {
  buildExitLineDrawing,
  buildTradeSetupDrawings,
  toBacktestSeriesMarkers,
  toCustomSignalsSeriesMarkers,
  toSignalSeriesMarkers,
} from "./chartMarkers";
import { subscribeSharedPoll } from "./sharedPoll";
import type { ChartEngineController, OrderLineStyle, ZoneColorStyle } from "./types";

// Matches ChartPanel's own MARKERS_POLL_MS — the live-bot eye view's
// signals/trades poll runs on the same cadence as the journal marker poll.
const MARKERS_POLL_MS = 5000;

export interface UseBacktestDataParams {
  backtestReportId: string | null;
  liveBotSkill: string | null;
  symbol: string;
  /** Chart engine handle (from `useChartEngine`) — the marker-application
   * effect below paints `backtestTrades`/`backtestSignals` through it. */
  chartController: ChartEngineController;
  /** All candles currently loaded for this symbol/timeframe, oldest first —
   * read (not written) here to resolve the replay cursor bar's time for the
   * "no lookahead while replaying" reveal gate. */
  candlesRef: RefObject<Candle[]>;
  replayActive: boolean;
  replayCursorIndex: number;
  /** "Run Custom Code" drawer's last evaluation result (owned by
   * `useStrategyEditor`/ChartPanel) — while set, its signals take over the
   * marker layer in place of the report's own trades/signals. */
  customCodeResult: EvaluateCustomCodeResponse | null;
  orderLineStyle: OrderLineStyle;
  zoneColorStyle: ZoneColorStyle;
  activeHighlightedTicket: string | number | null;
}

export function useBacktestData({
  backtestReportId,
  liveBotSkill,
  symbol,
  chartController,
  candlesRef,
  replayActive,
  replayCursorIndex,
  customCodeResult,
  orderLineStyle,
  zoneColorStyle,
  activeHighlightedTicket,
}: UseBacktestDataParams) {
  const accountId = useActiveAccount();

  // The replay cursor bar's own time — same value the marker-drawing effect
  // below computes locally as `cursorTime` (falling back to `Infinity` there
  // to mean "reveal everything"), but exposed as `null` here so SignalsDock
  // can tell "not replaying" apart from "cursor at time 0" and fall back to
  // its own flat trade list instead of an Active/History split.
  const replayCursorTime = replayActive
    ? ((candlesRef.current[replayCursorIndex]?.time as number | undefined) ?? null)
    : null;

  // Backtest-view state (§F): the report's trades, converted to markers once
  // fetched, and an error flag for the "View on Chart" banner.
  const [backtestTrades, setBacktestTrades] = useState<BacktestTrade[] | null>(
    null,
  );
  const [backtestError, setBacktestError] = useState<string | null>(null);
  // Strategy/symbol/period behind the current backtest report — fed to the
  // inline strategy editor so it can be tested/tweaked right on the chart.
  const [backtestMeta, setBacktestMeta] = useState<{
    strategy: string;
    symbol: string;
    period: string;
  } | null>(null);
  // Backtest report's own persisted activity log (signals/vetoes/fills for
  // this exact run, with simulated-clock timestamps) — distinct from
  // ActivityLogDock's default live/global poll, used to drive replay.
  const [backtestActivityLog, setBacktestActivityLog] = useState<
    ActivityLogEntry[] | null
  >(null);
  // Every signal the report's strategy emitted (taken AND vetoed) — drawn
  // as chart markers and listed in the SignalsDock for click-to-navigate.
  const [backtestSignals, setBacktestSignals] = useState<
    BacktestSignal[] | null
  >(null);

  // Index into `backtestTrades` of the trade selected from SignalsDock's
  // Trades tab (backtest report or a live bot's "eye" overlay — both feed
  // the same `backtestTrades` state) — draws that trade's entry/SL/TP/close
  // as highlighted dashed lines, same treatment as a selected live position.
  const [selectedTradeIndex, setSelectedTradeIndex] = useState<number | null>(null);
  // A new report/bot or symbol invalidates the index into the (now
  // different) trades array — stale selection would either point at the
  // wrong trade or highlight nothing.
  useEffect(() => {
    setSelectedTradeIndex(null);
  }, [backtestReportId, liveBotSkill, symbol]);

  // Same idea as `selectedTradeIndex`, for SignalsDock's Signals tab — a
  // signal has no lasting price level (unlike a trade's entry/SL/TP), so it
  // renders as a full-height vertical dashed line at that bar's time instead
  // of a horizontal one.
  const [selectedSignalIndex, setSelectedSignalIndex] = useState<number | null>(null);
  useEffect(() => {
    setSelectedSignalIndex(null);
  }, [backtestReportId, liveBotSkill, symbol]);

  // Live-bot "eye" view: fetch one bot's own signal trail + own trades and
  // feed them into the *same* backtestSignals/backtestTrades state backtest
  // view populates (see ChartPanel's history-loading effect) — the
  // marker-merge effect and SignalsDock already render whatever's in those
  // two slots, so nothing there needs to know or care which source filled
  // them. `TradeMarker` (live) doesn't carry `r_multiple`/`zone`/`pattern`/
  // `structure` the way a backtest's `BacktestTrade` does; SignalsDock only
  // reads those behind null checks, so filling them with `null`/`[]` renders
  // correctly with no changes to SignalsDock itself. Only closed trades go
  // into the dock/marker list — an open position has no profit to show yet
  // (it still appears as a live position via the broker/orders UI).
  useEffect(() => {
    if (!liveBotSkill || !accountId) return;
    // Shared across every window with the same bot's eye on (multi-chart
    // layout §) — the fetch is identical (account+symbol+skill), so N
    // windows don't each poll the journal/activity-log independently.
    const unsubscribe = subscribeSharedPoll(
      `live-bot-markers:${accountId}:${symbol}:${liveBotSkill}`,
      MARKERS_POLL_MS,
      () =>
        Promise.all([
          getLiveBotSignals(accountId, liveBotSkill),
          getTradeMarkers(accountId, symbol, liveBotSkill),
        ]),
      (result) => {
        if (!result) return; // Activity log / journal unreachable — leave whatever's already shown.
        const [signals, markers] = result;
        setBacktestSignals(signals);
        setBacktestTrades(
          markers
            .filter(
              (m): m is TradeMarker & { close_time: number; close_price: number } =>
                m.close_time !== null && m.close_price !== null,
            )
            .map((m) => ({
              side: m.side,
              volume: m.volume,
              open_time: m.open_time,
              open_price: m.open_price,
              sl: m.sl,
              tp: m.tp,
              close_time: m.close_time,
              close_price: m.close_price,
              profit: m.profit ?? 0,
              r_multiple: null,
              zone: null,
              pattern: null,
              structure: [],
              reason: "",
              confidence: null,
            })),
        );
      },
    );
    return unsubscribe;
  }, [accountId, symbol, liveBotSkill]);

  // Clears the live-bot overlay's data when the eye turns off, so a stale
  // bot's signals/trades don't linger after switching away.
  useEffect(() => {
    if (liveBotSkill || backtestReportId) return;
    setBacktestSignals(null);
    setBacktestTrades(null);
  }, [liveBotSkill, backtestReportId]);

  // Tracks the last "open:close count + exit-line-style" signature the
  // marker-application effect below actually rebuilt drawings for — a
  // no-op replay cursor tick (one that doesn't cross any trade's reveal
  // threshold) shouldn't pay for a full drawings rebuild. Only that effect
  // reads/writes it.
  const lastRevealedSignatureRef = useRef<string | null>(null);

  // Render the backtest report's (or eyed live bot's) trades as markers once
  // fetched, plus each trade's zone rectangle and SL/TP segments
  // (BACKTEST_DRAWING_PREFIX) — cleared and rebuilt here on every report
  // change rather than reusing recomputeIndicators's cadence, since this
  // only needs to run when the report's trades actually change, not on
  // every live candle tick.
  useEffect(() => {
    const manager = chartController.getDrawingManager();
    const zoneMeta = chartController.getZoneMetaMap();
    function clearBacktestDrawings() {
      if (!manager) return;
      for (const drawing of manager.getAllDrawings()) {
        if (
          drawing.id.startsWith(BACKTEST_DRAWING_PREFIX) ||
          // Also clear any live-view closed-trade lines left over from
          // before switching into the backtest report.
          drawing.id.startsWith(LIVE_TRADE_DRAWING_PREFIX)
        ) {
          manager.removeDrawing(drawing.id);
          zoneMeta.delete(drawing.id);
        }
      }
    }
    if (!(backtestReportId || liveBotSkill) || backtestTrades === null) {
      clearBacktestDrawings();
      lastRevealedSignatureRef.current = null;
      return;
    }
    const colors = { ok: cssVar('--color-ok'), err: cssVar('--color-err') };
    // While replaying, only reveal what would have been visible at the
    // cursor bar's close — entry/setup at `open_time`, exit at `close_time`
    // — same "no lookahead" contract as ChartPanel's `visibleCandles()`.
    // Off (the default), `cursorTime = Infinity` shows everything, unchanged
    // from before replay existed.
    const cursorTime = replayActive
      ? ((candlesRef.current[replayCursorIndex]?.time as number | undefined) ??
        0)
      : Infinity;
    if (customCodeResult) {
      chartController.getSeriesMarkersPrimitive()?.setMarkers(
        toCustomSignalsSeriesMarkers(
          customCodeResult.signals,
          colors,
          true,
        ).filter((m) => (m.time as number) <= cursorTime)
      );
      clearBacktestDrawings();
      lastRevealedSignatureRef.current = null;
      return;
    }
    chartController.getSeriesMarkersPrimitive()?.setMarkers(
      [
        ...toBacktestSeriesMarkers(backtestTrades, colors, true),
        // Vetoed/rejected signals as square markers — every valid setup the
        // strategy saw, not only the fills (opened signals ARE the trade
        // arrows above, so they're excluded from this builder).
        ...toSignalSeriesMarkers(backtestSignals ?? [], true),
      ]
        .sort((a, b) => (a.time as number) - (b.time as number))
        .filter((m) => (m.time as number) <= cursorTime),
    );
    if (manager) {
      // Rebuilding every trade's zone/SL/TP/exit-line drawings is O(trades)
      // — cheap once, but this effect reruns on every replay cursor tick,
      // and most single-bar advances don't cross any trade's reveal
      // threshold. Skip the rebuild entirely when the revealed set hasn't
      // actually changed since the last run (tracked as an "open:close
      // count" signature) — a no-op tick shouldn't pay for a full rebuild.
      const openCount = backtestTrades.reduce(
        (n, t) => (t.open_time <= cursorTime ? n + 1 : n),
        0,
      );
      const closeCount = backtestTrades.reduce(
        (n, t) => (t.close_time <= cursorTime ? n + 1 : n),
        0,
      );
      const signature = `${openCount}:${closeCount}:${orderLineStyle.showExitLine}:${orderLineStyle.exitLineDash}:${orderLineStyle.exitLineWidth}:${orderLineStyle.exitLineCustomColor}:${orderLineStyle.exitLineWinColor}:${orderLineStyle.exitLineLossColor}:${zoneColorStyle.customColors}:${zoneColorStyle.tradeZone.demandColor}:${zoneColorStyle.tradeZone.supplyColor}`;
      if (signature !== lastRevealedSignatureRef.current) {
        lastRevealedSignatureRef.current = signature;
        clearBacktestDrawings();
        const zoneColors = {
          demand: pickZoneColor(zoneColorStyle.tradeZone, true, false, zoneColorStyle.customColors),
          supply: pickZoneColor(zoneColorStyle.tradeZone, false, false, zoneColorStyle.customColors),
          sl: cssVar('--color-err'),
          tp: cssVar('--color-ok'),
        };
        backtestTrades.forEach((t, i) => {
          if (t.open_time > cursorTime) return;
          for (const drawing of buildTradeSetupDrawings(t, i, zoneColors)) {
            manager.addDrawing(drawing);
          }
          if (t.zone) {
            zoneMeta.set(`${BACKTEST_DRAWING_PREFIX}zone:${i}`, {
              indicator: 'trade_zone',
              indicatorLabel: 'Trade signal zone',
              pattern: t.zone.pattern,
              kind: t.zone.kind,
              priceLow: t.zone.price_low,
              priceHigh: t.zone.price_high,
              timeStart: t.zone.time_start,
              timeEnd: t.zone.time_end,
              state: 'triggered',
              extra: {
                ...(t.pattern ? { 'Confirming pattern': t.pattern } : {}),
                ...(t.reason ? { Reason: t.reason } : {}),
              },
            });
          }
          if (t.close_time <= cursorTime) {
            const exitDrawing = buildExitLineDrawing(
              BACKTEST_DRAWING_PREFIX,
              String(i),
              t.open_time as UTCTimestamp,
              t.open_price,
              t.close_time as UTCTimestamp,
              t.close_price,
              t.profit,
              { ok: zoneColors.tp, err: zoneColors.sl },
              orderLineStyle,
            );
            if (exitDrawing) manager.addDrawing(exitDrawing);
          }
        });
      }
    }
    // chartController/candlesRef are stable ref/handle objects across a
    // render, same as every other effect in this codebase that reads them —
    // omitted deliberately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    backtestReportId,
    liveBotSkill,
    backtestTrades,
    backtestSignals,
    customCodeResult,
    replayActive,
    replayCursorIndex,
    orderLineStyle,
    zoneColorStyle,
  ]);

  // The report's own time bounds — earliest trade open, latest trade close
  // or signal, whichever is later (mirrors the initial-candle-window anchor
  // logic in useCandleData.ts's resolveInitialCandles) — so a split-window
  // secondary pane can fetch the same period at its own timeframe and clip
  // it to the shared replay cursor, the same way session replay's explicit
  // from/to already lets it. Null while the report/trades haven't loaded yet.
  const backtestPeriod = useMemo(() => {
    if (!backtestTrades || backtestTrades.length === 0) return null;
    const times = [
      ...backtestTrades.flatMap((t) => [t.open_time, t.close_time]),
      ...(backtestSignals ?? []).map((s) => s.time),
    ];
    return { from: Math.min(...times), to: Math.max(...times) };
  }, [backtestTrades, backtestSignals]);

  return {
    backtestTrades,
    setBacktestTrades,
    backtestError,
    setBacktestError,
    backtestMeta,
    setBacktestMeta,
    backtestActivityLog,
    setBacktestActivityLog,
    backtestSignals,
    setBacktestSignals,
    selectedTradeIndex,
    setSelectedTradeIndex,
    selectedSignalIndex,
    setSelectedSignalIndex,
    replayCursorTime,
    backtestPeriod,
  };
}

export type BacktestData = ReturnType<typeof useBacktestData>;
