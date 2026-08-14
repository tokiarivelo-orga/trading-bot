'use client';

/**
 * Chart feature (Phase 2+3): lightweight-charts candlesticks + volume, live WS
 * updates, timeframe switcher, spread indicator, and trade markers (F7) from
 * the journal — entry arrows + exit circles, refreshed alongside the spread.
 * Drawing tools (F-draw): lightweight-charts-drawing DrawingManager attached to
 * the candleSeries — toolbar in DrawingToolbar.tsx, persistence in localStorage.
 */

import {
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { type IDrawing } from 'lightweight-charts-drawing';
import {
  ArrowDown,
  Play,
  Square,
  X,
} from 'lucide-react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  getNewsEvents,
  getTradeMarkers,
  type Candle,
  type NewsEventRecord,
  type TradeMarker,
  type StrategyVersionSummary,
  evaluateCustomCode,
} from '@/shared/api/client';
import { useActiveAccount } from '@/shared/api/account-context';
import type { Trading } from '@/features/trading/useTrading';
import { SIGNAL_OUTCOME_META } from '@/features/backtest/signalOutcome';
import { ActivityLogDock } from './ActivityLogDock';
import { ChartContextMenu } from './ChartContextMenu';
import { ChartOrderPopover } from './ChartOrderPopover';
import { ChartToolbar, type ChartToolbarProps } from './ChartToolbar';
import { DrawingContextMenu } from './DrawingContextMenu';
import { DrawingEditPopover } from './DrawingEditPopover';
import { DrawingToolbar } from './DrawingToolbar';
import { DrawingsList } from './DrawingsList';
import { IndicatorsDock } from './IndicatorsDock';
import { PositionEditPopover } from './PositionEditPopover';
import { ReplayControls } from './ReplayControls';
import { ReplayRevealOverlay } from './ReplayRevealOverlay';
import { ZoneInfoPopover } from './ZoneInfoPopover';
import { SignalInfoPopover } from './SignalInfoPopover';
import type { SignalTooltipState, ZoneTooltipState } from './types';
import { SessionReplayPicker } from './SessionReplayPicker';
import { SignalsDock } from './SignalsDock';
import { useBacktestData } from './useBacktestData';
import { FINER_TIMEFRAME, fetchCandlesForPeriod, TIMEFRAME_SECONDS, useCandleData } from './useCandleData';
import { fetchShared } from './sharedFetchCache';
import { useChartEngine } from './useChartEngine';
import { useChartUIToggles } from './useChartUIToggles';
import { useVolatilityGuard } from '@/features/settings/useVolatilityGuard';
import { useDrawingTools } from './useDrawingTools';
import { useIndicators } from './useIndicators';
import { useOrderPopovers } from './useOrderPopovers';
import { useReplayEngine } from './useReplayEngine';
import { useReplayReveal } from './useReplayReveal';
import { useStrategyEditor } from './useStrategyEditor';
import type {
  BotReplayControls,
  DrawingToolType,
  EntryLineSpec,
  OrderLineDash,
  OrderLineStyle,
  PriceLineSpec,
  ReplayUIState,
  SharedReplaySession,
} from './types';
import {
  cssVar,
  defaultOffset,
  derivePeriodParam,
  hexToRgba,
} from './chartFormat';
import {
  LAST_TIMEFRAME_KEY,
  LIVE_TRADE_DRAWING_PREFIX,
  loadLastTimeframe,
  TIMEFRAME_QUERY_KEY,
} from './chartStorage';
import { nearestCandleTime } from './chartData';
import {
  buildLiveTradeLineDrawings,
  toCustomSignalsSeriesMarkers,
  toNewsEventSeriesMarkers,
  toSeriesMarkers,
} from './chartMarkers';
import { subscribeSharedPoll } from './sharedPoll';

const MARKERS_POLL_MS = 5000;

// Lazy-loaded: BacktestStrategyEditor pulls in @uiw/react-codemirror +
// @codemirror/lang-python + @uiw/codemirror-theme-github, and only renders
// when the strategy-editor drawer is toggled open — keep that bundle out of
// the main chart route's initial JS.
const BacktestStrategyEditor = dynamic(
  () =>
    import('@/features/backtest/BacktestStrategyEditor').then(
      (mod) => mod.BacktestStrategyEditor,
    ),
  { ssr: false },
);

// Lazy-loaded for the same reason: CustomCodeDrawer owns its own copy of
// @uiw/react-codemirror + @codemirror/lang-python + @uiw/codemirror-theme-github
// (a second, independent user of those packages from BacktestStrategyEditor
// above) and only renders when the "Run Custom Code" drawer is open.
const CustomCodeDrawer = dynamic(
  () =>
    import('./CustomCodeDrawer').then((mod) => mod.CustomCodeDrawer),
  { ssr: false },
);

export function ChartPanel({
  symbol,
  trading,
  activeStrategy,
  backtestReportId = null,
  onExitBacktestView,
  onReportChange,
  liveBotSkill = null,
  highlightedTicket = null,
  onSelectTicket,
  onReplaySessionChange,
  onReplayCursorTime,
  onReplayUIChange,
  windowIndex = 0,
  windowCount = 1,
  selectedWindowIndex = 0,
  onSelectWindow,
  onCloseWindow,
  initialTimeframe,
  onTimeframeChange,
  sharedReplay = null,
  hideToolbar = false,
  onToolbarStateChange,
}: {
  symbol: string;
  trading: Trading;
  activeStrategy: StrategyVersionSummary | null;
  /** When set, the chart shows this backtest report's trades as markers
   * (§F: "test the bot in chart for candle history") instead of the live
   * journal's — anchored to the historical candle window the report's
   * trades actually happened in, and with live WS updates paused so a
   * present-day candle doesn't get appended after months of history. */
  backtestReportId?: string | null;
  /** Called when the user leaves backtest view (only rendered while
   * `backtestReportId` is set) — the caller owns clearing the id/URL param. */
  onExitBacktestView?: () => void;
  /** Called with a new report id after the inline strategy editor (below)
   * saves an edit and re-runs the backtest — the caller owns swapping
   * `backtestReportId`/the URL to it so the chart picks up the new report's
   * trades without leaving the chart. */
  onReportChange?: (reportId: string) => void;
  /** A bot's full id (from BotSelector's eye icon) whose live signal trail
   * and own positions/profit should overlay the chart, in place of the
   * unscoped "every trade on this symbol" markers — reuses the exact same
   * signals/trades dock and marker rendering as backtest view, just fed
   * from `getLiveBotSignals`/`getTradeMarkers(symbol, skill)` instead of a
   * backtest report. Mutually exclusive with `backtestReportId` (the caller
   * enforces this — see page.tsx's `toggleLiveBotSignals`). */
  liveBotSkill?: string | null;
  /** Ticket selected from the account-wide Active Orders / Positions panel
   * (see page.tsx's `selectedOrderTicket`) — when it belongs to a position or
   * pending order on this symbol, its entry/SL/TP lines render emphasized
   * (thicker, glowing) and every other line dims, mirroring TradingView's
   * "selected position" look. Null (the default) renders every line at its
   * normal, always-on style. */
  highlightedTicket?: string | number | null;
  /** Called when the user clicks an entry/SL/TP/pending line on the chart —
   * notifies parent to highlight all lines for this order/position and sync
   * with the OrdersDock table. */
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
  /** Multi-chart layout (§ split-window): fired whenever replay is entered/
   * exited or a session-replay period starts/ends, so a parent rendering
   * secondary chart windows alongside this one can mirror the same period
   * at their own timeframe. Null `sessionPeriod` with `active: true` means a
   * backtest-report replay (bounded by the report's own candle window, not
   * an explicit from/to) — secondary windows have no report to anchor to,
   * so they simply have nothing to sync in that case. */
  onReplaySessionChange?: (windowIndex: number, session: {
    active: boolean;
    sessionPeriod: { from: number; to: number } | null;
  }) => void;
  /** Fired on every replay-cursor tick with the cursor bar's own candle time
   * (null while not replaying) — the single "current position" secondary
   * windows follow, since a cursor *index* only means something within this
   * window's own timeframe's candle array. */
  onReplayCursorTime?: (windowIndex: number, time: number | null) => void;
  windowIndex?: number;
  windowCount?: number;
  selectedWindowIndex?: number;
  onSelectWindow?: (index: number) => void;
  onCloseWindow?: (index: number) => void;
  initialTimeframe?: Candle['timeframe'];
  onTimeframeChange?: (tf: Candle['timeframe']) => void;
  sharedReplay?: SharedReplaySession | null;
  hideToolbar?: boolean;
  onToolbarStateChange?: (windowIndex: number, props: ChartToolbarProps) => void;
  onReplayUIChange?: (windowIndex: number, ui: ReplayUIState | null) => void;
}) {
  // All candles currently on the chart for this symbol/timeframe, oldest
  // first — kept in sync with live updates so "load more" always pages back
  // from the true oldest bar, and mutated in place (no React re-render).
  // historyLoadedRef/hasMoreHistoryRef/loadingMoreRef used to live here too,
  // but (unlike this ref — see useCandleData.ts's module doc for why it's
  // still created here) they're read/written exclusively inside the fetch
  // effect, so they moved into useCandleData.ts wholesale.
  const candlesRef = useRef<Candle[]>([]);
  // Multi-timeframe replay synthesis: when this window is a *coarser* follower
  // (e.g. M5) than the master driving the replay (e.g. M1), it holds the
  // master's finer candles for the whole session period so `visibleCandles()`
  // can aggregate the ones that have landed inside the current not-yet-closed
  // bucket into a synthetic "forming" bar that grows tick-by-tick like a live
  // chart. Empty when this window isn't coarser than the master (or no replay
  // is active) — the naive cursor cutoff already handles those cases. Loaded
  // by the fetch effect further down.
  const masterCandlesRef = useRef<Candle[]>([]);
  // Tick-form replay: the next-finer timeframe's candles for the *local*
  // (single-window) replay period, so `visibleCandles()` can grow the current
  // bar into a live-like "forming" bar from the finer bars that have landed
  // inside it — the same synthesis `masterCandlesRef` drives for a coarser
  // multi-chart follower, here for this window's own replay. Empty when
  // tick-form is off, no replay is active, the timeframe has nothing finer
  // (M1), or the finer history isn't available for the period — all of which
  // fall back to the whole-bar reveal. Loaded by the fetch effect further down.
  const finerCandlesRef = useRef<Candle[]>([]);
  // useBacktestData's call moves further down (right after useChartEngine),
  // since phase 10 gave it a `chartController` param its marker-application
  // effect needs to paint through — see the call site below.
  const [showSignalsDock, setShowSignalsDock] = useState(false);
  const {
    showTfDropdown,
    setShowTfDropdown,
    showOverlaysDropdown,
    setShowOverlaysDropdown,
    tfDropdownRef,
    overlaysDropdownRef,
    showSeparators,
    showSeparatorsRef,
    toggleSeparators,
    showSpreadLine,
    toggleSpreadLine,
    showVolume,
    toggleVolume,
    showTradeLabels,
    toggleTradeLabels,
    showTradeMarkers,
    toggleTradeMarkers,
    orderLineStyle,
    updateOrderLineStyle,
    showOrderLineSettings,
    setShowOrderLineSettings,
    zoneColorStyle,
    updateZoneColorStyle,
    showZoneColorSettings,
    setShowZoneColorSettings,
    showDrawingToolbar,
    toggleDrawingToolbar,
  } = useChartUIToggles();
  // Shared with `features/settings/VolatilityGuardPanel.tsx` via the same
  // TanStack Query cache entry — toggling here or there stays in sync with
  // no prop drilling beyond this panel.
  const volatilityGuard = useVolatilityGuard();
  // Replay ("live session player", §F): progressively reveals the backtest
  // report's candles/indicators/trades/log up to a moving cursor instead of
  // drawing everything at once — see `visibleCandles()` below. `replayActive`
  // toggles the mode on/off (off = today's static full-report view,
  // unchanged); `replayActiveRef`/`replayCursorIndexRef` are the imperative
  // source of truth read by `visibleCandles()`/`render()`/`recomputeIndicators`
  // (closures created once per data-load, not re-created on every cursor
  // tick), while the `useState` pair only drives the player UI. Created here
  // rather than inside `useReplayEngine` (which owns the rest of this
  // concern) because `useChartEngine`/`useCandleData` — both called before
  // `useReplayEngine` can exist — need direct read/write access; see
  // useReplayEngine.ts's module doc for the full explanation.
  const [replayActive, setReplayActive] = useState(false);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replayCursorIndex, setReplayCursorIndex] = useState(0);
  const replayActiveRef = useRef(false);
  const replayCursorIndexRef = useRef(0);
  // Whether the view auto-centers on the cursor bar as it advances — turned
  // off by a manual drag/zoom (see useReplayEngine's mousedown/wheel
  // listener), back on via ReplayControls' "Center" button. `followingCursor`
  // mirrors it into state purely so the UI can show whether it's engaged.
  const followCursorRef = useRef(true);
  const [followingCursor, setFollowingCursor] = useState(true);
  // Tick-form replay toggle: when on (the default), the autoplay loop advances
  // a finer sub-cursor *within* each bar so the bar visibly forms — wick
  // growing, body flipping green/red — from real finer-timeframe candles
  // (`finerCandlesRef`), instead of popping in fully closed. `tickFormRef` is
  // the imperative copy `visibleCandles()`/`render()` (closures created once
  // per data-load) read; `tickForm` mirrors it for the ReplayControls toggle.
  // Auto-forced off (and disabled in the UI) when the current timeframe has no
  // finer level or the finer history isn't available — see `finerAvailable`.
  const tickFormRef = useRef(true);
  const [tickForm, setTickForm] = useState(true);
  const [finerAvailable, setFinerAvailable] = useState(false);
  // The finer sub-cursor: the time up to which the current forming bar has
  // been revealed from `finerCandlesRef`. Null outside animation (manual
  // step/scrub, tick-form off, no finer data) — `visibleCandles()` then shows
  // the current bar fully closed. Written by the autoplay loop in
  // useReplayEngine; read by `visibleCandles()` below.
  const replayFineTimeRef = useRef<number | null>(null);
  // The single gate every candle/indicator render path reads through: the
  // full loaded window normally, or a prefix up to the replay cursor while
  // replaying — so candles, EMA/SMA/RSI/MACD/Bollinger, and every manual
  // indicator (VWAP/ATR/structure/QML/patterns) all become cursor-gated for
  // free by switching their one `candlesRef.current` read to this call.
  function visibleCandles(): Candle[] {
    if (sharedReplay?.active && sharedReplay.cursorTime != null) {
      const all = candlesRef.current;
      const cursorTime = sharedReplay.cursorTime;
      // Coarser-follower synthesis: if this window's timeframe is strictly
      // coarser than the master driving the replay (e.g. M5 follower, M1
      // master) and the master's finer candles are loaded, reveal a partial
      // "forming" bar for the current bucket that grows as each finer bar
      // lands — instead of only jumping to a fully-closed bar every N ticks.
      const master = masterCandlesRef.current;
      if (
        sharedReplay.masterTimeframe &&
        TIMEFRAME_SECONDS[timeframe] > TIMEFRAME_SECONDS[sharedReplay.masterTimeframe] &&
        master.length > 0
      ) {
        // Derive the current bucket from this window's own last real candle
        // whose time <= cursor (scan backward) rather than flooring the cursor
        // arithmetically — H4/D1 depend on broker server-time offsets and W1
        // uses a server-side epoch shift, so a naive floor can pick the wrong
        // boundary. The follower's own candle times are already correct.
        let bucketIdx = -1;
        for (let i = all.length - 1; i >= 0; i--) {
          if ((all[i].time as number) <= cursorTime) {
            bucketIdx = i;
            break;
          }
        }
        if (bucketIdx !== -1) {
          const bucketOpen = all[bucketIdx].time as number;
          const closed = all.slice(0, bucketIdx);
          const fine = master.filter(
            (c) => (c.time as number) >= bucketOpen && (c.time as number) <= cursorTime,
          );
          if (fine.length > 0) {
            const synthetic: Candle = {
              symbol,
              timeframe,
              time: bucketOpen,
              open: fine[0].open,
              high: Math.max(...fine.map((c) => c.high)),
              low: Math.min(...fine.map((c) => c.low)),
              close: fine[fine.length - 1].close,
              tick_volume: fine.reduce((sum, c) => sum + c.tick_volume, 0),
              spread_points: fine[fine.length - 1].spread_points,
            };
            return [...closed, synthetic];
          }
        }
      }
      let idx = all.findIndex((c) => (c.time as number) > cursorTime);
      if (idx === -1) idx = all.length;
      return all.slice(0, idx);
    }
    if (!replayActiveRef.current) return candlesRef.current;
    const all = candlesRef.current;
    const idx = replayCursorIndexRef.current;
    const current = all[idx];
    const fineTime = replayFineTimeRef.current;
    const fine = finerCandlesRef.current;
    // Tick-form: grow the current bar into a live-like "forming" bar from the
    // finer candles that have landed inside it (up to the sub-cursor), so it
    // builds tick-by-tick — wick extending, body flipping — instead of popping
    // in fully closed. Same aggregation the coarser-follower branch above does
    // from `masterCandlesRef`, here from this window's own `finerCandlesRef`.
    // Falls back to the whole-bar reveal when tick-form is off, the sub-cursor
    // is idle (manual step/scrub), there's no finer data, or the current bar
    // hasn't loaded yet.
    if (tickFormRef.current && current && fineTime != null && fine.length > 0) {
      const bucketOpen = current.time as number;
      const inBucket = fine.filter(
        (c) => (c.time as number) >= bucketOpen && (c.time as number) <= fineTime,
      );
      if (inBucket.length > 0) {
        const synthetic: Candle = {
          symbol,
          timeframe,
          time: bucketOpen,
          open: inBucket[0].open,
          high: Math.max(...inBucket.map((c) => c.high)),
          low: Math.min(...inBucket.map((c) => c.low)),
          close: inBucket[inBucket.length - 1].close,
          tick_volume: inBucket.reduce((sum, c) => sum + c.tick_volume, 0),
          spread_points: inBucket[inBucket.length - 1].spread_points,
        };
        return [...all.slice(0, idx), synthetic];
      }
    }
    return all.slice(0, idx + 1);
  }
  // `visibleCandles` is re-created every render (it closes over props like
  // `sharedReplay.cursorTime`), but useChartEngine/useCandleData/useIndicators
  // capture whatever reference they got the last time *their own* effect ran —
  // effects whose deps never change on a replay cursor tick. So they'd keep
  // calling a frozen-in-time closure and the chart would paint once, then stop
  // following the cursor. Same "ref assigned during render, closures read
  // through it" idiom as `timeframeRef`/`symbolRef`/`onToolbarStateChangeRef`:
  // the wrapper's identity never changes (no consumer dep array needs it), but
  // calling it always runs the current render's logic.
  const visibleCandlesRef = useRef(visibleCandles);
  visibleCandlesRef.current = visibleCandles;
  const visibleCandlesStable = useCallback(() => visibleCandlesRef.current(), []);

  // Session replay: an arbitrary historical period, picked ad hoc (not tied
  // to a saved backtest report), replayed bar-by-bar like a live session.
  // `sessionReplayPeriod` drives the history-loading effect below the same
  // way `backtestReportId` does — pausing live WS and anchoring the initial
  // candle load to the picked window instead of "now" — and is cleared to
  // return to the live view. Stays here (rather than inside
  // `useReplayEngine`) for the same ordering reason as the replay-cursor
  // state above — useCandleData needs it directly. The picker's own input
  // state (from/to strings, visibility) has no such constraint and is owned
  // by useReplayEngine instead.
  const [sessionReplayPeriod, setSessionReplayPeriod] = useState<{
    from: number;
    to: number;
  } | null>(null);
  // Progress while `fetchCandlesForPeriod`'s loop is still paging — null once
  // the fetch settles (success or failure).
  const [sessionReplayLoadingPage, setSessionReplayLoadingPage] = useState<{
    page: number;
    loaded: number;
  } | null>(null);

  // Multi-chart layout: mirror this window's replay cursor position outward
  // so secondary chart windows (different timeframe, same symbol) can follow
  // along — see this component's onReplayCursorTime doc comment. The sibling
  // "mirror the replay session" effect lives further down, after
  // useBacktestData — it needs that hook's `backtestPeriod` to sync a
  // backtest-report replay too, not just session replay.
  useEffect(() => {
    if (!sharedReplay?.active) {
      if (!replayActive) {
        onReplayCursorTime?.(windowIndex, null);
        return;
      }
      onReplayCursorTime?.(windowIndex, (candlesRef.current[replayCursorIndex]?.time as number) ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayActive, replayCursorIndex, sharedReplay?.active, windowIndex]);

  const {
    showStrategyEditor,
    setShowStrategyEditor,
    drawerPosition,
    setDrawerPosition,
    strategyInfoExpanded,
    setStrategyInfoExpanded,
    showCustomCodeEditor,
    setShowCustomCodeEditor,
    customCodeDraft,
    setCustomCodeDraft,
    customCodeResult,
    setCustomCodeResult,
    customCodeResultRef,
    customCodeBusy,
    setCustomCodeBusy,
    customCodeError,
    setCustomCodeError,
    customCodeCopied,
    handleCopyCustomCode,
  } = useStrategyEditor();

  const [timeframe, setTimeframe] =
    useState<Candle['timeframe']>(windowIndex > 0 && initialTimeframe ? initialTimeframe : loadLastTimeframe);
  const timeframeRef = useRef(timeframe);
  timeframeRef.current = timeframe;

  // sessionReplayEstimate (derived from the session-replay picker's raw
  // input strings) now lives in useReplayEngine.ts, destructured from
  // `replayEngine` above.

  // Keep `?timeframe=` and the last-picked timeframe in sync so a refresh (or
  // a bookmarked/bare link) resumes on the same timeframe — same convention
  // as the `?symbol=`/`tb.lastSymbol` sync in page.tsx.
  const onTimeframeChangeRef = useRef(onTimeframeChange);
  onTimeframeChangeRef.current = onTimeframeChange;
  useEffect(() => {
    if (windowIndex > 0) {
      onTimeframeChangeRef.current?.(timeframe);
      return;
    }
    const url = new URL(window.location.href);
    url.searchParams.set(TIMEFRAME_QUERY_KEY, timeframe);
    window.history.replaceState(null, '', url);
    try {
      localStorage.setItem(LAST_TIMEFRAME_KEY, timeframe);
    } catch {
      // Ignore blocked/full localStorage — timeframe just won't persist.
    }
  }, [timeframe, windowIndex]);

  // symbolInfo/spreadPoints/error/loadingMore/switchingChart/newsBands now
  // live in useCandleData's ChartRenderController — see the
  // `chartRenderController` destructure below, wired up once useChartEngine/
  // useIndicators (which candlesRef/visibleCandles are needed by first) are
  // in scope.
  // Drawing-tools state (tool selection, drawings list, active color, the
  // two drawing popovers) mostly lives in useDrawingTools — but the pieces
  // below are created here rather than there, because useChartEngine's
  // init effect (highlight/select/save-and-sync, right-click routing) needs
  // to close over them, and useChartEngine runs before useDrawingTools can
  // exist (it needs `chartController`, which only useChartEngine produces).
  // See useDrawingTools.ts's module doc for the full explanation. Everything
  // else this concern owns (`showDrawingsList`, `pendingAnchorCount`, the
  // two popovers' DOM refs) is created inside that hook instead.
  const [drawingTool, setDrawingTool] = useState<DrawingToolType | null>(null);
  // Mirror of manager.getAllDrawings() — kept in React state so the
  // DrawingsList panel re-renders whenever drawings are added/removed.
  const [drawingsList, setDrawingsList] = useState<IDrawing[]>([]);
  const [showActivityLogDock, setShowActivityLogDock] = useState(false);

  // Drawing color selection state
  const [activeColor, setActiveColor] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('chart-active-drawing-color') || '#2962ff';
    }
    return '#2962ff';
  });

  // Stored original styles for drawings that are highlighted when selected
  const originalStylesRef = useRef<Record<string, any>>({});

  // Ref to invoke saveAndSync from outside the useEffect block
  const saveAndSyncRef = useRef<() => void>(() => {});

  // States for context menu and edit popover of drawings
  const [drawingContextMenu, setDrawingContextMenu] = useState<{
    x: number;
    y: number;
    drawingId: string;
    drawingType: string;
    containerWidth: number;
    containerHeight: number;
  } | null>(null);

  const [drawingEditPopover, setDrawingEditPopover] = useState<{
    x: number;
    y: number;
    drawingId: string;
    drawingType: string;
    containerWidth: number;
    containerHeight: number;
  } | null>(null);

  // Read-only info popover for a clicked zone rectangle (Quasimodo/S&D v1/v2/
  // trade zone) — see the zone-click effect below, which resolves the click
  // against `chartController.getZoneMetaMap()`.
  const [zoneTooltip, setZoneTooltip] = useState<ZoneTooltipState | null>(null);

  // Read-only info popover for a clicked rejected/vetoed signal marker — see
  // the signal-click effect below.
  const [signalTooltip, setSignalTooltip] = useState<SignalTooltipState | null>(
    null,
  );

  // Stable references for symbol and symbol-switching state to avoid stale
  // closures inside the chart-creation useEffect.
  const symbolRef = useRef(symbol);
  symbolRef.current = symbol;
  // Active account (Phase 8) — every candle/marker/signal fetch and the WS
  // room subscription below need it. Mirrors symbolRef: a ref so effects
  // that only fire on a symbol/timeframe change still read the latest
  // account id without re-running.
  const accountId = useActiveAccount();
  const accountIdRef = useRef(accountId);
  accountIdRef.current = accountId;
  const isSwitchingSymbolRef = useRef(false);
  const drawingToolRef = useRef(drawingTool);
  drawingToolRef.current = drawingTool;

  const {
    editingTicket,
    setEditingTicket,
    editBusy,
    setEditBusy,
    contextMenu,
    setContextMenu,
    orderPopover,
    setOrderPopover,
    contextMenuRef,
    orderPopoverRef,
    activeHighlightedTicket,
    closedTrades,
    setClosedTrades,
    handleTicketSelect,
    handleTicketSelectRef,
    drag,
    setDrag,
    dragRef,
    dragStartRef,
  } = useOrderPopovers(symbol, highlightedTicket, onSelectTicket);

  // Forces a re-render (to recompute price->pixel positions) on pan/zoom/resize,
  // same trigger set the news-band overlay below already reacts to.
  const [, bumpLines] = useState(0);
  // Keeps the click-to-trade subscription (below) stable across re-renders
  // instead of resubscribing on every `trading` poll tick.
  const placeFromClickRef = useRef(trading.placeFromClick);
  placeFromClickRef.current = trading.placeFromClick;

  // Chart engine: the lightweight-charts instance itself — chart, candle +
  // volume series, drawing manager, series-markers primitives, resize
  // observer, drawing-tool mouse/context-menu wiring, and teardown. See
  // useChartEngine.ts for what still lives here vs. what later phases
  // (useCandleData/useDrawingTools/useReplayEngine) will further extract.
  const {
    chartController,
    containerRef,
    chartRef,
    candleSeriesRef,
    seriesMarkersRef,
    drawingManagerRef,
  } = useChartEngine({
    visibleCandles: visibleCandlesStable,
    replayActiveRef,
    followCursorRef,
    symbolRef,
    isSwitchingSymbolRef,
    drawingToolRef,
    setDrawingTool,
    setDrawingsList,
    originalStylesRef,
    setActiveColor,
    saveAndSyncRef,
    setContextMenu,
    setOrderPopover,
    setDrawingContextMenu,
    setDrawingEditPopover,
    bumpLines,
    showVolume,
  });

  // Backtest-view / live-bot "eye" view data (trades/activity log/signals,
  // SignalsDock selection) plus the effect that paints those trades/signals
  // onto the chart as markers and zone/SL-TP/exit-line drawings — called
  // here (after useChartEngine, rather than at the very top) because that
  // marker-application effect needs `chartController` to paint through. See
  // useBacktestData.ts's module doc.
  const {
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
  } = useBacktestData({
    backtestReportId,
    liveBotSkill,
    symbol,
    chartController,
    candlesRef,
    replayActive,
    replayCursorIndex,
    customCodeResult,
    orderLineStyle,
    showTradeLabels,
    showTradeMarkers,
    zoneColorStyle,
  });

  // Transient blinking "BUY HERE"/"SELL HERE"/exit flashes emitted as the
  // replay cursor crosses each trade/signal — see useReplayReveal.ts. Inert
  // (and returns a stable empty array) whenever replay is off, so it costs
  // nothing in the normal live view.
  const replayRevealEvents = useReplayReveal({
    replayActive,
    replayCursorTime,
    signals: backtestSignals,
    trades: backtestTrades,
    candlesRef,
  });

  // Multi-chart layout: mirror this window's replay session (active + picked
  // period) outward so secondary chart windows can follow along — see this
  // component's onReplaySessionChange doc comment. A saved backtest report's
  // "period" is its trades'/signals' own time bounds (`backtestPeriod`,
  // derived in useBacktestData) rather than an explicit picked from/to like
  // session replay has, but MiniChartPanel/useMiniCandleData don't care which
  // produced the range — either way it's just "fetch this period at my own
  // timeframe and clip to the shared cursor."
  useEffect(() => {
    if (!sharedReplay?.active) {
      onReplaySessionChange?.(windowIndex, {
        active: replayActive,
        sessionPeriod: backtestReportId ? backtestPeriod : sessionReplayPeriod,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayActive, sessionReplayPeriod, backtestReportId, backtestPeriod, sharedReplay?.active, windowIndex, timeframe]);

  // Indicator state + the indicator-series-creation effect. See
  // useIndicators.ts for what's now reactive (manualIndicators/
  // activeStrategy/showSeparators/chart readiness) vs. still imperative
  // (recomputeIndicatorsRef, called directly from the candle-loading effect
  // below on every history load/live tick/replay step — that effect hasn't
  // moved into its own hook yet, phase 7's useCandleData).
  const {
    manualIndicators,
    addManualIndicator,
    removeManualIndicator,
    updateManualIndicator,
    showIndicatorsDock,
    setShowIndicatorsDock,
    liveBotIndicators,
    computeCustomIndicatorsRef,
    recomputeIndicatorsRef,
  } = useIndicators({
    chartController,
    symbol,
    symbolRef,
    timeframe,
    timeframeRef,
    activeStrategy,
    showSeparators,
    showSeparatorsRef,
    visibleCandles: visibleCandlesStable,
    getRawCandles: () => candlesRef.current,
    customCodeResultRef,
    accountId,
    liveBotSkill,
    setShowSignalsDock,
    zoneColorStyle,
  });

  // Latest signal list for the signal-marker click hit-test effect above —
  // a ref so that click subscription isn't torn down and re-added on every
  // 5s live-bot signals poll.
  const backtestSignalsRef = useRef(backtestSignals);
  backtestSignalsRef.current = backtestSignals;

  // Stable indirection so useCandleData (below, called before
  // useReplayEngine can exist — it needs useCandleData's own return value)
  // can still invoke useReplayEngine's real `handleEnterReplay` once a
  // session-replay period's candles finish loading. Assigned right after
  // useReplayEngine runs, below — the same "ref created early, assigned by
  // a later hook" shape useChartEngine's `saveAndSyncRef` param already
  // established for useDrawingTools. Safe even though it's called
  // asynchronously (from the candle-load promise's `.then()`): by the time
  // that ever fires, this render has long since finished and the ref has
  // been assigned.
  const handleEnterReplayRef = useRef<() => void>(() => {});
  const centerOnRef = useRef<(index: number) => void>(() => {});

  // Candle history fetch + live WS subscribe, the spread/symbol-info poll,
  // and news-window shading now live in useCandleData.ts — see that hook's
  // module doc for why candlesRef/visibleCandles/replay-cursor state stay
  // here as inputs instead of moving in with everything else. `paintUpTo`
  // replaces the old `renderRef.current()` ref-hack every replay call site
  // (now in useReplayEngine.ts) used to reach for.
  const chartRenderController = useCandleData({
    chartController,
    symbol,
    timeframe,
    accountId,
    backtestReportId,
    sessionReplayPeriod: sharedReplay?.active && sharedReplay.sessionPeriod ? sharedReplay.sessionPeriod : sessionReplayPeriod,
    setSessionReplayPeriod,
    setSessionReplayLoadingPage,
    candlesRef,
    visibleCandles: visibleCandlesStable,
    replayActiveRef,
    replayCursorIndexRef,
    followCursorRef,
    setReplayActive,
    setReplayPlaying,
    setReplayCursorIndex,
    setFollowingCursor,
    setContextMenu,
    setOrderPopover,
    setDrawingContextMenu,
    setDrawingEditPopover,
    setBacktestTrades,
    setBacktestActivityLog,
    setBacktestSignals,
    setBacktestError,
    setBacktestMeta,
    recomputeIndicatorsRef,
    computeCustomIndicatorsRef,
    bumpLines,
    onSessionReplayLoaded: () => {
      if (sharedReplay?.active) {
        followCursorRef.current = true;
        setFollowingCursor(true);
        chartRenderController.paintUpTo();
        recomputeIndicatorsRef.current?.();
        const bars = visibleCandles();
        if (bars.length > 0) {
          requestAnimationFrame(() => {
            centerOnRef.current(bars.length - 1);
          });
        }
        return;
      }
      handleEnterReplayRef.current();
    },
  });
  const {
    symbolInfo,
    spreadPoints,
    error,
    loadingMore,
    switchingChart,
    newsBands,
  } = chartRenderController;

  useEffect(() => {
    if (sharedReplay?.active && sharedReplay.cursorTime != null) {
      chartRenderController.paintUpTo();
      recomputeIndicatorsRef.current?.();
      if (followCursorRef.current) {
        const bars = visibleCandles();
        if (bars.length > 0) {
          requestAnimationFrame(() => {
            centerOnRef.current(bars.length - 1);
          });
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sharedReplay?.active, sharedReplay?.cursorTime, chartRenderController.paintUpTo, recomputeIndicatorsRef]);

  // Multi-timeframe replay synthesis (see `masterCandlesRef` above): when this
  // window is a strictly coarser follower than the master driving the replay,
  // fetch the master's finer candles for the whole session period so
  // `visibleCandles()` can aggregate a forming bar for the current bucket.
  // Deduped via `fetchShared` so an M5 and M15 follower over the same M1
  // master/period share one fetch. Reset to empty whenever synthesis doesn't
  // apply (replay ended, this window isn't coarser, or account not yet
  // resolved) so stale finer data never leaks into a later render.
  const sessionPeriod = sharedReplay?.sessionPeriod ?? null;
  const masterTimeframe = sharedReplay?.masterTimeframe ?? null;
  useEffect(() => {
    if (
      !accountId ||
      !sharedReplay?.active ||
      !sessionPeriod ||
      !masterTimeframe ||
      TIMEFRAME_SECONDS[timeframe] <= TIMEFRAME_SECONDS[masterTimeframe]
    ) {
      masterCandlesRef.current = [];
      return;
    }
    let cancelled = false;
    const key = `${accountId}:${symbol}:${masterTimeframe}:${sessionPeriod.from}:${sessionPeriod.to}`;
    fetchShared(key, () =>
      fetchCandlesForPeriod(accountId, symbol, masterTimeframe, sessionPeriod.from, sessionPeriod.to),
    )
      .then((candles) => {
        if (!cancelled) {
          masterCandlesRef.current = candles;
          if (sharedReplay?.active && sharedReplay.cursorTime != null) {
            chartRenderController.paintUpTo();
            recomputeIndicatorsRef.current?.();
          }
        }
      })
      .catch(() => {
        if (!cancelled) masterCandlesRef.current = [];
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accountId,
    symbol,
    timeframe,
    sharedReplay?.active,
    masterTimeframe,
    sessionPeriod?.from,
    sessionPeriod?.to,
  ]);

  // Tick-form replay (see `finerCandlesRef` above): once local replay is active,
  // fetch the next-finer timeframe's candles across the loaded window so
  // `visibleCandles()` can synthesize a forming bar. `replayActive` only flips
  // true *after* the window's candles have loaded (see `handleEnterReplay`), so
  // `candlesRef.current` is populated by the time this runs. Deduped via
  // `fetchShared`. Deliberately NOT gated on the `tickForm` toggle: the fetch
  // (and `finerAvailable`, which the toggle's enabled state depends on) must not
  // depend on the toggle, or turning tick-form off would disable the button and
  // trap it off. `tickFormRef` gates *use* of the data in `visibleCandles`/the
  // autoplay loop instead. Resets to empty (and `finerAvailable` false) whenever
  // synthesis can't apply — no replay, the timeframe has no finer level (M1),
  // account not resolved, or candles not yet loaded — so stale finer data never
  // leaks into a render, and the empty array falls back to the whole-bar reveal.
  useEffect(() => {
    const finerTf = FINER_TIMEFRAME[timeframe];
    const loaded = candlesRef.current;
    if (!accountId || !replayActive || !finerTf || loaded.length === 0) {
      finerCandlesRef.current = [];
      setFinerAvailable(false);
      return;
    }
    let cancelled = false;
    const from = loaded[0].time as number;
    const to = (loaded[loaded.length - 1].time as number) + TIMEFRAME_SECONDS[timeframe];
    const key = `finer:${accountId}:${symbol}:${finerTf}:${from}:${to}`;
    fetchShared(key, () => fetchCandlesForPeriod(accountId, symbol, finerTf, from, to))
      .then((candles) => {
        if (cancelled) return;
        finerCandlesRef.current = candles;
        setFinerAvailable(candles.length > 0);
        // A finer load that lands mid-replay repaints so the forming bar can
        // start synthesizing immediately rather than on the next cursor tick.
        if (replayActiveRef.current) chartRenderController.paintUpTo();
      })
      .catch(() => {
        if (cancelled) return;
        finerCandlesRef.current = [];
        setFinerAvailable(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountId, symbol, timeframe, replayActive]);

  // Keep the imperative tick-form flag (read by `visibleCandles()`/`render()`
  // closures created once per data-load) in sync with the toggle state, and
  // repaint immediately so toggling mid-replay (e.g. while paused) takes effect
  // now rather than on the next autoplay tick.
  useEffect(() => {
    tickFormRef.current = tickForm;
    if (replayActiveRef.current) chartRenderController.paintUpTo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tickForm]);

  // Overlay recompute on activeStrategy/manualIndicators/showSeparators
  // change, and custom (backend-Python) indicator compute on
  // manualIndicators/symbol/timeframe change, both now live in
  // useIndicators.ts as their own effects — see that hook's module doc.

  // Drawing-tools: tool selection, the anchor-collection click workflow,
  // drawings-list panel state, active color, the drawing context-menu/edit-
  // popover pair and their outside-click effect, and the per-symbol
  // clear+reload effect (the chart-creation effect only handles the initial
  // symbol) — see useDrawingTools.ts's module doc for what stays here vs.
  // what moved, and why.
  const drawingTools = useDrawingTools({
    chartController,
    symbol,
    drawingTool,
    setDrawingTool,
    drawingsList,
    setDrawingsList,
    activeColor,
    setActiveColor,
    drawingContextMenu,
    setDrawingContextMenu,
    drawingEditPopover,
    setDrawingEditPopover,
    originalStylesRef,
    saveAndSyncRef,
    isSwitchingSymbolRef,
  });

  // Replay engine: the cursor-driven "live session player" over a backtest
  // report's or session-replay period's candles — play/pause/step/seek,
  // session-replay period picking, and the SignalsDock/trade-history
  // click-to-navigate handlers. Called here (after useChartEngine and
  // useCandleData, before the effects below that need its outputs) because
  // it's the heaviest consumer of both controllers — see useReplayEngine.ts's
  // module doc for why several replay primitives still live above as
  // controlled inputs instead of moving into the hook outright.
  const replayEngine = useReplayEngine({
    chartController,
    chartRenderController,
    timeframe,
    replayActive,
    setReplayActive,
    replayActiveRef,
    replayPlaying,
    setReplayPlaying,
    setReplayCursorIndex,
    replayCursorIndexRef,
    setFollowingCursor,
    followCursorRef,
    setSessionReplayPeriod,
    setShowActivityLogDock,
    // `hideToolbar` alone is true for every window MultiChartLayout renders,
    // including a single-window split — `windowCount > 1` is what actually
    // distinguishes "real" multi-window mode (see the same pairing above,
    // the focus-ring header).
    suppressAutoActivityDock: hideToolbar && windowCount > 1,
    backtestTrades,
    backtestSignals,
    selectedTradeIndex,
    setSelectedTradeIndex,
    selectedSignalIndex,
    setSelectedSignalIndex,
    sharedReplayActive: sharedReplay?.active ?? false,
    finerCandlesRef,
    replayFineTimeRef,
    tickFormRef,
  });
  // See the `handleEnterReplayRef` declaration above useCandleData's call —
  // reassigned every render so useCandleData's `onSessionReplayLoaded`
  // always dispatches to the current `handleEnterReplay`.
  handleEnterReplayRef.current = replayEngine.handleEnterReplay;
  centerOnRef.current = replayEngine.centerOn;
  const {
    navigateToTime,
    seekTo,
    handleEnterReplay,
    handleExitReplay,
    handleStartSessionReplay,
    startSessionReplayForRange,
    handleExitSessionReplay,
    handleRecenterReplay,
    handleToggleTrade,
    handleNavigateTrade,
    handleToggleSignal,
    replaySpeed,
    setReplaySpeed,
    showSessionReplayPicker,
    setShowSessionReplayPicker,
    sessionReplayFromInput,
    setSessionReplayFromInput,
    sessionReplayToInput,
    setSessionReplayToInput,
    sessionReplayEstimate,
    // `lastRevealedSignatureRef` (also on `replayEngine`) is no longer
    // destructured here — its only consumer, the backtest-trade-drawing
    // effect, moved into useBacktestData.ts in phase 10, which now owns an
    // equivalent ref of its own (useReplayEngine.ts is out of scope for
    // this phase, so its copy stays declared there, unused).
  } = replayEngine;

  // --- Bot dock "Replay" tab (BOT_SESSION_REPLAY_PLAN phase 2) --------------
  // The eyed bot's own session replay, driven by the very same
  // `useReplayEngine` session-replay path the chart toolbar's picker uses —
  // only the period comes from the bot's signals/trades instead of the
  // picker's inputs. Built only for the live-bot eye view; a saved backtest
  // report already has the toolbar's replay controls, and its period isn't
  // user-picked.
  //
  // Every handler goes through a ref reassigned on each render, so the memo
  // below can depend on plain values (and keep `BotSessionReplayTab` from
  // re-rendering on every parent render) without ever calling a stale
  // closure over the replay engine.
  const replayHandlersRef = useRef({
    startSessionReplayForRange,
    handleExitSessionReplay,
    setReplayPlaying,
    setReplaySpeed,
  });
  replayHandlersRef.current = {
    startSessionReplayForRange,
    handleExitSessionReplay,
    setReplayPlaying,
    setReplaySpeed,
  };

  const botReplayControls = useMemo<BotReplayControls | undefined>(() => {
    if (!liveBotSkill || backtestReportId) return undefined;
    return {
      active: Boolean(sessionReplayPeriod && replayActive),
      playing: replayPlaying,
      speed: replaySpeed,
      cursorTime: replayCursorTime,
      loading: sessionReplayLoadingPage,
      onStart: (from, to) =>
        replayHandlersRef.current.startSessionReplayForRange(from, to),
      onPlayPause: () => replayHandlersRef.current.setReplayPlaying((p) => !p),
      onSpeedChange: (speed) => replayHandlersRef.current.setReplaySpeed(speed),
      onExit: () => {
        replayHandlersRef.current.setReplayPlaying(false);
        replayHandlersRef.current.handleExitSessionReplay();
      },
    };
  }, [
    liveBotSkill,
    backtestReportId,
    sessionReplayPeriod,
    replayActive,
    replayPlaying,
    replaySpeed,
    replayCursorTime,
    sessionReplayLoadingPage,
  ]);

  // Spread/symbol-info poll now lives in useCandleData.ts (its
  // `symbolInfo`/`spreadPoints` are destructured from `chartRenderController`
  // above).

  // News-event markers (Phase 7): persisted calendar-event history
  // (`GET .../news/events`), fetched once per account/symbol-view change —
  // the calendar barely changes minute to minute, unlike trade markers'
  // MARKERS_POLL_MS poll, so no polling here. Held in state and merged into
  // the live trade-markers effect below rather than calling `setMarkers`
  // here directly, since the series-markers plugin only accepts one combined
  // array per `setMarkers` call. Skipped in backtest/eyed-bot view for the
  // same reason the live trade-markers effect below is — that view's own
  // marker-application effect lives in useBacktestData.ts and isn't touched
  // by this phase. Always-on (no toolbar toggle): wiring one in would mean
  // touching `ChartToolbarProps`/`ChartToolbar.tsx`, out of scope here (see
  // this file's other news-marker comments).
  const [newsEventMarkers, setNewsEventMarkers] = useState<SeriesMarker<Time>[]>([]);
  useEffect(() => {
    if (backtestReportId || liveBotSkill || !accountId) {
      setNewsEventMarkers([]);
      return;
    }
    let cancelled = false;
    const nowSeconds = Math.floor(Date.now() / 1000);
    getNewsEvents(accountId, {
      start: nowSeconds - 180 * 86400,
      end: nowSeconds + 30 * 86400,
    })
      .then((events: NewsEventRecord[]) => {
        if (!cancelled) setNewsEventMarkers(toNewsEventSeriesMarkers(events));
      })
      .catch(() => {
        // Calendar unreachable — leave whatever news markers are already drawn.
      });
    return () => {
      cancelled = true;
    };
  }, [accountId, backtestReportId, liveBotSkill]);

  // Poll trade markers (F7): entry arrows + exit circles from the journal,
  // plus an entry->exit oblique line (LIVE_TRADE_DRAWING_PREFIX) for each
  // closed trade so a closed position stays visible on the chart instead of
  // just leaving behind a small circle. Skipped in backtest view — those
  // markers/lines come from the report's own trades (set below) instead of
  // the live journal. Also skipped while a bot's eye is on (`liveBotSkill`)
  // — that's a focused single-bot view (same backtestTrades/backtestSignals
  // state and marker-merge effect backtest view uses, just fed live data),
  // replacing this unscoped "every trade on the symbol" view rather than
  // layering on top of it.
  //
  useEffect(() => {
    if (backtestReportId || liveBotSkill || !accountId) {
      setClosedTrades([]);
      return;
    }
    const colors = {
      ok: cssVar('--color-ok'),
      err: cssVar('--color-err'),
    };
    const clearLiveTradeLines = () => {
      const manager = drawingManagerRef.current;
      if (!manager) return;
      for (const drawing of manager.getAllDrawings()) {
        if (drawing.id.startsWith(LIVE_TRADE_DRAWING_PREFIX)) {
          manager.removeDrawing(drawing.id);
        }
      }
    };
    if (customCodeResult) {
      seriesMarkersRef.current?.setMarkers(
        [
          ...(showTradeMarkers
            ? toCustomSignalsSeriesMarkers(customCodeResult.signals, colors, showTradeLabels)
            : []),
          ...newsEventMarkers,
        ].sort((a, b) => (a.time as number) - (b.time as number)),
      );
      clearLiveTradeLines();
      return;
    }
    // The fetch is identical for every window on the same account+symbol
    // (multi-chart layout §) — shared via subscribeSharedPoll so N windows
    // don't each poll the journal independently, while applying the result
    // (series markers, drawing lines, closedTrades state) stays per-window.
    const unsubscribe = subscribeSharedPoll(
      `trade-markers:${accountId}:${symbol}`,
      MARKERS_POLL_MS,
      () => getTradeMarkers(accountId, symbol),
      (trades) => {
        if (!trades) return; // Journal unreachable — leave whatever's already drawn.
        // A clicked active order or trade-history row (`activeHighlightedTicket`,
        // synced with OrdersDock/TradeHistoryTable via `onSelectTicket`) scopes
        // the arrow/label markers and the entry->exit oblique line down to just
        // that one trade — without this, every trade on the symbol kept its own
        // arrow/label/oblique-line regardless of what was clicked. An open
        // position naturally gets only its entry arrow+label here (`toSeriesMarkers`
        // only adds the exit circle/oblique-line once `close_time` is set), which
        // is also why a selected active order never grows an oblique line.
        const displayTrades =
          activeHighlightedTicket !== null
            ? trades.filter((t) => String(t.id) === String(activeHighlightedTicket))
            : trades;
        seriesMarkersRef.current?.setMarkers(
          [
            ...(showTradeMarkers ? toSeriesMarkers(displayTrades, colors, showTradeLabels) : []),
            ...newsEventMarkers,
          ].sort((a, b) => (a.time as number) - (b.time as number)),
        );
        setClosedTrades(trades);
        clearLiveTradeLines();
        const manager = drawingManagerRef.current;
        if (manager) {
          for (const drawing of buildLiveTradeLineDrawings(
            displayTrades,
            colors,
            candlesRef.current,
            orderLineStyle,
          )) {
            manager.addDrawing(drawing);
          }
        }
      },
    );
    return unsubscribe;
    // setClosedTrades comes from useOrderPopovers but is a plain setState
    // setter (stable identity) — safe to omit, same as every other setState
    // setter this effect already calls without listing as a dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accountId,
    symbol,
    backtestReportId,
    liveBotSkill,
    customCodeResult,
    orderLineStyle,
    showTradeLabels,
    activeHighlightedTicket,
    showTradeMarkers,
    newsEventMarkers,
  ]);

  // A trade-history row can be arbitrarily far in the past (unlike an open
  // position, which is already near "now"), so jump the chart there the same
  // way SignalsDock's Trades tab does for a backtest trade — once per
  // ticket, not on every `closedTrades` poll refresh (which would otherwise
  // fight the user's own panning every MARKERS_POLL_MS while a trade stays
  // selected).
  const lastNavigatedHistoryTicketRef = useRef<string | number | null>(null);
  useEffect(() => {
    if (activeHighlightedTicket === null) {
      lastNavigatedHistoryTicketRef.current = null;
      return;
    }
    if (lastNavigatedHistoryTicketRef.current === activeHighlightedTicket) return;
    const t = findHistoryTrade();
    if (t) {
      lastNavigatedHistoryTicketRef.current = activeHighlightedTicket;
      navigateToTime(t.open_time);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeHighlightedTicket, closedTrades, trading.positions]);

  // Resolves the eyed bot's own indicator list (`liveBotIndicators`) and
  // auto-activates its "S&D zones" manual indicator — now owned by
  // useIndicators.ts (see that hook's module doc and its own effect).

  // Rendering the backtest report's (or eyed live bot's) trades as chart
  // markers + zone/SL-TP/exit-line drawings now lives in useBacktestData.ts
  // (moved there in phase 10 once `chartController` existed to paint
  // through) — see that hook's module doc and its own marker-application
  // effect.

  // News-window shading poll now lives in useCandleData.ts (`newsBands` is
  // destructured from `chartRenderController` above).

  // Click-to-trade: while `trading.placementMode` is armed (from the order
  // ticket), a chart click converts its y-coordinate to a price and hands it
  // to the ticket for confirmation — it never fires an order directly.
  useEffect(() => {
    const chart = chartRef.current;
    const series = candleSeriesRef.current;
    if (!chart || !series) return;
    const handler = (param: MouseEventParams) => {
      if (!param.point) return;
      const price = series.coordinateToPrice(param.point.y);
      if (price !== null) placeFromClickRef.current(price);
    };
    chart.subscribeClick(handler);
    return () => chart.unsubscribeClick(handler);
    // chartRef/candleSeriesRef are stable ref objects returned from
    // useChartEngine — omitted deliberately, same as every other effect in
    // this file that reads them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Zone click-to-inspect: clicking a zone rectangle (Quasimodo/S&D v1/v2/
  // trade zone) hit-tests the drawing manager the same way the right-click
  // context menu already does (see useChartEngine's handleContextMenu), and
  // shows a read-only details popover when the hit drawing has an entry in
  // the zone-meta side map. User drawings (no meta entry) are left alone —
  // their own select/drag/right-click-edit behavior is unaffected.
  useEffect(() => {
    const chart = chartRef.current;
    const container = containerRef.current;
    if (!chart || !container || !chartController) return;
    const handler = (param: MouseEventParams) => {
      if (!param.point) return;
      const manager = chartController.getDrawingManager();
      if (!manager) return;
      const hit = manager.hitTest(param.point);
      if (!hit) return;
      const meta = chartController.getZoneMetaMap().get(hit.id);
      if (!meta) return;
      setZoneTooltip({
        x: param.point.x,
        y: param.point.y,
        meta,
        containerWidth: container.clientWidth,
        containerHeight: container.clientHeight,
      });
    };
    chart.subscribeClick(handler);
    return () => chart.unsubscribeClick(handler);
    // chartRef/containerRef are stable ref objects returned from
    // useChartEngine; chartController only changes identity on mount/unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartController]);

  // Signal click-to-inspect: clicking a rejected/vetoed signal's square
  // marker (from `toSignalSeriesMarkers`) opens a popover with the
  // strategy's own reason text. The series-markers plugin exposes no hit
  // API, so the hit test is done here: resolve the clicked x to the nearest
  // candle, keep that bar's non-`opened` signals, then require the click y
  // to be near the bar on the side the marker is drawn (belowBar for buy,
  // aboveBar for sell). Zone rectangles keep priority — a click that lands
  // on a zone drawing is left to the zone effect above. Works identically
  // during replay (only signals at/before the cursor bar are drawn, and the
  // same cutoff is applied here) and in the normal live/eye view.
  useEffect(() => {
    const chart = chartRef.current;
    const series = candleSeriesRef.current;
    const container = containerRef.current;
    if (!chart || !series || !container || !chartController) return;
    const handler = (param: MouseEventParams) => {
      if (!param.point) return;
      const signals = backtestSignalsRef.current;
      if (!signals || signals.length === 0) return;
      // A zone rectangle under the cursor wins — same click, one popover.
      const manager = chartController.getDrawingManager();
      const hit = manager?.hitTest(param.point);
      if (hit && chartController.getZoneMetaMap().has(hit.id)) return;

      const candles = candlesRef.current;
      const time =
        (param.time as number | undefined) ??
        (chart.timeScale().coordinateToTime(param.point.x) as number | null) ??
        null;
      if (time === null) return;
      const barTime = nearestCandleTime(candles, time);
      if (barTime === null) return;
      const candle = candles.find((c) => c.time === barTime);
      if (!candle) return;
      // "No lookahead": while replaying, bars past the cursor aren't drawn.
      const cursorTime = replayActiveRef.current
        ? ((candlesRef.current[replayCursorIndexRef.current]?.time as
            | number
            | undefined) ?? 0)
        : Infinity;
      if ((barTime as number) > cursorTime) return;

      const hits = signals.filter(
        (s) =>
          s.outcome !== 'opened' &&
          nearestCandleTime(candles, s.time) === barTime,
      );
      if (hits.length === 0) return;

      const lowY = series.priceToCoordinate(candle.low);
      const highY = series.priceToCoordinate(candle.high);
      if (lowY === null || highY === null) return;
      const y = param.point.y;
      // Marker band: a buy signal's square sits below the bar's low, a sell
      // signal's above its high. Allow a little slack into the bar itself so
      // clicking the marker's edge still counts.
      const BAND = 44;
      const SLACK = 6;
      const matched = hits.filter((s) =>
        s.direction === 'buy'
          ? y >= lowY - SLACK && y <= lowY + BAND
          : y <= highY + SLACK && y >= highY - BAND,
      );
      if (matched.length === 0) return;

      setSignalTooltip({
        x: param.point.x,
        y,
        signals: matched,
        containerWidth: container.clientWidth,
        containerHeight: container.clientHeight,
      });
    };
    chart.subscribeClick(handler);
    return () => chart.unsubscribeClick(handler);
    // chartRef/candleSeriesRef/containerRef/candlesRef and the replay refs are
    // stable ref objects; chartController only changes identity on
    // mount/unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartController]);

  // Draggable SL/TP/trigger-price lines: recompute pixel positions on
  // pan/zoom/resize (prices themselves come from `trading` polling, which
  // already triggers a re-render on its own).
  useEffect(() => {
    const chart = chartRef.current;
    const container = containerRef.current;
    if (!chart || !container) return;
    const bump = () => bumpLines((t) => t + 1);
    chart.timeScale().subscribeVisibleTimeRangeChange(bump);
    const resizeObserver = new ResizeObserver(bump);
    resizeObserver.observe(container);
    return () => {
      chart.timeScale().unsubscribeVisibleTimeRangeChange(bump);
      resizeObserver.disconnect();
    };
    // chartRef/containerRef are stable ref objects returned from
    // useChartEngine — omitted deliberately, same as every other effect in
    // this file that reads them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, timeframe]);

  // Live mousemove/mouseup for whichever line (if any) is currently being
  // dragged — subscribed once; `dragRef` always holds the current target so
  // this doesn't need to resubscribe on every drag start/stop.
  useEffect(() => {
    function onMove(e: MouseEvent) {
      const current = dragRef.current;
      const container = containerRef.current;
      const series = candleSeriesRef.current;
      if (!current || !container || !series) return;
      const rect = container.getBoundingClientRect();
      const price = series.coordinateToPrice(e.clientY - rect.top);
      if (price !== null) setDrag({ ...current, price });
    }
    function onUp(e: MouseEvent) {
      const current = dragRef.current;
      if (current) current.commit(current.price);
      setDrag(null);

      const start = dragStartRef.current;
      if (start) {
        const dist = Math.hypot(e.clientX - start.x, e.clientY - start.y);
        if (dist < 5 && start.wasSelected) {
          handleTicketSelectRef.current(start.ticket);
        }
        dragStartRef.current = null;
      }
    }
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    // dragRef/dragStartRef/handleTicketSelectRef/setDrag come from
    // useOrderPopovers but are stable-identity ref objects and a plain
    // setState setter — safe to omit, same as this effect always relying on
    // `dragRef` staying current without resubscribing (see comment above).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Close the drawing-tools context menu / edit popover on click outside —
  // now lives in useDrawingTools (completing phase 5's TODO; see that
  // hook's module doc). The matching effect for `contextMenu`/`orderPopover`
  // lives in useOrderPopovers — the two pairs are opened mutually
  // exclusively by useChartEngine's `contextmenu` handler.

  // Replay engine — `centerOn`/`navigateToTime`/`seekTo`/`handleEnterReplay`/
  // `handleExitReplay`/`handleStartSessionReplay`/`handleExitSessionReplay`/
  // `handleRecenterReplay`/`handleToggleTrade`/`handleNavigateTrade`/
  // `handleToggleSignal`, the mousedown-pauses-follow effect, and the
  // autoplay tick effect now live in useReplayEngine.ts — destructured from
  // `replayEngine` above.

  // handleColorChange/handleModifyDrawingColor now live in useDrawingTools
  // (`drawingTools.handleColorChange`/`drawingTools.handleModifyDrawingColor`).

  function buildPriceLines(): PriceLineSpec[] {
    const okColor = cssVar('--color-ok');
    const errColor = cssVar('--color-err');
    const accentColor = cssVar('--color-accent');
    const specs: PriceLineSpec[] = [];
    for (const p of trading.positions) {
      const offset = defaultOffset(p.open_price);
      const direction = p.side === 'buy' ? 1 : -1;
      if (p.sl !== null) {
        specs.push({
          key: `pos-${p.ticket}-sl`,
          ticket: p.ticket,
          price: p.sl,
          color: errColor,
          label: `SL ${p.sl}`,
          commit: (np) => trading.modifyPositionSlTp(p.ticket, np, p.tp),
          pnlOpenPrice: p.open_price,
          pnlSide: p.side,
          pnlVolume: p.volume,
        });
      } else {
        specs.push({
          key: `pos-${p.ticket}-sl`,
          ticket: p.ticket,
          price: p.open_price - direction * offset,
          color: errColor,
          label: '+ SL',
          placeholder: true,
          commit: (np) => trading.modifyPositionSlTp(p.ticket, np, p.tp),
          pnlOpenPrice: p.open_price,
          pnlSide: p.side,
          pnlVolume: p.volume,
        });
      }
      if (p.tp !== null) {
        specs.push({
          key: `pos-${p.ticket}-tp`,
          ticket: p.ticket,
          price: p.tp,
          color: okColor,
          label: `TP ${p.tp}`,
          commit: (np) => trading.modifyPositionSlTp(p.ticket, p.sl, np),
          pnlOpenPrice: p.open_price,
          pnlSide: p.side,
          pnlVolume: p.volume,
        });
      } else {
        specs.push({
          key: `pos-${p.ticket}-tp`,
          ticket: p.ticket,
          price: p.open_price + direction * offset,
          color: okColor,
          label: '+ TP',
          placeholder: true,
          commit: (np) => trading.modifyPositionSlTp(p.ticket, p.sl, np),
          pnlOpenPrice: p.open_price,
          pnlSide: p.side,
          pnlVolume: p.volume,
        });
      }
    }
    for (const o of trading.pendingOrders) {
      const offset = defaultOffset(o.price);
      const direction = o.side === 'buy' ? 1 : -1;
      specs.push({
        key: `pend-${o.ticket}-price`,
        ticket: o.ticket,
        price: o.price,
        color: accentColor,
        label: `${o.side} ${o.order_type} ${o.price}`,
        commit: (np) => trading.modifyPending(o.ticket, np, o.sl, o.tp),
      });
      if (o.sl !== null) {
        specs.push({
          key: `pend-${o.ticket}-sl`,
          ticket: o.ticket,
          price: o.sl,
          color: errColor,
          label: `SL ${o.sl}`,
          commit: (np) => trading.modifyPending(o.ticket, null, np, o.tp),
          pnlOpenPrice: o.price,
          pnlSide: o.side,
          pnlVolume: o.volume,
        });
      } else {
        specs.push({
          key: `pend-${o.ticket}-sl`,
          ticket: o.ticket,
          price: o.price - direction * offset,
          color: errColor,
          label: '+ SL',
          placeholder: true,
          commit: (np) => trading.modifyPending(o.ticket, null, np, o.tp),
          pnlOpenPrice: o.price,
          pnlSide: o.side,
          pnlVolume: o.volume,
        });
      }
      if (o.tp !== null) {
        specs.push({
          key: `pend-${o.ticket}-tp`,
          ticket: o.ticket,
          price: o.tp,
          color: okColor,
          label: `TP ${o.tp}`,
          commit: (np) => trading.modifyPending(o.ticket, null, o.sl, np),
          pnlOpenPrice: o.price,
          pnlSide: o.side,
          pnlVolume: o.volume,
        });
      } else {
        specs.push({
          key: `pend-${o.ticket}-tp`,
          ticket: o.ticket,
          price: o.price + direction * offset,
          color: okColor,
          label: '+ TP',
          placeholder: true,
          commit: (np) => trading.modifyPending(o.ticket, null, o.sl, np),
          pnlOpenPrice: o.price,
          pnlSide: o.side,
          pnlVolume: o.volume,
        });
      }
    }
    return specs;
  }

  // One dashed line per running position at its entry (open) price — separate
  // from the SL/TP lines above so it reads as "this is where the trade is
  // running from", not a modifiable trigger. Color is by side (buy/sell), not
  // ok/err, since those are already reserved for TP/SL regardless of side.
  function buildEntryLines(): EntryLineSpec[] {
    const buyColor = cssVar('--color-buy');
    const sellColor = cssVar('--color-sell');
    return trading.positions.map((p) => ({
      key: `entry-${p.ticket}`,
      position: p,
      color: orderLineStyle.customColors
        ? orderLineStyle.openColor
        : p.side === 'buy'
          ? buyColor
          : sellColor,
      label: `${p.side.toUpperCase()} ${p.volume} @ ${p.open_price}`,
    }));
  }

  // SL/TP lines for the trade selected in SignalsDock's Trades tab —
  // read-only (no drag/commit, unlike buildPriceLines) since these are
  // closed backtest/live-bot trades, not orders that can still be modified.
  // Open/close lines are handled separately by `buildSelectedTradeOpenClose`
  // (styleable + bounded to the trade's own time span).
  function buildSelectedTradeLines(): { key: string; price: number; color: string; leftLabel?: string; rightLabel?: string }[] {
    if (selectedTradeIndex === null || !backtestTrades) return [];
    const t = backtestTrades[selectedTradeIndex];
    if (!t) return [];
    const okColor = cssVar('--color-ok');
    const errColor = cssVar('--color-err');
    const contractSize = symbolInfo?.contract_size ?? 1;
    const calcPnl = (targetPrice: number) => {
      const rawPnl = (targetPrice - t.open_price) * (t.side === 'buy' ? 1 : -1) * contractSize * t.volume;
      const pnl = Math.abs(rawPnl) < 0.005 ? 0 : rawPnl;
      return `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}$`;
    };
    const specs = [];
    if (t.sl !== null) {
      specs.push({ key: 'trade-sl', price: t.sl, color: errColor, leftLabel: calcPnl(t.sl), rightLabel: `SL ${t.sl}` });
    }
    if (t.tp !== null) {
      specs.push({ key: 'trade-tp', price: t.tp, color: okColor, leftLabel: calcPnl(t.tp), rightLabel: `TP ${t.tp}` });
    }
    return specs;
  }

  // Open/close pair for the trade selected in SignalsDock's Trades tab,
  // rendered as two segments bounded to [open_time, close_time] — styled via
  // `orderLineStyle` (see the toolbar's "Order lines" control) instead of the
  // fixed dashed full-width look `buildSelectedTradeLines` used to give them.
  function buildSelectedTradeOpenClose(): {
    openTime: UTCTimestamp;
    openPrice: number;
    openColor: string;
    openLabel: string;
    closeTime: UTCTimestamp;
    closePrice: number;
    closeColor: string;
    closeLabel: string;
  } | null {
    if (!orderLineStyle.visible) return null;
    if (selectedTradeIndex === null || !backtestTrades) return null;
    const t = backtestTrades[selectedTradeIndex];
    if (!t) return null;
    const buyColor = cssVar('--color-buy');
    const sellColor = cssVar('--color-sell');
    const accentColor = cssVar('--color-accent');
    return {
      openTime: t.open_time as UTCTimestamp,
      openPrice: t.open_price,
      openColor: orderLineStyle.customColors
        ? orderLineStyle.openColor
        : t.side === 'buy'
          ? buyColor
          : sellColor,
      openLabel: `${t.side.toUpperCase()} ${t.volume} @ ${t.open_price}`,
      closeTime: t.close_time as UTCTimestamp,
      closePrice: t.close_price,
      closeColor: orderLineStyle.customColors ? orderLineStyle.closeColor : accentColor,
      closeLabel: `${t.profit >= 0 ? '+' : ''}${t.profit.toFixed(2)} @ ${t.close_price}`,
    };
  }

  // The trade-history row clicked in TradeHistoryTable, if it's an open or
  // closed trade on this symbol — same `activeHighlightedTicket` conduit as an open
  // position/pending order (see the `highlightedTicket` prop doc), looked up in
  // `trading.positions` or `closedTrades` (which holds recent journal trade markers).
  function findHistoryTrade(): TradeMarker | null {
    if (activeHighlightedTicket === null) return null;
    const pos = trading.positions.find((p) => p.ticket === Number(activeHighlightedTicket));
    if (pos) {
      return {
        id: String(pos.ticket),
        symbol: pos.symbol,
        side: pos.side,
        volume: pos.volume,
        open_price: pos.open_price,
        open_time: Math.floor(Date.parse(pos.open_time) / 1000),
        sl: pos.sl,
        tp: pos.tp,
        close_price: null,
        close_time: null,
        profit: pos.profit,
        comment: pos.comment,
      };
    }
    return (
      closedTrades.find(
        (t) => String(t.id) === String(activeHighlightedTicket),
      ) ?? null
    );
  }

  // Entry/SL/TP/close lines for a highlighted trade-history row, all full-
  // width horizontal lines like `buildSelectedTradeLines`/SL-TP treatment —
  // deliberately NOT the time-bounded segment `buildSelectedTradeOpenClose`
  // draws for a backtest trade. That approach anchors both ends via
  // `timeToCoordinate`, which silently drops the whole line the moment
  // either timestamp isn't resolvable in whatever candle window happens to
  // be loaded — exactly the case for an old history row, and the entry price
  // is the one thing a click on this row must always show.
  function buildHistoryTradeLines(): { key: string; price: number; color: string; leftLabel?: string; rightLabel?: string }[] {
    const t = findHistoryTrade();
    if (!t) return [];
    const okColor = cssVar('--color-ok');
    const errColor = cssVar('--color-err');
    const buyColor = cssVar('--color-buy');
    const sellColor = cssVar('--color-sell');
    const accentColor = cssVar('--color-accent');
    const contractSize = symbolInfo?.contract_size ?? 1;
    const calcPnl = (targetPrice: number) => {
      const rawPnl = (targetPrice - t.open_price) * (t.side === 'buy' ? 1 : -1) * contractSize * t.volume;
      const pnl = Math.abs(rawPnl) < 0.005 ? 0 : rawPnl;
      return `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}$`;
    };
    const specs: { key: string; price: number; color: string; leftLabel?: string; rightLabel?: string }[] = [
      {
        key: 'history-open',
        price: t.open_price,
        color: t.side === 'buy' ? buyColor : sellColor,
        leftLabel: `${t.side.toUpperCase()} ${t.volume} @ ${t.open_price}`,
      },
    ];
    if (t.sl !== null) {
      specs.push({ key: 'history-sl', price: t.sl, color: errColor, leftLabel: calcPnl(t.sl), rightLabel: `SL ${t.sl}` });
    }
    if (t.tp !== null) {
      specs.push({ key: 'history-tp', price: t.tp, color: okColor, leftLabel: calcPnl(t.tp), rightLabel: `TP ${t.tp}` });
    }
    if (t.close_price !== null) {
      specs.push({
        key: 'history-close',
        price: t.close_price,
        color: accentColor,
        leftLabel: `${(t.profit ?? 0) >= 0 ? '+' : ''}${(t.profit ?? 0).toFixed(2)} @ ${t.close_price}`,
      });
    }
    return specs;
  }

  async function handleSaveEdit(
    ticket: number,
    sl: number | null,
    tp: number | null,
  ) {
    setEditBusy(true);
    try {
      await trading.modifyPositionSlTp(ticket, sl, tp);
      setEditingTicket(null);
    } finally {
      setEditBusy(false);
    }
  }

  async function handleCloseFromEdit(ticket: number) {
    if (!window.confirm(`Close position #${ticket}?`)) return;
    setEditBusy(true);
    try {
      await trading.close(ticket);
      setEditingTicket(null);
    } finally {
      setEditBusy(false);
    }
  }

  // Add/remove a manual indicator — see useIndicators.ts's
  // addManualIndicator/removeManualIndicator (persists + updates state).
  async function runCustomCode(codeOverride?: string) {
    const code = codeOverride ?? customCodeDraft;
    if (!code) return;
    setCustomCodeBusy(true);
    setCustomCodeError(null);
    try {
      const periodParam = derivePeriodParam(candlesRef.current);
      if (!periodParam) {
        throw new Error('No historical candles loaded on the chart yet.');
      }

      const res = await evaluateCustomCode({
        code,
        symbol,
        timeframe,
        period: periodParam,
      });

      if (res.error) {
        setCustomCodeError(res.error);
      } else {
        setCustomCodeResult(res);
        customCodeResultRef.current = res;
        recomputeIndicatorsRef.current();
      }
    } catch (err) {
      setCustomCodeError(
        err instanceof Error ? err.message : 'Execution failed',
      );
    } finally {
      setCustomCodeBusy(false);
    }
  }

  function clearCustomCode() {
    setCustomCodeResult(null);
    customCodeResultRef.current = null;
    setCustomCodeError(null);
    recomputeIndicatorsRef.current();
  }

  const toolbarProps: ChartToolbarProps = {
    symbol,
    timeframe,
    showTfDropdown,
    onSelectTimeframe: (tf) => {
      setTimeframe(tf);
      setShowTfDropdown(false);
    },
    onToggleTfDropdown: () => setShowTfDropdown((v) => !v),
    tfDropdownRef,
    onScrollToLatest: () => chartRef.current?.timeScale().scrollToRealTime(),
    onResetZoom: () => chartRef.current?.timeScale().resetTimeScale(),
    showIndicatorsDock,
    onToggleIndicatorsDock: () => setShowIndicatorsDock((v) => !v),
    manualIndicatorsCount: manualIndicators.length,
    showDrawingsList: drawingTools.showDrawingsList,
    onToggleDrawingsList: () => drawingTools.setShowDrawingsList((v) => !v),
    drawingsListCount: drawingsList.length,
    showDrawingToolbar,
    onToggleDrawingToolbar: toggleDrawingToolbar,
    showCustomCodeEditor,
    onToggleCodeEditor: () => {
      setShowCustomCodeEditor((v) => !v);
      setShowStrategyEditor(false);
    },
    showActivityLogDock,
    onToggleActivityLogDock: () => setShowActivityLogDock((v) => !v),
    showOverlaysDropdown,
    onToggleOverlaysDropdown: () => setShowOverlaysDropdown((v) => !v),
    overlaysDropdownRef,
    showSeparators,
    onToggleSeparators: toggleSeparators,
    showSpreadLine,
    onToggleSpreadLine: toggleSpreadLine,
    showVolume,
    onToggleVolume: toggleVolume,
    showTradeLabels,
    onToggleTradeLabels: toggleTradeLabels,
    showTradeMarkers,
    onToggleTradeMarkers: toggleTradeMarkers,
    orderLineVisible: orderLineStyle.visible,
    onToggleOrderLinesVisible: () => updateOrderLineStyle({ visible: !orderLineStyle.visible }),
    showOrderLineSettings,
    onToggleOrderLineSettings: () => setShowOrderLineSettings((v) => !v),
    showZoneColorSettings,
    onToggleZoneColorSettings: () => setShowZoneColorSettings((v) => !v),
    backtestReportId,
    sessionReplayPeriod,
    showSessionReplayPicker,
    onSessionReplayToggle: () => {
      if (sessionReplayPeriod) {
        handleExitSessionReplay();
      } else {
        setShowSessionReplayPicker((v) => !v);
      }
    },
    drawingTool: drawingTools.drawingTool,
    pendingAnchorCount: drawingTools.pendingAnchorCount,
    spreadPoints: chartRenderController.spreadPoints,
    volatilityGuardEnabled: volatilityGuard.config?.enabled ?? null,
    volatilityGuardSaving: volatilityGuard.isSaving,
    onToggleVolatilityGuard: () => {
      if (volatilityGuard.config) volatilityGuard.setEnabled(!volatilityGuard.config.enabled);
    },
  };

  const onToolbarStateChangeRef = useRef(onToolbarStateChange);
  onToolbarStateChangeRef.current = onToolbarStateChange;
  useEffect(() => {
    onToolbarStateChangeRef.current?.(windowIndex, toolbarProps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    windowIndex,
    symbol,
    timeframe,
    showTfDropdown,
    showIndicatorsDock,
    manualIndicators.length,
    drawingTools.showDrawingsList,
    drawingsList.length,
    showDrawingToolbar,
    showCustomCodeEditor,
    showActivityLogDock,
    showOverlaysDropdown,
    showSeparators,
    showSpreadLine,
    showVolume,
    showTradeLabels,
    showTradeMarkers,
    orderLineStyle.visible,
    showOrderLineSettings,
    showZoneColorSettings,
    backtestReportId,
    sessionReplayPeriod,
    showSessionReplayPicker,
    drawingTools.drawingTool,
    drawingTools.pendingAnchorCount,
    chartRenderController.spreadPoints,
    volatilityGuard.config?.enabled,
    volatilityGuard.isSaving,
  ]);

  const onReplayUIChangeRef = useRef(onReplayUIChange);
  onReplayUIChangeRef.current = onReplayUIChange;
  useEffect(() => {
    if (!hideToolbar || sharedReplay?.active) {
      onReplayUIChangeRef.current?.(windowIndex, null);
      return;
    }
    const showPicker = Boolean(showSessionReplayPicker && !sessionReplayPeriod);
    const isReplayActive = Boolean((backtestReportId || sessionReplayPeriod) && replayActive);
    onReplayUIChangeRef.current?.(windowIndex, {
      showPicker,
      pickerProps: showPicker
        ? {
            fromValue: sessionReplayFromInput,
            toValue: sessionReplayToInput,
            onFromChange: setSessionReplayFromInput,
            onToChange: setSessionReplayToInput,
            estimate: sessionReplayEstimate,
            onCancel: () => setShowSessionReplayPicker(false),
            onStart: handleStartSessionReplay,
          }
        : null,
      sessionPeriod: sessionReplayPeriod,
      loadingPage: sessionReplayLoadingPage,
      replayActive: isReplayActive,
      replayControlsProps: isReplayActive
        ? {
            playing: replayPlaying,
            onPlayPause: () => setReplayPlaying((p) => !p),
            onStepBack: () => {
              setReplayPlaying(false);
              seekTo(replayCursorIndexRef.current - 1);
            },
            onStepForward: () => {
              setReplayPlaying(false);
              seekTo(replayCursorIndexRef.current + 1);
            },
            speed: replaySpeed,
            onSpeedChange: setReplaySpeed,
            cursorIndex: replayCursorIndex,
            totalBars: candlesRef.current.length,
            currentTime:
              candlesRef.current[replayCursorIndex]
                ? new Date(candlesRef.current[replayCursorIndex].time * 1000)
                    .toISOString()
                    .replace('T', ' ')
                    .slice(0, 19)
                : '—',
            onSeek: (index) => {
              setReplayPlaying(false);
              seekTo(index);
            },
            following: followingCursor,
            onRecenter: handleRecenterReplay,
          }
        : null,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    windowIndex,
    hideToolbar,
    sharedReplay?.active,
    showSessionReplayPicker,
    sessionReplayPeriod,
    sessionReplayFromInput,
    sessionReplayToInput,
    sessionReplayEstimate,
    sessionReplayLoadingPage,
    backtestReportId,
    replayActive,
    replayPlaying,
    replaySpeed,
    replayCursorIndex,
    followingCursor,
  ]);

  return (
    <section className='flex min-h-0 flex-1 flex-col rounded-md border border-line bg-panel'>
      {hideToolbar && windowCount > 1 && (
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSelectWindow?.(windowIndex);
          }}
          className={`flex items-center justify-between border-b px-3 py-1.5 text-xs select-none cursor-pointer transition-colors duration-150 ${
            selectedWindowIndex === windowIndex
              ? 'bg-accent/15 border-accent/30 text-ink font-semibold shadow-[0_1px_6px_rgba(41,98,255,0.15)]'
              : 'bg-panel/80 border-line text-ink-muted hover:bg-line/20 hover:text-ink'
          }`}
        >
          <div className='flex items-center gap-2'>
            <span
              className={`h-2.5 w-2.5 rounded-full transition-all duration-200 ${
                selectedWindowIndex === windowIndex
                  ? 'bg-accent scale-110 shadow-[0_0_8px_var(--color-accent)] animate-pulse'
                  : 'bg-ink-muted/40'
              }`}
            />
            <span className='font-bold text-ink tracking-wide'>{symbol}</span>
            <span className='rounded bg-bg/80 border border-line/60 px-2 py-0.5 font-mono text-[11px] text-ink-muted'>
              {timeframe}
            </span>
            {selectedWindowIndex === windowIndex && (
              <span className='text-[10px] uppercase font-bold tracking-wider text-accent bg-accent/20 px-1.5 py-0.5 rounded border border-accent/30 shadow-2xs'>
                Active
              </span>
            )}
          </div>
          {windowIndex > 0 && onCloseWindow && (
            <button
              type='button'
              onClick={(e) => {
                e.stopPropagation();
                onCloseWindow(windowIndex);
              }}
              className='cursor-pointer rounded p-1 text-ink-muted hover:bg-err/20 hover:text-err transition-colors'
              title='Close window'
            >
              <X size={14} />
            </button>
          )}
        </div>
      )}
      {!hideToolbar && <ChartToolbar {...toolbarProps} />}
      {error && <p className='px-4 py-1 text-xs text-err'>{error}</p>}
      {!hideToolbar && showSessionReplayPicker && !sessionReplayPeriod && (
        <SessionReplayPicker
          fromValue={sessionReplayFromInput}
          toValue={sessionReplayToInput}
          onFromChange={setSessionReplayFromInput}
          onToChange={setSessionReplayToInput}
          estimate={sessionReplayEstimate}
          onCancel={() => setShowSessionReplayPicker(false)}
          onStart={handleStartSessionReplay}
        />
      )}
      {!hideToolbar && sessionReplayPeriod && (
        <div className='flex items-center gap-2 border-b border-line bg-accent/10 px-4 py-1 text-xs text-accent'>
          <span>
            Session replay —{' '}
            {new Date(sessionReplayPeriod.from * 1000)
              .toISOString()
              .replace('T', ' ')
              .slice(0, 16)}{' '}
            →{' '}
            {new Date(sessionReplayPeriod.to * 1000)
              .toISOString()
              .replace('T', ' ')
              .slice(0, 16)}
            {sessionReplayLoadingPage &&
              ` — loading… page ${sessionReplayLoadingPage.page} (${sessionReplayLoadingPage.loaded.toLocaleString()} candles so far)`}
          </span>
        </div>
      )}
      {backtestReportId && (
        <div className='flex items-center gap-2 border-b border-line bg-accent/10 px-4 py-1 text-xs text-accent'>
          <span>
            Backtest view
            {backtestTrades !== null &&
              ` — ${backtestTrades.length} trade${backtestTrades.length === 1 ? '' : 's'}`}
            {backtestError && ` — ${backtestError}`}
          </span>
          <Link
            href={`/backtest/${encodeURIComponent(backtestReportId)}`}
            className='rounded border border-accent px-2 py-0.5 text-accent hover:bg-accent/20'
          >
            ← Back to report
          </Link>
          {backtestMeta && (
            <button
              className={`cursor-pointer rounded border px-2 py-0.5 ${
                showStrategyEditor
                  ? 'border-accent bg-accent/20 text-accent'
                  : 'border-accent text-accent hover:bg-accent/20'
              }`}
              onClick={() => setShowStrategyEditor((v) => !v)}
              title="Edit this strategy's source code and re-run the backtest in place"
            >
              {showStrategyEditor ? 'Hide code' : 'Edit code'}
            </button>
          )}
          <button
            className={`cursor-pointer rounded border px-2 py-0.5 ${
              showSignalsDock
                ? 'border-accent bg-accent/20 text-accent'
                : 'border-accent text-accent hover:bg-accent/20'
            }`}
            onClick={() => setShowSignalsDock((v) => !v)}
            title="List every signal and trade the strategy emitted — click one to jump to it on the chart"
          >
            Signals & Trades
            {backtestSignals !== null && backtestTrades !== null && ` (${backtestSignals.length} / ${backtestTrades.length})`}
          </button>
          <button
            className={`flex cursor-pointer items-center gap-1 rounded border px-2 py-0.5 ${
              replayActive
                ? 'border-accent bg-accent/20 text-accent'
                : 'border-accent text-accent hover:bg-accent/20'
            }`}
            onClick={replayActive ? handleExitReplay : handleEnterReplay}
            title="Watch this backtest unfold bar by bar — candles, indicators, trades and the activity log revealed progressively, like a live session"
          >
            {replayActive ? (
              <>
                <Square size={12} fill="currentColor" /> Exit replay
              </>
            ) : (
              <>
                <Play size={12} fill="currentColor" /> Replay
              </>
            )}
          </button>
          {onExitBacktestView && (
            <button
              className='ml-auto cursor-pointer rounded border border-accent px-2 py-0.5 text-accent hover:bg-accent/20'
              onClick={onExitBacktestView}
            >
              Exit backtest view
            </button>
          )}
        </div>
      )}
      {/* Live-bot "eye" view banner — same Signals & Trades dock as backtest
          view, fed live data instead; click the bot's eye again to exit. */}
      {liveBotSkill && (
        <div className='flex items-center gap-2 border-b border-line bg-accent/10 px-4 py-1 text-xs text-accent'>
          <span>
            Bot view — {liveBotSkill.split('/').pop()}
            {backtestTrades !== null &&
              ` — ${backtestTrades.length} closed trade${backtestTrades.length === 1 ? '' : 's'}`}
          </span>
          <button
            className={`cursor-pointer rounded border px-2 py-0.5 ${
              showSignalsDock
                ? 'border-accent bg-accent/20 text-accent'
                : 'border-accent text-accent hover:bg-accent/20'
            }`}
            onClick={() => setShowSignalsDock((v) => !v)}
            title="List every signal and trade this bot emitted, plus its strategy's indicators — click a signal or trade to jump to it on the chart"
          >
            Signals & Trades
            {backtestSignals !== null && backtestTrades !== null && (
              <>
                {` (${backtestSignals.length} / ${backtestTrades.length}`}
                {liveBotIndicators !== null ? ` / ${liveBotIndicators.length} ind.)` : ')'}
              </>
            )}
          </button>
        </div>
      )}
      {/* Replay player — shown while replaying a backtest report (§F) or a
          session-replay period */}
      {!hideToolbar && (backtestReportId || sessionReplayPeriod) && replayActive && (
        <ReplayControls
          playing={replayPlaying}
          onPlayPause={() => setReplayPlaying((p) => !p)}
          onStepBack={() => {
            setReplayPlaying(false);
            seekTo(replayCursorIndexRef.current - 1);
          }}
          onStepForward={() => {
            setReplayPlaying(false);
            seekTo(replayCursorIndexRef.current + 1);
          }}
          speed={replaySpeed}
          onSpeedChange={setReplaySpeed}
          cursorIndex={replayCursorIndex}
          totalBars={candlesRef.current.length}
          currentTime={
            candlesRef.current[replayCursorIndex]
              ? new Date(candlesRef.current[replayCursorIndex].time * 1000)
                  .toISOString()
                  .replace('T', ' ')
                  .slice(0, 19)
              : '—'
          }
          onSeek={(index) => {
            setReplayPlaying(false);
            seekTo(index);
          }}
          following={followingCursor}
          onRecenter={handleRecenterReplay}
          tickForm={tickForm}
          onToggleTickForm={() => setTickForm((t) => !t)}
          finerAvailable={finerAvailable}
        />
      )}
      {/* Strategy info — collapsed by default, shows only name + toggle */}
      {activeStrategy?.spec &&
        (activeStrategy.spec.unrecognized_indicators.length > 0 ||
          activeStrategy.spec.chart_notes.length > 0) && (
          <div className='flex flex-col border-b border-line text-xs'>
            <div className='flex items-center gap-2 px-4 py-1'>
              <span className='font-semibold text-ink'>
                {activeStrategy.name}
              </span>
              <span className='text-ink-muted/70 text-[10px]'>
                {activeStrategy.spec.unrecognized_indicators.length +
                  activeStrategy.spec.chart_notes.length}{' '}
                item(s) not auto-drawn
              </span>
              <button
                onClick={() => setStrategyInfoExpanded((v) => !v)}
                className='ml-auto cursor-pointer rounded border border-line px-2 py-0.5 text-[10px] text-ink-muted hover:border-accent hover:text-accent transition-colors'
                title={
                  strategyInfoExpanded
                    ? 'Collapse strategy info'
                    : 'Expand strategy info'
                }
              >
                {strategyInfoExpanded ? '▲ Hide info' : '▼ Show info'}
              </button>
            </div>
            {strategyInfoExpanded && (
              <div className='flex flex-wrap items-center gap-1.5 border-t border-line px-4 py-1.5 text-ink-muted bg-panel/50'>
                <span className='text-ink-muted/70 mr-1'>
                  Mentions (not auto-drawn):
                </span>
                {activeStrategy.spec.unrecognized_indicators.map((name) => (
                  <span
                    key={`ind:${name}`}
                    className='rounded border border-line px-1.5 py-0.5 bg-panel'
                    title='Indicator outside the 5 plottable families (EMA/SMA/RSI/MACD/Bollinger)'
                  >
                    {name}
                  </span>
                ))}
                {activeStrategy.spec.chart_notes.map((note) => (
                  <span
                    key={`note:${note}`}
                    className='rounded border border-line px-1.5 py-0.5 bg-panel'
                    title='No explicit price level in the source document — not turned into chart geometry'
                  >
                    {note}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      {/* Order lines style panel — shown when the toggle is active */}
      {showOrderLineSettings && (
        <div className='flex flex-wrap items-center gap-3 border-b border-line bg-panel px-3 py-1.5 text-xs'>
          <span className='text-ink-muted'>Trade open/close lines</span>
          <label className='flex items-center gap-1.5'>
            <span className='text-ink-muted'>Type</span>
            <select
              value={orderLineStyle.dash}
              onChange={(e) =>
                updateOrderLineStyle({ dash: e.target.value as OrderLineDash })
              }
              className='cursor-pointer rounded border border-line bg-panel px-1 py-0.5 text-ink'
            >
              <option value='solid'>Solid</option>
              <option value='dashed'>Dashed</option>
              <option value='dotted'>Dotted</option>
            </select>
          </label>
          <label className='flex items-center gap-1.5'>
            <span className='text-ink-muted'>Size</span>
            <select
              value={orderLineStyle.width}
              onChange={(e) =>
                updateOrderLineStyle({
                  width: Number(e.target.value) as OrderLineStyle['width'],
                })
              }
              className='cursor-pointer rounded border border-line bg-panel px-1 py-0.5 text-ink'
            >
              <option value={1}>1px</option>
              <option value={2}>2px</option>
              <option value={3}>3px</option>
              <option value={4}>4px</option>
            </select>
          </label>
          <label className='flex items-center gap-1.5 cursor-pointer'>
            <input
              type='checkbox'
              checked={orderLineStyle.customColors}
              onChange={(e) => updateOrderLineStyle({ customColors: e.target.checked })}
            />
            <span className='text-ink-muted'>Custom colors</span>
          </label>
          {orderLineStyle.customColors && (
            <>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Open</span>
                <input
                  type='color'
                  value={orderLineStyle.openColor}
                  onChange={(e) => updateOrderLineStyle({ openColor: e.target.value })}
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Close</span>
                <input
                  type='color'
                  value={orderLineStyle.closeColor}
                  onChange={(e) => updateOrderLineStyle({ closeColor: e.target.value })}
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
            </>
          )}

          <div className='h-3.5 w-px bg-line/80 my-auto mx-1' />

          {/* Oblique trade path line controls */}
          <label className='flex items-center gap-1.5 cursor-pointer font-medium text-ink'>
            <input
              type='checkbox'
              checked={orderLineStyle.showExitLine}
              onChange={(e) => updateOrderLineStyle({ showExitLine: e.target.checked })}
            />
            <span>Oblique exit path line</span>
          </label>
          {orderLineStyle.showExitLine && (
            <>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Type</span>
                <select
                  value={orderLineStyle.exitLineDash}
                  onChange={(e) =>
                    updateOrderLineStyle({ exitLineDash: e.target.value as OrderLineDash })
                  }
                  className='cursor-pointer rounded border border-line bg-panel px-1 py-0.5 text-ink'
                >
                  <option value='solid'>Solid</option>
                  <option value='dashed'>Dashed</option>
                  <option value='dotted'>Dotted</option>
                </select>
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Size</span>
                <select
                  value={orderLineStyle.exitLineWidth}
                  onChange={(e) =>
                    updateOrderLineStyle({
                      exitLineWidth: Number(e.target.value) as OrderLineStyle['width'],
                    })
                  }
                  className='cursor-pointer rounded border border-line bg-panel px-1 py-0.5 text-ink'
                >
                  <option value={1}>1px</option>
                  <option value={2}>2px</option>
                  <option value={3}>3px</option>
                  <option value={4}>4px</option>
                </select>
              </label>
              <label className='flex items-center gap-1.5 cursor-pointer'>
                <input
                  type='checkbox'
                  checked={orderLineStyle.exitLineCustomColor}
                  onChange={(e) => updateOrderLineStyle({ exitLineCustomColor: e.target.checked })}
                />
                <span className='text-ink-muted'>Custom colors</span>
              </label>
              {orderLineStyle.exitLineCustomColor && (
                <>
                  <label className='flex items-center gap-1.5'>
                    <span className='text-ink-muted'>Win</span>
                    <input
                      type='color'
                      value={orderLineStyle.exitLineWinColor}
                      onChange={(e) => updateOrderLineStyle({ exitLineWinColor: e.target.value })}
                      className='h-4 w-6 cursor-pointer'
                    />
                  </label>
                  <label className='flex items-center gap-1.5'>
                    <span className='text-ink-muted'>Loss</span>
                    <input
                      type='color'
                      value={orderLineStyle.exitLineLossColor}
                      onChange={(e) => updateOrderLineStyle({ exitLineLossColor: e.target.value })}
                      className='h-4 w-6 cursor-pointer'
                    />
                  </label>
                </>
              )}
            </>
          )}
        </div>
      )}
      {/* Zone colors settings panel — shown when the toggle is active. One
          "Custom zone colors" checkbox gates all the per-indicator pickers
          below, same shape as the order-lines panel's `customColors` above;
          off (the default) keeps today's hardcoded buy/sell/muted theme
          colors — see `pickZoneColor` in chartFormat.ts. */}
      {showZoneColorSettings && (
        <div className='flex flex-wrap items-center gap-3 border-b border-line bg-panel px-3 py-1.5 text-xs'>
          <span className='text-ink-muted'>Zone rectangle colors</span>
          <label className='flex items-center gap-1.5 cursor-pointer font-medium text-ink'>
            <input
              type='checkbox'
              checked={zoneColorStyle.customColors}
              onChange={(e) => updateZoneColorStyle({ customColors: e.target.checked })}
            />
            <span>Custom zone colors</span>
          </label>
          {zoneColorStyle.customColors && (
            <>
              <div className='h-3.5 w-px bg-line/80 my-auto mx-1' />
              <span className='text-ink-muted font-medium'>Quasimodo</span>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Demand</span>
                <input
                  type='color'
                  value={zoneColorStyle.qml.demandColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ qml: { ...zoneColorStyle.qml, demandColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Supply</span>
                <input
                  type='color'
                  value={zoneColorStyle.qml.supplyColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ qml: { ...zoneColorStyle.qml, supplyColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Touched</span>
                <input
                  type='color'
                  value={zoneColorStyle.qml.touchedColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ qml: { ...zoneColorStyle.qml, touchedColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>

              <div className='h-3.5 w-px bg-line/80 my-auto mx-1' />
              <span className='text-ink-muted font-medium'>S&D v1</span>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Demand</span>
                <input
                  type='color'
                  value={zoneColorStyle.snd.demandColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ snd: { ...zoneColorStyle.snd, demandColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Supply</span>
                <input
                  type='color'
                  value={zoneColorStyle.snd.supplyColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ snd: { ...zoneColorStyle.snd, supplyColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Touched</span>
                <input
                  type='color'
                  value={zoneColorStyle.snd.touchedColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ snd: { ...zoneColorStyle.snd, touchedColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>

              <div className='h-3.5 w-px bg-line/80 my-auto mx-1' />
              <span className='text-ink-muted font-medium'>S&D v2</span>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Demand</span>
                <input
                  type='color'
                  value={zoneColorStyle.sndV2.demandColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ sndV2: { ...zoneColorStyle.sndV2, demandColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Supply</span>
                <input
                  type='color'
                  value={zoneColorStyle.sndV2.supplyColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ sndV2: { ...zoneColorStyle.sndV2, supplyColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Touched</span>
                <input
                  type='color'
                  value={zoneColorStyle.sndV2.touchedColor}
                  onChange={(e) =>
                    updateZoneColorStyle({ sndV2: { ...zoneColorStyle.sndV2, touchedColor: e.target.value } })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>

              <div className='h-3.5 w-px bg-line/80 my-auto mx-1' />
              <span className='text-ink-muted font-medium'>Trade zone</span>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Demand</span>
                <input
                  type='color'
                  value={zoneColorStyle.tradeZone.demandColor}
                  onChange={(e) =>
                    updateZoneColorStyle({
                      tradeZone: { ...zoneColorStyle.tradeZone, demandColor: e.target.value },
                    })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
              <label className='flex items-center gap-1.5'>
                <span className='text-ink-muted'>Supply</span>
                <input
                  type='color'
                  value={zoneColorStyle.tradeZone.supplyColor}
                  onChange={(e) =>
                    updateZoneColorStyle({
                      tradeZone: { ...zoneColorStyle.tradeZone, supplyColor: e.target.value },
                    })
                  }
                  className='h-4 w-6 cursor-pointer'
                />
              </label>
            </>
          )}
        </div>
      )}
      {/* Drawings list panel — shown when the toggle is active */}
      {drawingTools.showDrawingsList && (
        <DrawingsList
          drawings={drawingsList}
          onRemove={drawingTools.removeDrawing}
          onToggleVisible={drawingTools.toggleDrawingVisible}
          onColorChange={drawingTools.handleModifyDrawingColor}
        />
      )}
      {/* Indicators dock — shown when the toggle is active */}
      {showIndicatorsDock && (
        <IndicatorsDock
          indicators={manualIndicators}
          onAdd={addManualIndicator}
          onRemove={removeManualIndicator}
          onUpdate={updateManualIndicator}
          onCustomIndicatorCodeSaved={() => computeCustomIndicatorsRef.current()}
        />
      )}
      {/* Activity log dock — shown when the toggle is active. During replay,
          feeds the backtest report's own activity log (its persisted trail
          of signals/vetoes/fills, simulated-clock timestamps) filtered up
          to the cursor bar, instead of the default live/global poll. */}
      {/* Signals dock has been moved to the right side of the chart */}
      {showActivityLogDock && (
        <ActivityLogDock
          symbol={symbol}
          replayEntries={
            replayActive && backtestActivityLog
              ? backtestActivityLog
                  .filter(
                    (e) =>
                      e.time <=
                      ((candlesRef.current[replayCursorIndex]?.time as
                        | number
                        | undefined) ?? 0),
                  )
                  .map((e, i) => ({
                    id: i,
                    created_at: e.time,
                    level: e.level,
                    logger: e.logger,
                    message: e.message,
                  }))
              : undefined
          }
        />
      )}
      <div className='relative min-h-0 flex-1 flex flex-row overflow-hidden'>
        <div className='relative flex-1 min-w-0 h-full'>
          <div ref={containerRef} className='h-full w-full' />
        {/* Strategy code drawer — slides in from the configured edge */}
        {backtestReportId && backtestMeta && showStrategyEditor && (
          <div
            className={`pointer-events-auto absolute z-40 flex flex-col bg-panel border-line shadow-2xl overflow-hidden ${
              drawerPosition === 'right'
                ? 'right-0 top-0 h-full border-l'
                : drawerPosition === 'left'
                  ? 'left-0 top-0 h-full border-r'
                  : drawerPosition === 'bottom'
                    ? 'bottom-0 left-0 w-full border-t'
                    : 'top-0 left-0 w-full border-b'
            }`}
            style={{
              width:
                drawerPosition === 'right' || drawerPosition === 'left'
                  ? '420px'
                  : '100%',
              height:
                drawerPosition === 'bottom' || drawerPosition === 'top'
                  ? '340px'
                  : '100%',
              maxWidth:
                drawerPosition === 'right' || drawerPosition === 'left'
                  ? '55%'
                  : undefined,
              maxHeight:
                drawerPosition === 'bottom' || drawerPosition === 'top'
                  ? '55%'
                  : undefined,
            }}
          >
            {/* Drawer header */}
            <div className='flex items-center gap-2 border-b border-line px-3 py-1.5 bg-panel shrink-0'>
              <span className='text-xs font-semibold text-ink truncate'>
                Edit Strategy Code
              </span>
              {/* Position controls */}
              <div className='flex items-center gap-0.5 ml-auto'>
                {(['right', 'bottom', 'left', 'top'] as const).map((pos) => (
                  <button
                    key={pos}
                    onClick={() => setDrawerPosition(pos)}
                    title={`Move to ${pos}`}
                    className={`cursor-pointer rounded px-1.5 py-0.5 text-[10px] border transition-colors ${
                      drawerPosition === pos
                        ? 'border-accent text-accent bg-accent/10'
                        : 'border-line text-ink-muted hover:text-ink'
                    }`}
                  >
                    {pos === 'right'
                      ? '⇥'
                      : pos === 'left'
                        ? '⇤'
                        : pos === 'bottom'
                          ? '⇓'
                          : '⇑'}
                  </button>
                ))}
                <button
                  onClick={() => setShowStrategyEditor(false)}
                  className='cursor-pointer ml-1 rounded border border-line px-1.5 py-0.5 text-[10px] text-ink-muted hover:border-err hover:text-err transition-colors'
                  title='Close drawer'
                >
                  ✕
                </button>
              </div>
            </div>
            {/* Drawer content */}
            <div className='flex-1 flex flex-col p-2 min-h-0'>
              <BacktestStrategyEditor
                strategyName={backtestMeta.strategy}
                symbol={backtestMeta.symbol}
                period={backtestMeta.period}
                className='flex-1 min-h-0'
                onSaved={(newReportId) => {
                  setShowStrategyEditor(false);
                  onReportChange?.(newReportId);
                }}
                onRunPreview={runCustomCode}
                previewBusy={customCodeBusy}
                previewError={customCodeError}
                previewResult={customCodeResult}
                onResetPreview={clearCustomCode}
              />
            </div>
          </div>
        )}
        {/* Custom script code drawer — slides in from the configured edge */}
        {showCustomCodeEditor && (
          <CustomCodeDrawer
            drawerPosition={drawerPosition}
            setDrawerPosition={setDrawerPosition}
            customCodeDraft={customCodeDraft}
            setCustomCodeDraft={setCustomCodeDraft}
            customCodeCopied={customCodeCopied}
            handleCopyCustomCode={handleCopyCustomCode}
            customCodeBusy={customCodeBusy}
            customCodeError={customCodeError}
            customCodeResult={customCodeResult}
            runCustomCode={() => runCustomCode()}
            clearCustomCode={clearCustomCode}
            onClose={() => {
              setShowCustomCodeEditor(false);
              clearCustomCode();
            }}
          />
        )}
        {/* Drawing toolbar — floats on the left edge of the chart canvas,
            toggled via the toolbar's "Tools" button (showDrawingToolbar) */}
        {showDrawingToolbar && (
          <DrawingToolbar
            activeTool={drawingTool}
            onToolSelect={setDrawingTool}
            onClearAll={drawingTools.clearAllDrawings}
            activeColor={activeColor}
            onColorChange={drawingTools.handleColorChange}
          />
        )}
        {contextMenu && (
          <ChartContextMenu
            ref={contextMenuRef}
            x={contextMenu.x}
            y={contextMenu.y}
            price={contextMenu.price}
            containerWidth={contextMenu.containerWidth}
            containerHeight={contextMenu.containerHeight}
            onSelectOption={(side, type) => {
              setOrderPopover({
                x: contextMenu.x,
                y: contextMenu.y,
                price: contextMenu.price,
                side,
                orderType: type,
                containerWidth: contextMenu.containerWidth,
                containerHeight: contextMenu.containerHeight,
              });
              setContextMenu(null);
            }}
          />
        )}
        {orderPopover && (
          <ChartOrderPopover
            ref={orderPopoverRef}
            x={orderPopover.x}
            y={orderPopover.y}
            price={orderPopover.price}
            side={orderPopover.side}
            orderType={orderPopover.orderType}
            containerWidth={orderPopover.containerWidth}
            containerHeight={orderPopover.containerHeight}
            busy={editBusy}
            onClose={() => setOrderPopover(null)}
            onPlace={async (volume, price, sl, tp) => {
              await trading.placePending(
                orderPopover.side,
                orderPopover.orderType,
                volume,
                price,
                sl,
                tp,
              );
            }}
          />
        )}
        {drawingContextMenu && (
          <DrawingContextMenu
            ref={drawingTools.drawingContextMenuRef}
            x={drawingContextMenu.x}
            y={drawingContextMenu.y}
            drawingType={drawingContextMenu.drawingType}
            containerWidth={drawingContextMenu.containerWidth}
            containerHeight={drawingContextMenu.containerHeight}
            onSelectEdit={() => {
              setDrawingEditPopover({
                x: drawingContextMenu.x,
                y: drawingContextMenu.y,
                drawingId: drawingContextMenu.drawingId,
                drawingType: drawingContextMenu.drawingType,
                containerWidth: drawingContextMenu.containerWidth,
                containerHeight: drawingContextMenu.containerHeight,
              });
              setDrawingContextMenu(null);
            }}
            onDelete={() => {
              drawingTools.removeDrawing(drawingContextMenu.drawingId);
              setDrawingContextMenu(null);
            }}
          />
        )}
        {drawingEditPopover && (
          <DrawingEditPopover
            ref={drawingTools.drawingEditPopoverRef}
            x={drawingEditPopover.x}
            y={drawingEditPopover.y}
            drawingId={drawingEditPopover.drawingId}
            drawingType={drawingEditPopover.drawingType}
            containerWidth={drawingEditPopover.containerWidth}
            containerHeight={drawingEditPopover.containerHeight}
            manager={drawingManagerRef.current}
            originalStylesRef={originalStylesRef}
            onClose={() => setDrawingEditPopover(null)}
            onSaveAndSync={saveAndSyncRef.current}
            onColorChange={drawingTools.handleModifyDrawingColor}
            onWidthChange={drawingTools.handleModifyDrawingWidth}
          />
        )}
        <ReplayRevealOverlay
          events={replayRevealEvents}
          chartRef={chartRef}
          candleSeriesRef={candleSeriesRef}
        />
        {signalTooltip && (
          <SignalInfoPopover
            x={signalTooltip.x}
            y={signalTooltip.y}
            signals={signalTooltip.signals}
            containerWidth={signalTooltip.containerWidth}
            containerHeight={signalTooltip.containerHeight}
            onClose={() => setSignalTooltip(null)}
          />
        )}
        {zoneTooltip && (
          <ZoneInfoPopover
            x={zoneTooltip.x}
            y={zoneTooltip.y}
            meta={zoneTooltip.meta}
            containerWidth={zoneTooltip.containerWidth}
            containerHeight={zoneTooltip.containerHeight}
            onClose={() => setZoneTooltip(null)}
          />
        )}
        {newsBands.map((b) => {
          const color = cssVar(
            b.phase === 'pre' ? '--color-err' : '--color-accent',
          );
          return (
            <div
              key={b.key}
              className='pointer-events-none absolute top-0 h-full border-x border-dashed'
              style={{
                left: b.left,
                width: b.width,
                backgroundColor: hexToRgba(color, 0.1),
                borderColor: color,
              }}
              title={`${b.label} (${b.phase}-event news window)`}
            />
          );
        })}
        {loadingMore && (
          <div className='pointer-events-none absolute left-2 top-2 rounded border border-line bg-panel px-2 py-1 text-xs text-ink-muted'>
            Loading history…
          </div>
        )}
        {switchingChart && (
          // The chart keeps rendering the *previous* symbol/timeframe's bars
          // until fresh candles land (there's no cheap way to blank a
          // lightweight-charts series without a flash) — without this badge
          // a switch looks like it silently did nothing until data arrives.
          <div className='pointer-events-none absolute left-1/2 top-2 z-10 -translate-x-1/2 rounded border border-line bg-panel px-2 py-1 text-xs text-ink-muted shadow-sm'>
            Loading {timeframe}…
          </div>
        )}
        {(() => {
          const priceScaleWidth = candleSeriesRef.current?.priceScale().width() || 65;
          const priceLineSpecs = orderLineStyle.visible ? buildPriceLines() : [];
          const entryLineSpecs = orderLineStyle.visible ? buildEntryLines() : [];

          const highlightActive =
            activeHighlightedTicket !== null &&
            (priceLineSpecs.some((s) => s.ticket === activeHighlightedTicket) ||
              entryLineSpecs.some((s) => s.position.ticket === activeHighlightedTicket));

          // Translucent Risk/Reward bands for highlighted position / order
          let riskRewardBands: React.ReactNode = null;
          if (highlightActive && activeHighlightedTicket !== null) {
            const pos = trading.positions.find((p) => p.ticket === activeHighlightedTicket);
            const pend = trading.pendingOrders.find((o) => o.ticket === activeHighlightedTicket);
            const entryPrice = pos ? pos.open_price : pend ? pend.price : null;
            const slPrice = pos ? pos.sl : pend ? pend.sl : null;
            const tpPrice = pos ? pos.tp : pend ? pend.tp : null;
            const series = candleSeriesRef.current;

            if (entryPrice !== null && series) {
              const yEntry = series.priceToCoordinate(entryPrice);
              const ySl = slPrice !== null ? series.priceToCoordinate(slPrice) : null;
              const yTp = tpPrice !== null ? series.priceToCoordinate(tpPrice) : null;

              riskRewardBands = (
                <>
                  {ySl !== null && yEntry !== null && (
                    <div
                      className='pointer-events-none absolute w-56 z-0 bg-err/10 border-r-4 border-err/50 transition-all rounded-l'
                      style={{
                        top: `${Math.min(yEntry, ySl)}px`,
                        height: `${Math.abs(yEntry - ySl)}px`,
                        right: `${priceScaleWidth}px`,
                      }}
                    />
                  )}
                  {yTp !== null && yEntry !== null && (
                    <div
                      className='pointer-events-none absolute w-56 z-0 bg-ok/10 border-r-4 border-ok/50 transition-all rounded-l'
                      style={{
                        top: `${Math.min(yEntry, yTp)}px`,
                        height: `${Math.abs(yEntry - yTp)}px`,
                        right: `${priceScaleWidth}px`,
                      }}
                    />
                  )}
                </>
              );
            }
          }

          const dashClass =
            orderLineStyle.dash === 'solid'
              ? 'border-solid'
              : orderLineStyle.dash === 'dotted'
                ? 'border-dotted'
                : 'border-dashed';

          const priceLines = priceLineSpecs.map((spec) => {
            const dragging = drag?.key === spec.key;
            const price = dragging ? drag.price : spec.price;
            const top = candleSeriesRef.current?.priceToCoordinate(price);
            if (top === null || top === undefined) return null;
            // Placeholders (no sl/tp set yet) render faint until dragged/clicked
            // — once that happens `dragging` takes over the "live" style so the
            // user gets feedback that it's now a real, about-to-commit value.
            const faint = spec.placeholder && !dragging;
            const selected = highlightActive && spec.ticket === activeHighlightedTicket;
            const dimmed = highlightActive && !selected;
            return (
              <div
                key={spec.key}
                className='pointer-events-auto absolute left-0 h-4 -translate-y-1/2 cursor-ns-resize z-10 flex items-center select-none'
                style={{
                  top: `${top}px`,
                  right: `${priceScaleWidth}px`,
                  opacity: dimmed ? 0.3 : faint ? 0.45 : 1,
                }}
                onMouseDown={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const wasSelected = activeHighlightedTicket === spec.ticket;
                  dragStartRef.current = {
                    x: e.clientX,
                    y: e.clientY,
                    ticket: spec.ticket,
                    wasSelected,
                  };
                  if (!wasSelected) {
                    handleTicketSelect(spec.ticket);
                  }
                  setDrag({
                    key: spec.key,
                    price: spec.price,
                    commit: spec.commit,
                  });
                }}
              >
                <div
                  className={`w-full border-t ${dashClass}`}
                  style={{
                    borderColor: spec.color,
                    borderTopWidth: `${selected ? Math.max(orderLineStyle.width + 1, 3) : orderLineStyle.width}px`,
                    filter: selected ? `drop-shadow(0 0 6px ${spec.color})` : undefined,
                  }}
                />
                {spec.pnlOpenPrice !== undefined && spec.pnlSide !== undefined && spec.pnlVolume !== undefined && (() => {
                  const contractSize = symbolInfo?.contract_size ?? 1;
                  const rawPnl = (price - spec.pnlOpenPrice!) * (spec.pnlSide === 'buy' ? 1 : -1) * contractSize * spec.pnlVolume!;
                  const pnl = Math.abs(rawPnl) < 0.005 ? 0 : rawPnl;
                  const pnlLabel = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}$`;
                  return (
                    <div
                      className={`absolute left-2 top-1/2 -translate-y-1/2 rounded px-1 text-[10px] font-bold ${
                        selected ? 'px-2 py-0.5 text-xs ring-2 ring-white shadow-md' : ''
                      }`}
                      style={{
                        backgroundColor: spec.color,
                        color: '#ffffff',
                        opacity: faint ? 0.7 : 1,
                        boxShadow: selected ? `0 0 8px ${spec.color}` : undefined,
                      }}
                      title={`Estimated P/L for ${spec.pnlVolume} lot (${spec.pnlSide})`}
                    >
                      {pnlLabel}
                    </div>
                  );
                })()}
                <div
                  className={`absolute right-2 top-1/2 -translate-y-1/2 rounded px-1 text-[10px] font-bold ${
                    selected ? 'px-2 py-0.5 text-xs ring-2 ring-white shadow-md' : ''
                  }`}
                  style={{
                    backgroundColor: spec.color,
                    color: '#ffffff',
                    opacity: faint ? 0.7 : 1,
                    boxShadow: selected ? `0 0 8px ${spec.color}` : undefined,
                  }}
                  title={
                    spec.placeholder
                      ? 'Drag to set — not saved yet'
                      : 'Drag to modify'
                  }
                >
                  {dragging ? price.toFixed(5) : spec.label}
                </div>
              </div>
            );
          });

          const entryLines = entryLineSpecs.map((spec) => {
            const top = candleSeriesRef.current?.priceToCoordinate(
              spec.position.open_price,
            );
            if (top === null || top === undefined) return null;
            const selected = highlightActive && spec.position.ticket === activeHighlightedTicket;
            const dimmed = highlightActive && !selected;
            return (
              <div
                key={spec.key}
                className='pointer-events-auto absolute left-0 h-4 -translate-y-1/2 cursor-pointer z-10 flex items-center select-none'
                style={{ top: `${top}px`, right: `${priceScaleWidth}px`, opacity: dimmed ? 0.3 : 1 }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  handleTicketSelect(spec.position.ticket);
                }}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  setEditingTicket(spec.position.ticket);
                }}
              >
                <div
                  className={`w-full border-t ${dashClass}`}
                  style={{
                    borderColor: spec.color,
                    borderTopWidth: `${selected ? Math.max(orderLineStyle.width + 1, 3) : Math.max(orderLineStyle.width, 2)}px`,
                    filter: selected ? `drop-shadow(0 0 6px ${spec.color})` : undefined,
                  }}
                />
                <div
                  className={`absolute left-2 top-1/2 -translate-y-1/2 rounded px-1 text-[10px] font-bold ${
                    selected ? 'px-2 py-0.5 text-xs ring-2 ring-white shadow-md' : ''
                  }`}
                  style={{
                    backgroundColor: spec.color,
                    color: '#ffffff',
                    boxShadow: selected ? `0 0 8px ${spec.color}` : undefined,
                  }}
                  title='Double-click to modify this position'
                >
                  {spec.label}
                </div>
              </div>
            );
          });

          return (
            <>
              {riskRewardBands}
              {priceLines}
              {entryLines}
            </>
          );
        })()}
        {[...buildSelectedTradeLines(), ...buildHistoryTradeLines()].map((spec) => {
          const top = candleSeriesRef.current?.priceToCoordinate(spec.price);
          const priceScaleWidth = candleSeriesRef.current?.priceScale().width() || 65;
          if (top === null || top === undefined) return null;
          return (
            <div
              key={spec.key}
              className='pointer-events-none absolute left-0 h-4 -translate-y-1/2 z-10 flex items-center select-none'
              style={{ top: `${top}px`, right: `${priceScaleWidth}px` }}
            >
              <div
                className='w-full border-t-[3px] border-dashed'
                style={{ borderColor: spec.color, filter: `drop-shadow(0 0 4px ${spec.color})` }}
              />
              {spec.leftLabel && (
                <div
                  className='absolute left-2 top-1/2 -translate-y-1/2 rounded px-1.5 py-0.5 text-[11px] font-bold ring-2 ring-ink'
                  style={{ backgroundColor: spec.color, color: '#ffffff', boxShadow: `0 0 6px ${spec.color}` }}
                >
                  {spec.leftLabel}
                </div>
              )}
              {spec.rightLabel && (
                <div
                  className='absolute right-2 top-1/2 -translate-y-1/2 rounded px-1.5 py-0.5 text-[11px] font-bold ring-2 ring-ink'
                  style={{ backgroundColor: spec.color, color: '#ffffff', boxShadow: `0 0 6px ${spec.color}` }}
                >
                  {spec.rightLabel}
                </div>
              )}
            </div>
          );
        })}
        {(() => {
          // Highlighted history-row "zone": a low-opacity rectangle spanning
          // the trade's own open->close time window (x-axis) and its SL/TP
          // price band (y-axis, falling back to the open/close price path
          // when SL or TP wasn't set) — gives a clicked TradeHistoryTable row
          // a visible "where this traded" box on the chart, on top of the
          // exact-price dashed lines `buildHistoryTradeLines` already draws.
          // Deliberately a plain absolute div (not a `Rectangle` drawing
          // primitive) so it recomputes every render exactly like the
          // backtest openClose segments below, instead of needing its own
          // drawingManager add/remove lifecycle.
          const t = findHistoryTrade();
          if (!t) return null;
          const chart = chartRef.current;
          const series = candleSeriesRef.current;
          if (!chart || !series) return null;
          const candles = candlesRef.current;

          // If close time or price are not set, it means the trade is open.
          // We span the rectangle to the latest candle on the chart.
          const lastCandle = candles[candles.length - 1];
          const closeTime = t.close_time ?? (lastCandle ? lastCandle.time : Math.floor(Date.now() / 1000));
          const closePrice = t.close_price ?? (lastCandle ? lastCandle.close : t.open_price);

          const openSnap = nearestCandleTime(candles, t.open_time);
          const closeSnap = nearestCandleTime(candles, closeTime);
          if (openSnap === null || closeSnap === null) return null;
          const x1 = chart.timeScale().timeToCoordinate(openSnap);
          const x2 = chart.timeScale().timeToCoordinate(closeSnap);
          if (x1 === null || x2 === null) return null;

          const prices = [t.open_price, closePrice, t.sl, t.tp].filter(
            (p): p is number => p !== null,
          );
          const yTop = series.priceToCoordinate(Math.max(...prices));
          const yBottom = series.priceToCoordinate(Math.min(...prices));
          if (yTop === null || yBottom === null) return null;
          const color = (t.profit ?? 0) >= 0 ? cssVar('--color-ok') : cssVar('--color-err');

          // Make fill color opacity more transparent to see the candles behind (e.g. 0.04), and highlighted border (0.8 opacity + glow)
          const fillOpacity = 0.04;
          const borderOpacity = 0.8;
          const glowOpacity = 0.5;
          const borderStyle = t.close_time === null ? 'dashed' : 'solid';

          return (
            <div
              key={`history-zone-${t.id}`}
              className='pointer-events-none absolute z-0 rounded border-2 transition-all'
              style={{
                left: `${Math.min(x1, x2)}px`,
                width: `${Math.max(1, Math.abs(x2 - x1))}px`,
                top: `${Math.min(yTop, yBottom)}px`,
                height: `${Math.max(1, Math.abs(yBottom - yTop))}px`,
                backgroundColor: hexToRgba(color, fillOpacity),
                borderColor: hexToRgba(color, borderOpacity),
                borderStyle: borderStyle,
                boxShadow: `0 0 12px ${hexToRgba(color, glowOpacity)}`,
              }}
              title={`#${t.id} zone (${t.close_time === null ? 'OPEN' : 'CLOSED'}): ${t.side.toUpperCase()} ${t.open_price} → ${t.close_price ?? 'current'}${
                t.sl !== null ? `, SL ${t.sl}` : ''
              }${t.tp !== null ? `, TP ${t.tp}` : ''}`}
            />
          );
        })()}
        {(() => {
          // "Opened here" callout for a clicked active order or trade-history
          // row (`findHistoryTrade`, same conduit the history-zone rectangle
          // above and `navigateToTime`'s auto-scroll both use) — a pulsing
          // ping-ring + solid dot exactly on the opening candle/price, plus a
          // blinking side label with a bouncing down-arrow pointing at it, so
          // the spot is unmistakable even once the chart has auto-centered on
          // it (the plain series-marker arrow from `toSeriesMarkers` can be
          // easy to miss at a glance).
          const t = findHistoryTrade();
          if (!t) return null;
          const chart = chartRef.current;
          const series = candleSeriesRef.current;
          if (!chart || !series) return null;
          const openSnap = nearestCandleTime(candlesRef.current, t.open_time);
          if (openSnap === null) return null;
          const x = chart.timeScale().timeToCoordinate(openSnap);
          const y = series.priceToCoordinate(t.open_price);
          if (x === null || y === null) return null;
          const color = t.side === 'buy' ? cssVar('--color-buy') : cssVar('--color-sell');
          return (
            <div
              key={`opened-here-${t.id}`}
              className='pointer-events-none absolute z-20'
              style={{ left: `${x}px`, top: `${y}px` }}
            >
              <div className='absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2'>
                <span
                  className='absolute left-1/2 top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 animate-ping rounded-full'
                  style={{ backgroundColor: color, opacity: 0.6 }}
                />
                <span
                  className='relative block h-3 w-3 rounded-full ring-2 ring-white'
                  style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
                />
              </div>
              <div
                className='absolute left-1/2 -translate-x-1/2 -translate-y-full flex flex-col items-center gap-0.5'
                style={{ top: '-6px' }}
              >
                <span
                  className='animate-pulse whitespace-nowrap rounded px-2 py-0.5 text-[11px] font-bold'
                  style={{ backgroundColor: color, color: '#ffffff', boxShadow: `0 0 8px ${color}` }}
                >
                  {t.side.toUpperCase()} opened here
                </span>
                <ArrowDown size={14} className='animate-bounce' style={{ color }} />
              </div>
            </div>
          );
        })()}
        {(() => {
          const openClose = buildSelectedTradeOpenClose();
          if (!openClose) return null;
          const chart = chartRef.current;
          const series = candleSeriesRef.current;
          if (!chart || !series) return null;
          // `timeToCoordinate` only resolves a time that exactly matches a
          // loaded bar's timestamp — true when the chart's timeframe equals
          // the strategy's entry timeframe, but a scalp bot's M1 fill time
          // won't match an M5/M15 chart bar. Snap to the nearest loaded
          // candle (same fix `buildLiveTradeLineDrawings` applies) so the
          // line still resolves instead of silently disappearing.
          const candles = candlesRef.current;
          const openSnap = nearestCandleTime(candles, openClose.openTime);
          const closeSnap = nearestCandleTime(candles, openClose.closeTime);
          if (openSnap === null || closeSnap === null) return null;
          const x1 = chart.timeScale().timeToCoordinate(openSnap);
          const x2 = chart.timeScale().timeToCoordinate(closeSnap);
          if (x1 === null || x2 === null) return null;
          const left = Math.min(x1, x2);
          const width = Math.max(1, Math.abs(x2 - x1));
          const segments = [
            {
              key: 'trade-open',
              top: series.priceToCoordinate(openClose.openPrice),
              color: openClose.openColor,
              label: openClose.openLabel,
            },
            {
              key: 'trade-close',
              top: series.priceToCoordinate(openClose.closePrice),
              color: openClose.closeColor,
              label: openClose.closeLabel,
            },
          ];
          return segments.map((seg) => {
            if (seg.top === null || seg.top === undefined) return null;
            return (
              <div
                key={seg.key}
                className='pointer-events-none absolute h-4 -translate-y-1/2 z-10 flex items-center select-none'
                style={{ top: `${seg.top}px`, left: `${left}px`, width: `${width}px` }}
              >
                <div
                  className='w-full'
                  style={{
                    borderTopWidth: orderLineStyle.width,
                    borderTopStyle: orderLineStyle.dash,
                    borderColor: seg.color,
                    filter: `drop-shadow(0 0 4px ${seg.color})`,
                  }}
                />
                <div
                  className='absolute left-2 top-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-bold ring-2 ring-ink'
                  style={{ backgroundColor: seg.color, color: '#ffffff', boxShadow: `0 0 6px ${seg.color}` }}
                >
                  {seg.label}
                </div>
              </div>
            );
          });
        })()}
        {selectedSignalIndex !== null &&
          backtestSignals?.[selectedSignalIndex] &&
          (() => {
            const s = backtestSignals[selectedSignalIndex];
            // `s.time` is the bar's *close* time, but `timeToCoordinate`
            // only resolves times that exactly match an existing candle's
            // (open) time — same mismatch `navigateToTime` already accounts
            // for, so snap to that candle's own time the same way it does.
            const candles = candlesRef.current;
            let barIndex = candles.findIndex((c) => (c.time as number) >= s.time);
            if (barIndex === -1) barIndex = candles.length - 1;
            const barTime = candles[barIndex]?.time;
            const x =
              barTime === undefined
                ? undefined
                : chartRef.current?.timeScale().timeToCoordinate(barTime as UTCTimestamp);
            if (x === null || x === undefined) return null;
            const color = cssVar(SIGNAL_OUTCOME_META[s.outcome].token);
            return (
              <div
                className='pointer-events-none absolute top-0 h-full z-10 select-none'
                style={{ left: `${x}px` }}
              >
                <div
                  className='h-full border-l-[3px] border-dashed'
                  style={{ borderColor: color, filter: `drop-shadow(0 0 4px ${color})` }}
                />
                <div
                  className='absolute top-2 left-2 whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-bold ring-2 ring-ink'
                  style={{ backgroundColor: color, color: '#ffffff', boxShadow: `0 0 6px ${color}` }}
                >
                  {s.direction.toUpperCase()} signal · {SIGNAL_OUTCOME_META[s.outcome].label}
                </div>
              </div>
            );
          })()}
        {showSpreadLine &&
          (() => {
            const currentCandle = replayActive
              ? candlesRef.current[replayCursorIndex]
              : candlesRef.current[candlesRef.current.length - 1];
            if (!currentCandle) return null;

            const currentSpread = replayActive
              ? (currentCandle.spread_points ?? spreadPoints ?? 0)
              : (spreadPoints ?? currentCandle.spread_points ?? 0);

            const bidPrice = replayActive
              ? currentCandle.close
              : (symbolInfo?.bid ?? currentCandle.close);

            const digits = symbolInfo?.digits ?? (bidPrice > 500 ? 2 : 5);
            const point = symbolInfo?.point ?? (digits === 2 ? 0.01 : 0.00001);

            const askPrice = replayActive
              ? (bidPrice != null && currentSpread > 0 ? bidPrice + currentSpread * point : null)
              : (symbolInfo?.ask ?? (bidPrice != null && currentSpread > 0 ? bidPrice + currentSpread * point : null));

            if (askPrice === null) return null;

            const top = candleSeriesRef.current?.priceToCoordinate(askPrice);
            const priceScaleWidth = candleSeriesRef.current?.priceScale().width() || 65;
            if (top === null || top === undefined) return null;

            const accentColor = cssVar('--color-accent');
            return (
              <div
                key='spread-line'
                className='pointer-events-none absolute left-0 h-4 -translate-y-1/2 z-10 flex items-center select-none'
                style={{ top: `${top}px`, right: `${priceScaleWidth}px` }}
              >
                <div
                  className='w-full border-t border-dashed'
                  style={{
                    borderColor: accentColor,
                    filter: `drop-shadow(0 0 3px ${accentColor})`,
                  }}
                />
                <div
                  className='absolute right-2 top-1/2 -translate-y-1/2 rounded px-1.5 py-0.5 text-[10px] font-bold ring-1 ring-ink'
                  style={{
                    backgroundColor: accentColor,
                    color: '#ffffff',
                    boxShadow: `0 0 4px ${accentColor}`,
                  }}
                  title={`Ask price / Spread line (${currentSpread} pts)`}
                >
                  Ask {askPrice.toFixed(digits)} ({currentSpread} pts)
                </div>
              </div>
            );
          })()}
        {editingTicket !== null &&
          (() => {
            const position = trading.positions.find(
              (p) => p.ticket === editingTicket,
            );
            if (!position) return null;
            const top = candleSeriesRef.current?.priceToCoordinate(
              position.open_price,
            );
            if (top === null || top === undefined) return null;
            return (
              <PositionEditPopover
                position={position}
                top={top}
                busy={editBusy}
                onClose={() => setEditingTicket(null)}
                onSave={(sl, tp) => handleSaveEdit(position.ticket, sl, tp)}
                onClosePosition={() => handleCloseFromEdit(position.ticket)}
              />
            );
          })()}
        </div>
        {(backtestReportId || liveBotSkill) && showSignalsDock && (
          <SignalsDock
            signals={backtestSignals ?? []}
            trades={backtestTrades ?? []}
            indicators={liveBotSkill ? (liveBotIndicators ?? []) : undefined}
            backtestMeta={backtestMeta}
            selectedTradeIndex={selectedTradeIndex}
            onSelectTrade={handleToggleTrade}
            onNavigateTrade={handleNavigateTrade}
            selectedSignalIndex={selectedSignalIndex}
            onSelectSignal={handleToggleSignal}
            replayCursorTime={backtestReportId ? replayCursorTime : null}
            replay={botReplayControls}
          />
        )}
      </div>
    </section>
  );
}
