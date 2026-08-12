'use client';

/**
 * Candle data: the history fetch + live WS subscribe/resubscribe effect for
 * the current symbol/timeframe (or backtest report / session-replay period),
 * the spread/symbol-info poll, the news-window shading poll, and the single
 * `render`/`paintUpTo` function every candle/indicator paint path (live tick,
 * "load more" pagination, WS-reconnect patch, and — from outside this hook —
 * replay's `seekTo`/`handleEnterReplay`/`handleExitReplay`) goes through to
 * put candles on the chart.
 *
 * This is a straight structural move of ChartPanel's original combined
 * history-loading `useEffect` (deps: `[accountId, symbol, timeframe,
 * backtestReportId, sessionReplayPeriod]`) plus its two smaller siblings
 * (spread poll, news-band poll) — same fetch/subscribe/teardown logic, same
 * triggers, with exactly one thing changed: the `render` closure defined
 * inside that effect used to be handed out via `renderRef.current = render`
 * (a mutable ref ChartPanel.tsx's replay code read through); it's now
 * wrapped in `paintUpTo`, a stable `useCallback` returned by this hook, so
 * external callers get a real function instead of a bare ref.
 *
 * `paintUpTo` still indirects through an internal `renderRef` rather than
 * being `render` itself, and deliberately so: `render` closes over several
 * variables (`cancelled`, the overlay-recompute throttle's `overlayTimer`/
 * `lastOverlayRun`) that are scoped to one particular run of the fetch
 * effect, not to the hook's lifetime — a fresh `render` closure (and thus a
 * fresh set of those variables) is created every time symbol/timeframe/
 * report changes, exactly like before. `paintUpTo`'s job is only to always
 * call through to *whichever* `render` is current, with a stable identity
 * so passing it to other hooks/effects as a dependency never causes a false
 * "changed" trigger — the ref-hack survives internally, it just no longer
 * leaks to callers outside this hook.
 *
 * Not owned here — deliberately left in ChartPanel.tsx and passed in as
 * params, because both are still fused with replay state that hasn't been
 * extracted yet (phase 9's `useReplayEngine`):
 *   - `candlesRef`: the mutable candle array itself. It has to be created in
 *     ChartPanel.tsx *before* `useChartEngine`/`useIndicators` are called,
 *     because both of those hooks take a `visibleCandles()` reader over this
 *     same ref as a constructor param, and `useCandleData` can't be called
 *     until *after* `useChartEngine` returns (it needs the resulting
 *     `ChartEngineController` for `getChart()`/`getCandleSeries()`/etc.).
 *     That ordering makes it impossible for this hook to be the one that
 *     creates the ref — so it accepts it as an external input and is simply
 *     the (near-)exclusive writer of its contents instead, the same
 *     ref-ownership-vs-ref-creation split `useChartEngine` already uses for
 *     params like `originalStylesRef`.
 *   - `visibleCandles()`: reads `candlesRef` *and* `replayActiveRef`/
 *     `replayCursorIndexRef` (replay-cursor state, still ChartPanel-owned)
 *     to decide whether to return the full loaded window or a prefix up to
 *     the replay cursor — it can't move here without either duplicating
 *     replay-cursor state or reaching back into ChartPanel for it.
 *   - The replay refs/setters themselves (`replayActiveRef`,
 *     `replayCursorIndexRef`, `followCursorRef`, `setReplayActive`,
 *     `setReplayPlaying`, `setReplayCursorIndex`, `setFollowingCursor`) —
 *     the fetch effect resets them on every symbol/timeframe/report switch
 *     (a fresh load invalidates whatever replay cursor was mid-flight), and
 *     `onSessionReplayLoaded` (ChartPanel's `handleEnterReplay`) is called
 *     once a session-replay period finishes loading — still real replay
 *     *behavior*, just triggered from here.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from 'react';
import { type LogicalRange, type UTCTimestamp } from 'lightweight-charts';
import {
  getActiveNewsWindows,
  getBacktestReport,
  getCandles,
  getSymbolInfo,
  type ActivityLogEntry,
  type BacktestSignal,
  type BacktestTrade,
  type Candle,
  type NewsWindow,
  type SymbolInfo,
} from '@/shared/api/client';
import { onSocketConnect, subscribeRoom } from '@/shared/api/ws';
import { cssVar } from './chartFormat';
import { isCandleMessage, toBar, toVolumeBar } from './chartData';
import { subscribeSharedPoll } from './sharedPoll';
import type { ChartEngineController, ChartRenderController, NewsBand } from './types';
import type { ContextMenuState, OrderPopoverState } from './useOrderPopovers';
import type { DrawingMenuState } from './useDrawingTools';

const CANDLE_COUNT = 300;
// Seconds per bar, used to anchor backtest-view history loads (see
// `resolveInitialCandles` below) and session-replay paging — approximate
// for W1/MN is fine since it only sizes a buffer, never the bars
// themselves. Exported: ChartPanel's session-replay picker estimate
// (`sessionReplayEstimate`) needs the same table to size its own preview.
export const TIMEFRAME_SECONDS: Record<Candle['timeframe'], number> = {
  M1: 60,
  M5: 300,
  M15: 900,
  M30: 1800,
  H1: 3600,
  H4: 14_400,
  D1: 86_400,
  W1: 604_800,
  MN: 2_592_000,
};

// The next-finer timeframe to drive tick-form ("live-like") replay: replaying
// a coarse bar, we fetch these finer bars for the same window and aggregate
// the ones inside the current bucket into a growing "forming" bar — same
// synthesis `visibleCandles()` already does for a coarser multi-chart follower,
// applied to single-window local replay. `M1 → null`: there is no candle
// timeframe below M1 (the API exposes no tick/quote endpoint), so an M1 replay
// has nothing finer to form from and falls back to the whole-bar reveal. The
// mapping keeps each step to a single finer level (not all the way to M1) so a
// bar forms from a sane number of frames (5 for M5, 4 for H1→M15, …) rather
// than hundreds.
export const FINER_TIMEFRAME: Record<
  Candle['timeframe'],
  Candle['timeframe'] | null
> = {
  M1: null,
  M5: 'M1',
  M15: 'M5',
  M30: 'M5',
  H1: 'M15',
  H4: 'H1',
  D1: 'H1',
  W1: 'D1',
  MN: 'D1',
};

// Session replay ("live session player" over an arbitrary historical period,
// independent of any backtest report): backend's `/market-data/candles` caps
// `count` at 5000 (market_data/api/routes.py), so a period wider than one
// page needs multiple requests paged backward via `before` — the "looping
// fetch" in `fetchCandlesForPeriod` below. Exported: ChartPanel's picker
// estimate divides by the same chunk size to report an expected page count.
export const SESSION_REPLAY_CHUNK_SIZE = 5000;
// Hard ceiling so a mis-picked period (e.g. years of M1) can't hang the tab
// on dozens of sequential requests or hold an enormous array in memory.
// Exported: ChartPanel's picker blocks a period past this same threshold.
// lightweight-charts' `coordinateToTime`/drawing anchors can only resolve a
// Time for a coordinate that lines up with an existing data point (real or
// whitespace) — clicking past the last loaded candle has no bar there, so
// the drawing tools silently drop the anchor. Padding the series with this
// many `time`-only whitespace bars past the last real candle gives the time
// scale future points to resolve against, so trendlines/rectangles/etc. can
// be drawn (and dragged) into that space. Paired with `FUTURE_RIGHT_OFFSET`
// below, which reserves on-screen room for it.
const FUTURE_WHITESPACE_BARS = 200;
// Empty bar-widths kept visible to the right of the last candle by default
// (lightweight-charts' `timeScale.rightOffset`) — without this there's no
// visible space to draw into even once whitespace bars make it resolvable.
// Exported: set as the chart's initial `rightOffset` in useChartEngine.ts.
export const FUTURE_RIGHT_OFFSET = 20;

export const SESSION_REPLAY_MAX_CANDLES = 60_000;
// Safety valve on the fetch loop itself (defense in depth beyond the
// picker's own block threshold) — a couple of pages of slack past what
// SESSION_REPLAY_MAX_CANDLES should ever require.
const SESSION_REPLAY_MAX_PAGES =
  Math.ceil(SESSION_REPLAY_MAX_CANDLES / SESSION_REPLAY_CHUNK_SIZE) + 2;

const SPREAD_POLL_MS = 3000;
// Matches the backend's own news-window transition-check cadence — no point
// polling faster than the window state can actually change.
const NEWS_POLL_MS = 30_000;
// Start fetching the next page of history once the visible window's left
// edge gets this close to the oldest bar currently loaded, so more arrives
// before the user actually scrolls past the end of the data.
const LOAD_MORE_THRESHOLD = 50;

/** Builds `FUTURE_WHITESPACE_BARS` time-only bars past the last loaded
 * candle so the chart's time scale (and thus drawing-tool anchors, which go
 * through `coordinateToTime`) has real points to resolve to beyond the last
 * candle — see `FUTURE_WHITESPACE_BARS`'s doc comment. Time-only entries
 * render as nothing on the candle series itself. */
function buildFutureWhitespace(
  bars: Candle[],
  timeframe: Candle['timeframe'],
): { time: UTCTimestamp }[] {
  if (bars.length === 0) return [];
  const stepSec = TIMEFRAME_SECONDS[timeframe];
  const lastTime = bars[bars.length - 1].time;
  return Array.from({ length: FUTURE_WHITESPACE_BARS }, (_, i) => ({
    time: (lastTime + stepSec * (i + 1)) as UTCTimestamp,
  }));
}

/** Fetches every candle in `[fromSec, toSec]`, paging backward one
 * `SESSION_REPLAY_CHUNK_SIZE`-sized page at a time (same `before`-cursor
 * pattern as the chart's own "load more") until the range is covered.
 * `onPage` reports progress for the picker/banner UI. Exported: multi-chart
 * layout's secondary windows (`useMiniCandleData`) reuse this to load the
 * same session-replay period at their own timeframe, instead of
 * duplicating the chunked-pagination logic. */
export async function fetchCandlesForPeriod(
  accountId: string,
  symbol: string,
  timeframe: Candle['timeframe'],
  fromSec: number,
  toSec: number,
  onPage?: (page: number, loaded: number) => void,
  signal?: AbortSignal,
): Promise<Candle[]> {
  // Pages come back newest-first (page 1 = most recent); collect them in
  // that order and concatenate once at the end instead of prepending each
  // batch onto a growing array, which would copy the whole accumulator on
  // every page (O(n^2) for a period spanning many chunks).
  const pages: Candle[][] = [];
  let loaded = 0;
  // `before` excludes the cursor bar itself — nudge one bar past `toSec` so
  // the bar covering the period's end is still included in the first page.
  let cursor = toSec + TIMEFRAME_SECONDS[timeframe];
  for (let page = 1; page <= SESSION_REPLAY_MAX_PAGES; page++) {
    const batch = await getCandles(
      accountId,
      symbol,
      timeframe,
      SESSION_REPLAY_CHUNK_SIZE,
      cursor,
      signal,
    );
    if (batch.length === 0) break;
    pages.push(batch);
    loaded += batch.length;
    onPage?.(page, loaded);
    const oldest = batch[0];
    if (oldest.time <= fromSec || batch.length < SESSION_REPLAY_CHUNK_SIZE) break;
    cursor = oldest.time;
  }
  const acc = pages.reverse().flat();
  return acc.filter((c) => c.time >= fromSec && c.time <= toSec);
}

export interface UseCandleDataParams {
  chartController: ChartEngineController;
  symbol: string;
  timeframe: Candle['timeframe'];
  /** Resolved active account id — null while GET /accounts is still in
   * flight, in which case the fetch effect waits rather than firing a
   * request with no valid account. */
  accountId: string | null;
  backtestReportId?: string | null;
  sessionReplayPeriod: { from: number; to: number } | null;
  /** Cleared (set to null) if the session-replay fetch itself fails. */
  setSessionReplayPeriod: Dispatch<SetStateAction<{ from: number; to: number } | null>>;
  setSessionReplayLoadingPage: Dispatch<
    SetStateAction<{ page: number; loaded: number } | null>
  >;
  /** See this module's doc comment: created in ChartPanel.tsx, owned
   * (written) here. */
  candlesRef: RefObject<Candle[]>;
  visibleCandles: () => Candle[];
  replayActiveRef: RefObject<boolean>;
  replayCursorIndexRef: RefObject<number>;
  followCursorRef: RefObject<boolean>;
  setReplayActive: Dispatch<SetStateAction<boolean>>;
  setReplayPlaying: Dispatch<SetStateAction<boolean>>;
  setReplayCursorIndex: Dispatch<SetStateAction<number>>;
  setFollowingCursor: Dispatch<SetStateAction<boolean>>;
  setContextMenu: Dispatch<SetStateAction<ContextMenuState | null>>;
  setOrderPopover: Dispatch<SetStateAction<OrderPopoverState | null>>;
  setDrawingContextMenu: Dispatch<SetStateAction<DrawingMenuState | null>>;
  setDrawingEditPopover: Dispatch<SetStateAction<DrawingMenuState | null>>;
  setBacktestTrades: Dispatch<SetStateAction<BacktestTrade[] | null>>;
  setBacktestActivityLog: Dispatch<SetStateAction<ActivityLogEntry[] | null>>;
  setBacktestSignals: Dispatch<SetStateAction<BacktestSignal[] | null>>;
  setBacktestError: Dispatch<SetStateAction<string | null>>;
  setBacktestMeta: Dispatch<
    SetStateAction<{ strategy: string; symbol: string; period: string } | null>
  >;
  /** Written by `useIndicators` — called on every history load/live tick/
   * replay step, same as before this split. */
  recomputeIndicatorsRef: RefObject<() => void>;
  computeCustomIndicatorsRef: RefObject<() => void>;
  bumpLines: Dispatch<SetStateAction<number>>;
  /** ChartPanel's `handleEnterReplay` — called once a session-replay
   * period's candles finish loading, since session replay has no separate
   * "static full view" step (entering the mode always means playing
   * through the picked period). Still genuine replay behavior, not owned
   * by this hook — see module doc. */
  onSessionReplayLoaded: () => void;
}

export function useCandleData(params: UseCandleDataParams): ChartRenderController {
  const {
    chartController,
    symbol,
    timeframe,
    accountId,
    backtestReportId,
    sessionReplayPeriod,
    setSessionReplayPeriod,
    setSessionReplayLoadingPage,
    candlesRef,
    visibleCandles,
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
    onSessionReplayLoaded,
  } = params;

  // Guards against applying a live WS update before the REST history load
  // for the current symbol/timeframe has landed — see the effect below.
  const historyLoadedRef = useRef(false);
  const hasMoreHistoryRef = useRef(true);
  const loadingMoreRef = useRef(false);
  // Set at the end of the candle-loading effect below to the `render()`
  // closure created there, so `paintUpTo` (and, previously, the replay tick
  // loop directly) can trigger a redraw without duplicating candle/volume
  // `setData()` logic. See this module's doc comment for why this
  // indirection is still needed even with `paintUpTo` as the real,
  // stable-identity export.
  const renderRef = useRef<() => void>(() => {});
  // Same ref-per-effect-run indirection as `renderRef`, for the same reason:
  // `reloadWindow` closes over the current run's `cancelled`/abort controller.
  const reloadRef = useRef<() => Promise<void>>(async () => {});

  const [symbolInfo, setSymbolInfo] = useState<SymbolInfo | null>(null);
  const [spreadPoints, setSpreadPoints] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  // True from the moment a symbol/timeframe/report switch starts clearing
  // state until fresh candles actually land — the chart keeps the previous
  // symbol/timeframe's bars on screen the whole time (there's no cheap way
  // to blank a lightweight-charts series without a visible flash), so
  // without this the switch looks like nothing happened until data arrives.
  const [switchingChart, setSwitchingChart] = useState(false);
  const [newsBands, setNewsBands] = useState<NewsBand[]>([]);

  // Load history + subscribe to live updates whenever symbol/timeframe
  // changes — or, in backtest view, whenever the report being inspected
  // changes (§F: "test the bot in chart for candle history").
  useEffect(() => {
    // Not resolved yet (GET /accounts still in flight) — wait for it rather
    // than firing a request with no valid account id.
    if (!accountId) return;
    const account = accountId;
    let cancelled = false;
    // Cancels the initial-history fetch (and, since it's the one
    // `AbortController` shared for this symbol/timeframe/report's whole
    // lifetime, `loadMore`'s pan-left pagination fetch too) in flight for
    // the *previous* symbol/timeframe/report as soon as a newer one is
    // picked, instead of leaving it to finish on its own. Without this,
    // rapidly clicking through timeframes (or panning left then switching
    // symbol mid-fetch) queues up real HTTP requests competing for the same
    // connection pool whose result the client would just discard anyway.
    const initialLoadController = new AbortController();
    setError(null);
    setLoadingMore(false);
    setSwitchingChart(true);
    setContextMenu(null);
    setOrderPopover(null);
    setDrawingContextMenu(null);
    setDrawingEditPopover(null);
    setBacktestTrades(null);
    setBacktestActivityLog(null);
    setBacktestSignals(null);
    setBacktestError(null);
    // A new symbol/timeframe/report invalidates any in-progress replay —
    // the cursor index no longer lines up with the freshly-loaded candles.
    replayActiveRef.current = false;
    replayCursorIndexRef.current = 0;
    followCursorRef.current = true;
    setReplayActive(false);
    setReplayPlaying(false);
    setReplayCursorIndex(0);
    setFollowingCursor(true);
    // WS updates for the new room can start arriving before the REST
    // history call below resolves. Applying one to the still-stale
    // previous symbol/timeframe's data can move time backwards (e.g.
    // switching from M1 to D1: the D1 forming bar's open time is earlier
    // than the M1 bar still on screen) and lightweight-charts throws.
    // Dropping live updates until history for *this* symbol/timeframe is
    // actually on the chart avoids that race.
    historyLoadedRef.current = false;
    candlesRef.current = [];
    hasMoreHistoryRef.current = true;
    loadingMoreRef.current = false;

    const chart = chartController.getChart();

    // `recomputeIndicators` tears down and recreates every indicator series
    // (`chart.removeSeries`/`addSeries` per EMA/RSI/MACD/Bollinger line, plus
    // rebuilding every period-separator drawing) — fine to call on every live
    // tick (~once/1.5s) but far too expensive to run on literally every
    // replay bar at speed. `scheduleOverlayRecompute` throttles just that
    // part (leading + trailing edge: the first call after an idle period
    // runs immediately, rapid follow-up calls coalesce into one trailing
    // update `OVERLAY_THROTTLE_MS` later) while candle/volume `setData()`
    // below — cheap, just fills existing series — still runs every tick so
    // playback itself stays smooth.
    const OVERLAY_THROTTLE_MS = 200;
    let overlayTimer: ReturnType<typeof setTimeout> | null = null;
    let lastOverlayRun = 0;

    function runOverlaysNow() {
      lastOverlayRun = Date.now();
      recomputeIndicatorsRef.current();
      // Custom-code overlays stay full-range/ungated during replay (§F scope
      // decision) — skip re-running the sandboxed backend eval on every
      // cursor tick, which would otherwise fire an HTTP request per bar.
      if (!replayActiveRef.current) computeCustomIndicatorsRef.current();
    }

    function scheduleOverlayRecompute() {
      if (!replayActiveRef.current) {
        runOverlaysNow();
        return;
      }
      const elapsed = Date.now() - lastOverlayRun;
      if (elapsed >= OVERLAY_THROTTLE_MS) {
        runOverlaysNow();
        return;
      }
      if (overlayTimer) return;
      overlayTimer = setTimeout(() => {
        overlayTimer = null;
        runOverlaysNow();
      }, OVERLAY_THROTTLE_MS - elapsed);
    }

    function render() {
      const upColor = cssVar('--color-ok');
      const downColor = cssVar('--color-err');
      const bars = visibleCandles();
      chartController.getCandleSeries()?.setData(bars.map(toBar));
      chartController
        .getVolumeSeries()
        ?.setData(bars.map((c) => toVolumeBar(c, upColor, downColor)));
      chartController
        .getWhitespaceSeries()
        ?.setData(buildFutureWhitespace(bars, timeframe));
      scheduleOverlayRecompute();
      setTimeout(() => {
        if (!cancelled) requestAnimationFrame(() => bumpLines((t) => t + 1));
      }, 50);
    }

    // Re-fetches the *currently loaded* time window from scratch and swaps it
    // in, for when the bars on screen are known to be wrong rather than
    // merely incomplete — today that means after `useCandleGaps` has had the
    // backend re-download missing history (`POST /market-data/candle-gaps/
    // repair`), where the window keeps its time span but gains bars in the
    // middle. Deliberately not `loadMore` (which only ever prepends older
    // pages) and not `patchLatestHistoryOnReconnect` (which only replaces the
    // newest `CANDLE_COUNT`).
    //
    // Filling a hole shifts every bar to its right by the number of recovered
    // bars, so the visible *logical* range no longer points at the same bars.
    // Snapshotting and restoring the visible *time* range instead keeps the
    // user looking at the same moment in the market, and the replay cursor is
    // re-resolved by time for the same reason.
    async function reloadWindow() {
      const bars = candlesRef.current;
      if (bars.length === 0) return;
      const from = bars[0].time;
      const to = bars[bars.length - 1].time;
      const cursorTime = replayActiveRef.current
        ? bars[replayCursorIndexRef.current]?.time
        : undefined;
      const timeRange = chart?.timeScale().getVisibleRange();
      const fresh = await fetchCandlesForPeriod(
        account,
        symbol,
        timeframe,
        from,
        to,
        undefined,
        initialLoadController.signal,
      );
      if (cancelled || fresh.length === 0) return;
      candlesRef.current = fresh;
      if (cursorTime !== undefined) {
        const index = fresh.findIndex((c) => c.time >= cursorTime);
        const resolved = index === -1 ? fresh.length - 1 : index;
        replayCursorIndexRef.current = resolved;
        setReplayCursorIndex(resolved);
      }
      render();
      if (timeRange) {
        requestAnimationFrame(() => {
          if (!cancelled) chart?.timeScale().setVisibleRange(timeRange);
        });
      }
    }

    // Fetches the next page of older bars once the user pans near the left
    // edge of what's loaded — the chart's "fetch more" is this auto-trigger
    // plus the `loadingMore` indicator rendered below, not a manual button.
    async function loadMore() {
      if (
        loadingMoreRef.current ||
        !hasMoreHistoryRef.current ||
        candlesRef.current.length === 0
      ) {
        return;
      }
      loadingMoreRef.current = true;
      setLoadingMore(true);
      const oldest = candlesRef.current[0];
      try {
        const older = await getCandles(
          account,
          symbol,
          timeframe,
          CANDLE_COUNT,
          oldest.time,
          initialLoadController.signal,
        );
        if (cancelled) return;
        if (older.length === 0) {
          hasMoreHistoryRef.current = false;
        } else {
          hasMoreHistoryRef.current = older.length >= CANDLE_COUNT;
          candlesRef.current = [...older, ...candlesRef.current];
          // Prepending shifts every existing bar's logical index forward by
          // the number of new bars, so the visible window must shift with
          // it or the chart jumps — lightweight-charts has no "prepend"
          // primitive, this is the documented workaround for setData().
          // We snapshot the range *before* setData, call setData, then restore
          // via requestAnimationFrame so the adjustment runs after the new
          // layout pass — applying it synchronously can land before the bars
          // are actually committed and produce an off-by-N shift.
          const range = chart?.timeScale().getVisibleLogicalRange();
          if (replayActiveRef.current) {
            replayCursorIndexRef.current += older.length;
            setReplayCursorIndex((prev) => prev + older.length);
          }
          render();
          if (range) {
            requestAnimationFrame(() => {
              if (!cancelled) {
                chart?.timeScale().setVisibleLogicalRange({
                  from: range.from + older.length,
                  to: range.to + older.length,
                });
              }
            });
          }
        }
      } catch {
        // Transient failure — leave hasMore true so the next pan retries.
      } finally {
        if (!cancelled) {
          loadingMoreRef.current = false;
          setLoadingMore(false);
        }
      }
    }

    const onVisibleRangeChange = (range: LogicalRange | null) => {
      if (range && range.from < LOAD_MORE_THRESHOLD) void loadMore();
    };
    chart?.timeScale().subscribeVisibleLogicalRangeChange(onVisibleRangeChange);

    // Backtest view anchors history to the report's own trades instead of
    // "now" — just past the last trade's close, scaled to a couple of bars
    // of the current timeframe, so the anchor guarantees that trade's candle
    // is included without burning most of CANDLE_COUNT's budget on empty
    // time past the trades (a flat multi-hour buffer would eat most of a
    // 300-bar M5 window and push every earlier trade off the loaded page).
    async function resolveInitialCandles(): Promise<Candle[]> {
      if (sessionReplayPeriod) {
        setSessionReplayLoadingPage({ page: 0, loaded: 0 });
        return fetchCandlesForPeriod(
          account,
          symbol,
          timeframe,
          sessionReplayPeriod.from,
          sessionReplayPeriod.to,
          (page, loaded) => {
            if (!cancelled) setSessionReplayLoadingPage({ page, loaded });
          },
          initialLoadController.signal,
        );
      }
      if (!backtestReportId)
        return getCandles(
          account,
          symbol,
          timeframe,
          CANDLE_COUNT,
          undefined,
          initialLoadController.signal,
        );
      const report = await getBacktestReport(backtestReportId);
      if (cancelled) return [];
      setBacktestTrades(report.trades);
      setBacktestActivityLog(report.activity_log);
      setBacktestSignals(report.signals ?? []);
      setBacktestMeta({
        strategy: report.strategy,
        symbol: report.symbol,
        period: report.period,
      });
      // Anchor the candle window at the last *event* in the report — the
      // final trade close or the final signal, whichever is later. Anchoring
      // on trades alone left signals emitted after the last trade (vetoed
      // setups near the period's end) beyond the loaded candles, where their
      // markers clamp misleadingly onto the last visible bar.
      const lastClose = [
        ...report.trades.map((t) => t.close_time),
        ...(report.signals ?? []).map((s) => s.time),
      ].reduce((max, t) => Math.max(max, t), 0);
      const anchor =
        lastClose > 0
          ? lastClose + 2 * TIMEFRAME_SECONDS[timeframe]
          : undefined;
      return getCandles(
        account,
        symbol,
        timeframe,
        CANDLE_COUNT,
        anchor,
        initialLoadController.signal,
      );
    }

    renderRef.current = render;
    reloadRef.current = reloadWindow;

    resolveInitialCandles()
      .then((candles) => {
        if (cancelled) return;
        candlesRef.current = candles;
        // A backtest report can be older than the local candle DB (or ask
        // for a timeframe that was never backfilled) — the anchored fetch
        // then comes back empty and the chart would just render blank.
        // Say so instead of leaving a silent void.
        if (backtestReportId && candles.length === 0) {
          setBacktestError(
            `no ${timeframe} candle history covering this report's period — ` +
              'backfill it (POST /market-data/backfill) or switch to a timeframe with history',
          );
        }
        // A picked session-replay range can fall entirely inside a gap in
        // the local candle DB (e.g. before backfill started) — without this,
        // the fetch "succeeds" with zero candles and replay silently enters
        // an empty, permanently-blank session with no indication why.
        if (sessionReplayPeriod && candles.length === 0) {
          setError(
            `no ${timeframe} candle history in this replay period — ` +
              'pick a different range or timeframe',
          );
        }
        // Session replay's window is deliberately bounded by the picked
        // period — panning left shouldn't silently pull in history from
        // before it, unlike the live/backtest views' open-ended paging.
        hasMoreHistoryRef.current = sessionReplayPeriod
          ? false
          : candles.length >= CANDLE_COUNT;
        render();
        historyLoadedRef.current = true;
        setSessionReplayLoadingPage(null);
        setSwitchingChart(false);
        // A symbol/timeframe switch loads a fresh price/time range, but
        // lightweight-charts keeps whatever pan/zoom/price-scale state was
        // active for the previous symbol. `scrollToRealTime()` alone only
        // moves the time axis — it doesn't reset the logical range or price
        // scale, so e.g. switching from BTCUSD (~60000) to XAGUSD (~30) can
        // leave the new candles partly or fully outside the viewport.
        // `fitContent()` plus forcing `autoScale` back on fixes both.
        chartController
          .getCandleSeries()
          ?.priceScale()
          .applyOptions({ autoScale: true });
        chart?.timeScale().fitContent();
        setTimeout(() => {
          if (!cancelled) bumpLines((t) => t + 1);
        }, 50);
        // Session replay has no separate "static full view" step — entering
        // the mode always means playing through the picked period.
        if (sessionReplayPeriod) onSessionReplayLoaded();
      })
      .catch(() => {
        if (cancelled) return;
        setSessionReplayLoadingPage(null);
        setSwitchingChart(false);
        setError(
          backtestReportId
            ? 'failed to load backtest report'
            : sessionReplayPeriod
              ? 'failed to load session replay candles'
              : 'failed to load candles',
        );
        if (backtestReportId)
          setBacktestError('failed to load backtest report');
        if (sessionReplayPeriod) setSessionReplayPeriod(null);
      });

    // Live candle updates only make sense against "now" — in backtest view
    // or session replay the chart is anchored to a historical window, so a
    // fresh WS tick would just append a stray present-day bar after a
    // months-wide gap.
    if (backtestReportId || sessionReplayPeriod) {
      return () => {
        cancelled = true;
        initialLoadController.abort();
        if (overlayTimer) clearTimeout(overlayTimer);
        historyLoadedRef.current = false;
        chart
          ?.timeScale()
          .unsubscribeVisibleLogicalRangeChange(onVisibleRangeChange);
      };
    }

    // `candle_update` streams the in-progress bar every ~1.5s so the
    // rightmost candle moves continuously like MT5; `candle_closed` is the
    // authoritative final print once the bar completes. Both are handled
    // identically here — lightweight-charts' `update()` amends the last bar
    // in place when the timestamp matches, or appends a new one otherwise.
    const unsubscribe = subscribeRoom(
      ['candle_closed', 'candle_update'],
      { accountId, symbol, timeframe },
      (message) => {
        if (!isCandleMessage(message)) return;
        if (!historyLoadedRef.current) return;
        const { candle } = message;
        // The socket carries every open room's events on one connection —
        // Socket.IO room scoping only filters what the *server* emits, not
        // which of a client's several `.on(event, ...)` handlers a given
        // message reaches. A multi-chart layout (this chart + the other
        // windows' own candle subscriptions) or the brief overlap window
        // during a symbol/timeframe switch means this handler can be called
        // with another room's candle.
        // Without this check that candle gets spliced into `bars` and pushed
        // into the chart series regardless of scale, producing exactly the
        // out-of-order drops/crashes this file's other guards are catching.
        if (candle.symbol !== symbol || candle.timeframe !== timeframe) return;
        const bars = candlesRef.current;
        const lastTime =
          bars.length > 0 ? bars[bars.length - 1].time : undefined;
        if (lastTime !== undefined && candle.time < lastTime) {
          // Stale/out-of-order message (e.g. stream jitter) — pushing this
          // would break the ascending-time invariant every indicator and
          // lightweight-charts itself relies on, so drop it instead.
          console.warn(
            'chart: dropped out-of-order candle update',
            candle.time,
            'last',
            lastTime,
          );
          return;
        }
        if (lastTime === candle.time) {
          bars[bars.length - 1] = candle;
        } else {
          bars.push(candle);
          chartController
            .getWhitespaceSeries()
            ?.setData(buildFutureWhitespace(bars, timeframe));
        }
        try {
          chartController.getCandleSeries()?.update(toBar(candle));
          chartController
            .getVolumeSeries()
            ?.update(toVolumeBar(candle, cssVar('--color-ok'), cssVar('--color-err')));
          recomputeIndicatorsRef.current();
          setTimeout(() => {
            if (!cancelled) bumpLines((t) => t + 1);
          }, 50);
        } catch (err) {
          // Defensive: lightweight-charts throws if a live update's time is
          // older than what the *series* itself last received — which can
          // still happen even after the `bars`-ref check above, since `bars`
          // and the series' own internal state can momentarily disagree
          // (e.g. mid-flight during a symbol/timeframe switch's async
          // history load). Previously this just dropped the tick and warned,
          // leaving the chart one bar stale until the next tick happened to
          // land cleanly; falling back to a full `render()` re-`setData()`s
          // the series from `candlesRef.current` (already updated above),
          // resyncing it immediately instead of hoping the next tick fixes it.
          console.warn('chart: dropped out-of-order live update, resyncing', err);
          render();
        }
      },
    );

    // A reconnect (network blip, backend restart) can leave a hole between
    // the last bar we have and "now" — `candle_closed`/`candle_update` only
    // stream deltas going forward, they never backfill what was missed while
    // disconnected. Refetch the tail on every `connect` after the first
    // (guarded by `historyLoadedRef`, which is false during the initial
    // load's own fetch) and splice it in: bars older than the refetched
    // window are left untouched, so paged-in history from `loadMore` above
    // survives.
    let patchingReconnect = false;
    async function patchLatestHistoryOnReconnect() {
      if (!historyLoadedRef.current || patchingReconnect) return;
      patchingReconnect = true;
      try {
        const latest = await getCandles(account, symbol, timeframe, CANDLE_COUNT);
        if (cancelled || latest.length === 0) return;
        const cutoff = latest[0].time;
        candlesRef.current = [
          ...candlesRef.current.filter((c) => c.time < cutoff),
          ...latest,
        ];
        render();
      } catch {
        // Transient failure — the next reconnect (or the regular live-tick
        // stream, once it catches up) gets another chance.
      } finally {
        patchingReconnect = false;
      }
    }
    const unsubscribeReconnect = onSocketConnect(() => {
      void patchLatestHistoryOnReconnect();
    });

    return () => {
      cancelled = true;
      initialLoadController.abort();
      if (overlayTimer) clearTimeout(overlayTimer);
      historyLoadedRef.current = false;
      chart
        ?.timeScale()
        .unsubscribeVisibleLogicalRangeChange(onVisibleRangeChange);
      unsubscribe();
      unsubscribeReconnect();
    };
    // setBacktestTrades/setBacktestError/setBacktestMeta/setBacktestActivityLog/
    // setBacktestSignals/setContextMenu/setOrderPopover/setDrawingContextMenu/
    // setDrawingEditPopover/setReplayActive/setReplayPlaying/
    // setReplayCursorIndex/setFollowingCursor/setSessionReplayLoadingPage/
    // setSessionReplayPeriod/bumpLines are all plain setState setters (stable
    // identity) — safe to omit, same as every other effect in this file that
    // calls setState setters without listing them as deps. chartController/
    // candlesRef/visibleCandles/replayActiveRef/replayCursorIndexRef/
    // followCursorRef/recomputeIndicatorsRef/computeCustomIndicatorsRef/
    // onSessionReplayLoaded are stable across a render the same way the
    // equivalent params are in useChartEngine/useIndicators — omitted
    // deliberately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountId, symbol, timeframe, backtestReportId, sessionReplayPeriod]);

  // Poll live spread and symbol info for header indicator and spread line —
  // shared across every window on the same account+symbol (multi-chart
  // layout §) so N windows don't each fire an identical request every tick.
  useEffect(() => {
    if (!accountId) return;
    const unsubscribe = subscribeSharedPoll(
      `symbol-info:${accountId}:${symbol}`,
      SPREAD_POLL_MS,
      () => getSymbolInfo(accountId, symbol),
      (info, error) => {
        requestAnimationFrame(() => {
          if (error || !info) {
            setSymbolInfo(null);
            setSpreadPoints(null);
          } else {
            setSymbolInfo(info);
            setSpreadPoints(info.spread_points);
          }
        });
      },
    );
    return unsubscribe;
  }, [accountId, symbol]);

  // News window shading (§8, F8): shade the pre/post-event window of any
  // active news window that affects this symbol. Pixel positions are
  // recomputed on every news poll, pan/zoom, and resize since they depend on
  // the chart's current visible time range, not just the window's own times.
  useEffect(() => {
    let cancelled = false;
    let currentWindows: NewsWindow[] = [];
    const chart = chartController.getChart();
    const container = chartController.containerRef.current;

    function recompute() {
      if (!chart || !container) {
        setNewsBands((prev) => (prev.length === 0 ? prev : []));
        return;
      }
      const visible = chart.timeScale().getVisibleRange();
      if (!visible) {
        setNewsBands((prev) => (prev.length === 0 ? prev : []));
        return;
      }
      const from = visible.from as number;
      const to = visible.to as number;
      const bands: NewsBand[] = [];
      for (const w of currentWindows) {
        if (!w.symbols.includes(symbol)) continue;
        if (w.window_end < from || w.window_start > to) continue;
        const x1 = chart
          .timeScale()
          .timeToCoordinate(Math.max(from, w.window_start) as UTCTimestamp);
        const x2 = chart
          .timeScale()
          .timeToCoordinate(Math.min(to, w.window_end) as UTCTimestamp);
        if (x1 === null || x2 === null) continue;
        bands.push({
          key: `${w.event.name}-${w.window_start}`,
          left: Math.min(x1, x2),
          width: Math.max(1, Math.abs(x2 - x1)),
          label: w.event.name,
          phase: w.phase,
        });
      }
      requestAnimationFrame(() => {
        setNewsBands((prev) => {
          if (prev.length === 0 && bands.length === 0) return prev;
          if (
            prev.length === bands.length &&
            prev.every(
              (b, i) =>
                b.key === bands[i].key &&
                b.left === bands[i].left &&
                b.width === bands[i].width &&
                b.label === bands[i].label &&
                b.phase === bands[i].phase,
            )
          ) {
            return prev;
          }
          return bands;
        });
      });
    }

    // The fetch itself is account-wide (no symbol/timeframe param) and
    // identical for every open window — shared across all of them (multi-
    // chart layout §), while `recompute()` above (which reads the result
    // through `currentWindows`) stays per-window since it depends on this
    // window's own visible range/container size.
    const unsubscribe = subscribeSharedPoll(
      'news-windows',
      NEWS_POLL_MS,
      () => getActiveNewsWindows(),
      (windows, error) => {
        if (cancelled) return;
        currentWindows = error || !windows ? [] : windows;
        recompute();
      },
    );
    chart?.timeScale().subscribeVisibleTimeRangeChange(recompute);
    const resizeObserver = new ResizeObserver(recompute);
    if (container) resizeObserver.observe(container);

    return () => {
      cancelled = true;
      unsubscribe();
      chart?.timeScale().unsubscribeVisibleTimeRangeChange(recompute);
      resizeObserver.disconnect();
    };
    // `timeframe` deliberately excluded — this poll/recompute doesn't depend
    // on it (getActiveNewsWindows takes no timeframe param, and `recompute`
    // only reads `symbol`), so including it just tore down and restarted
    // the poll/observer on every timeframe switch for no reason.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  // Stable wrapper around `renderRef` — see this module's doc comment for
  // why the indirection through a ref survives even though this is now a
  // "real" callback: `render` itself is recreated fresh on every effect run
  // (one per symbol/timeframe/report), and needs to be, since it closes
  // over that run's own `cancelled`/overlay-throttle state. `paintUpTo`'s
  // only job is to always dispatch to whichever `render` is current, with
  // an identity that never changes so passing it around (e.g. as another
  // effect's dependency) is safe.
  const paintUpTo = useCallback(() => {
    renderRef.current();
  }, []);

  /** Stable wrapper around `reloadRef`, same rationale as `paintUpTo`. */
  const reloadWindow = useCallback(() => reloadRef.current(), []);

  return {
    candlesRef,
    paintUpTo,
    reloadWindow,
    symbolInfo,
    spreadPoints,
    error,
    loadingMore,
    switchingChart,
    newsBands,
  };
}

export type CandleData = ReturnType<typeof useCandleData>;
