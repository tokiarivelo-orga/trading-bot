'use client';

/**
 * Replay engine ("live session player", §F): the moving cursor over a
 * backtest report's (or live-bot eye's) candles, session replay's own
 * arbitrary-period picker + chunked-fetch progress UI, the autoplay tick
 * loop, and every click-to-navigate handler (SignalsDock's trade/signal
 * rows, a trade-history row via `navigateToTime`) that routes through the
 * cursor instead of a plain pan while replaying. This is phase 9's
 * extraction — a straight structural move of ChartPanel's `centerOn`/
 * `navigateToTime`/`seekTo`/`handleEnterReplay`/`handleExitReplay`/
 * `handleStartSessionReplay`/`handleExitSessionReplay`/
 * `handleRecenterReplay`/`handleToggleTrade`/`handleNavigateTrade`/
 * `handleToggleSignal` functions and its two replay `useEffect`s (autoplay
 * tick, mousedown-pauses-follow) — same logic, same triggers, same reads/
 * writes, just relocated and routed through `chartController`/
 * `chartRenderController` instead of raw `chartRef.current`/
 * `candleSeriesRef.current`/`renderRef.current()`.
 *
 * Ownership split — why `candlesRef`/`visibleCandles`/several replay
 * primitives still aren't created here, even though this hook is the
 * "owner" of the replay concern:
 *
 * `useCandleData.ts`'s module doc left `candlesRef`/`visibleCandles` in
 * ChartPanel.tsx pending this hook's arrival. Now that it exists, the
 * verdict is: they still can't move here. `useChartEngine`'s one-time `[]`
 * chart-creation effect takes `visibleCandles` (closed over `candlesRef` +
 * the replay-cursor refs) as a constructor param, and `useChartEngine` is
 * called *before* `useCandleData` — which itself must run *before* this
 * hook, since this hook takes `chartRenderController` (`useCandleData`'s
 * return value) as an input. A hook invoked third in that chain cannot
 * retroactively supply a value the first hook in the chain already needed.
 * So `candlesRef` stays created in ChartPanel.tsx, exactly like
 * `originalStylesRef`/`saveAndSyncRef` do for `useChartEngine`/
 * `useDrawingTools` — this hook simply reads it through
 * `chartRenderController.candlesRef` (already exposed there for this exact
 * purpose, see `types.ts`'s `ChartRenderController` doc) instead of taking
 * a second, redundant, separately-threaded ref param. `visibleCandles`
 * itself is never called from here at all: every replay handler below
 * (`seekTo`, `navigateToTime`, the autoplay tick, …) deliberately reads the
 * *raw* `candlesRef.current` — cursor math needs the full underlying array
 * to index into, not the already-cursor-gated view.
 *
 * The same ordering constraint applies to `replayActive`/`replayPlaying`/
 * `replayCursorIndex`/`followingCursor` (+ their refs) and
 * `sessionReplayPeriod`/`sessionReplayLoadingPage`: `useChartEngine` reads
 * `replayActiveRef`/`followCursorRef` directly, and `useCandleData` resets
 * every one of these on every symbol/timeframe/report switch (a fresh load
 * invalidates whatever replay cursor was mid-flight) and reads
 * `sessionReplayPeriod` to anchor its own fetch — both hooks run before this
 * one can exist. ChartPanel.tsx therefore still creates exactly those
 * primitives and passes them in as controlled inputs (the same "hook
 * creates vs. receives" split `useDrawingTools.ts`'s module doc documents
 * for its own inputs); everything with no such constraint —
 * `replaySpeed`, `isMouseDownRef`/`animationFrameRef`/
 * `lastRevealedSignatureRef`, and the session-replay *picker's own* UI
 * state (`showSessionReplayPicker`/`sessionReplayFromInput`/
 * `sessionReplayToInput`) — is created fresh here instead.
 * `lastRevealedSignatureRef` is a special case: nothing in this hook reads
 * or writes it (its only consumer is ChartPanel.tsx's backtest-trade-
 * drawing effect, which stays there — a different concern, see that
 * effect's own comments), but it's genuinely unconstrained by any *other*
 * hook's call ordering, so it's created and returned here rather than
 * left as a ChartPanel.tsx local with no hook backing it.
 *
 * The mirror-image ordering problem also applies to `onSessionReplayLoaded`:
 * `useCandleData` needs to call this hook's `handleEnterReplay` once a
 * session-replay period's candles land, but `useCandleData` runs *before*
 * this hook (which needs `useCandleData`'s own return value). ChartPanel.tsx
 * bridges that with a small ref-forwarding shim (`handleEnterReplayRef`,
 * declared before the `useCandleData` call, assigned to this hook's
 * `handleEnterReplay` right after this hook runs) — the same "ref created
 * early, assigned by a later hook" shape `useChartEngine`'s `saveAndSyncRef`
 * param already established for `useDrawingTools`, not a new pattern.
 */

import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from 'react';
import type {
  BacktestSignal,
  BacktestTrade,
  Candle,
} from '@/shared/api/client';
import { SESSION_REPLAY_CHUNK_SIZE, SESSION_REPLAY_MAX_CANDLES, TIMEFRAME_SECONDS } from './useCandleData';
import type { ChartEngineController, ChartRenderController } from './types';

// Beyond this many candles the picker warns (but still allows) the period —
// it'll take more than one request to load.
const SESSION_REPLAY_WARN_CANDLES = 8000;

// How many bars of context to start replay with instead of a single bar —
// one candle against the still-full-range price scale renders as a
// barely-visible sliver squashed to one edge until the next fit; starting
// with real context avoids that and gives a sane first frame.
const REPLAY_START_CONTEXT_BARS = 50;

export interface UseReplayEngineParams {
  chartController: ChartEngineController;
  chartRenderController: ChartRenderController;
  timeframe: Candle['timeframe'];

  // --- Controlled inputs: created in ChartPanel.tsx because useChartEngine
  // (replay-gated autoscale) and useCandleData (resets these on every
  // symbol/timeframe/report switch, anchors its fetch on
  // sessionReplayPeriod) both need them before this hook can exist — see
  // module doc. Only the setters/refs/values this hook's own logic
  // actually reads or writes are listed; state ChartPanel.tsx needs for
  // JSX but this hook's logic never touches (`replayCursorIndex`,
  // `followingCursor`, `sessionReplayLoadingPage`, …) stays its own local
  // rather than round-tripping through this hook's params/return.
  /** Read by the mousedown-pauses-follow effect (only *active* while
   * replaying) and as its dependency. */
  replayActive: boolean;
  setReplayActive: Dispatch<SetStateAction<boolean>>;
  replayActiveRef: RefObject<boolean>;
  /** Read by the autoplay tick effect (only *ticks* while playing) and as
   * its dependency. */
  replayPlaying: boolean;
  setReplayPlaying: Dispatch<SetStateAction<boolean>>;
  setReplayCursorIndex: Dispatch<SetStateAction<number>>;
  replayCursorIndexRef: RefObject<number>;
  setFollowingCursor: Dispatch<SetStateAction<boolean>>;
  followCursorRef: RefObject<boolean>;
  setSessionReplayPeriod: Dispatch<
    SetStateAction<{ from: number; to: number } | null>
  >;
  /** Auto-opened on entering replay so the report's own activity log is
   * visible without an extra click. Plain ChartPanel.tsx-local UI toggle —
   * unrelated to any hook-ordering constraint, just simplest to receive
   * as a param rather than duplicate. */
  setShowActivityLogDock: Dispatch<SetStateAction<boolean>>;
  /** Skips the auto-open above — set for any window in a multi-window split
   * layout (`hideToolbar`), where popping the activity dock open over
   * whichever window started replay is disruptive rather than helpful; the
   * single full-chart view keeps the original auto-open behavior. */
  suppressAutoActivityDock?: boolean;

  // --- Backtest/live-bot trade+signal selection (useBacktestData) — the
  // click targets `handleToggleTrade`/`handleNavigateTrade`/
  // `handleToggleSignal` navigate to.
  backtestTrades: BacktestTrade[] | null;
  backtestSignals: BacktestSignal[] | null;
  selectedTradeIndex: number | null;
  setSelectedTradeIndex: Dispatch<SetStateAction<number | null>>;
  selectedSignalIndex: number | null;
  setSelectedSignalIndex: Dispatch<SetStateAction<number | null>>;
  sharedReplayActive?: boolean;

  // --- Tick-form replay (created in ChartPanel.tsx alongside the other replay
  // refs, since `visibleCandles`/`render` — closures built in useCandleData,
  // called before this hook — read them; see ChartPanel's `finerCandlesRef`/
  // `tickFormRef`/`replayFineTimeRef` docs). The autoplay loop below advances
  // `replayFineTimeRef` through `finerCandlesRef` so each bar visibly forms
  // from its finer constituents instead of appearing fully closed.
  /** The next-finer timeframe's candles for the replay window, or empty when
   * tick-form is off / unavailable — then the loop reveals whole bars. */
  finerCandlesRef: RefObject<Candle[]>;
  /** Sub-cursor: time up to which the current forming bar is revealed. Null =
   * show the current bar fully closed (manual seek, tick-form off). */
  replayFineTimeRef: RefObject<number | null>;
  /** Whether tick-form is enabled (mirror of ChartPanel's `tickForm`). */
  tickFormRef: RefObject<boolean>;
}

export function useReplayEngine(params: UseReplayEngineParams) {
  const {
    chartController,
    timeframe,
    replayActive,
    sharedReplayActive = false,
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
    suppressAutoActivityDock = false,
    backtestTrades,
    backtestSignals,
    selectedTradeIndex,
    setSelectedTradeIndex,
    selectedSignalIndex,
    setSelectedSignalIndex,
    finerCandlesRef,
    replayFineTimeRef,
    tickFormRef,
  } = params;
  const chartRenderController = params.chartRenderController;

  // Playback speed multiplier for the autoplay tick effect below — no
  // ordering constraint from any other hook, owned outright here.
  const [replaySpeed, setReplaySpeed] = useState(1);

  // Track dragging/scrolling interaction and animation-frame handles for
  // replay panning.
  const isMouseDownRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);
  // "open-count:close-count" signature of the trades revealed as of the last
  // trade-drawing rebuild in ChartPanel.tsx's own effect — lets that effect
  // skip rebuilding when a cursor tick didn't actually cross any trade's
  // reveal threshold. Owned here (no other hook needs it before this one
  // exists) but consumed entirely by that ChartPanel.tsx effect — see
  // module doc.
  const lastRevealedSignatureRef = useRef<string | null>(null);

  // Session replay: an arbitrary historical period, picked ad hoc (not tied
  // to a saved backtest report), replayed bar-by-bar like a live session.
  // The picker's own input/visibility state — `sessionReplayPeriod` itself
  // (what actually drives useCandleData's fetch) stays a ChartPanel.tsx
  // controlled input, see module doc.
  const [showSessionReplayPicker, setShowSessionReplayPicker] = useState(false);
  const [sessionReplayFromInput, setSessionReplayFromInput] = useState('');
  const [sessionReplayToInput, setSessionReplayToInput] = useState('');

  function parseDateTimeLocal(value: string): number | null {
    if (!value) return null;
    const ms = new Date(value).getTime();
    return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
  }

  // Derived from the picker's raw input strings — recomputed each render
  // (cheap) rather than kept in state, since it always follows directly
  // from sessionReplayFromInput/ToInput/timeframe.
  const sessionReplayFromSec = parseDateTimeLocal(sessionReplayFromInput);
  const sessionReplayToSec = parseDateTimeLocal(sessionReplayToInput);
  const sessionReplayEstimate =
    sessionReplayFromSec !== null &&
    sessionReplayToSec !== null &&
    sessionReplayToSec > sessionReplayFromSec
      ? (() => {
          const candles = Math.ceil(
            (sessionReplayToSec - sessionReplayFromSec) /
              TIMEFRAME_SECONDS[timeframe],
          );
          const pages = Math.ceil(candles / SESSION_REPLAY_CHUNK_SIZE);
          const level: 'ok' | 'warn' | 'block' =
            candles > SESSION_REPLAY_MAX_CANDLES
              ? 'block'
              : candles > SESSION_REPLAY_WARN_CANDLES
                ? 'warn'
                : 'ok';
          return { candles, pages, level };
        })()
      : null;

  // Replay ("live session player", §F): the single place the cursor moves —
  // used by step forward/back, the scrubber, and the autoplay tick below.
  // Reads `candlesRef.current.length` live (not a captured snapshot) since
  // panning near the left edge during replay can still trigger `loadMore`
  // and grow the array.
  //
  // Keeps the cursor bar centered (history to its left, reserved empty space
  // to its right, like a currently-forming live bar) by setting an explicit
  // logical range every tick — not `scrollToPosition`, which anchors the
  // latest bar to the *right edge*, not the middle. `followCursorRef` is the
  // on/off switch: a manual drag/zoom (see the mousedown/wheel listener
  // below) turns it off so playback stops fighting the user's pan, and
  // `centerOn`'s width is *read from* the current visible range so it
  // preserves whatever zoom level the user left it at rather than resetting.
  function centerOn(index: number) {
    const chart = chartController.getChart();
    if (!chart) return;
    const current = chart.timeScale().getVisibleLogicalRange();
    const width = current
      ? Math.max(10, current.to - current.from)
      : 2 * REPLAY_START_CONTEXT_BARS;
    chart.timeScale().setVisibleLogicalRange({
      from: index - width / 2,
      to: index + width / 2,
    });
  }

  /** Center the chart on the bar at (or nearest after) `time` — the
   * SignalsDock's click-to-navigate. During replay the cursor is moved
   * there instead, so the revealed candles/markers stay consistent with
   * the "no lookahead" contract rather than panning past the cursor. */
  function navigateToTime(time: number) {
    const candles = chartRenderController.candlesRef.current;
    if (candles.length === 0) return;
    let index = candles.findIndex((c) => (c.time as number) >= time);
    if (index === -1) index = candles.length - 1;
    if (replayActiveRef.current) {
      followCursorRef.current = true;
      seekTo(index);
      return;
    }
    centerOn(index);
  }

  // `keepFine` is set only by the tick-form autoplay loop, which sets
  // `replayFineTimeRef` itself right before committing a frame; every other
  // caller (manual step/scrub, jump-to-signal) leaves it default so the target
  // bar is shown fully closed rather than mid-formation from a stale sub-cursor.
  function seekTo(index: number, opts?: { keepFine?: boolean }) {
    if (!opts?.keepFine) replayFineTimeRef.current = null;
    const total = chartRenderController.candlesRef.current.length;
    if (total === 0) return;
    const clamped = Math.max(0, Math.min(index, total - 1));
    replayCursorIndexRef.current = clamped;
    // Decouple React state update from current call stack to break "Maximum update depth" loops
    requestAnimationFrame(() => setReplayCursorIndex(clamped));
    const chart = chartController.getChart();
    // Not following: capture the user's current view before `paintUpTo()`
    // touches the series data, and restore it exactly afterward — immune to
    // whatever `setData()` itself does to the visible range internally, so
    // a manual pan/zoom is never fought no matter how fast replay is ticking.
    const preservedRange = followCursorRef.current
      ? null
      : chart?.timeScale().getVisibleLogicalRange();
    chartRenderController.paintUpTo();

    // Cancel any pending animation frame to prevent layout queue accumulation
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (followCursorRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        animationFrameRef.current = null;
        if (followCursorRef.current) {
          centerOn(clamped);
        }
      });
    } else if (chart && preservedRange && !isMouseDownRef.current) {
      animationFrameRef.current = requestAnimationFrame(() => {
        animationFrameRef.current = null;
        if (!followCursorRef.current && !isMouseDownRef.current) {
          chart.timeScale().setVisibleLogicalRange(preservedRange);
        }
      });
    }
  }

  function handleEnterReplay() {
    const total = chartRenderController.candlesRef.current.length;
    const startIndex = Math.min(REPLAY_START_CONTEXT_BARS, Math.max(0, total - 1));
    replayCursorIndexRef.current = startIndex;
    setReplayCursorIndex(startIndex);
    replayFineTimeRef.current = null;
    setReplayPlaying(false);
    replayActiveRef.current = true;
    setReplayActive(true);
    followCursorRef.current = true;
    setFollowingCursor(true);
    if (!suppressAutoActivityDock) setShowActivityLogDock(true);
    chartRenderController.paintUpTo();
    // Re-fit the price scale to the (now much smaller) revealed window
    // instead of leaving the full report's price range applied — otherwise
    // the first bars render as a squashed sliver at one edge of the old
    // range. Center the time axis on the cursor with a fixed initial
    // window (not `centerOn`, which would inherit the old full-report
    // width and start zoomed miles out).
    chartController.getCandleSeries()?.priceScale().applyOptions({ autoScale: true });

    // Cancel any pending animation frame
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      chartController.getChart()?.timeScale().setVisibleLogicalRange({
        from: startIndex - REPLAY_START_CONTEXT_BARS,
        to: startIndex + REPLAY_START_CONTEXT_BARS,
      });
    });
  }

  function handleExitReplay() {
    replayActiveRef.current = false;
    setReplayActive(false);
    setReplayPlaying(false);
    replayFineTimeRef.current = null;
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    chartRenderController.paintUpTo();
    chartController.getCandleSeries()?.priceScale().applyOptions({ autoScale: true });
    chartController.getChart()?.timeScale().fitContent();
  }

  // Starts session replay: validates the picker's current from/to inputs
  // (mirrors the picker's own disabled-Start-button guard, in case this ever
  // gets called some other way) and hands the parsed range to
  // useCandleData's fetch effect via `sessionReplayPeriod`, which fetches it
  // (chunked, if needed) and auto-enters replay once it lands.
  function handleStartSessionReplay() {
    if (
      sessionReplayFromSec === null ||
      sessionReplayToSec === null ||
      sessionReplayToSec <= sessionReplayFromSec ||
      !sessionReplayEstimate ||
      sessionReplayEstimate.level === 'block'
    ) {
      return;
    }
    setShowSessionReplayPicker(false);
    setSessionReplayPeriod({ from: sessionReplayFromSec, to: sessionReplayToSec });
  }

  // Same as `handleStartSessionReplay`, but for a caller that already knows
  // the period it wants (the bot dock's "Replay" tab derives it from the
  // eyed bot's own signals/trades) instead of the picker's input strings.
  // Deliberately a *sibling* rather than an optional-arg overload of
  // `handleStartSessionReplay`: that one is wired straight to a button's
  // `onClick` and would receive a MouseEvent as its first argument.
  // Applies the identical max-candles guard and goes through the same
  // `setSessionReplayPeriod` path, so useCandleData fetches the range
  // (chunked if needed) and auto-enters replay once it lands.
  function startSessionReplayForRange(fromSec: number, toSec: number) {
    if (!Number.isFinite(fromSec) || !Number.isFinite(toSec)) return;
    if (toSec <= fromSec) return;
    const candles = Math.ceil((toSec - fromSec) / TIMEFRAME_SECONDS[timeframe]);
    if (candles > SESSION_REPLAY_MAX_CANDLES) return;
    setShowSessionReplayPicker(false);
    setSessionReplayPeriod({ from: Math.floor(fromSec), to: Math.floor(toSec) });
  }

  // Leaves session replay entirely (not just pausing the player) — clearing
  // `sessionReplayPeriod` re-triggers useCandleData's fetch effect, which
  // reloads live "now" candles and resubscribes to WS updates.
  function handleExitSessionReplay() {
    handleExitReplay();
    setSessionReplayPeriod(null);
  }

  function handleRecenterReplay() {
    followCursorRef.current = true;
    setFollowingCursor(true);
    chartController.getCandleSeries()?.priceScale().applyOptions({ autoScale: true });
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      if (followCursorRef.current) {
        centerOn(replayCursorIndexRef.current);
      }
    });
  }

  // Card click in SignalsDock's Trades tab: clicking the already-selected
  // trade clears it (same toggle as the Active Orders panel's rows);
  // clicking any other trade selects it and jumps the chart to its entry.
  function handleToggleTrade(index: number) {
    if (selectedTradeIndex === index) {
      setSelectedTradeIndex(null);
      return;
    }
    setSelectedTradeIndex(index);
    const trade = backtestTrades?.[index];
    if (trade) navigateToTime(trade.open_time);
  }

  // The Entry/Exit nav buttons always select (never toggle off) — they're
  // an explicit "look at this" action, not a selection toggle.
  function handleNavigateTrade(index: number, time: number) {
    setSelectedTradeIndex(index);
    navigateToTime(time);
  }

  // Row click in SignalsDock's Signals tab — same toggle as trades above.
  function handleToggleSignal(index: number) {
    if (selectedSignalIndex === index) {
      setSelectedSignalIndex(null);
      return;
    }
    setSelectedSignalIndex(index);
    const signal = backtestSignals?.[index];
    if (signal) navigateToTime(signal.time);
  }

  // A manual drag, touch-pan, or wheel/scroll zoom means the user wants to
  // look at something other than the cursor bar — stop fighting it and
  // leave the view exactly where they left it across every subsequent tick,
  // until they explicitly re-engage via the "Center" button in
  // ReplayControls. Only active during replay; the live/static views never
  // had auto-follow to begin with.
  useEffect(() => {
    if (!replayActive && !sharedReplayActive) {
      isMouseDownRef.current = false;
      return;
    }
    const container = chartController.containerRef.current;
    if (!container) return;

    const disengage = () => {
      followCursorRef.current = false;
      setFollowingCursor(false);
      isMouseDownRef.current = true;
    };

    const handleMouseUp = () => {
      isMouseDownRef.current = false;
    };

    container.addEventListener('mousedown', disengage);
    container.addEventListener('touchstart', disengage, { passive: true });
    container.addEventListener('wheel', disengage, { passive: true });

    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleMouseUp);

    return () => {
      container.removeEventListener('mousedown', disengage);
      container.removeEventListener('touchstart', disengage);
      container.removeEventListener('wheel', disengage);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
    // chartController.containerRef is a stable ref object returned from
    // useChartEngine — omitted deliberately, same as every other effect in
    // this file that reads it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayActive]);

  // One autoplay step. Mutates the replay refs but does NOT paint — the tick
  // below paints once after running however many steps the speed calls for, so
  // the expensive recompute inside `paintUpTo()` still fires at most once per
  // tick no matter how fast playback is. Returns false at the end of the data.
  //
  // Tick-form (finer candles loaded + toggle on): walk a sub-cursor
  // (`replayFineTimeRef`) through the current bar's finer constituents so the
  // bar visibly forms; once the bar's finer candles are exhausted, close it and
  // open the next one anchored at *its* first finer candle (so the new bar
  // starts as a thin sliver rather than flashing fully-closed for a frame).
  // A bar with no finer coverage (e.g. before the broker's M1 history begins)
  // just falls through to the whole-bar advance, same as tick-form off.
  function advanceReplayState(): boolean {
    const candles = chartRenderController.candlesRef.current;
    const total = candles.length;
    if (total === 0) return false;
    const idx = replayCursorIndexRef.current;
    const fine = finerCandlesRef.current;
    const current = candles[idx];
    if (tickFormRef.current && fine.length > 0 && current) {
      const step = TIMEFRAME_SECONDS[timeframe];
      const barOpen = current.time as number;
      const barEnd = barOpen + step;
      const fineTime = replayFineTimeRef.current;
      // `barOpen - 1` when the sub-cursor is idle/before this bar means "start
      // from this bar's first finer candle"; times are whole seconds so this is
      // exactly `>= barOpen`.
      const from = fineTime == null || fineTime < barOpen ? barOpen - 1 : fineTime;
      const next = fine.find(
        (c) => (c.time as number) > from && (c.time as number) < barEnd,
      );
      if (next) {
        replayFineTimeRef.current = next.time as number;
        return true; // same bar, still forming
      }
      // This bar's finer candles are exhausted — reveal the next bar.
      if (idx + 1 >= total) return false;
      const newOpen = candles[idx + 1].time as number;
      const firstFine = fine.find(
        (c) => (c.time as number) >= newOpen && (c.time as number) < newOpen + step,
      );
      replayFineTimeRef.current = firstFine ? (firstFine.time as number) : null;
      replayCursorIndexRef.current = idx + 1;
      return true;
    }
    // Whole-bar reveal (tick-form off or no finer data).
    replayFineTimeRef.current = null;
    if (idx + 1 >= total) return false;
    replayCursorIndexRef.current = idx + 1;
    return true;
  }

  // Autoplay: advances the replay at an interval scaled by `replaySpeed`,
  // pausing once it reaches the last loaded bar. `paintUpTo()` re-runs the
  // full indicator/structure/pattern recompute (and ChartPanel.tsx's
  // trade-drawing effect reruns on every cursor change too) — expensive
  // enough over a few hundred bars that firing it once per *step* at high
  // speed (e.g. every ~37ms at 16x) visibly stutters. Ticks are floored at
  // MIN_TICK_MS and more steps run per tick to compensate (whole bars when
  // tick-form is off, finer sub-frames when on), while `seekTo` paints just
  // once per tick, capping how often the recompute actually runs regardless
  // of the speed the user picks.
  useEffect(() => {
    if (!replayPlaying) return;
    const MIN_TICK_MS = 100;
    const rawIntervalMs = 600 / replaySpeed;
    const tickMs = Math.max(MIN_TICK_MS, rawIntervalMs);
    const stepsPerTick = Math.max(1, Math.round(tickMs / rawIntervalMs));
    const id = setInterval(() => {
      let reachedEnd = false;
      for (let s = 0; s < stepsPerTick; s++) {
        if (!advanceReplayState()) {
          reachedEnd = true;
          break;
        }
      }
      // Commit the accumulated refs with a single paint+center; `keepFine`
      // preserves the sub-cursor `advanceReplayState` just set (a plain
      // `seekTo` would clear it and paint a whole bar).
      seekTo(replayCursorIndexRef.current, { keepFine: true });
      if (reachedEnd) setReplayPlaying(false);
    }, tickMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayPlaying, replaySpeed]);

  return {
    // Owned state.
    replaySpeed,
    setReplaySpeed,
    showSessionReplayPicker,
    setShowSessionReplayPicker,
    sessionReplayFromInput,
    setSessionReplayFromInput,
    sessionReplayToInput,
    setSessionReplayToInput,
    sessionReplayEstimate,
    lastRevealedSignatureRef,

    // Handlers.
    centerOn,
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
  };
}

export type ReplayEngine = ReturnType<typeof useReplayEngine>;
