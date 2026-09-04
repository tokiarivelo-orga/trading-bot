'use client';

/**
 * Split-window chart layout (§): up to 4 chart windows on screen at once, all
 * showing the same symbol. Window 1 is always the full ChartPanel — drawing
 * tools, indicators dock, order popovers, trade placement, backtest/live-bot
 * overlay, everything unchanged from the single-chart view. Windows 2-4 are
 * lightweight MiniChartPanels (candles + volume + trade markers only) with
 * their own independent timeframe, so this is a multi-timeframe layout
 * (TradingView-style), not 4 independent charts — see the frontend-feature
 * skill conversation this was scoped from.
 *
 * Replay sync: only the primary window drives entering/exiting replay,
 * playing, and seeking (its existing ReplayControls/SessionReplayPicker UI,
 * untouched). This component just mirrors that window's replay session
 * (`onReplaySessionChange`/`onReplayCursorTime`, added to ChartPanel for
 * this feature) into `sharedReplay` state, which every MiniChartPanel reads
 * to fetch the same period at its own timeframe and clip its display to the
 * same cursor position. `sessionPeriod` covers both an ad-hoc session replay
 * (the picked from/to) and a saved backtest report's replay (its trades'/
 * signals' own time bounds, computed in useBacktestData.ts) — ChartPanel
 * picks whichever applies before calling `onReplaySessionChange`, so this
 * component doesn't need to know which one it's looking at. Only when
 * neither is active (replay off, or the live-bot eye view) do secondary
 * windows fall back to their own live view.
 */

import { useCallback, useEffect, useRef, useState, type ComponentProps } from 'react';
import type { Candle } from '@/shared/api/client';
import { isTimeframe } from './chartFormat';
import { ChartPanel } from './ChartPanel';
import { ChartToolbar, type ChartToolbarProps } from './ChartToolbar';
import { ReplayControls } from './ReplayControls';
import { SessionReplayPicker } from './SessionReplayPicker';
import type { ReplayUIState, SharedReplaySession } from './types';

const WINDOW_COUNT_KEY = 'tb.chartWindowCount';
const WINDOW_TIMEFRAMES_KEY = 'tb.chartWindowTimeframes';
const WINDOW_COUNT_QUERY_KEY = 'windows';
const WINDOW_TIMEFRAMES_QUERY_KEY = 'tfs';
const MAX_WINDOWS = 4;
// Distinct default timeframes for windows 2-4 so a first-time split shows
// genuinely different views rather than three identical M5 charts.
const DEFAULT_SECONDARY_TIMEFRAMES: Candle['timeframe'][] = ['M15', 'H1', 'H4'];

function loadWindowCount(): number {
  try {
    const urlCount = Number(new URLSearchParams(window.location.search).get(WINDOW_COUNT_QUERY_KEY));
    if (!isNaN(urlCount) && urlCount >= 1 && urlCount <= MAX_WINDOWS) {
      return urlCount;
    }
    const stored = Number(localStorage.getItem(WINDOW_COUNT_KEY));
    return stored >= 1 && stored <= MAX_WINDOWS ? stored : 1;
  } catch {
    return 1;
  }
}

function loadWindowTimeframes(): Candle['timeframe'][] {
  try {
    const urlTfs = new URLSearchParams(window.location.search).get(WINDOW_TIMEFRAMES_QUERY_KEY);
    if (urlTfs) {
      const parsed = urlTfs.split(',');
      return DEFAULT_SECONDARY_TIMEFRAMES.map((fallback, i) =>
        isTimeframe(parsed[i]) ? (parsed[i] as Candle['timeframe']) : fallback,
      );
    }
    const raw = localStorage.getItem(WINDOW_TIMEFRAMES_KEY);
    if (!raw) return DEFAULT_SECONDARY_TIMEFRAMES;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return DEFAULT_SECONDARY_TIMEFRAMES;
    return DEFAULT_SECONDARY_TIMEFRAMES.map((fallback, i) =>
      isTimeframe(parsed[i]) ? parsed[i] : fallback,
    );
  } catch {
    return DEFAULT_SECONDARY_TIMEFRAMES;
  }
}

function saveWindowTimeframes(timeframes: Candle['timeframe'][]) {
  try {
    localStorage.setItem(WINDOW_TIMEFRAMES_KEY, JSON.stringify(timeframes));
  } catch {
    // Ignore blocked/full localStorage — timeframes just won't persist.
  }
}

type ChartPanelProps = ComponentProps<typeof ChartPanel>;

// Grid arrangement per window count — 3 gives the primary window the full
// left column (its toolbars need more width than a mini window) with the
// two secondary windows stacked to the right, matching common multi-chart
// layout presets rather than an uneven 3-equal-column split.
const GRID_CLASS: Record<number, string> = {
  1: 'grid-cols-1 grid-rows-1',
  2: 'grid-cols-2 grid-rows-1',
  3: 'grid-cols-2 grid-rows-2',
  4: 'grid-cols-2 grid-rows-2',
};

export function MultiChartLayout(props: ChartPanelProps) {
  const [windowCount, setWindowCount] = useState(loadWindowCount);
  const [secondaryTimeframes, setSecondaryTimeframes] = useState(loadWindowTimeframes);

  useEffect(() => {
    try {
      const url = new URL(window.location.href);
      if (windowCount > 1) {
        url.searchParams.set(WINDOW_COUNT_QUERY_KEY, String(windowCount));
        url.searchParams.set(WINDOW_TIMEFRAMES_QUERY_KEY, secondaryTimeframes.join(','));
      } else {
        url.searchParams.delete(WINDOW_COUNT_QUERY_KEY);
        url.searchParams.delete(WINDOW_TIMEFRAMES_QUERY_KEY);
      }
      window.history.replaceState(null, '', url);
    } catch {
      // Ignore URL parsing errors during SSR or edge environments.
    }
  }, [windowCount, secondaryTimeframes]);

  const [sharedReplay, setSharedReplay] = useState<SharedReplaySession | null>(null);
  const [syncedWindows, setSyncedWindows] = useState<Record<number, boolean>>({});
  const windowReplayUIRef = useRef<Record<number, ReplayUIState>>({});
  const [activeReplayUI, setActiveReplayUI] = useState<{ windowIndex: number; ui: ReplayUIState } | null>(null);

  const [selectedWindow, setSelectedWindow] = useState(0);
  const selectedWindowRef = useRef(0);
  selectedWindowRef.current = selectedWindow;

  const windowToolbarsRef = useRef<Record<number, ChartToolbarProps>>({});
  const [activeToolbarProps, setActiveToolbarProps] = useState<ChartToolbarProps | null>(null);
  // Live per-window timeframe, for the "Sync Replay" chips below — kept
  // separate from `activeToolbarProps` (which only updates for the
  // *selected* window) and from `secondaryTimeframes` (which only reflects
  // each window's timeframe at its own mount time, never a later change).
  // Without this, a non-selected window's timeframe change updated
  // `windowToolbarsRef` (a ref, no re-render) but nothing else, so the chip
  // for that window kept showing its stale initial timeframe indefinitely.
  const [windowTimeframes, setWindowTimeframes] = useState<Record<number, Candle['timeframe']>>({});

  function handleSelectWindow(idx: number) {
    setSelectedWindow(idx);
    selectedWindowRef.current = idx;
    if (windowToolbarsRef.current[idx]) {
      setActiveToolbarProps(windowToolbarsRef.current[idx]);
    }
  }

  const handleToolbarStateChange = useCallback((idx: number, tp: ChartToolbarProps) => {
    windowToolbarsRef.current[idx] = tp;
    setWindowTimeframes((prev) =>
      prev[idx] === tp.timeframe ? prev : { ...prev, [idx]: tp.timeframe },
    );
    setSharedReplay((prev) => {
      if (prev?.masterIndex === idx && prev.masterTimeframe !== tp.timeframe) {
        return { ...prev, masterTimeframe: tp.timeframe };
      }
      return prev;
    });
    if (idx === selectedWindowRef.current) {
      setActiveToolbarProps((prev) => {
        if (!prev) return tp;
        if (
          prev.symbol === tp.symbol &&
          prev.timeframe === tp.timeframe &&
          prev.showTfDropdown === tp.showTfDropdown &&
          prev.showIndicatorsDock === tp.showIndicatorsDock &&
          prev.manualIndicatorsCount === tp.manualIndicatorsCount &&
          prev.showDrawingsList === tp.showDrawingsList &&
          prev.drawingsListCount === tp.drawingsListCount &&
          prev.showDrawingToolbar === tp.showDrawingToolbar &&
          prev.showCustomCodeEditor === tp.showCustomCodeEditor &&
          prev.showActivityLogDock === tp.showActivityLogDock &&
          prev.showOverlaysDropdown === tp.showOverlaysDropdown &&
          prev.showSeparators === tp.showSeparators &&
          prev.showSpreadLine === tp.showSpreadLine &&
          prev.showVolume === tp.showVolume &&
          prev.showTradeBadges === tp.showTradeBadges &&
          prev.orderLineVisible === tp.orderLineVisible &&
          prev.showOrderLineSettings === tp.showOrderLineSettings &&
          (prev.backtestReportId ?? null) === (tp.backtestReportId ?? null) &&
          (prev.sessionReplayPeriod?.from ?? null) === (tp.sessionReplayPeriod?.from ?? null) &&
          (prev.sessionReplayPeriod?.to ?? null) === (tp.sessionReplayPeriod?.to ?? null) &&
          prev.showSessionReplayPicker === tp.showSessionReplayPicker &&
          (prev.drawingTool ?? null) === (tp.drawingTool ?? null) &&
          (prev.pendingAnchorCount ?? null) === (tp.pendingAnchorCount ?? null) &&
          (prev.spreadPoints ?? null) === (tp.spreadPoints ?? null)
        ) {
          return prev;
        }
        return tp;
      });
    }
  }, []);

  const handleReplayUIChange = useCallback((idx: number, ui: ReplayUIState | null) => {
    if (ui) {
      windowReplayUIRef.current[idx] = ui;
    } else {
      delete windowReplayUIRef.current[idx];
    }
    setActiveReplayUI((prev) => {
      const activeIdx = Object.keys(windowReplayUIRef.current).map(Number).find((i) => {
        const s = windowReplayUIRef.current[i];
        return s?.showPicker || s?.sessionPeriod || s?.replayActive;
      });
      if (activeIdx !== undefined) {
        const cur = windowReplayUIRef.current[activeIdx];
        if (
          prev &&
          prev.windowIndex === activeIdx &&
          prev.ui.showPicker === cur.showPicker &&
          (prev.ui.pickerProps?.fromValue ?? null) === (cur.pickerProps?.fromValue ?? null) &&
          (prev.ui.pickerProps?.toValue ?? null) === (cur.pickerProps?.toValue ?? null) &&
          (prev.ui.pickerProps?.estimate?.candles ?? null) === (cur.pickerProps?.estimate?.candles ?? null) &&
          (prev.ui.pickerProps?.estimate?.pages ?? null) === (cur.pickerProps?.estimate?.pages ?? null) &&
          (prev.ui.pickerProps?.estimate?.level ?? null) === (cur.pickerProps?.estimate?.level ?? null) &&
          (prev.ui.sessionPeriod?.from ?? null) === (cur.sessionPeriod?.from ?? null) &&
          (prev.ui.sessionPeriod?.to ?? null) === (cur.sessionPeriod?.to ?? null) &&
          (prev.ui.loadingPage?.page ?? null) === (cur.loadingPage?.page ?? null) &&
          (prev.ui.loadingPage?.loaded ?? null) === (cur.loadingPage?.loaded ?? null) &&
          prev.ui.replayActive === cur.replayActive &&
          (prev.ui.replayControlsProps?.playing ?? null) === (cur.replayControlsProps?.playing ?? null) &&
          (prev.ui.replayControlsProps?.speed ?? null) === (cur.replayControlsProps?.speed ?? null) &&
          (prev.ui.replayControlsProps?.cursorIndex ?? null) === (cur.replayControlsProps?.cursorIndex ?? null) &&
          (prev.ui.replayControlsProps?.totalBars ?? null) === (cur.replayControlsProps?.totalBars ?? null) &&
          (prev.ui.replayControlsProps?.currentTime ?? null) === (cur.replayControlsProps?.currentTime ?? null) &&
          (prev.ui.replayControlsProps?.following ?? null) === (cur.replayControlsProps?.following ?? null)
        ) {
          return prev;
        }
        return { windowIndex: activeIdx, ui: cur };
      }
      return null;
    });
  }, []);

  const handleReplaySessionChange = useCallback(
    (idx: number, session: { active: boolean; sessionPeriod: { from: number; to: number } | null }) => {
      if (!session.active && !session.sessionPeriod) {
        setSharedReplay((prev) => (prev?.masterIndex === idx ? null : prev));
      } else {
        setSharedReplay((prev) => ({
          active: session.active,
          sessionPeriod: session.sessionPeriod,
          cursorTime: prev?.masterIndex === idx ? prev.cursorTime : null,
          masterIndex: idx,
          masterTimeframe: windowToolbarsRef.current[idx]?.timeframe ?? null,
        }));
      }
    },
    [],
  );

  const handleReplayCursorTime = useCallback((idx: number, time: number | null) => {
    setSharedReplay((prev) => {
      if (prev?.masterIndex === idx) {
        const curTf = windowToolbarsRef.current[idx]?.timeframe ?? prev.masterTimeframe;
        if (prev.cursorTime !== time || prev.masterTimeframe !== curTf) {
          return { ...prev, cursorTime: time, masterTimeframe: curTf };
        }
      }
      return prev;
    });
  }, []);

  function setCount(count: number) {
    setWindowCount(count);
    if (selectedWindow >= count) {
      const newIdx = Math.max(0, count - 1);
      handleSelectWindow(newIdx);
    }
    try {
      localStorage.setItem(WINDOW_COUNT_KEY, String(count));
    } catch {
      // Ignore blocked/full localStorage — window count just won't persist.
    }
  }

  const setSecondaryTimeframe = useCallback((slot: number, tf: Candle['timeframe']) => {
    setSecondaryTimeframes((prev) => {
      if (prev[slot] === tf) return prev;
      const updated = prev.map((t, i) => (i === slot ? tf : t));
      saveWindowTimeframes(updated);
      return updated;
    });
  }, []);

  // Removes this specific window's slot (not just the last one) so closing
  // window 2 out of {2,3,4} leaves {3,4} on screen, not {2,3} — the other
  // open windows' timeframes/positions shouldn't shuffle just because a
  // different one closed.
  function closeSecondaryWindow(slot: number) {
    if (selectedWindow === slot + 1) {
      handleSelectWindow(0);
    } else if (selectedWindow > slot + 1) {
      handleSelectWindow(selectedWindow - 1);
    }
    const remaining = secondaryTimeframes.filter((_, i) => i !== slot);
    const padded = [...remaining, DEFAULT_SECONDARY_TIMEFRAMES[remaining.length] ?? 'M15'];
    setSecondaryTimeframes(padded);
    saveWindowTimeframes(padded);
    setCount(Math.max(1, windowCount - 1));
  }

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
      {activeToolbarProps ? (
        <ChartToolbar
          {...activeToolbarProps}
          onSelectTimeframe={(tf) => windowToolbarsRef.current[selectedWindow]?.onSelectTimeframe(tf)}
          onToggleTfDropdown={() => windowToolbarsRef.current[selectedWindow]?.onToggleTfDropdown()}
          onScrollToLatest={() => windowToolbarsRef.current[selectedWindow]?.onScrollToLatest()}
          onResetZoom={() => windowToolbarsRef.current[selectedWindow]?.onResetZoom()}
          onToggleIndicatorsDock={() => windowToolbarsRef.current[selectedWindow]?.onToggleIndicatorsDock()}
          onToggleDrawingsList={() => windowToolbarsRef.current[selectedWindow]?.onToggleDrawingsList()}
          onToggleDrawingToolbar={() => windowToolbarsRef.current[selectedWindow]?.onToggleDrawingToolbar()}
          onToggleCodeEditor={() => windowToolbarsRef.current[selectedWindow]?.onToggleCodeEditor()}
          onToggleActivityLogDock={() => windowToolbarsRef.current[selectedWindow]?.onToggleActivityLogDock()}
          onToggleOverlaysDropdown={() => windowToolbarsRef.current[selectedWindow]?.onToggleOverlaysDropdown()}
          onToggleSeparators={() => windowToolbarsRef.current[selectedWindow]?.onToggleSeparators()}
          onToggleSpreadLine={() => windowToolbarsRef.current[selectedWindow]?.onToggleSpreadLine()}
          onToggleVolume={() => windowToolbarsRef.current[selectedWindow]?.onToggleVolume()}
          onToggleTradeBadges={() => windowToolbarsRef.current[selectedWindow]?.onToggleTradeBadges()}
          onToggleOrderLinesVisible={() => windowToolbarsRef.current[selectedWindow]?.onToggleOrderLinesVisible()}
          onToggleOrderLineSettings={() => windowToolbarsRef.current[selectedWindow]?.onToggleOrderLineSettings()}
          onSessionReplayToggle={() => windowToolbarsRef.current[selectedWindow]?.onSessionReplayToggle()}
          windowCount={windowCount}
          onSelectWindowCount={setCount}
        />
      ) : (
        <div className="h-11 border-b border-line bg-panel/90 px-3 py-1.5 animate-pulse rounded-md" />
      )}

      {/* Centralized Global Replay Controls & Multi-Window Sync Selector */}
      {activeReplayUI?.ui.showPicker && activeReplayUI.ui.pickerProps && (
        <div className="rounded-md border border-line bg-panel shadow-xs overflow-hidden">
          <SessionReplayPicker {...activeReplayUI.ui.pickerProps} />
        </div>
      )}
      {activeReplayUI?.ui.sessionPeriod && (
        <div className="flex items-center gap-2 rounded-md border border-line bg-accent/10 px-4 py-1.5 text-xs text-accent shadow-xs">
          <span>
            Session replay —{' '}
            {new Date(activeReplayUI.ui.sessionPeriod.from * 1000)
              .toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}{' '}
            →{' '}
            {new Date(activeReplayUI.ui.sessionPeriod.to * 1000)
              .toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
            {activeReplayUI.ui.loadingPage &&
              ` — loading… page ${activeReplayUI.ui.loadingPage.page} (${activeReplayUI.ui.loadingPage.loaded.toLocaleString()} candles so far)`}
          </span>
        </div>
      )}
      {activeReplayUI?.ui.replayActive && activeReplayUI.ui.replayControlsProps && (
        <div className="rounded-md border border-line bg-panel shadow-xs overflow-hidden">
          <ReplayControls
            {...activeReplayUI.ui.replayControlsProps}
            prefixContent={
              windowCount > 1 ? (
                <div className="flex flex-wrap items-center gap-1.5 border-r border-line/80 pr-2.5 mr-0.5 my-0.5">
                  <span className="font-semibold text-ink-muted text-[11px] uppercase tracking-wider select-none">
                    Sync Replay:
                  </span>
                  {Array.from({ length: windowCount }, (_, i) => {
                    const isMaster = i === activeReplayUI.windowIndex;
                    const isSynced = syncedWindows[i] !== false;
                    const tf =
                      windowTimeframes[i] ??
                      (i === 0
                        ? (activeToolbarProps?.timeframe ?? 'M15')
                        : (secondaryTimeframes[i - 1] ?? DEFAULT_SECONDARY_TIMEFRAMES[i - 1]));
                    return (
                      <button
                        key={i}
                        type="button"
                        disabled={isMaster}
                        onClick={() =>
                          setSyncedWindows((prev) => ({
                            ...prev,
                            [i]: !isSynced,
                          }))
                        }
                        className={`flex cursor-pointer items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium transition-all ${
                          isMaster
                            ? 'bg-accent/20 text-accent border border-accent/50 cursor-default font-bold shadow-2xs'
                            : isSynced
                              ? 'bg-accent/10 text-accent border border-accent/30 hover:bg-accent/20 hover:border-accent/50'
                              : 'bg-bg/80 text-ink-muted border border-line/80 opacity-70 hover:opacity-100 hover:text-ink hover:border-line'
                        }`}
                        title={
                          isMaster
                            ? `Window ${i + 1} (${tf}) is driving this replay session`
                            : isSynced
                              ? `Window ${i + 1} (${tf}) is synchronized — click to unlink and return to live view`
                              : `Window ${i + 1} (${tf}) is unlinked (live view) — click to sync with replay session`
                        }
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            isMaster ? 'bg-accent animate-pulse' : isSynced ? 'bg-accent' : 'bg-ink-muted/40'
                          }`}
                        />
                        <span>
                          Win {i + 1} ({tf})
                        </span>
                        {isMaster && (
                          <span className="text-[9px] uppercase bg-accent text-white font-black px-1 py-0.2 rounded leading-none ml-0.5">
                            Master
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              ) : null
            }
          />
        </div>
      )}

      <div className={`grid min-h-0 flex-1 gap-2 ${GRID_CLASS[windowCount]}`}>
        <div
          onPointerDownCapture={() => handleSelectWindow(0)}
          className={`relative flex min-h-0 min-w-0 flex-1 flex-col rounded-md transition-all duration-200 ${
            windowCount === 3 ? 'row-span-2' : ''
          } ${
            windowCount > 1 && selectedWindow === 0
              ? 'ring-2 ring-accent border border-accent/80 shadow-[0_0_15px_rgba(41,98,255,0.25)] z-10'
              : windowCount > 1
                ? 'border border-line hover:border-line/80 opacity-95 hover:opacity-100'
                : ''
          }`}
        >
          <ChartPanel
            {...props}
            windowIndex={0}
            windowCount={windowCount}
            selectedWindowIndex={selectedWindow}
            onSelectWindow={handleSelectWindow}
            hideToolbar={true}
            onToolbarStateChange={handleToolbarStateChange}
            sharedReplay={(sharedReplay?.masterIndex ?? 0) !== 0 && syncedWindows[0] !== false && sharedReplay?.active ? sharedReplay : null}
            onReplaySessionChange={handleReplaySessionChange}
            onReplayCursorTime={handleReplayCursorTime}
            onReplayUIChange={handleReplayUIChange}
          />
        </div>
        {Array.from({ length: windowCount - 1 }, (_, i) => {
          const slot = i + 1;
          return (
            <div
              key={slot}
              onPointerDownCapture={() => handleSelectWindow(slot)}
              className={`relative flex min-h-0 min-w-0 flex-1 flex-col rounded-md transition-all duration-200 ${
                selectedWindow === slot
                  ? 'ring-2 ring-accent border border-accent/80 shadow-[0_0_15px_rgba(41,98,255,0.25)] z-10'
                  : 'border border-line hover:border-line/80 opacity-95 hover:opacity-100'
              }`}
            >
              <ChartPanel
                {...props}
                windowIndex={slot}
                windowCount={windowCount}
                selectedWindowIndex={selectedWindow}
                onSelectWindow={handleSelectWindow}
                onCloseWindow={() => closeSecondaryWindow(i)}
                initialTimeframe={secondaryTimeframes[i] ?? DEFAULT_SECONDARY_TIMEFRAMES[i]}
                onTimeframeChange={(tf) => setSecondaryTimeframe(i, tf)}
                sharedReplay={sharedReplay?.masterIndex !== slot && syncedWindows[slot] !== false && sharedReplay?.active ? sharedReplay : null}
                hideToolbar={true}
                onToolbarStateChange={handleToolbarStateChange}
                onReplaySessionChange={handleReplaySessionChange}
                onReplayCursorTime={handleReplayCursorTime}
                onReplayUIChange={handleReplayUIChange}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
