import type { RefObject } from 'react';
import type {
  IChartApi,
  ISeriesApi,
  ISeriesMarkersPluginApi,
  Time,
} from 'lightweight-charts';
import type { DrawingManager } from 'lightweight-charts-drawing';
import {
  type BacktestSignal,
  type Candle,
  type PositionOut,
  type SymbolInfo,
} from '@/shared/api/client';

/** Manually added indicator (via IndicatorsDock), independent of whatever
 * the active strategy's spec auto-draws — see `recomputeIndicators` below,
 * which plots both together. */
export type ManualIndicatorType =
  | 'ema'
  | 'sma'
  | 'rsi'
  | 'macd'
  | 'bollinger'
  | 'vwap'
  | 'atr'
  | 'volatility'
  | 'volume_profile'
  | 'structure'
  | 'qml'
  | 'snd'
  | 'snd_v2'
  | 'base'
  | 'patterns'
  | 'smc'
  | 'custom';

export type IndicatorLineStyle = 'solid' | 'dashed' | 'dotted';
export type IndicatorLineWidth = 1 | 2 | 3 | 4;

export interface ManualIndicator {
  id: string;
  type: ManualIndicatorType;
  period: number;
  color: string;
  lineStyle?: IndicatorLineStyle;
  lineWidth?: IndicatorLineWidth;
  label: string;
  /** Set only when type === 'custom': the saved backend indicator's id
   * (GET /indicators/{id}) whose compute() output this instance plots.
   * Mutually exclusive with `previewCode`. */
  indicatorId?: string;
  /** Set only when type === 'custom' and this instance is ad-hoc code not
   * (yet) saved as an indicator — computed via POST /indicators/preview
   * instead of GET /indicators/{id}/compute. Mutually exclusive with
   * `indicatorId`. See IndicatorsDock's "Write new code…" option. */
  previewCode?: string;
  /** Set only when type === 'volume_profile': number of price buckets the
   * histogram bins tick_volume into. Reuses `period` for the lookback (how
   * many recent candles to profile) to match the existing optional-field
   * pattern rather than adding a whole parallel settings shape. */
  bucketCount?: number;
  /** Set only when type === 'volume_profile': which side of the pane the
   * horizontal bars extend from. Defaults to 'right' when unset. */
  side?: 'left' | 'right';
  /** Which chart "screen" (pane) this indicator renders on — see
   * `paneTargets.ts` for the full contract and resolution rules.
   * `'main'` = the price pane; `{ paneKey }` = a bottom pane identified by a
   * stable logical key (NOT a raw pane index, which shifts as panes are
   * added/removed) — two indicators sharing the same `paneKey` render
   * stacked in the same real chart pane. `undefined` = legacy default
   * (oscillators get their own bottom pane, everything else goes to main),
   * preserving where already-persisted indicators render. */
  paneTarget?: 'main' | { paneKey: string };
  /** If true, the indicator is temporarily hidden from the chart but remains in the list. */
  hidden?: boolean;
}

export type DrawingToolType =
  | 'trend-line'
  | 'extended-line'
  | 'horizontal-line'
  | 'vertical-line'
  | 'rectangle'
  | 'fib-retracement'
  | 'parallel-channel'
  | 'circle'
  | 'long-position'
  | 'short-position'
  | 'price-label'
  | 'text-annotation';

/** Chart-annotation detail for one zone rectangle drawing, keyed by that
 * drawing's id in the `DrawingManager` (see `ChartEngineController.getZoneMetaMap`)
 * — the `Rectangle` primitive itself has no free-form metadata field, so
 * this side map is how a click handler resolves "what indicator drew this
 * and what does it mean" back from a hit-tested drawing id. Populated
 * alongside `manager.addDrawing(...)` by whichever hook owns that zone type
 * (`useIndicators.ts` for qml/snd/snd_v2, `useBacktestData.ts` for the
 * per-trade backend zone) and cleared on the same rebuild cycle as the
 * drawings themselves, so it never outlives the rectangle it describes. */
export interface ZoneMeta {
  indicator: 'qml' | 'snd' | 'snd_v2' | 'trade_zone' | 'smc';
  indicatorLabel: string;
  /** Zone subtype, e.g. "RBR"/"DBD"/"RBD"/"DBR"/"QML"/"QML_INV"/"DZ"/"SZ" —
   * null when the source (an older backend report/trade) didn't report one. */
  pattern: string | null;
  kind: 'demand' | 'supply';
  priceLow: number;
  priceHigh: number;
  timeStart: number; // epoch seconds UTC
  /** Null while the zone is still open/fresh (right edge follows the latest candle). */
  timeEnd: number | null;
  /** 'fresh' = still valid/untouched, 'touched' = consumed by a retest/fill,
   * 'triggered' = the per-trade backend zone, which has no fresh/touched
   * concept of its own since a trade was, by definition, taken from it. */
  state: 'fresh' | 'touched' | 'triggered';
  /** Free-form extra facts worth surfacing in the tooltip but not worth a
   * dedicated field — e.g. QML's neckline/head price, or a trade zone's
   * originating trade reason. */
  extra?: Record<string, string | number>;
}

/** Floating read-only popover state for a clicked zone rectangle — opened by
 * ChartPanel's zone-click effect (`chart.subscribeClick`, same API the
 * click-to-trade handler already uses) once `manager.hitTest(point)` resolves
 * to a drawing id present in `ChartEngineController.getZoneMetaMap()`. */
export interface ZoneTooltipState {
  x: number;
  y: number;
  meta: ZoneMeta;
  containerWidth: number;
  containerHeight: number;
}

/** Floating read-only popover state for a clicked rejected/vetoed signal
 * marker — opened by ChartPanel's signal-click effect. The markers plugin
 * exposes no hit API, so that effect resolves the clicked bar via
 * `nearestCandleTime` and keeps every non-`opened` signal on it. */
export interface SignalTooltipState {
  x: number;
  y: number;
  signals: BacktestSignal[];
  containerWidth: number;
  containerHeight: number;
}

/** Multi-chart layout (split-window §): the primary ChartPanel's replay
 * session/cursor, mirrored into secondary MiniChartPanel windows so they
 * follow the same replayed period at their own timeframe. `sessionPeriod` is
 * an ad-hoc session replay's picked from/to, or — while viewing a saved
 * backtest report — that report's own trades'/signals' time bounds
 * (`useBacktestData.ts`'s `backtestPeriod`); either way secondary windows
 * fetch and cursor-clip it the same way. Null only while nothing is being
 * replayed (or during the live-bot eye view), meaning secondary windows have
 * nothing to sync to and fall back to their own live view. */
export interface SharedReplaySession {
  active: boolean;
  sessionPeriod: { from: number; to: number } | null;
  cursorTime: number | null;
  /** In multi-window layouts, identifies which window index initiated and drives the master replay clock. */
  masterIndex?: number;
  /** The master window's current timeframe — lets a coarser follower window
   * know both whether it needs to synthesize a forming bar (its own timeframe
   * is coarser than this) and which finer timeframe's candles to aggregate
   * from. Null while unknown (not yet populated). */
  masterTimeframe: Candle['timeframe'] | null;
}

export interface ReplayUIState {
  showPicker: boolean;
  pickerProps: {
    fromValue: string;
    toValue: string;
    onFromChange: (val: string) => void;
    onToChange: (val: string) => void;
    estimate: {
      candles: number;
      pages: number;
      level: 'ok' | 'warn' | 'block';
    } | null;
    onCancel: () => void;
    onStart: () => void;
  } | null;
  sessionPeriod: { from: number; to: number } | null;
  loadingPage: { page: number; loaded: number } | null;
  replayActive: boolean;
  replayControlsProps: {
    playing: boolean;
    onPlayPause: () => void;
    onStepBack: () => void;
    onStepForward: () => void;
    speed: number;
    onSpeedChange: (speed: number) => void;
    cursorIndex: number;
    totalBars: number;
    currentTime: string;
    onSeek: (index: number) => void;
    following: boolean;
    onRecenter: () => void;
  } | null;
}

/** Everything the bot dock's "Replay" tab (`BotSessionReplayTab`) needs to
 * drive the existing session-replay engine for the bot currently under the
 * eye. Built in ChartPanel.tsx (only when a bot's eye is on and no saved
 * backtest report is loaded) and threaded through `SignalsDock`'s optional
 * `replay` prop; every handler is a thin wrapper over `useReplayEngine` —
 * the tab never owns replay state of its own. */
export interface BotReplayControls {
  /** True once the picked period's candles landed and the player entered
   * replay — the tab then swaps its From/To form for playback status. */
  active: boolean;
  /** Autoplay is currently ticking (drives Pause vs. Resume). */
  playing: boolean;
  /** Playback speed multiplier (`useReplayEngine`'s `replaySpeed`). */
  speed: number;
  /** Cursor bar's epoch seconds while replaying, null otherwise. Also what
   * the tab counts "revealed" signals/trades against. */
  cursorTime: number | null;
  /** Chunked-fetch progress between pressing Play and replay going active;
   * null when not loading. */
  loading: { page: number; loaded: number } | null;
  /** Hand a period (epoch seconds) to the replay engine — same guard and
   * same path as the toolbar's session-replay picker. */
  onStart: (fromSec: number, toSec: number) => void;
  /** Toggle autoplay pause/resume. */
  onPlayPause: () => void;
  /** Change the speed multiplier. */
  onSpeedChange: (speed: number) => void;
  /** Leave replay entirely and return the chart to live candles. */
  onExit: () => void;
}

export interface NewsBand {
  key: string;
  left: number;
  width: number;
  label: string;
  phase: 'pre' | 'post';
}

// User-configurable look for the selected trade's open/close lines (see
// `buildSelectedTradeLines`) — persisted globally (not per-symbol) since
// it's a display preference, like `chart-show-separators`.
export type OrderLineDash = 'solid' | 'dashed' | 'dotted';

export interface OrderLineStyle {
  visible: boolean;
  dash: OrderLineDash;
  width: 1 | 2 | 3 | 4;
  customColors: boolean;
  openColor: string;
  closeColor: string;
  showExitLine: boolean;
  exitLineDash: OrderLineDash;
  exitLineWidth: 1 | 2 | 3 | 4;
  exitLineCustomColor: boolean;
  exitLineWinColor: string;
  exitLineLossColor: string;
}

// User-configurable colors for zone rectangles (Quasimodo, S&D v1/v2, the
// per-trade backend zone), persisted globally like `OrderLineStyle` — see
// `pickZoneColor` in chartFormat.ts, which falls back to the existing
// hardcoded buy/sell/muted theme tokens when `customColors` is false, so
// behavior is unchanged until a user opts in.
export interface ZoneIndicatorColors {
  demandColor: string;
  supplyColor: string;
  touchedColor: string;
}

export interface ZoneColorStyle {
  customColors: boolean;
  qml: ZoneIndicatorColors;
  snd: ZoneIndicatorColors;
  sndV2: ZoneIndicatorColors;
  smc: ZoneIndicatorColors;
  fvg: ZoneIndicatorColors;
  // The per-trade backend zone has no fresh/touched state of its own — a
  // trade was, by definition, taken from it — so no `touchedColor`.
  tradeZone: Omit<ZoneIndicatorColors, 'touchedColor'>;
}

export interface PriceLineSpec {
  key: string;
  ticket: number;
  price: number;
  color: string;
  label: string;
  commit: (newPrice: number) => void;
  placeholder?: boolean; // no sl/tp set yet — drag (or click) this to add one
  pnlOpenPrice?: number;
  pnlSide?: 'buy' | 'sell';
  pnlVolume?: number;
}

export interface EntryLineSpec {
  key: string;
  position: PositionOut;
  color: string;
  label: string;
}

/** Handle onto the lightweight-charts engine — chart instance, the two base
 * series, the drawing manager, and the two series-markers plugin instances
 * (trade markers + the independent structure/QML marker layer) — created
 * once by `useChartEngine` and consumed by every other chart hook that needs
 * to read or draw onto the chart (`useIndicators`, and later phases'
 * `useCandleData`/`useDrawingTools`/`useReplayEngine`). Getter functions
 * (rather than plain fields) so a stable controller object can always report
 * the current instance without itself needing to change identity on every
 * chart mutation — only `isReady` flipping true (mount) or false (unmount)
 * changes the controller's identity, which is what consumer effects should
 * key off of. */
export interface ChartEngineController {
  containerRef: RefObject<HTMLDivElement | null>;
  getChart(): IChartApi | null;
  getCandleSeries(): ISeriesApi<'Candlestick'> | null;
  getVolumeSeries(): ISeriesApi<'Histogram'> | null;
  /** Dedicated invisible series holding future time-scale points (whitespace bars)
   * so trendlines/drawings can resolve future anchors without contaminating the
   * main candle series and breaking out-of-order checks on live updates. */
  getWhitespaceSeries(): ISeriesApi<'Line'> | null;
  getDrawingManager(): DrawingManager | null;
  /** Zone-rectangle metadata side map — see `ZoneMeta`'s doc comment. Same
   * `Map` instance for the life of the chart (not recreated per render),
   * mutated in place by whichever hook owns a given zone type. */
  getZoneMetaMap(): Map<string, ZoneMeta>;
  /** Trade entry/exit markers (backtest/live/custom-code) — see chartMarkers.ts. */
  getSeriesMarkersPrimitive(): ISeriesMarkersPluginApi<Time> | null;
  /** Independent marker layer for the 'structure'/'qml'/'snd'/'patterns'
   * manual-indicator types — kept separate so toggling structure markers
   * never touches the trade-marker plugin instance above. */
  getStructureMarkersPrimitive(): ISeriesMarkersPluginApi<Time> | null;
  /** True once `createChart`/series/`DrawingManager` setup has completed
   * (mount), false again once torn down (unmount). Consumers should treat
   * `false` as "nothing above is safe to call yet" rather than relying on
   * null-checking every getter individually. */
  isReady: boolean;
}

/** Handle onto the current symbol/timeframe's candle data and its paint
 * pipeline — created by `useCandleData` (phase 7), consumed by ChartPanel's
 * not-yet-extracted replay code (`seekTo`/`handleEnterReplay`/
 * `handleExitReplay`/`navigateToTime`, phase 9's `useReplayEngine`
 * territory) wherever it used to reach for the old `renderRef.current()`
 * ref-hack. */
export interface ChartRenderController {
  /** All candles currently loaded for this symbol/timeframe, oldest first.
   * Still a ref (not React state) — it's mutated on every live tick and by
   * `loadMore`'s pagination, and re-rendering the whole component on every
   * tick would be wasteful; consumers that need to react to it read through
   * `paintUpTo()`/`visibleCandles()` instead of expecting fresh renders.
   * Owned (created) by ChartPanel.tsx, not this controller, because
   * `useChartEngine`/`useIndicators` are called before `useCandleData` can
   * exist (they need `chartController`, which only `useChartEngine` can
   * produce) yet both need a `visibleCandles()` reader over this same ref —
   * see useCandleData.ts's module doc for the full explanation. */
  candlesRef: RefObject<Candle[]>;
  /** Repaints the candle/volume series from the current `visibleCandles()`
   * window (the full loaded history, or a prefix up to the replay cursor)
   * and schedules an indicator recompute — a real, stable-identity function
   * replacing the old `renderRef.current()` mutable-ref-assignment hack. */
  paintUpTo(): void;
  /** Re-fetches the currently loaded time window from the backend and swaps
   * it in, preserving the visible time range and the replay cursor's position
   * in time. For when the loaded bars are wrong rather than merely
   * incomplete — after `useCandleGaps` repairs missing history, the window
   * spans the same period but gains bars in the middle. */
  reloadWindow(): Promise<void>;
  symbolInfo: SymbolInfo | null;
  spreadPoints: number | null;
  error: string | null;
  loadingMore: boolean;
  switchingChart: boolean;
  newsBands: NewsBand[];
}

/** One transient "here's what the bot just did" flash emitted by
 * `useReplayReveal` as the replay cursor crosses a trade's open/close time or
 * a rejected signal's time, and rendered by `ReplayRevealOverlay`. Purely
 * ephemeral: every event self-expires a couple of seconds after it appears,
 * and none are ever produced while replay is inactive. */
export interface ReplayRevealEvent {
  /** Unique per emission (a bar can legitimately produce two identical-looking
   * flashes), used as the React key and the expiry-timer handle. */
  id: string;
  /** Epoch seconds of the bar the flash anchors to — `timeToCoordinate`. */
  time: number;
  /** Price the flash anchors to — `priceToCoordinate`. Null when it could not
   * be resolved (no loaded candle at that time), in which case the overlay
   * skips the event rather than guessing a position. */
  price: number | null;
  /** 'buy'/'sell' = a position opened, 'exit' = a position closed,
   * 'rejected' = a signal the engine vetoed/refused. Drives the colour. */
  kind: 'buy' | 'sell' | 'exit' | 'rejected';
  /** Short all-caps text, e.g. "BUY HERE". */
  label: string;
}

/** Outcome of one "Fill gaps" run, shown transiently on the toolbar — see
 * `useCandleGaps`. */
export interface GapRepairSummary {
  /** Bars that were missing and are stored now. Counted in bars rather than
   * holes because a hole often only shrinks — an outage that ran into the
   * broker's nightly break gives back its trading-hours bars and keeps the
   * closure, which a "holes closed" count would report as nothing gained. */
  barsRecovered: number;
  /** Holes the broker itself could not fill (holiday, halt, symbol listed
   * later). Real market closures, so the UI says so rather than inviting
   * another click. */
  remaining: number;
  /** Set instead of the counts when the repair request itself failed (the
   * gateway is down, MT5 not logged in). */
  error: string | null;
}
