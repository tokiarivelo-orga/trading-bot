import {
  Activity,
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronsRight,
  Code,
  Columns2,
  Eye,
  EyeOff,
  Grid2x2,
  History,
  Layers,
  Pencil,
  PenTool,
  RefreshCw,
  RotateCcw,
  Settings,
  Shield,
  ShieldOff,
  Sliders,
  Square,
} from 'lucide-react';
import { memo, type Ref } from 'react';
import type { Candle } from '@/shared/api/client';
import { REQUIRED_ANCHORS, TIMEFRAMES } from './chartFormat';
import type { DrawingToolType, GapRepairSummary } from './types';

export interface ChartToolbarProps {
  symbol: string;

  // Timeframe selector
  timeframe: Candle['timeframe'];
  showTfDropdown: boolean;
  onSelectTimeframe: (tf: Candle['timeframe']) => void;
  onToggleTfDropdown: () => void;
  /** Forwarded to the pills+dropdown wrapper so ChartPanel's outside-click
   * detection can scope to this specific instance. */
  tfDropdownRef?: Ref<HTMLDivElement>;

  // Chart navigation
  onScrollToLatest: () => void;
  onResetZoom: () => void;

  // Docks toggled from this row (the docks themselves render elsewhere)
  showIndicatorsDock: boolean;
  onToggleIndicatorsDock: () => void;
  manualIndicatorsCount: number;
  showDrawingsList: boolean;
  onToggleDrawingsList: () => void;
  drawingsListCount: number;
  /** Floating drawing-tool palette (DrawingToolbar) on the chart's left
   * edge — hidden drawings themselves are unaffected, only the tool-picker
   * strip is toggled. */
  showDrawingToolbar: boolean;
  onToggleDrawingToolbar: () => void;
  showCustomCodeEditor: boolean;
  onToggleCodeEditor: () => void;
  showActivityLogDock: boolean;
  onToggleActivityLogDock: () => void;

  // Overlays dropdown (spread line, trade labels, period separators, order lines)
  showOverlaysDropdown: boolean;
  onToggleOverlaysDropdown: () => void;
  /** Forwarded to the dropdown wrapper for the same outside-click reason as
   * `tfDropdownRef`. */
  overlaysDropdownRef?: Ref<HTMLDivElement>;
  showSeparators: boolean;
  onToggleSeparators: () => void;
  showSpreadLine: boolean;
  onToggleSpreadLine: () => void;
  showVolume: boolean;
  onToggleVolume: () => void;
  showTradeLabels: boolean;
  onToggleTradeLabels: () => void;
  showTradeMarkers: boolean;
  onToggleTradeMarkers: () => void;
  orderLineVisible: boolean;
  onToggleOrderLinesVisible: () => void;
  showOrderLineSettings: boolean;
  onToggleOrderLineSettings: () => void;
  showZoneColorSettings: boolean;
  onToggleZoneColorSettings: () => void;

  // Missing-candle repair ("Fill gaps") — see useCandleGaps.ts
  /** Actionable holes in the loaded window (weekend closures and holes the
   * broker already refused to fill are excluded). */
  candleGapCount: number;
  candleGapMissingBars: number;
  candleGapRepairing: boolean;
  candleGapResult: GapRepairSummary | null;
  onRepairCandleGaps: () => void;

  // Session replay
  backtestReportId?: string | null;
  sessionReplayPeriod: { from: number; to: number } | null;
  showSessionReplayPicker: boolean;
  onSessionReplayToggle: () => void;

  // Drawing-tool status & spread badge
  drawingTool: DrawingToolType | null;
  pendingAnchorCount: number;
  spreadPoints: number | null;

  // Split-window controls (when rendered as master toolbar in MultiChartLayout)
  windowCount?: number;
  onSelectWindowCount?: (count: number) => void;

  /** Live on/off switch for the ATR-percentile volatility guard, shared
   * state with `features/settings/VolatilityGuardPanel.tsx` via TanStack
   * Query — `null` while the config is still loading. */
  volatilityGuardEnabled: boolean | null;
  volatilityGuardSaving: boolean;
  onToggleVolatilityGuard: () => void;
}

/** One line of feedback after a "Fill gaps" run. Leads with bars recovered
 * rather than holes closed: a hole that shrank (an outage running into the
 * broker's nightly break) still gave back real candles, and reporting only
 * the leftover closure would read as "nothing happened". */
function gapResultText(result: GapRepairSummary): string {
  if (result.error) return `Refetch failed: ${result.error}`;
  const recovered =
    result.barsRecovered > 0 ? `Recovered ${result.barsRecovered} candle(s)` : '';
  const closed =
    result.remaining > 0 ? `${result.remaining} gap(s) left — market closed` : '';
  if (recovered && closed) return `${recovered} · ${closed}`;
  return recovered || closed || 'History complete';
}

/** Top toolbar row of the chart panel: symbol tag, timeframe pills/dropdown,
 * chart navigation, dock toggle buttons, overlays dropdown, session-replay
 * button and the drawing-tool/spread status badges. Purely presentational —
 * every value it reads and every action it can trigger comes in as a prop;
 * ChartPanel still owns all of the underlying state (see its `showTfDropdown`
 * / `showOverlaysDropdown` / dock-visibility `useState`s). */
export const ChartToolbar = memo(function ChartToolbar({
  symbol,
  timeframe,
  showTfDropdown,
  onSelectTimeframe,
  onToggleTfDropdown,
  tfDropdownRef,
  onScrollToLatest,
  onResetZoom,
  showIndicatorsDock,
  onToggleIndicatorsDock,
  manualIndicatorsCount,
  showDrawingsList,
  onToggleDrawingsList,
  drawingsListCount,
  showDrawingToolbar,
  onToggleDrawingToolbar,
  showCustomCodeEditor,
  onToggleCodeEditor,
  showActivityLogDock,
  onToggleActivityLogDock,
  showOverlaysDropdown,
  onToggleOverlaysDropdown,
  overlaysDropdownRef,
  showSeparators,
  onToggleSeparators,
  showSpreadLine,
  onToggleSpreadLine,
  showVolume,
  onToggleVolume,
  showTradeLabels,
  onToggleTradeLabels,
  showTradeMarkers,
  onToggleTradeMarkers,
  orderLineVisible,
  onToggleOrderLinesVisible,
  showOrderLineSettings,
  onToggleOrderLineSettings,
  showZoneColorSettings,
  onToggleZoneColorSettings,
  candleGapCount,
  candleGapMissingBars,
  candleGapRepairing,
  candleGapResult,
  onRepairCandleGaps,
  backtestReportId,
  sessionReplayPeriod,
  showSessionReplayPicker,
  onSessionReplayToggle,
  drawingTool,
  pendingAnchorCount,
  spreadPoints,
  windowCount,
  onSelectWindowCount,
  volatilityGuardEnabled,
  volatilityGuardSaving,
  onToggleVolatilityGuard,
}: ChartToolbarProps) {
  return (
    <header className='flex items-center justify-between flex-wrap gap-2 border-b border-line bg-panel/90 px-3 py-1.5 backdrop-blur-sm z-20'>
      {/* Left Section: Symbol, Timeframes, Navigation */}
      <div className='flex items-center gap-2 flex-wrap'>
        {/* Split Window Selector */}
        {onSelectWindowCount && (
          <>
            <div className='flex items-center bg-bg/70 border border-line rounded-md p-0.5 text-xs shadow-2xs'>
              <button
                type='button'
                onClick={() => onSelectWindowCount(1)}
                title='Single chart'
                className={`cursor-pointer rounded p-1 transition-all ${
                  windowCount === 1 ? 'bg-accent text-white shadow-xs font-bold' : 'text-ink-muted hover:text-accent hover:bg-line/40'
                }`}
              >
                <Square size={14} />
              </button>
              <button
                type='button'
                onClick={() => onSelectWindowCount(2)}
                title='Split into 2 windows'
                className={`cursor-pointer rounded p-1 transition-all ${
                  windowCount === 2 ? 'bg-accent text-white shadow-xs font-bold' : 'text-ink-muted hover:text-accent hover:bg-line/40'
                }`}
              >
                <Columns2 size={14} />
              </button>
              <button
                type='button'
                onClick={() => onSelectWindowCount(3)}
                title='Split into 3 windows'
                className={`cursor-pointer rounded px-1.5 py-0.5 text-[11px] font-bold transition-all ${
                  windowCount === 3 ? 'bg-accent text-white shadow-xs' : 'text-ink-muted hover:text-accent hover:bg-line/40'
                }`}
              >
                3
              </button>
              <button
                type='button'
                onClick={() => onSelectWindowCount(4)}
                title='Split into 4 windows'
                className={`cursor-pointer rounded p-1 transition-all ${
                  windowCount === 4 ? 'bg-accent text-white shadow-xs font-bold' : 'text-ink-muted hover:text-accent hover:bg-line/40'
                }`}
              >
                <Grid2x2 size={14} />
              </button>
            </div>
            <div className='h-4 w-px bg-line/60 mx-0.5 hidden sm:block' />
          </>
        )}

        {/* Symbol Tag */}
        <div className='flex items-center gap-1.5 font-bold text-xs text-ink bg-line/40 border border-line/60 rounded px-2.5 py-1 select-none shadow-2xs'>
          <span className='tracking-wide'>{symbol}</span>
        </div>

        <div className='h-4 w-px bg-line/60 mx-0.5 hidden sm:block' />

        {/* Timeframe Selector Pills + Dropdown */}
        <div className='relative flex items-center bg-bg/70 border border-line rounded-md p-0.5 text-xs shadow-2xs' ref={tfDropdownRef}>
          {(['M1', 'M5', 'M15', 'H1', 'D1'] as Candle['timeframe'][]).map((tf) => (
            <button
              key={tf}
              type='button'
              className={`cursor-pointer rounded px-2 py-1 font-medium transition-all ${
                tf === timeframe
                  ? 'bg-accent text-white shadow-xs'
                  : 'text-ink-muted hover:text-ink hover:bg-line/40'
              }`}
              onClick={() => onSelectTimeframe(tf)}
            >
              {tf}
            </button>
          ))}

          {/* Dropdown for remaining timeframes */}
          <div className='relative border-l border-line/60 ml-0.5 pl-0.5'>
            <button
              type='button'
              className={`flex cursor-pointer items-center gap-1 rounded px-1.5 py-1 text-xs font-medium transition-all ${
                !['M1', 'M5', 'M15', 'H1', 'D1'].includes(timeframe)
                  ? 'bg-accent text-white shadow-xs'
                  : 'text-ink-muted hover:text-ink hover:bg-line/40'
              }`}
              onClick={onToggleTfDropdown}
              title='More timeframes'
            >
              {!['M1', 'M5', 'M15', 'H1', 'D1'].includes(timeframe) ? timeframe : 'More'}
              <ChevronDown size={12} className={`transition-transform ${showTfDropdown ? 'rotate-180' : ''}`} />
            </button>

            {showTfDropdown && (
              <div className='absolute left-0 top-full z-30 mt-1.5 w-28 rounded-md border border-line bg-panel p-1 shadow-lg backdrop-blur-md animate-in fade-in zoom-in-95'>
                {TIMEFRAMES.map((tf) => (
                  <button
                    key={tf}
                    type='button'
                    className={`flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      tf === timeframe
                        ? 'bg-accent/20 text-accent font-semibold'
                        : 'text-ink-muted hover:bg-line/50 hover:text-ink'
                    }`}
                    onClick={() => onSelectTimeframe(tf)}
                  >
                    <span>{tf}</span>
                    {tf === timeframe && <Check size={12} className='text-accent' />}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className='h-4 w-px bg-line/60 mx-0.5 hidden sm:block' />

        {/* Navigation Controls Group */}
        <div className='flex items-center bg-bg/70 border border-line rounded-md p-0.5 text-xs shadow-2xs'>
          <button
            type='button'
            className='flex cursor-pointer items-center gap-1 rounded px-2 py-1 text-ink-muted hover:text-accent hover:bg-line/40 transition-colors'
            onClick={onScrollToLatest}
            title='Scroll chart to latest real-time bar'
          >
            <ChevronsRight size={13} />
            <span>Latest</span>
          </button>
          <button
            type='button'
            className='flex cursor-pointer items-center gap-1 rounded px-2 py-1 text-ink-muted hover:text-accent hover:bg-line/40 transition-colors'
            onClick={onResetZoom}
            title='Reset chart zoom and scale'
          >
            <RotateCcw size={13} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Center/Right Section: Feature Tools & Overlays */}
      <div className='flex items-center gap-2 flex-wrap'>
        {/* Main Tools Group (Indicators, Drawings, Code, Activity) */}
        <div className='flex items-center bg-bg/70 border border-line rounded-md p-0.5 text-xs shadow-2xs'>
          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-all ${
              showIndicatorsDock
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink hover:bg-line/40'
            }`}
            onClick={onToggleIndicatorsDock}
            title='Add or configure technical indicators'
          >
            <Sliders size={13} />
            <span>Indicators</span>
            {manualIndicatorsCount > 0 && (
              <span className='rounded-full bg-accent text-white text-[10px] px-1.5 py-0.2 font-bold leading-none'>
                {manualIndicatorsCount}
              </span>
            )}
          </button>

          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-all ${
              showDrawingsList
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink hover:bg-line/40'
            }`}
            onClick={onToggleDrawingsList}
            title='Show or manage chart drawings'
          >
            <Pencil size={13} />
            <span>Drawings</span>
            {drawingsListCount > 0 && (
              <span className='rounded-full bg-accent text-white text-[10px] px-1.5 py-0.2 font-bold leading-none'>
                {drawingsListCount}
              </span>
            )}
          </button>

          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-all ${
              showDrawingToolbar
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink hover:bg-line/40'
            }`}
            onClick={onToggleDrawingToolbar}
            title='Show or hide the drawing tools palette'
          >
            <PenTool size={13} />
            <span>Tools</span>
          </button>

          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-all ${
              showCustomCodeEditor
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink hover:bg-line/40'
            }`}
            onClick={onToggleCodeEditor}
            title='Write custom script and run directly on chart'
          >
            <Code size={13} />
            <span>Code</span>
          </button>

          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-all ${
              showActivityLogDock
                ? 'bg-accent/20 text-accent border border-accent/30'
                : 'text-ink-muted hover:text-ink hover:bg-line/40'
            }`}
            onClick={onToggleActivityLogDock}
            title='See what the bot is doing on this symbol and why'
          >
            <Activity size={13} />
            <span>Activity</span>
          </button>
        </div>

        {/* Overlays Dropdown */}
        <div className='relative' ref={overlaysDropdownRef}>
          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border border-line bg-bg/70 px-2.5 py-1.5 text-xs font-medium transition-all ${
              showSeparators || showSpreadLine || orderLineVisible || showVolume
                ? 'text-accent border-accent/40 bg-accent/10'
                : 'text-ink-muted hover:text-ink hover:border-line/80'
            }`}
            onClick={onToggleOverlaysDropdown}
            title='Chart overlay settings (Spread line, Period separators, Order lines, Volume)'
          >
            <Layers size={13} />
            <span>Overlays</span>
            <ChevronDown size={12} className={`transition-transform ${showOverlaysDropdown ? 'rotate-180' : ''}`} />
          </button>

          {showOverlaysDropdown && (
            <div className='absolute right-0 top-full z-30 mt-1.5 w-56 rounded-md border border-line bg-panel p-1.5 shadow-lg backdrop-blur-md animate-in fade-in zoom-in-95'>
              <div className='px-2 py-1 text-[10px] font-semibold tracking-wider text-ink-muted uppercase border-b border-line/60 mb-1'>
                Display Overlays
              </div>

              {/* Spread Line Toggle */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleSpreadLine}
              >
                <span className='flex items-center gap-2'>
                  {showSpreadLine ? <Eye size={13} className='text-accent' /> : <EyeOff size={13} />}
                  <span>Spread line (Ask)</span>
                </span>
                {showSpreadLine && <Check size={12} className='text-accent' />}
              </button>

              {/* Volume Toggle */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleVolume}
              >
                <span className='flex items-center gap-2'>
                  {showVolume ? <Eye size={13} className='text-accent' /> : <EyeOff size={13} />}
                  <span>Volume histogram</span>
                </span>
                {showVolume && <Check size={12} className='text-accent' />}
              </button>

              {/* Trade Labels Toggle */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleTradeLabels}
                title='Show/hide the BUY/SELL text under trade markers — arrows stay visible either way'
              >
                <span className='flex items-center gap-2'>
                  {showTradeLabels ? <Eye size={13} className='text-accent' /> : <EyeOff size={13} />}
                  <span>Trade labels (BUY/SELL)</span>
                </span>
                {showTradeLabels && <Check size={12} className='text-accent' />}
              </button>

              {/* Trade Markers (arrows) Toggle */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleTradeMarkers}
                title='Show/hide the BUY/SELL arrows themselves — independent of the text label toggle'
              >
                <span className='flex items-center gap-2'>
                  {showTradeMarkers ? <Eye size={13} className='text-accent' /> : <EyeOff size={13} />}
                  <span>Trade arrows (BUY/SELL)</span>
                </span>
                {showTradeMarkers && <Check size={12} className='text-accent' />}
              </button>

              {/* Period Separators Toggle */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleSeparators}
              >
                <span className='flex items-center gap-2'>
                  <Layers size={13} className={showSeparators ? 'text-accent' : ''} />
                  <span>Period separators</span>
                </span>
                {showSeparators && <Check size={12} className='text-accent' />}
              </button>

              {/* Order Lines Toggle & Gear */}
              <div className='flex items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 transition-colors'>
                <button
                  type='button'
                  className='flex items-center gap-2 cursor-pointer flex-1 text-left hover:text-ink'
                  onClick={onToggleOrderLinesVisible}
                >
                  <Settings size={13} className={orderLineVisible ? 'text-accent' : ''} />
                  <span>Order lines</span>
                </button>
                <div className='flex items-center gap-1.5'>
                  {orderLineVisible && <Check size={12} className='text-accent' />}
                  <button
                    type='button'
                    className={`cursor-pointer rounded p-0.5 hover:bg-line text-xs ${
                      showOrderLineSettings ? 'text-accent' : 'text-ink-muted hover:text-ink'
                    }`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleOrderLineSettings();
                    }}
                    title='Style open/close lines (type, width, colors)'
                  >
                    ⚙
                  </button>
                </div>
              </div>

              {/* Zone Colors Gear — no separate visibility toggle (zone
                  rectangles are shown/hidden per-indicator from
                  IndicatorsDock), just opens the color settings panel. */}
              <button
                type='button'
                className='flex w-full cursor-pointer items-center justify-between rounded px-2.5 py-1.5 text-xs text-ink-muted hover:bg-line/50 hover:text-ink transition-colors'
                onClick={onToggleZoneColorSettings}
                title='Customize demand/supply/touched colors per zone indicator'
              >
                <span className='flex items-center gap-2'>
                  <Settings size={13} className={showZoneColorSettings ? 'text-accent' : ''} />
                  <span>Zone colors</span>
                </span>
              </button>
            </div>
          )}
        </div>

        {/* Volatility Guard Toggle */}
        {volatilityGuardEnabled !== null && (
          <button
            type='button'
            disabled={volatilityGuardSaving}
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-all disabled:opacity-60 ${
              volatilityGuardEnabled
                ? 'border-accent/40 bg-accent/10 text-accent'
                : 'border-line bg-bg/70 text-ink-muted hover:border-line/80 hover:text-ink'
            }`}
            onClick={onToggleVolatilityGuard}
            title='Live on/off switch for the ATR-percentile volatility guard (scales bot SL/TP and can force-close/trail positions in high volatility)'
          >
            {volatilityGuardEnabled ? <Shield size={13} /> : <ShieldOff size={13} />}
            <span>Volatility guard: {volatilityGuardSaving ? '…' : volatilityGuardEnabled ? 'ON' : 'OFF'}</span>
          </button>
        )}

        {/* Missing-candle repair. Always clickable, not just when a gap was
            detected: the scan only sees the local database, so this doubles
            as "refetch this window from the broker" for bars that look wrong
            for any other reason. */}
        <div className='flex items-center gap-1.5'>
          <button
            type='button'
            disabled={candleGapRepairing}
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-all disabled:cursor-wait disabled:opacity-70 ${
              candleGapCount > 0
                ? 'border-sell/50 bg-sell/10 text-sell'
                : 'border-line bg-bg/70 text-ink-muted hover:border-accent/60 hover:text-accent'
            }`}
            onClick={onRepairCandleGaps}
            title={
              candleGapCount > 0
                ? `${candleGapCount} gap(s) — ${candleGapMissingBars} candle(s) missing from this window. ` +
                  'The chart draws straight across them, so indicators and bot analysis on this range are wrong. ' +
                  'Click to re-download them from the broker.'
                : 'Re-download this window’s candles from the broker and redraw the chart (no gaps detected in local history)'
            }
          >
            {candleGapCount > 0 && !candleGapRepairing ? (
              <AlertTriangle size={13} />
            ) : (
              <RefreshCw size={13} className={candleGapRepairing ? 'animate-spin' : ''} />
            )}
            <span>{candleGapRepairing ? 'Filling gaps…' : 'Fill gaps'}</span>
            {candleGapCount > 0 && !candleGapRepairing && (
              <span className='rounded-full bg-sell px-1.5 py-0.2 text-[10px] font-bold leading-none text-white'>
                {candleGapCount}
              </span>
            )}
          </button>

          {candleGapResult && !candleGapRepairing && (
            <span
              className={`text-[11px] ${candleGapResult.error ? 'text-err' : 'text-ink-muted'}`}
            >
              {gapResultText(candleGapResult)}
            </span>
          )}
        </div>

        {/* Session Replay Button */}
        {!backtestReportId && (
          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-all ${
              sessionReplayPeriod
                ? 'border-accent bg-accent/20 text-accent shadow-xs'
                : showSessionReplayPicker
                  ? 'border-accent text-accent'
                  : 'border-line bg-bg/70 text-ink-muted hover:border-accent/60 hover:text-accent'
            }`}
            onClick={onSessionReplayToggle}
            title='Replay an arbitrary historical period bar-by-bar, like a live session'
          >
            {sessionReplayPeriod ? (
              <>
                <Square size={12} fill='currentColor' /> Exit replay
              </>
            ) : (
              <>
                <History size={13} /> Session replay
              </>
            )}
          </button>
        )}

        {/* Drawing Tool Status & Spread Badge */}
        <div className='flex items-center gap-2 text-xs'>
          {drawingTool && (
            <span className='text-accent font-medium animate-pulse bg-accent/10 border border-accent/30 rounded px-2 py-0.5'>
              {pendingAnchorCount === 0 ? (
                <>Drawing {drawingTool}: click 1st point</>
              ) : (
                <>
                  Drawing {drawingTool}: {REQUIRED_ANCHORS[drawingTool] - pendingAnchorCount} more point(s)
                </>
              )}
            </span>
          )}

          {/* Interactive Spread Badge */}
          <button
            type='button'
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-mono transition-all ${
              showSpreadLine
                ? 'border-accent/40 bg-accent/10 text-accent font-medium shadow-2xs'
                : 'border-line bg-bg/70 text-ink-muted hover:text-accent hover:border-line/80'
            }`}
            onClick={onToggleSpreadLine}
            title='Click to toggle spread line (Ask price) on chart'
          >
            <span className={`h-1.5 w-1.5 rounded-full ${showSpreadLine ? 'bg-accent animate-pulse' : 'bg-ink-muted/50'}`} />
            <span>Spread: {spreadPoints === null ? '—' : `${spreadPoints} pts`}</span>
          </button>
        </div>
      </div>
    </header>
  );
});
