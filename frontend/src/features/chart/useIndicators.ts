'use client';

/**
 * Indicator state + rendering: manually added indicators (IndicatorsDock),
 * their per-symbol localStorage persistence, the live-bot "eye" view's own
 * indicator list, saved custom (backend-Python) indicator compute, and the
 * indicator-series-creation effect itself (`recomputeIndicators`).
 *
 * `recomputeIndicators` used to be defined inline inside ChartPanel's single
 * `[]`-dependency chart-creation effect (see `useChartEngine.ts`'s module
 * doc) — meaning it was created once and only ever re-run when something
 * else remembered to call `recomputeIndicatorsRef.current()` by hand. That's
 * still true for the high-frequency callers (a live candle tick, a replay
 * step) that can't wait for a React re-render — see `recomputeIndicatorsRef`
 * below, still exposed for ChartPanel.tsx's not-yet-extracted candle-loading
 * effect to call directly. What's fixed here is the *declarative* trigger:
 * a real `useEffect` below (mirroring ChartPanel's previous
 * `[activeStrategy, manualIndicators, showSeparators]` effect, now also
 * gated on the chart actually existing) means toggling an indicator no
 * longer depends on some unrelated effect happening to fire first.
 *
 * Not owned here: `candlesRef`/`visibleCandles`/replay state (not extracted
 * yet — phases 7/9), the drawing-manager/mouse-event wiring (still tightly
 * coupled to `useChartEngine`'s `manager`, phase 8 territory), and
 * `showSignalsDock` (plain UI toggle still local to ChartPanel.tsx, flipped
 * as a side effect of the live-bot-indicator fetch below — same coupling
 * the original combined effect had, not something this phase changes).
 */

import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type RefObject,
  type SetStateAction,
} from 'react';
import {
  HistogramSeries,
  LineSeries,
  LineStyle,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { HorizontalLine, Rectangle, VerticalLine } from 'lightweight-charts-drawing';
import {
  computeIndicator,
  getSkillAssignments,
  getStrategyVersions,
  previewIndicatorCode,
  type Candle,
  type ComputeIndicatorResponse,
  type EvaluateCustomCodeResponse,
  type IndicatorSpec,
  type StrategyVersionSummary,
} from '@/shared/api/client';
import {
  atr,
  bollinger,
  detectBases,
  detectPatterns,
  ema,
  historicalVolatility,
  macd,
  quasimodoLevels,
  rsi,
  sma,
  sndZones,
  sndZonesV2,
  swingStructure,
  vwap,
  smcConcepts,
} from './indicators';
import { VolumeProfilePrimitive } from './volumeProfilePrimitive';
import { cssVar, derivePeriodParam, hexToRgba, pickZoneColor, usesSndZones } from './chartFormat';
import { isOscillatorType, paneKeyOf } from './paneTargets';
import {
  loadManualIndicators,
  saveManualIndicators,
  SEPARATOR_DRAWING_PREFIX,
  STRATEGY_DRAWING_PREFIX,
} from './chartStorage';
import type { ChartEngineController, ManualIndicator, ZoneColorStyle } from './types';

// Shared swing-detection constants for the 'structure'/'qml' indicators,
// matching the backend vix75 strategy's defaults (atr_period: 14,
// structure_margin_atr_mult: 0.1) so the chart's reading of "HH"/"QML"
// agrees with what the strategy itself computes per trade. Swing lookback
// itself is user-editable per instance (ManualIndicator.period).
const STRUCTURE_ATR_PERIOD = 14;
const STRUCTURE_MARGIN_ATR_MULT = 0.1;

// `recomputeIndicators` recomputes every EMA/RSI/MACD/Bollinger/VWAP/ATR/
// structure/QML/SND/pattern overlay from scratch on every live tick and,
// throttled, on every replay step. Left unbounded it recomputes over however
// much history `loadMore()`'s paging (or session replay's up to 60k-candle
// window) has accumulated in `candlesRef.current` — cost that only grows the
// longer a chart stays open, on a cadence (~1.5s live, 200ms replay) that
// doesn't care how much history is loaded. Capping the window this feeds
// indicator computation makes that cost constant regardless of accumulated
// history; recursive indicators (EMA/RSI/ATR) get a few bars of warm-up at
// the window's start, invisible once scrolled off. The candle/volume price
// series themselves are unaffected — they always render the full loaded
// history via `visibleCandles()` directly, not this cap.
const MAX_INDICATOR_CANDLES = 3000;

export interface UseIndicatorsParams {
  chartController: ChartEngineController | null;
  symbol: string;
  symbolRef: RefObject<string>;
  timeframe: Candle['timeframe'];
  timeframeRef: RefObject<Candle['timeframe']>;
  activeStrategy: StrategyVersionSummary | null;
  showSeparators: boolean;
  showSeparatorsRef: RefObject<boolean>;
  /** The full loaded window normally, or a prefix up to the replay cursor
   * while replaying — see ChartPanel's `visibleCandles()` (not extracted
   * yet — phases 7/9). Read fresh on every `recomputeIndicators()` call. */
  visibleCandles: () => Candle[];
  /** Raw (non-replay-gated) loaded candles — `candlesRef.current` in
   * ChartPanel.tsx — used only to size the custom-code compute request the
   * same way the original code did (deliberately un-gated by replay; see
   * `runOverlaysNow`'s comment in ChartPanel.tsx on why custom-code
   * overlays stay full-range during replay). */
  getRawCandles: () => Candle[];
  /** Mirror of the "Run Custom Code" drawer's last result — owned by
   * `useStrategyEditor`. */
  customCodeResultRef: RefObject<EvaluateCustomCodeResponse | null>;
  accountId: string | null;
  liveBotSkill: string | null;
  /** The live-bot eye view flips SignalsDock open when it resolves a bot's
   * indicators — same side effect the original combined effect had. */
  setShowSignalsDock: Dispatch<SetStateAction<boolean>>;
  /** User-configurable zone-rectangle colors (Zone colors settings panel) —
   * see `pickZoneColor` in chartFormat.ts. */
  zoneColorStyle: ZoneColorStyle;
}

export function useIndicators(params: UseIndicatorsParams) {
  const {
    chartController,
    symbol,
    symbolRef,
    timeframe,
    timeframeRef,
    activeStrategy,
    showSeparators,
    showSeparatorsRef,
    visibleCandles,
    getRawCandles,
    customCodeResultRef,
    accountId,
    liveBotSkill,
    setShowSignalsDock,
    zoneColorStyle,
  } = params;

  // Manually added indicators (independent of the active strategy's spec),
  // persisted per-symbol in localStorage — same convention as drawings.
  const [manualIndicators, setManualIndicators] = useState<ManualIndicator[]>(
    () => loadManualIndicators(symbol),
  );
  // User-added indicators (via the IndicatorsDock), read fresh inside
  // recomputeIndicators the same way activeStrategyRef is.
  const manualIndicatorsRef = useRef<ManualIndicator[]>([]);
  manualIndicatorsRef.current = manualIndicators;

  const [showIndicatorsDock, setShowIndicatorsDock] = useState(false);

  // The live-bot eye view's own indicator list (from its strategy version's
  // spec) — null while unresolved/no bot selected, [] when resolved but the
  // spec has none. Fed to SignalsDock's "Indicators" tab.
  const [liveBotIndicators, setLiveBotIndicators] = useState<IndicatorSpec[] | null>(null);

  // Series added for the active strategy's PDF-derived indicators (EMA/SMA/
  // RSI/MACD/Bollinger) — replaced wholesale on every recompute.
  const indicatorSeriesRef = useRef<ISeriesApi<'Line' | 'Histogram'>[]>([]);
  // Volume Profile primitives attached to the main candle series (not a
  // series of their own) — tracked separately so they can be detached
  // wholesale each recompute, same "wholesale replace" convention as
  // `indicatorSeriesRef`, since `series.attachPrimitive`/`detachPrimitive`
  // isn't touched by `chart.removeSeries`.
  const volumeProfilePrimitivesRef = useRef<VolumeProfilePrimitive[]>([]);
  const activeStrategyRef = useRef<StrategyVersionSummary | null>(activeStrategy);
  activeStrategyRef.current = activeStrategy;

  // Computed series for each 'custom' (saved, backend-Python) manual
  // indicator instance, keyed by ManualIndicator.id — populated by
  // computeCustomIndicatorsRef below (an API round trip, so it can't live
  // inside the synchronous recomputeIndicators) and read fresh inside
  // recomputeIndicators the same way customCodeResultRef already is.
  const customIndicatorResultsRef = useRef<Record<string, ComputeIndicatorResponse>>({});

  // Reassigned every render (see the assignment below) so it always closes
  // over the current symbol/timeframe/manualIndicators — called both from
  // this hook's own effect (add/remove/symbol/timeframe changes) and from
  // ChartPanel.tsx's chart-creation effect `render()` once the initial
  // history lands, so an indicator added before candles finish loading
  // still computes.
  const computeCustomIndicatorsRef = useRef<() => void>(() => {});

  // Invoked from: this hook's own reactive effect below, ChartPanel.tsx's
  // candle-loading effect on every history load/live tick/replay step, and
  // the custom-code editor's run/clear handlers — all via this stable ref,
  // since none of those callers can rely on React re-rendering between a
  // data change and needing the chart updated. Body defined once below
  // (not per-render) and reads every input through refs, exactly like the
  // original inline version — see the module doc above.
  const recomputeIndicatorsRef = useRef<() => void>(() => {});

  useEffect(() => {
    recomputeIndicatorsRef.current = () => {
      const chart = chartController?.getChart();
      const manager = chartController?.getDrawingManager();
      if (!chart || !manager) return;
      // chart/manager only resolve non-null when chartController itself is
      // non-null (both come from it via optional chaining above).
      const zoneMeta = chartController!.getZoneMetaMap();

      for (const series of indicatorSeriesRef.current) {
        try {
          chart.removeSeries(series);
        } catch {
          // Series may already be gone if the chart is mid-teardown.
        }
      }
      indicatorSeriesRef.current = [];

      const candleSeries = chartController!.getCandleSeries();
      for (const primitive of volumeProfilePrimitivesRef.current) {
        try {
          candleSeries?.detachPrimitive(primitive);
        } catch {
          // Series may already be gone if the chart is mid-teardown.
        }
      }
      volumeProfilePrimitivesRef.current = [];
      for (const drawing of manager.getAllDrawings()) {
        if (
          drawing.id.startsWith(STRATEGY_DRAWING_PREFIX) ||
          drawing.id.startsWith(SEPARATOR_DRAWING_PREFIX)
        ) {
          manager.removeDrawing(drawing.id);
          zoneMeta.delete(drawing.id);
        }
      }
      // Note: BACKTEST_DRAWING_PREFIX drawings are intentionally left alone
      // here — they're cleared/rebuilt by the backtest-trades effect only,
      // not on every recomputeIndicators call (candle tick, symbol switch).

      const spec = activeStrategyRef.current?.spec;
      const allCandles = visibleCandles();
      if (allCandles.length === 0) return;
      // See MAX_INDICATOR_CANDLES: bound recompute cost to a constant window
      // instead of however much history is currently loaded.
      const candles =
        allCandles.length > MAX_INDICATOR_CANDLES
          ? allCandles.slice(allCandles.length - MAX_INDICATOR_CANDLES)
          : allCandles;

      // Real bottom panes (lightweight-charts v5 `chart.addPane()` /
      // `chart.addSeries(Series, options, paneIndex)`) instead of the old
      // same-pane `scaleMargins` band trick — each oscillator group gets its
      // own independent price axis below the main price/volume pane (pane
      // 0, untouched). Panes beyond 0 are fully torn down and rebuilt every
      // recompute, same "wholesale replace" convention as `indicatorSeriesRef`
      // above, so a toggled-off oscillator's pane never lingers empty.
      for (let i = chart.panes().length - 1; i >= 1; i--) {
        try {
          chart.removePane(i);
        } catch {
          // Already gone if the chart is mid-teardown.
        }
      }
      const paneIndexForGroup = new Map<string, number>();
      const OSCILLATOR_PANE_HEIGHT = 130; // px, ~ one of a few stacked bottom panes
      const getOscillatorPane = (group: string): number => {
        let idx = paneIndexForGroup.get(group);
        if (idx === undefined) {
          const pane = chart.addPane();
          idx = pane.paneIndex();
          pane.setHeight(OSCILLATOR_PANE_HEIGHT);
          paneIndexForGroup.set(group, idx);
        }
        return idx;
      };
      // Manual-indicator pane resolution (see paneTargets.ts): a manual
      // indicator's `paneTarget` resolves via `paneKeyOf()` to either
      // 'main' or a bottom-pane key, which this maps to a real pane index —
      // reusing `getOscillatorPane` (and its dedup map) so an explicit
      // `paneKey` two indicators share lands them in the SAME pane, exactly
      // like the legacy `legacy:<type>` grouping already did for same-type
      // oscillators.
      const getPaneForManualIndicator = (
        manualIndicator: ManualIndicator,
      ): { paneIndex: number; ownPriceScaleId?: string } => {
        const key = paneKeyOf(manualIndicator);
        if (key === 'main') {
          // Oscillators explicitly forced onto the main pane need their own
          // price scale so RSI (0-100) or an unbounded MACD histogram don't
          // distort the candle scale — same trick the old overlay hack
          // used, now opt-in via the pane picker instead of automatic.
          if (isOscillatorType(manualIndicator.type)) {
            return { paneIndex: 0, ownPriceScaleId: `manual-osc-${manualIndicator.id}` };
          }
          return { paneIndex: 0 };
        }
        return { paneIndex: getOscillatorPane(key) };
      };

      for (const indicator of spec?.indicators ?? []) {
        switch (indicator.type) {
          case 'ema': {
            const series = chart.addSeries(LineSeries, {
              color: '#42a5f5',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              title: indicator.label,
            });
            series.setData(ema(candles, indicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'sma': {
            const series = chart.addSeries(LineSeries, {
              color: '#ffa726',
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              title: indicator.label,
            });
            series.setData(sma(candles, indicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'rsi': {
            const rsiPaneIndex = getOscillatorPane('rsi');
            const series = chart.addSeries(
              LineSeries,
              {
                color: '#ab47bc',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
                title: indicator.label,
                autoscaleInfoProvider: () => ({
                  priceRange: { minValue: 0, maxValue: 100 },
                }),
              },
              rsiPaneIndex,
            );
            series.setData(rsi(candles, indicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'macd': {
            const slow = indicator.params.slow ?? 26;
            const signal = indicator.params.signal ?? 9;
            const { macdLine, signalLine, histogram } = macd(
              candles,
              indicator.period,
              slow,
              signal,
            );
            const macdPaneIndex = getOscillatorPane('macd');
            const macdSeries = chart.addSeries(
              LineSeries,
              {
                color: '#26a69a',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${indicator.label} macd`,
              },
              macdPaneIndex,
            );
            const signalSeries = chart.addSeries(
              LineSeries,
              {
                color: '#ef5350',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${indicator.label} signal`,
              },
              macdPaneIndex,
            );
            const histSeries = chart.addSeries(
              HistogramSeries,
              {
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${indicator.label} hist`,
              },
              macdPaneIndex,
            );
            macdSeries.setData(macdLine);
            signalSeries.setData(signalLine);
            histSeries.setData(histogram);
            indicatorSeriesRef.current.push(macdSeries, signalSeries, histSeries);
            break;
          }
          case 'bollinger': {
            const stdDev = indicator.params.std_dev ?? 2;
            const { upper, middle, lower } = bollinger(
              candles,
              indicator.period,
              stdDev,
            );
            for (const [data, opacity] of [
              [upper, 1],
              [middle, 0.6],
              [lower, 1],
            ] as const) {
              const series = chart.addSeries(LineSeries, {
                color: hexToRgba('#78909c', opacity),
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
                title: indicator.label,
              });
              series.setData(data);
              indicatorSeriesRef.current.push(series);
            }
            break;
          }
        }
      }

      if (spec) {
        const anchorTime = candles[0].time as UTCTimestamp;
        spec.price_levels.forEach((level, i) => {
          const color = level.type === 'support' ? '#26a69a' : '#ab47bc';
          const drawing = HorizontalLine.create(
            `${STRATEGY_DRAWING_PREFIX}${symbolRef.current}:${i}`,
            level.price,
            anchorTime,
            { lineColor: color, lineWidth: 1, lineDash: [4, 4] },
            {
              locked: true,
              showPrice: true,
              showLabel: true,
              labelText: level.label,
            },
          );
          manager.addDrawing(drawing);
        });
      }

      // User-added indicators from IndicatorsDock — plotted alongside
      // whatever the strategy spec above already drew. RSI/MACD reuse the
      // same panes (`strategy-rsi`/`strategy-macd`) as the strategy-derived
      // ones so oscillators from both sources stack in one place rather than
      // each opening a second pane.
      const structureMarkers: SeriesMarker<Time>[] = [];
      // 'structure' and 'qml' both need swingStructure() at the same
      // (STRUCTURE_ATR_PERIOD, STRUCTURE_MARGIN_ATR_MULT) — only the
      // indicator's own `period` (lookback) varies. Cache by that period so
      // having both active doesn't recompute the same O(n) pass twice.
      const swingStructureCache = new Map<number, ReturnType<typeof swingStructure>>();
      const cachedSwingStructure = (period: number) => {
        let points = swingStructureCache.get(period);
        if (!points) {
          points = swingStructure(
            candles,
            period,
            STRUCTURE_ATR_PERIOD,
            STRUCTURE_MARGIN_ATR_MULT,
          );
          swingStructureCache.set(period, points);
        }
        return points;
      };
      for (const manualIndicator of manualIndicatorsRef.current) {
        if (manualIndicator.hidden) continue;
        const lineStyleVal =
          manualIndicator.lineStyle === 'dashed'
            ? LineStyle.Dashed
            : manualIndicator.lineStyle === 'dotted'
              ? LineStyle.Dotted
              : LineStyle.Solid;
        const lineWidthVal = (manualIndicator.lineWidth ?? 1) as any;
        const lineDashVal =
          manualIndicator.lineStyle === 'dashed'
            ? [4, 4]
            : manualIndicator.lineStyle === 'dotted'
              ? [2, 2]
              : undefined;

        const { paneIndex: manualPaneIndex, ownPriceScaleId } =
          getPaneForManualIndicator(manualIndicator);

        switch (manualIndicator.type) {
          case 'ema': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
              },
              manualPaneIndex,
            );
            series.setData(ema(candles, manualIndicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'sma': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
              },
              manualPaneIndex,
            );
            series.setData(sma(candles, manualIndicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'vwap': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
              },
              manualPaneIndex,
            );
            series.setData(vwap(candles));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'rsi': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
                priceScaleId: ownPriceScaleId,
                autoscaleInfoProvider: () => ({
                  priceRange: { minValue: 0, maxValue: 100 },
                }),
              },
              manualPaneIndex,
            );
            if (ownPriceScaleId) {
              // Forced onto the main pane: pin its own price scale to a thin
              // band at the bottom so it doesn't distort the candle scale —
              // same trick the old overlay hack used automatically, now
              // opt-in via the pane picker.
              chart.priceScale(ownPriceScaleId).applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
              });
            }
            series.setData(rsi(candles, manualIndicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'atr': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
                priceScaleId: ownPriceScaleId,
              },
              manualPaneIndex,
            );
            if (ownPriceScaleId) {
              chart.priceScale(ownPriceScaleId).applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
              });
            }
            series.setData(atr(candles, manualIndicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'volatility': {
            const series = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: manualIndicator.label,
                priceScaleId: ownPriceScaleId,
              },
              manualPaneIndex,
            );
            if (ownPriceScaleId) {
              chart.priceScale(ownPriceScaleId).applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
              });
            }
            series.setData(historicalVolatility(candles, manualIndicator.period));
            indicatorSeriesRef.current.push(series);
            break;
          }
          case 'macd': {
            const { macdLine, signalLine, histogram } = macd(candles, 12, 26, 9);
            if (ownPriceScaleId) {
              chart.priceScale(ownPriceScaleId).applyOptions({
                scaleMargins: { top: 0.8, bottom: 0 },
              });
            }
            const macdSeries = chart.addSeries(
              LineSeries,
              {
                color: manualIndicator.color,
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${manualIndicator.label} macd`,
                priceScaleId: ownPriceScaleId,
              },
              manualPaneIndex,
            );
            const signalSeries = chart.addSeries(
              LineSeries,
              {
                color: '#ef5350',
                lineWidth: lineWidthVal,
                lineStyle: lineStyleVal,
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${manualIndicator.label} signal`,
                priceScaleId: ownPriceScaleId,
              },
              manualPaneIndex,
            );
            const histSeries = chart.addSeries(
              HistogramSeries,
              {
                priceLineVisible: false,
                lastValueVisible: false,
                title: `${manualIndicator.label} hist`,
                priceScaleId: ownPriceScaleId,
              },
              manualPaneIndex,
            );
            macdSeries.setData(macdLine);
            signalSeries.setData(signalLine);
            histSeries.setData(histogram);
            indicatorSeriesRef.current.push(macdSeries, signalSeries, histSeries);
            break;
          }
          case 'bollinger': {
            const { upper, middle, lower } = bollinger(
              candles,
              manualIndicator.period,
              2,
            );
            for (const [data, opacity] of [
              [upper, 1],
              [middle, 0.6],
              [lower, 1],
            ] as const) {
              const series = chart.addSeries(
                LineSeries,
                {
                  color: hexToRgba(manualIndicator.color, opacity),
                  lineWidth: lineWidthVal,
                  lineStyle: lineStyleVal,
                  priceLineVisible: false,
                  lastValueVisible: false,
                  title: manualIndicator.label,
                },
                manualPaneIndex,
              );
              series.setData(data);
              indicatorSeriesRef.current.push(series);
            }
            break;
          }
          case 'structure': {
            const points = cachedSwingStructure(manualIndicator.period);
            for (const p of points) {
              structureMarkers.push({
                time: p.time,
                position:
                  p.label === 'HH' || p.label === 'LH' ? 'aboveBar' : 'belowBar',
                color: manualIndicator.color,
                shape: 'circle',
                size: 0,
                text: p.label,
              });
            }
            break;
          }
          case 'qml': {
            const points = cachedSwingStructure(manualIndicator.period);
            const lastTime = candles[candles.length - 1].time as UTCTimestamp;
            quasimodoLevels(points, candles).forEach((zone, zoneIdx) => {
              // Confirmation: the neckline-break candle, tagged at the QML
              // level (the left shoulder) where the retest entry sits.
              structureMarkers.push({
                time: zone.time,
                position: 'atPriceMiddle',
                price: zone.price,
                color: manualIndicator.color,
                shape: zone.kind === 'QML' ? 'arrowDown' : 'arrowUp',
                text: zone.kind === 'QML' ? 'QML' : 'QML-INV',
              });
              // The QM zone band between the QML level (left shoulder) and
              // the head (maximum pain level), from the head until the
              // zone is broken past the head — or still-open to the latest candle.
              // Supply (sell) tint for QML, demand (buy) for the inverse.
              const qmlTouched = zone.retestTime !== undefined;
              const zoneColor = pickZoneColor(
                zoneColorStyle.qml,
                zone.kind === 'QML_INV',
                qmlTouched,
                zoneColorStyle.customColors,
              );
              const qmlZoneId = `${STRATEGY_DRAWING_PREFIX}qml-zone:${manualIndicator.id}:${zoneIdx}`;
              manager.addDrawing(
                new Rectangle(
                  qmlZoneId,
                  [
                    { time: zone.headTime, price: zone.headPrice },
                    { time: zone.retestTime ?? lastTime, price: zone.price },
                  ],
                  {
                    lineColor: zoneColor,
                    lineWidth: lineWidthVal,
                    lineDash: lineDashVal,
                    fillColor: hexToRgba(zoneColor, 0.15),
                  },
                  { filled: true, locked: true },
                ),
              );
              zoneMeta.set(qmlZoneId, {
                indicator: 'qml',
                indicatorLabel: 'Quasimodo',
                pattern: zone.kind,
                kind: zone.kind === 'QML' ? 'supply' : 'demand',
                priceLow: Math.min(zone.headPrice, zone.price),
                priceHigh: Math.max(zone.headPrice, zone.price),
                timeStart: zone.headTime as number,
                timeEnd: zone.retestTime ? (zone.retestTime as number) : null,
                state: qmlTouched ? 'touched' : 'fresh',
                extra: { 'Neckline price': zone.necklinePrice, 'Head price': zone.headPrice },
              });
              // Retest of the QML level after the break = the actual
              // entry signal (sell for QML, buy for the inversed pattern).
              if (zone.retestTime) {
                structureMarkers.push({
                  time: zone.retestTime,
                  position: 'atPriceMiddle',
                  price: zone.price,
                  color: manualIndicator.color,
                  shape: zone.kind === 'QML' ? 'arrowDown' : 'arrowUp',
                  text: zone.kind === 'QML' ? 'SELL' : 'BUY',
                });
              }
            });
            break;
          }
          case 'snd': {
            // PoB supply & demand entry points (RBR/DBD/RBD/DBR): the base
            // candles' band drawn as a zone rectangle, same treatment as
            // the QML zone above. `period` here is the max base-candle
            // count, not a lookback. A zone is valid only until it is first
            // TOUCHED — the rectangle ends at that touch and the zone is
            // greyed to mark it consumed.
            const lastTime = candles[candles.length - 1].time as UTCTimestamp;
            sndZones(candles, manualIndicator.period, STRUCTURE_ATR_PERIOD).forEach(
              (zone, zoneIdx) => {
                const demand = zone.kind === 'demand';
                // Retest entries sit at the proximal edge — the side of the
                // base that price approaches first when it comes back.
                const proximal = demand ? zone.priceHigh : zone.priceLow;
                const touched = zone.touchedTime !== undefined;
                // Demand (buy) tint for RBR/DBR, supply (sell) for DBD/RBD;
                // a touched (consumed, invalid) zone is greyed instead.
                const zoneColor = pickZoneColor(
                  zoneColorStyle.snd,
                  demand,
                  touched,
                  zoneColorStyle.customColors,
                );
                structureMarkers.push({
                  time: zone.time,
                  position: 'atPriceMiddle',
                  price: proximal,
                  color: touched ? cssVar('--color-ink-muted') : manualIndicator.color,
                  shape: demand ? 'arrowUp' : 'arrowDown',
                  text: touched ? `${zone.pattern} touched` : zone.pattern,
                });
                // The rectangle spans the base candles' extremes from the
                // first base candle until the zone is first touched — or
                // still-open to the latest candle while fresh.
                const sndZoneId = `${STRATEGY_DRAWING_PREFIX}snd-zone:${manualIndicator.id}:${zoneIdx}`;
                manager.addDrawing(
                  new Rectangle(
                    sndZoneId,
                    [
                      { time: zone.baseStartTime, price: zone.priceHigh },
                      { time: zone.touchedTime ?? lastTime, price: zone.priceLow },
                    ],
                    {
                      lineColor: zoneColor,
                      lineWidth: lineWidthVal,
                      lineDash: lineDashVal,
                      fillColor: hexToRgba(zoneColor, 0.15),
                    },
                    { filled: true, locked: true },
                  ),
                );
                zoneMeta.set(sndZoneId, {
                  indicator: 'snd',
                  indicatorLabel: 'S&D Zones v1',
                  pattern: zone.pattern,
                  kind: zone.kind,
                  priceLow: zone.priceLow,
                  priceHigh: zone.priceHigh,
                  timeStart: zone.baseStartTime as number,
                  timeEnd: zone.touchedTime ? (zone.touchedTime as number) : null,
                  state: touched ? 'touched' : 'fresh',
                });
                // The touch itself is the entry (buy the demand base, sell
                // the supply base) and where the zone was consumed.
                if (zone.touchedTime) {
                  structureMarkers.push({
                    time: zone.touchedTime,
                    position: 'atPriceMiddle',
                    price: proximal,
                    color: manualIndicator.color,
                    shape: demand ? 'arrowUp' : 'arrowDown',
                    text: demand ? 'BUY' : 'SELL',
                  });
                }
              },
            );
            break;
          }
          case 'snd_v2': {
            // PoB S&D zones v2 (see `sndZonesV2` in indicators.ts): the
            // departure-first detector that also surfaces origin bases (DZ/SZ,
            // no leg-in) and wide ranges, with the "basing candle" height
            // refinement so the drawn band is roughly one candle tall. A zone
            // is valid only while FRESH — the first touch consumes it, ending
            // the rectangle there and greying it. `period` is the max
            // base-candle count, not a lookback.
            const lastTime = candles[candles.length - 1].time as UTCTimestamp;
            sndZonesV2(candles, manualIndicator.period, STRUCTURE_ATR_PERIOD).forEach(
              (zone, zoneIdx) => {
                const demand = zone.kind === 'demand';
                const touched = zone.state === 'touched';
                // A touched (consumed, invalid) zone greys out; a fresh one
                // keeps its buy/sell tint and stays drawn to the right edge.
                // The BUY/SELL touch marker below always keeps the untouched
                // buy/sell tint (it marks the entry action itself), even
                // though the rectangle it sits inside has gone muted.
                const freshColor = pickZoneColor(
                  zoneColorStyle.sndV2,
                  demand,
                  false,
                  zoneColorStyle.customColors,
                );
                const lineColor = touched
                  ? pickZoneColor(zoneColorStyle.sndV2, demand, true, zoneColorStyle.customColors)
                  : freshColor;
                const fillColor = touched
                  ? hexToRgba(lineColor, 0.06)
                  : hexToRgba(lineColor, 0.2);
                const sndV2ZoneId = `${STRATEGY_DRAWING_PREFIX}snd2-zone:${manualIndicator.id}:${zoneIdx}`;
                manager.addDrawing(
                  new Rectangle(
                    sndV2ZoneId,
                    [
                      { time: zone.baseStartTime, price: zone.priceHigh },
                      { time: zone.touchedTime ?? lastTime, price: zone.priceLow },
                    ],
                    {
                      lineColor,
                      lineWidth: lineWidthVal,
                      lineDash: lineDashVal,
                      fillColor,
                    },
                    { filled: true, locked: true },
                  ),
                );
                zoneMeta.set(sndV2ZoneId, {
                  indicator: 'snd_v2',
                  indicatorLabel: 'S&D Zones v2',
                  pattern: zone.pattern,
                  kind: zone.kind,
                  priceLow: zone.priceLow,
                  priceHigh: zone.priceHigh,
                  timeStart: zone.baseStartTime as number,
                  timeEnd: zone.touchedTime ? (zone.touchedTime as number) : null,
                  state: touched ? 'touched' : 'fresh',
                  extra: zone.hasLegIn ? undefined : { Origin: 'No leg-in (origin base)' },
                });
                // Confirmation marker at the proximal edge, tagged with the
                // pattern (RBR/DBR/RBD/DBD, or DZ/SZ for an origin base) and
                // whether the zone is still fresh or has been touched.
                structureMarkers.push({
                  time: zone.time,
                  position: 'atPriceMiddle',
                  price: zone.proximal,
                  color: lineColor,
                  shape: demand ? 'arrowUp' : 'arrowDown',
                  text: `${zone.pattern} ${touched ? 'touched' : 'fresh'}`,
                });
                // The first touch is the entry (buy demand / sell supply) and
                // where the zone was consumed.
                if (zone.touchedTime) {
                  structureMarkers.push({
                    time: zone.touchedTime,
                    position: 'atPriceMiddle',
                    price: zone.proximal,
                    color: freshColor,
                    shape: demand ? 'arrowUp' : 'arrowDown',
                    text: demand ? 'BUY' : 'SELL',
                  });
                }
              },
            );
            break;
          }
          case 'base': {
            // Consolidation bases (see `detectBases` in indicators.ts) drawn
            // as two locked, dashed horizontal lines per base — the range's
            // high and low. `period` is the base's minimum candle count. Only
            // the most recent 3 bases are drawn to avoid clutter.
            const anchorTime = candles[0].time as UTCTimestamp;
            const bases = detectBases(candles, manualIndicator.period, STRUCTURE_ATR_PERIOD);
            const baseColor = cssVar('--color-ink-muted');
            const targetColor = manualIndicator.color || baseColor;
            const targetDash = manualIndicator.lineStyle === 'solid' ? undefined : manualIndicator.lineStyle === 'dotted' ? [2, 2] : [4, 4];
            const slicedBases = bases.slice(-3);
            slicedBases.forEach((base, baseIdx) => {
              const opacity = slicedBases.length > 1
                ? 0.4 + (baseIdx / (slicedBases.length - 1)) * 0.6
                : 1.0;
              const lineColor = targetColor.startsWith('#')
                ? hexToRgba(targetColor, opacity)
                : targetColor;
              for (const [edge, price, labelText] of [
                ['hi', base.high, 'Base high'],
                ['lo', base.low, 'Base low'],
              ] as const) {
                manager.addDrawing(
                  HorizontalLine.create(
                    `${STRATEGY_DRAWING_PREFIX}base:${manualIndicator.id}:${baseIdx}:${edge}`,
                    price,
                    anchorTime,
                    {
                      lineColor,
                      lineWidth: lineWidthVal,
                      lineDash: targetDash,
                    },
                    {
                      locked: true,
                      showPrice: true,
                      showLabel: true,
                      labelText,
                    },
                  ),
                );
              }
            });
            break;
          }
          case 'volume_profile': {
            // Horizontal volume-by-price histogram overlaid on the main
            // price pane — a custom series primitive (not a line/histogram
            // series), attached to the candle series. See
            // volumeProfilePrimitive.ts.
            const lookback = Math.max(10, manualIndicator.period > 0 ? manualIndicator.period * 20 : candles.length);
            const profileCandles =
              candles.length > lookback ? candles.slice(candles.length - lookback) : candles;
            const bucketCount = manualIndicator.bucketCount ?? Math.max(5, manualIndicator.period || 24);
            const primitive = new VolumeProfilePrimitive({
              candles: profileCandles,
              bucketCount,
              side: manualIndicator.side ?? 'right',
              color: manualIndicator.color,
            });
            if (candleSeries) {
              candleSeries.attachPrimitive(primitive);
              volumeProfilePrimitivesRef.current.push(primitive);
            }
            break;
          }
          case 'patterns': {
            for (const p of detectPatterns(candles)) {
              structureMarkers.push({
                time: p.time,
                position: p.label.startsWith('bullish') ? 'belowBar' : 'aboveBar',
                color: manualIndicator.color,
                shape: 'circle',
                size: 0,
                text: p.label,
              });
            }
            break;
          }
          case 'smc': {
            const { obs, fvgs, structures } = smcConcepts(candles, manualIndicator.period, STRUCTURE_ATR_PERIOD);
            
            // Plot structures (BOS / CHoCH)
            for (const st of structures) {
              structureMarkers.push({
                time: st.time,
                position: st.direction === 'bullish' ? 'belowBar' : 'aboveBar',
                color: manualIndicator.color,
                shape: st.direction === 'bullish' ? 'arrowUp' : 'arrowDown',
                size: 0,
                text: `${st.kind} (${st.direction === 'bullish' ? '+' : '-'})`,
              });
            }

            const lastTime = candles[candles.length - 1].time as UTCTimestamp;
            let zoneIdx = 0;

            const drawZone = (z: any, type: string) => {
              const demand = z.kind === 'bullish';
              const touched = !!z.mitigatedTime;
              const isFvg = type === 'FVG';
              
              const smcColors = isFvg 
                ? (zoneColorStyle.fvg || { demandColor: '#8a2be2', supplyColor: '#8a2be2', touchedColor: '#787b86' })
                : (zoneColorStyle.smc || { demandColor: '#42a5f5', supplyColor: '#ff9800', touchedColor: '#787b86' });
                
              const freshColor = pickZoneColor(smcColors, demand, false, zoneColorStyle.customColors);
              const lineColor = touched ? pickZoneColor(smcColors, demand, true, zoneColorStyle.customColors) : freshColor;
              const fillColor = touched ? hexToRgba(lineColor, 0.06) : hexToRgba(lineColor, isFvg ? 0.1 : 0.2);
              const smcZoneId = `${STRATEGY_DRAWING_PREFIX}smc-zone:${manualIndicator.id}:${zoneIdx++}`;
              manager.addDrawing(
                new Rectangle(
                  smcZoneId,
                  [
                    { time: z.time, price: z.high },
                    { time: z.mitigatedTime ?? lastTime, price: z.low },
                  ],
                  { lineColor, lineWidth: lineWidthVal, lineDash: lineDashVal, fillColor },
                  { filled: true, locked: true },
                ),
              );
              zoneMeta.set(smcZoneId, {
                indicator: 'smc',
                indicatorLabel: `SMC ${type}`,
                pattern: type,
                kind: demand ? 'demand' : 'supply',
                priceLow: z.low,
                priceHigh: z.high,
                timeStart: z.time as number,
                timeEnd: z.mitigatedTime ? (z.mitigatedTime as number) : null,
                state: touched ? 'touched' : 'fresh',
              });
              
              if (!isFvg) {
                structureMarkers.push({
                  time: z.time,
                  position: 'atPriceMiddle',
                  price: z.high,
                  color: lineColor,
                  shape: demand ? 'arrowUp' : 'arrowDown',
                  text: `${type} ${demand ? 'Buy' : 'Sell'}`,
                });
              }
            };

            for (const ob of obs) drawZone(ob, 'OB');
            for (const fvg of fvgs) drawZone(fvg, 'FVG');
            break;
          }
        }
      }
      // If we have custom code results, plot their custom indicators
      if (customCodeResultRef.current) {
        const { indicators, candles: customCandles } = customCodeResultRef.current;
        let colorIdx = 0;
        const CUSTOM_INDICATOR_COLORS = [
          '#00f0ff',
          '#e0aaff',
          '#ffd166',
          '#06d6a0',
          '#ff70a6',
        ];
        for (const [name, values] of Object.entries(indicators)) {
          const lineData = [];
          for (let i = 0; i < customCandles.length; i++) {
            const val = values[i];
            if (val !== null && val !== undefined) {
              lineData.push({
                time: customCandles[i].time as UTCTimestamp,
                value: val,
              });
            }
          }
          if (lineData.length > 0) {
            const series = chart.addSeries(LineSeries, {
              color: CUSTOM_INDICATOR_COLORS[colorIdx % CUSTOM_INDICATOR_COLORS.length],
              lineWidth: 2,
              priceLineVisible: false,
              lastValueVisible: false,
              title: name,
            });
            series.setData(lineData);
            indicatorSeriesRef.current.push(series);
            colorIdx++;
          }
        }
      }

      // Saved custom (backend-Python) indicators added via IndicatorsDock —
      // computed asynchronously by the computeCustomIndicators effect below
      // and cached per manual-indicator instance id in
      // customIndicatorResultsRef, so this stays a synchronous read like
      // every other case here. Silently skipped if the result hasn't
      // arrived yet or carries an error (surfaced in IndicatorsDock instead
      // of breaking the rest of the chart).
      //
      // A series name ending in `_marker_up`/`_marker_down`/`_marker` is a
      // reserved convention (see the PoB pattern/confirmation indicators)
      // for discrete one-off events — a candle pattern, a swing point, an
      // entry retest — rather than a continuous line. Values for those bars
      // would otherwise get silently connected by a straight line across
      // whatever gap separates two occurrences (LineSeries has no concept
      // of "these two points aren't related"), so they're routed through
      // the same `structureMarkers`/structure-markers plugin the built-in
      // structure/QML/pattern indicators already use instead.
      for (const manualIndicator of manualIndicatorsRef.current) {
        if (manualIndicator.hidden) continue;
        if (
          manualIndicator.type !== 'custom' ||
          !(manualIndicator.indicatorId || manualIndicator.previewCode)
        )
          continue;
        const result = customIndicatorResultsRef.current[manualIndicator.id];
        if (!result || result.error) continue;
        let colorIdx = 0;
        for (const [seriesName, values] of Object.entries(result.series)) {
          const markerKind = seriesName.endsWith('_marker_up')
            ? 'up'
            : seriesName.endsWith('_marker_down')
              ? 'down'
              : seriesName.endsWith('_marker')
                ? 'neutral'
                : null;

          if (markerKind) {
            const label = seriesName.replace(/_marker(_up|_down)?$/, '').replace(/_/g, ' ');
            for (let i = 0; i < result.times.length; i++) {
              const val = values[i];
              if (val === null || val === undefined) continue;
              structureMarkers.push({
                time: result.times[i] as UTCTimestamp,
                position:
                  markerKind === 'up'
                    ? 'belowBar'
                    : markerKind === 'down'
                      ? 'aboveBar'
                      : 'atPriceMiddle',
                price: markerKind === 'neutral' ? val : undefined,
                color:
                  markerKind === 'up'
                    ? cssVar('--color-ok')
                    : markerKind === 'down'
                      ? cssVar('--color-err')
                      : manualIndicator.color,
                shape:
                  markerKind === 'up'
                    ? 'arrowUp'
                    : markerKind === 'down'
                      ? 'arrowDown'
                      : 'circle',
                size: markerKind === 'neutral' ? 0 : undefined,
                text: label,
              } as SeriesMarker<Time>);
            }
            continue;
          }

          const lineData: { time: UTCTimestamp; value: number }[] = [];
          for (let i = 0; i < result.times.length; i++) {
            const val = values[i];
            if (val !== null && val !== undefined) {
              lineData.push({ time: result.times[i] as UTCTimestamp, value: val });
            }
          }
          if (lineData.length === 0) continue;
          const lineStyleVal =
            manualIndicator.lineStyle === 'dashed'
              ? LineStyle.Dashed
              : manualIndicator.lineStyle === 'dotted'
                ? LineStyle.Dotted
                : LineStyle.Solid;
          const lineWidthVal = (manualIndicator.lineWidth ?? 1) as any;
          const series = chart.addSeries(LineSeries, {
            color: hexToRgba(manualIndicator.color, colorIdx === 0 ? 1 : 0.6),
            lineWidth: lineWidthVal,
            lineStyle: lineStyleVal,
            priceLineVisible: false,
            lastValueVisible: false,
            title: `${manualIndicator.label} ${seriesName}`,
          });
          series.setData(lineData);
          indicatorSeriesRef.current.push(series);
          colorIdx++;
        }
      }
      chartController?.getStructureMarkersPrimitive()?.setMarkers(
        structureMarkers.sort((a, b) => (a.time as number) - (b.time as number)),
      );

      // Draw day/period separators if enabled
      if (showSeparatorsRef.current) {
        const tf = timeframeRef.current;
        for (let i = 1; i < candles.length; i++) {
          const prev = candles[i - 1];
          const curr = candles[i];

          let isNew = false;
          const prevDate = new Date(prev.time * 1000);
          const currDate = new Date(curr.time * 1000);

          if (tf === 'W1') {
            isNew =
              prevDate.getUTCMonth() !== currDate.getUTCMonth() ||
              prevDate.getUTCFullYear() !== currDate.getUTCFullYear();
          } else if (tf === 'MN') {
            isNew = prevDate.getUTCFullYear() !== currDate.getUTCFullYear();
          } else if (tf === 'D1') {
            const prevWeek = Math.floor((prev.time / 86400 + 3) / 7);
            const currWeek = Math.floor((curr.time / 86400 + 3) / 7);
            isNew = prevWeek !== currWeek;
          } else {
            // M1, M5, M15, M30, H1, H4
            isNew =
              prevDate.getUTCDate() !== currDate.getUTCDate() ||
              prevDate.getUTCMonth() !== currDate.getUTCMonth() ||
              prevDate.getUTCFullYear() !== currDate.getUTCFullYear();
          }

          if (isNew) {
            const t = curr.time as UTCTimestamp;
            const drawing = VerticalLine.create(
              `${SEPARATOR_DRAWING_PREFIX}${symbolRef.current}:${i}`,
              t,
              curr.open,
              {
                lineColor: hexToRgba(cssVar('--color-ink'), 0.5),
                lineWidth: 1,
                lineDash: [4, 4],
              },
              {
                locked: true,
              },
            );
            manager.addDrawing(drawing);
          }
        }
      }
    };

    // Declarative trigger: recompute overlays when the active strategy
    // changes (activated, deactivated, or a different one picked up for
    // this symbol), when the user adds/removes a manual indicator, or when
    // the separators toggle changes — without waiting for the next candle.
    // Also re-runs once the chart engine becomes ready (mount), so an
    // indicator added before the chart exists still renders as soon as it
    // does, instead of needing a later unrelated trigger.
    if (chartController?.isReady) {
      recomputeIndicatorsRef.current();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartController, manualIndicators, activeStrategy, showSeparators, zoneColorStyle]);

  // Saved custom (backend-Python) indicators need an API round trip to
  // compute, unlike every other manual-indicator type, so they can't live
  // inside the synchronous recomputeIndicators above. Reassigned every
  // render so `computeCustomIndicatorsRef.current()` (called below, and
  // from ChartPanel's chart-creation effect `render()`) always sees the
  // latest symbol/timeframe/manualIndicators.
  computeCustomIndicatorsRef.current = () => {
    const customInstances = manualIndicators.filter(
      (ind) => ind.type === 'custom' && (!!ind.indicatorId || !!ind.previewCode),
    );
    if (customInstances.length === 0) return;
    const periodParam = derivePeriodParam(getRawCandles());
    if (!periodParam) return;

    Promise.all(
      customInstances.map(async (ind) => {
        try {
          const result = ind.indicatorId
            ? await computeIndicator(ind.indicatorId, {
                symbol,
                timeframe,
                period: periodParam,
              })
            : await previewIndicatorCode({
                code: ind.previewCode as string,
                symbol,
                timeframe,
                period: periodParam,
              });
          return [ind.id, result] as const;
        } catch (err) {
          return [
            ind.id,
            {
              times: [],
              series: {},
              error: err instanceof Error ? err.message : 'compute failed',
            } as ComputeIndicatorResponse,
          ] as const;
        }
      }),
    ).then((entries) => {
      const presentIds = new Set(customInstances.map((ind) => ind.id));
      const next: Record<string, ComputeIndicatorResponse> = {};
      for (const [id, result] of entries) next[id] = result;
      // Keep results for instances still present that weren't in this
      // batch (shouldn't happen given the filter above, but avoids
      // silently dropping data if this is ever narrowed later).
      for (const [id, result] of Object.entries(customIndicatorResultsRef.current)) {
        if (presentIds.has(id) && !(id in next)) next[id] = result;
      }
      customIndicatorResultsRef.current = next;
      recomputeIndicatorsRef.current();
    });
  };

  // Add/remove a custom indicator, or switch symbol/timeframe, without
  // waiting for the next candle. The initial-history-load case (candles not
  // loaded yet on mount) is covered separately by `render()` inside
  // ChartPanel's chart-creation effect.
  useEffect(() => {
    computeCustomIndicatorsRef.current();
  }, [manualIndicators, symbol, timeframe]);

  // Manual indicators follow the same per-symbol load convention as
  // drawings — reload the new symbol's saved set whenever it changes. Split
  // out of ChartPanel's drawing-reload effect (which handles the
  // drawing-manager half of the same symbol switch, phase 8 territory) —
  // the two halves don't depend on each other's ordering.
  useEffect(() => {
    setManualIndicators(loadManualIndicators(symbol));
  }, [symbol]);

  // The live-bot eye view (`liveBotSkill`) mirrors the assigned strategy
  // version's indicator spec onto the chart itself — the same chip a trader
  // would add by hand from the Indicators dock — and removes it again when
  // the eye turns off (or moves to a different bot), so the chip only stays
  // lit while its bot is being watched. It won't touch a chip the user
  // added manually themselves.
  useEffect(() => {
    if (!liveBotSkill || !accountId) {
      setLiveBotIndicators(null);
      return;
    }
    setShowSignalsDock(true);
    let cancelled = false;
    let addedIndicatorId: string | null = null;
    Promise.all([getSkillAssignments(), getStrategyVersions(accountId)])
      .then(([assignments, versions]) => {
        if (cancelled) return;
        const assignment = assignments.find((a) => a.name === liveBotSkill);
        const version = assignment
          ? versions.find((v) => v.name === assignment.strategy && v.status === 'active')
          : undefined;
        setLiveBotIndicators(version?.spec?.indicators ?? []);

        if (assignment && usesSndZones(assignment.strategy)) {
          setManualIndicators((prev) => {
            if (prev.some((i) => i.type === 'snd' || i.type === 'snd_v2')) return prev;
            const id = crypto.randomUUID();
            addedIndicatorId = id;
            const next: ManualIndicator[] = [
              ...prev,
              {
                id,
                type: 'snd_v2',
                period: 30,
                color: '#42a5f5',
                label: 'S&D zones v2 (base ≤ 30)',
              },
            ];
            saveManualIndicators(symbol, next);
            return next;
          });
        }
      })
      .catch(() => {
        if (!cancelled) setLiveBotIndicators([]);
      });
    return () => {
      cancelled = true;
      if (addedIndicatorId) {
        setManualIndicators((prev) => {
          const next = prev.filter((i) => i.id !== addedIndicatorId);
          saveManualIndicators(symbol, next);
          return next;
        });
      }
    };
  }, [accountId, liveBotSkill, symbol, setShowSignalsDock]);

  function addManualIndicator(indicator: ManualIndicator) {
    setManualIndicators((prev) => {
      const next = [...prev, indicator];
      saveManualIndicators(symbolRef.current, next);
      return next;
    });
  }

  function removeManualIndicator(id: string) {
    setManualIndicators((prev) => {
      const next = prev.filter((ind) => ind.id !== id);
      saveManualIndicators(symbolRef.current, next);
      return next;
    });
  }

  function updateManualIndicator(id: string, patch: Partial<ManualIndicator>) {
    setManualIndicators((prev) => {
      const next = prev.map((ind) => (ind.id === id ? { ...ind, ...patch } : ind));
      saveManualIndicators(symbolRef.current, next);
      return next;
    });
  }

  return {
    manualIndicators,
    setManualIndicators,
    manualIndicatorsRef,
    addManualIndicator,
    removeManualIndicator,
    updateManualIndicator,
    showIndicatorsDock,
    setShowIndicatorsDock,
    liveBotIndicators,
    indicatorSeriesRef,
    customIndicatorResultsRef,
    computeCustomIndicatorsRef,
    recomputeIndicatorsRef,
  };
}

export type Indicators = ReturnType<typeof useIndicators>;
