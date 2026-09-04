import type { SeriesMarker, Time, UTCTimestamp } from 'lightweight-charts';
import {
  type IDrawing,
  TrendLine,
  ExtendedLine,
  HorizontalLine,
  VerticalLine,
  Rectangle,
  FibRetracement,
  ParallelChannel,
  Circle,
  LongPosition,
  ShortPosition,
  PriceLabel,
  TextAnnotation,
} from 'lightweight-charts-drawing';
import type {
  BacktestSignal,
  BacktestTrade,
  Candle,
  CustomSignal,
  NewsEventRecord,
  TradeMarker,
} from '@/shared/api/client';
import { SIGNAL_OUTCOME_META } from '@/features/backtest/signalOutcome';
import type { DrawingToolType, OrderLineStyle } from './types';
import { cssVar, hexToRgba } from './chartFormat';
import { groupByKey, nearestCandleTime } from './chartData';
import {
  BACKTEST_DRAWING_PREFIX,
  LIVE_TRADE_DRAWING_PREFIX,
} from './chartStorage';

export function createDrawingInstance(
  tool: DrawingToolType,
  id: string,
  anchors: Array<{ price: number; time: UTCTimestamp }>,
  style: any,
): IDrawing | null {
  switch (tool) {
    case 'trend-line':
      return new TrendLine(id, anchors, style);
    case 'extended-line':
      return new ExtendedLine(id, anchors, style);
    case 'horizontal-line':
      return new HorizontalLine(id, anchors, style);
    case 'vertical-line':
      return new VerticalLine(id, anchors, style);
    case 'rectangle':
      return new Rectangle(id, anchors, style);
    case 'fib-retracement':
      return new FibRetracement(id, anchors, style);
    case 'parallel-channel':
      return new ParallelChannel(id, anchors, style);
    case 'circle':
      return new Circle(id, anchors, style);
    case 'long-position':
      return new LongPosition(id, anchors, style);
    case 'short-position':
      return new ShortPosition(id, anchors, style);
    case 'price-label':
      return new PriceLabel(id, anchors, style);
    case 'text-annotation':
      return new TextAnnotation(id, anchors, style, { text: 'Label', fontSize: 14 });
    default:
      return null;
  }
}

export function toSeriesMarkers(
  trades: TradeMarker[],
  colors: { ok: string; err: string },
  showLabels = true,
): SeriesMarker<Time>[] {
  return [];
}

/** Entry->exit oblique line for each closed live trade (LIVE_TRADE_DRAWING_PREFIX)
 * — mirrors the SL/TP-style segments `buildTradeSetupDrawings` draws for a
 * backtest report, but sourced from the journal's `TradeMarker[]` poll so a
 * closed live position is visible on the chart the same way. Open trades
 * (close_time/close_price still null) are skipped — there's no exit yet. */
export function buildLiveTradeLineDrawings(
  trades: TradeMarker[],
  colors: { ok: string; err: string },
  candles: Candle[],
  style?: OrderLineStyle,
): IDrawing[] {
  if (style && !style.showExitLine) return [];
  const drawings: IDrawing[] = [];
  for (const t of trades) {
    if (t.close_time === null || t.close_price === null) continue;
    const openTime = nearestCandleTime(candles, t.open_time);
    const closeTime = nearestCandleTime(candles, t.close_time);
    if (openTime === null || closeTime === null) continue;
    const drawing = buildExitLineDrawing(
      LIVE_TRADE_DRAWING_PREFIX,
      t.id,
      openTime,
      t.open_price,
      closeTime,
      t.close_price,
      t.profit ?? 0,
      colors,
      style,
    );
    if (drawing) drawings.push(drawing);
  }
  return drawings;
}

/** Same entry-arrow/exit-circle rendering as `toSeriesMarkers`, but for a
 * backtest report's closed trades (§F: "test the bot against candle
 * history") — a `BacktestTrade` always has a `close_time`/`close_price`
 * (the run is over), unlike a live `TradeMarker` which is null while open.
 * Also folds in the trade's `pattern` into the entry marker's text when the
 * strategy reports one. `t.structure` (the swing window that validated this
 * trade's zone) is intentionally NOT drawn here — it only covers each
 * trade's own ~100-bar lookback, so real swings between/around trades were
 * silently missing; the 'structure' manual indicator draws HH/HL/LH/LL over
 * the whole chart instead (see `swingStructure()` in indicators.ts). */
export function toBacktestSeriesMarkers(
  trades: BacktestTrade[],
  colors: { ok: string; err: string },
  showLabels = true,
): SeriesMarker<Time>[] {
  return [];
}

/** Square markers for the report's signals that did NOT become trades
 * (vetoed / rejected / skipped) — the opened ones already render as the
 * trade entry arrows above, so drawing them again would double up. Colored
 * by outcome via the shared SIGNAL_OUTCOME_META design tokens, so the
 * chart, the SignalsDock and the report page all read the same. */
export function toSignalSeriesMarkers(
  signals: BacktestSignal[],
  showLabels = true,
): SeriesMarker<Time>[] {
  const groups = groupByKey(
    signals.filter((s) => s.outcome !== 'opened'),
    (s) => `${s.time}:${s.direction}:${s.outcome}`,
  );
  return groups.map<SeriesMarker<Time>>((group) => {
    const s = group[0];
    return {
      time: s.time as UTCTimestamp,
      position: s.direction === 'buy' ? 'belowBar' : 'aboveBar',
      color: cssVar(SIGNAL_OUTCOME_META[s.outcome].token),
      shape: 'square',
      text: showLabels
        ? group.length > 1
          ? `${s.direction.toUpperCase()} ×${group.length} · ${SIGNAL_OUTCOME_META[s.outcome].label}`
          : `${s.direction.toUpperCase()} signal · ${SIGNAL_OUTCOME_META[s.outcome].label}`
        : '',
    };
  });
}

/** Oblique line from a closed trade's entry (open_time, open_price) to its
 * exit (close_time, close_price) — a closed position previously only left
 * behind an entry arrow + a small exit circle, with nothing tying them
 * together or showing the price path between them. Colored ok/err by
 * profit, same as the exit marker. */
export function buildExitLineDrawing(
  idPrefix: string,
  tradeId: string,
  openTime: UTCTimestamp,
  openPrice: number,
  closeTime: UTCTimestamp,
  closePrice: number,
  profit: number,
  colors: { ok: string; err: string },
  style?: OrderLineStyle,
): IDrawing | null {
  if (style && !style.showExitLine) return null;
  const dashStyle =
    style?.exitLineDash === 'dashed'
      ? [4, 4]
      : style?.exitLineDash === 'dotted'
        ? [2, 2]
        : undefined;
  const winColor = style?.exitLineCustomColor ? (style.exitLineWinColor || colors.ok) : colors.ok;
  const lossColor = style?.exitLineCustomColor ? (style.exitLineLossColor || colors.err) : colors.err;
  return new TrendLine(
    `${idPrefix}exit-line:${tradeId}`,
    [
      { time: openTime, price: openPrice },
      { time: closeTime, price: closePrice },
    ],
    {
      lineColor: profit >= 0 ? winColor : lossColor,
      lineWidth: style?.exitLineWidth ?? 2,
      lineDash: dashStyle,
    },
    { locked: true },
  );
}

/** Zone rectangle + SL/TP segments for a single backtest trade — the part
 * that's known the moment the trade opens (unlike the exit line below, whose
 * color/endpoint depend on the close). Split out from the old
 * `buildBacktestZoneDrawings` so replay (§F, ChartPanel's trade-drawing
 * effect) can reveal a trade's setup at `open_time` and its exit separately
 * at `close_time`, instead of the whole bundle appearing at once. Each
 * segment is bounded to the trade's own open→close time span (unlike live's
 * full-chart `buildPriceLines()`), since a report can have many trades on
 * screen at once. Only strategies that set `Signal.zone`/`sl`/`tp` produce
 * anything here; trades without one are skipped for that piece. */
export function buildTradeSetupDrawings(
  t: BacktestTrade,
  i: number,
  colors: { demand: string; supply: string; sl: string; tp: string },
): IDrawing[] {
  const drawings: IDrawing[] = [];
  if (t.zone) {
    const zoneColor = t.zone.kind === 'demand' ? colors.demand : colors.supply;
    drawings.push(
      new Rectangle(
        `${BACKTEST_DRAWING_PREFIX}zone:${i}`,
        [
          { time: t.zone.time_start as UTCTimestamp, price: t.zone.price_high },
          { time: t.zone.time_end as UTCTimestamp, price: t.zone.price_low },
        ],
        {
          lineColor: zoneColor,
          lineWidth: 1,
          fillColor: hexToRgba(zoneColor, 0.15),
        },
        { filled: true, locked: true },
      ),
    );
  }
  const openTime = t.open_time as UTCTimestamp;
  const closeTime = t.close_time as UTCTimestamp;
  if (t.sl !== null) {
    drawings.push(
      new TrendLine(
        `${BACKTEST_DRAWING_PREFIX}sl:${i}`,
        [
          { time: openTime, price: t.sl },
          { time: closeTime, price: t.sl },
        ],
        { lineColor: colors.sl, lineWidth: 1, lineDash: [4, 4] },
        { locked: true },
      ),
    );
  }
  if (t.tp !== null) {
    drawings.push(
      new TrendLine(
        `${BACKTEST_DRAWING_PREFIX}tp:${i}`,
        [
          { time: openTime, price: t.tp },
          { time: closeTime, price: t.tp },
        ],
        { lineColor: colors.tp, lineWidth: 1, lineDash: [4, 4] },
        { locked: true },
      ),
    );
  }
  return drawings;
}

/** One marker per persisted calendar event (`GET .../news/events`, Phase 6
 * Part B / Phase 7) — square markers, colored by impact level, positioned
 * above the bar so they never collide with the arrowUp/arrowDown/circle
 * shapes `toSeriesMarkers`/`toBacktestSeriesMarkers` use for trades. Square
 * is also `toSignalSeriesMarkers`'s shape, but that builder's output is
 * never combined into the same `setMarkers` array as this one (vetoed
 * signals are backtest/eyed-bot-only; news markers are wired into the live
 * view), so there's no on-chart collision despite the shared shape. */
export function toNewsEventSeriesMarkers(events: NewsEventRecord[]): SeriesMarker<Time>[] {
  const colorByImpact: Record<string, string> = {
    high: cssVar('--color-err'),
    medium: cssVar('--color-sell'),
    low: cssVar('--color-ink-muted'),
  };
  
  const groups = groupByKey(events, (e) => String(e.time));
  const markers: SeriesMarker<Time>[] = [];
  for (const group of groups) {
    const hasHigh = group.some(e => e.impact === 'high');
    const hasMedium = group.some(e => e.impact === 'medium');
    const color = hasHigh ? colorByImpact['high'] : (hasMedium ? colorByImpact['medium'] : colorByImpact['low']);
    
    markers.push({
      time: group[0].time as UTCTimestamp,
      position: 'belowBar' as const,
      color: color ?? cssVar('--color-accent'),
      shape: 'circle' as const,
      text: '',
    });
  }
  return markers.sort((a, b) => (a.time as number) - (b.time as number));
}

export function toCustomSignalsSeriesMarkers(
  signals: CustomSignal[],
  colors: { ok: string; err: string },
  showLabels = true,
): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];
  const groups = groupByKey(signals, (s) => `${s.time}:${s.direction}`);
  for (const group of groups) {
    const s = group[0];
    markers.push({
      time: s.time as UTCTimestamp,
      position: s.direction === 'buy' ? 'belowBar' : 'aboveBar',
      color: s.direction === 'buy' ? colors.ok : colors.err,
      shape: s.direction === 'buy' ? 'arrowUp' : 'arrowDown',
      text: showLabels
        ? group.length > 1
          ? `${s.direction.toUpperCase()} ×${group.length}`
          : `${s.direction.toUpperCase()}: ${s.reason}`
        : '',
    });
  }
  return markers.sort((a, b) => (a.time as number) - (b.time as number));
}
