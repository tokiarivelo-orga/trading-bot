import {
  DrawingManager,
  type SerializedDrawing,
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
import type { Candle } from '@/shared/api/client';
import type { ManualIndicator, OrderLineStyle, ZoneColorStyle } from './types';
import { isTimeframe } from './chartFormat';

// Prefix for drawings this component adds itself (from the active strategy's
// PDF-derived price levels) so they can be told apart from the user's own —
// never persisted to localStorage, never removed by "Clear All".
export const STRATEGY_DRAWING_PREFIX = 'strategy-derived:';
// Prefix for drawings rendered from a backtest report's trades (zone
// rectangles, SL/TP segments) — same "not user data" treatment as
// STRATEGY_DRAWING_PREFIX, but cleared/rebuilt on its own lifecycle (when
// the backtest report's trades change) rather than on every candle tick.
export const BACKTEST_DRAWING_PREFIX = 'backtest-derived:';
// Prefix for the entry->exit oblique line drawn for closed *live* trades
// (journal-backed) — same "not user data" treatment as BACKTEST_DRAWING_PREFIX,
// but rebuilt on the live trade-markers poll cadence instead of the backtest
// report lifecycle.
export const LIVE_TRADE_DRAWING_PREFIX = 'live-trade-derived:';
// Prefix for daily/period separators drawn on the chart.
export const SEPARATOR_DRAWING_PREFIX = 'separator:';

/** True for any drawing this component added itself (strategy price levels,
 * backtest zone/SL annotations, live closed-trade lines, or period
 * separators) — never user data, so excluded from persistence, the
 * drawings-list panel, and "Clear All". */
export function isProgrammaticDrawingId(id: string): boolean {
  return (
    id.startsWith(STRATEGY_DRAWING_PREFIX) ||
    id.startsWith(BACKTEST_DRAWING_PREFIX) ||
    id.startsWith(LIVE_TRADE_DRAWING_PREFIX) ||
    id.startsWith(SEPARATOR_DRAWING_PREFIX) ||
    id === 'drawing-preview'
  );
}

/** Removes every drawing except strategy-derived ones (see
 * `STRATEGY_DRAWING_PREFIX`) — used in place of `manager.clearAll()`
 * wherever the intent is "clear *my* drawings", not the strategy's
 * auto-plotted price levels. */
export function clearUserDrawings(manager: DrawingManager): void {
  for (const drawing of manager.getAllDrawings()) {
    if (!isProgrammaticDrawingId(drawing.id)) {
      manager.removeDrawing(drawing.id);
    }
  }
}

/**
 * Restores saved drawings for `symbol` from localStorage into `manager`.
 * Uses a minimal factory that maps the serialised `type` string back to
 * the appropriate Drawing subclass — only the tools we expose in the toolbar
 * are covered; unknown types are silently skipped so old/unknown data can
 * never crash the chart.
 */
export function loadDrawingsFromStorage(
  manager: DrawingManager,
  symbol: string,
): void {
  try {
    const raw = localStorage.getItem(`chart-drawings:${symbol}`);
    if (!raw) return;
    const data: SerializedDrawing[] = JSON.parse(raw);
    manager.importDrawings(data, (type, d) => {
      switch (type) {
        case 'trend-line':
          return new TrendLine(d.id, d.anchors, d.style, d.options);
        case 'extended-line':
          return new ExtendedLine(d.id, d.anchors, d.style, d.options);
        case 'horizontal-line':
          return new HorizontalLine(d.id, d.anchors, d.style, d.options);
        case 'vertical-line':
          return new VerticalLine(d.id, d.anchors, d.style, d.options);
        case 'rectangle':
          return new Rectangle(d.id, d.anchors, d.style, d.options);
        case 'fib-retracement':
          return new FibRetracement(d.id, d.anchors, d.style, d.options);
        case 'parallel-channel':
          return new ParallelChannel(d.id, d.anchors, d.style, d.options);
        case 'circle':
          return new Circle(d.id, d.anchors, d.style, d.options);
        case 'long-position':
          return new LongPosition(d.id, d.anchors, d.style, d.options);
        case 'short-position':
          return new ShortPosition(d.id, d.anchors, d.style, d.options);
        case 'price-label':
          return new PriceLabel(d.id, d.anchors, d.style, d.options);
        case 'text-annotation':
          return new TextAnnotation(d.id, d.anchors, d.style, d.options);
        default:
          return null;
      }
    });
  } catch {
    // Corrupt or missing localStorage data is silently ignored.
  }
}

export const LAST_TIMEFRAME_KEY = 'chart-last-timeframe';
export const TIMEFRAME_QUERY_KEY = 'timeframe';

/**
 * Restores the timeframe to open on load — `?timeframe=` wins over the last
 * one picked on any chart (`chart-last-timeframe`), same priority order as
 * the symbol resolution in page.tsx.
 */
export function loadLastTimeframe(): Candle['timeframe'] {
  try {
    const urlTimeframe = new URLSearchParams(window.location.search).get(
      TIMEFRAME_QUERY_KEY,
    );
    if (isTimeframe(urlTimeframe)) return urlTimeframe;
    const stored = localStorage.getItem(LAST_TIMEFRAME_KEY);
    return isTimeframe(stored) ? stored : 'M5';
  } catch {
    return 'M5';
  }
}

/** Restores manually-added indicators for `symbol` from localStorage. */
export function loadManualIndicators(symbol: string): ManualIndicator[] {
  try {
    const raw = localStorage.getItem(`chart-indicators:${symbol}`);
    if (!raw) return [];
    return JSON.parse(raw) as ManualIndicator[];
  } catch {
    return [];
  }
}

export function saveManualIndicators(
  symbol: string,
  indicators: ManualIndicator[],
): void {
  try {
    localStorage.setItem(
      `chart-indicators:${symbol}`,
      JSON.stringify(indicators),
    );
  } catch {
    // localStorage quota or serialisation errors are non-fatal.
  }
}

const ORDER_LINE_STYLE_KEY = 'chart-order-line-style';

const DEFAULT_ORDER_LINE_STYLE: OrderLineStyle = {
  visible: true,
  dash: 'dashed',
  width: 3,
  customColors: false,
  openColor: '#22c55e',
  closeColor: '#6366f1',
  showExitLine: true,
  exitLineDash: 'solid',
  exitLineWidth: 2,
  exitLineCustomColor: false,
  exitLineWinColor: '#26a69a',
  exitLineLossColor: '#ef5350',
};

export function loadOrderLineStyle(): OrderLineStyle {
  try {
    const raw = localStorage.getItem(ORDER_LINE_STYLE_KEY);
    if (!raw) return DEFAULT_ORDER_LINE_STYLE;
    return { ...DEFAULT_ORDER_LINE_STYLE, ...(JSON.parse(raw) as Partial<OrderLineStyle>) };
  } catch {
    return DEFAULT_ORDER_LINE_STYLE;
  }
}

export function saveOrderLineStyle(style: OrderLineStyle): void {
  try {
    localStorage.setItem(ORDER_LINE_STYLE_KEY, JSON.stringify(style));
  } catch {
    // localStorage quota or serialisation errors are non-fatal.
  }
}

const ZONE_COLOR_STYLE_KEY = 'chart-zone-color-style';

const DEFAULT_ZONE_COLOR_STYLE: ZoneColorStyle = {
  customColors: false,
  qml: { demandColor: '#42a5f5', supplyColor: '#ff9800', touchedColor: '#787b86' },
  snd: { demandColor: '#42a5f5', supplyColor: '#ff9800', touchedColor: '#787b86' },
  sndV2: { demandColor: '#42a5f5', supplyColor: '#ff9800', touchedColor: '#787b86' },
  smc: { demandColor: '#42a5f5', supplyColor: '#ff9800', touchedColor: '#787b86' },
  fvg: { demandColor: '#8a2be2', supplyColor: '#8a2be2', touchedColor: '#787b86' },
  tradeZone: { demandColor: '#42a5f5', supplyColor: '#ff9800' },
};

export function loadZoneColorStyle(): ZoneColorStyle {
  try {
    const raw = localStorage.getItem(ZONE_COLOR_STYLE_KEY);
    if (!raw) return DEFAULT_ZONE_COLOR_STYLE;
    const parsed = JSON.parse(raw) as Partial<ZoneColorStyle>;
    return {
      ...DEFAULT_ZONE_COLOR_STYLE,
      ...parsed,
      qml: { ...DEFAULT_ZONE_COLOR_STYLE.qml, ...parsed.qml },
      snd: { ...DEFAULT_ZONE_COLOR_STYLE.snd, ...parsed.snd },
      sndV2: { ...DEFAULT_ZONE_COLOR_STYLE.sndV2, ...parsed.sndV2 },
      smc: { ...DEFAULT_ZONE_COLOR_STYLE.smc, ...parsed.smc },
      fvg: { ...DEFAULT_ZONE_COLOR_STYLE.fvg, ...parsed.fvg },
      tradeZone: { ...DEFAULT_ZONE_COLOR_STYLE.tradeZone, ...parsed.tradeZone },
    };
  } catch {
    return DEFAULT_ZONE_COLOR_STYLE;
  }
}

export function saveZoneColorStyle(style: ZoneColorStyle): void {
  try {
    localStorage.setItem(ZONE_COLOR_STYLE_KEY, JSON.stringify(style));
  } catch {
    // localStorage quota or serialisation errors are non-fatal.
  }
}
