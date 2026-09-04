import type { Candle } from '@/shared/api/client';
import type { DrawingToolType } from './types';

// Number of anchor points (clicks) each drawing tool needs before it's
// committed — shared by the drawing-creation flow (ChartPanel's pointer
// handling) and the toolbar's "Drawing <tool>: N more point(s)" status text.
export const REQUIRED_ANCHORS: Record<DrawingToolType, number> = {
  'trend-line': 2,
  'extended-line': 2,
  'horizontal-line': 1,
  'vertical-line': 1,
  rectangle: 2,
  'fib-retracement': 2,
  'parallel-channel': 3,
  circle: 2,
  'long-position': 3,
  'short-position': 3,
  'price-label': 1,
  'text-annotation': 1,
};

export const TIMEFRAMES: Candle['timeframe'][] = [
  'M1',
  'M5',
  'M15',
  'M30',
  'H1',
  'H4',
  'D1',
  'W1',
  'MN',
];

export function isTimeframe(value: string | null): value is Candle['timeframe'] {
  return TIMEFRAMES.includes(value as Candle['timeframe']);
}

export function cssVar(name: string): string {
  const val = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  if (val) return val;
  // Fallbacks in case the document stylesheets haven't parsed yet:
  switch (name) {
    case '--color-bg':
      return '#131722';
    case '--color-panel':
      return '#1e222d';
    case '--color-line':
      return '#2a2e39';
    case '--color-ink':
      return '#d1d4dc';
    case '--color-ink-muted':
      return '#5d606b';
    case '--color-accent':
      return '#2962ff';
    case '--color-ok':
      return '#26a69a';
    case '--color-err':
      return '#ef5350';
    case '--color-buy':
      return '#42a5f5';
    case '--color-sell':
      return '#ff9800';
    default:
      return '';
  }
}

export function hexToRgba(hex: string, alpha: number): string {
  const clean = hex.replace('#', '');
  const value = parseInt(clean, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Resolves a zone rectangle's fill/border color for one indicator's
 * `ZoneIndicatorColors` (see types.ts) — falls back to the original hardcoded
 * theme tokens (buy/sell/muted) when `customColors` is off, so nothing
 * changes visually until a user opts into the "Zone colors" settings panel.
 * `colors.touchedColor` is optional because the per-trade backend zone type
 * has no fresh/touched state of its own (a trade was, by definition, taken
 * from it) — `touched` is always false for that caller. */
export function pickZoneColor(
  colors: { demandColor: string; supplyColor: string; touchedColor?: string },
  demand: boolean,
  touched: boolean,
  customColors: boolean,
): string {
  if (!customColors) {
    return touched
      ? cssVar('--color-ink-muted')
      : cssVar(demand ? '--color-buy' : '--color-sell');
  }
  if (touched && colors.touchedColor) return colors.touchedColor;
  return demand ? colors.demandColor : colors.supplyColor;
}

/** Strategy families whose entries are S&D (RBR/DBD/RBD/DBR) zone retests —
 * matched by name rather than an explicit allowlist so newly generated
 * zone-based bots (the `new-strategy` skill scaffolds `pob_snd_zones_*`/
 * `rbr_dbd_zones_*`/`pob_price_action_snd*` per symbol) are picked up
 * without this list needing an update every time. */
export function usesSndZones(strategyName: string): boolean {
  return /snd|rbr_dbd/i.test(strategyName);
}

// Default distance for a not-yet-set SL/TP placeholder line: a flat points
// value would be meaningless across arbitrary instruments (gold vs. a
// synthetic index vs. BTC), so scale it off the reference price instead.
export function defaultOffset(referencePrice: number): number {
  return Math.abs(referencePrice) * 0.005 || 1;
}

export function numOrNull(value: string): number | null {
  if (value.trim() === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

/** "YYYY-MM:YYYY-MM" spanning the currently-loaded candle range — the
 * period format `parse_period` (backend/src/backtest/application/period.py)
 * expects. Returns null with nothing loaded yet. */
export function derivePeriodParam(candles: Candle[]): string | null {
  const oldestCandle = candles[0];
  const newestCandle = candles[candles.length - 1];
  if (!oldestCandle || !newestCandle) return null;
  const oldestDate = new Date(oldestCandle.time * 1000);
  const newestDate = new Date(newestCandle.time * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  const oldestStr = `${oldestDate.getUTCFullYear()}-${pad(oldestDate.getUTCMonth() + 1)}`;
  const newestStr = `${newestDate.getUTCFullYear()}-${pad(newestDate.getUTCMonth() + 1)}`;
  return `${oldestStr}:${newestStr}`;
}

export function getLocalTimeZoneOptions() {
  return {
    localization: {
      timeFormatter: (time: number) => {
        const date = new Date(time * 1000);
        return date.toLocaleString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        });
      },
    },
    timeScale: {
      tickMarkFormatter: (time: number, tickMarkType: number, locale: string) => {
        const date = new Date(time * 1000);
        switch (tickMarkType) {
          case 0: // Year
            return date.toLocaleString(locale, { year: 'numeric' });
          case 1: // Month
            return date.toLocaleString(locale, { month: 'short' });
          case 2: // DayOfMonth
            return date.toLocaleString(locale, { day: 'numeric' });
          case 3: // Time
            return date.toLocaleString(locale, { hour: '2-digit', minute: '2-digit' });
          case 4: // TimeWithSeconds
            return date.toLocaleString(locale, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
          default:
            return date.toLocaleString(locale, { hour: '2-digit', minute: '2-digit' });
        }
      }
    }
  };
}
