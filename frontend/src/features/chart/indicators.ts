/**
 * Pure indicator math over `Candle[]`, computed client-side so the backend
 * stays limited to emitting the structured `IndicatorSpec` (type/period/
 * params) extracted from a strategy's source PDF — see `ChartPanel.tsx`,
 * which feeds these into `chart.addSeries(LineSeries/HistogramSeries, ...)`.
 *
 * EMA matches this project's existing backend convention
 * (`backend/src/engine/application/mtf_confirm.py`: `ewm(span, adjust=False)`).
 * RSI uses standard Wilder smoothing since no backend RSI exists to match.
 */

import type { UTCTimestamp } from "lightweight-charts";
import type { Candle } from "@/shared/api/client";

export interface LinePoint {
  time: UTCTimestamp;
  value: number;
}

function toPoint(candle: Candle, value: number): LinePoint {
  return { time: candle.time as UTCTimestamp, value };
}

export function ema(candles: Candle[], period: number): LinePoint[] {
  if (candles.length === 0) return [];
  const alpha = 2 / (period + 1);
  const points: LinePoint[] = [];
  let prev = candles[0].close;
  points.push(toPoint(candles[0], prev));
  for (let i = 1; i < candles.length; i++) {
    prev = candles[i].close * alpha + prev * (1 - alpha);
    points.push(toPoint(candles[i], prev));
  }
  return points;
}

export function sma(candles: Candle[], period: number): LinePoint[] {
  const points: LinePoint[] = [];
  let sum = 0;
  for (let i = 0; i < candles.length; i++) {
    sum += candles[i].close;
    if (i >= period) sum -= candles[i - period].close;
    if (i >= period - 1) points.push(toPoint(candles[i], sum / period));
  }
  return points;
}

export function rsi(candles: Candle[], period: number): LinePoint[] {
  if (candles.length < period + 1) return [];
  const points: LinePoint[] = [];
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const change = candles[i].close - candles[i - 1].close;
    avgGain += Math.max(change, 0);
    avgLoss += Math.max(-change, 0);
  }
  avgGain /= period;
  avgLoss /= period;
  const rsiAt = (gain: number, loss: number) => (loss === 0 ? 100 : 100 - 100 / (1 + gain / loss));
  points.push(toPoint(candles[period], rsiAt(avgGain, avgLoss)));

  for (let i = period + 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = Math.max(change, 0);
    const loss = Math.max(-change, 0);
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    points.push(toPoint(candles[i], rsiAt(avgGain, avgLoss)));
  }
  return points;
}

function emaSeries(closes: number[], period: number): number[] {
  const alpha = 2 / (period + 1);
  const out: number[] = [closes[0]];
  for (let i = 1; i < closes.length; i++) {
    out.push(closes[i] * alpha + out[i - 1] * (1 - alpha));
  }
  return out;
}

export interface MacdResult {
  macdLine: LinePoint[];
  signalLine: LinePoint[];
  histogram: LinePoint[];
}

export function macd(
  candles: Candle[],
  fast: number,
  slow: number,
  signal: number,
): MacdResult {
  if (candles.length === 0) {
    return { macdLine: [], signalLine: [], histogram: [] };
  }
  const closes = candles.map((c) => c.close);
  const fastEma = emaSeries(closes, fast);
  const slowEma = emaSeries(closes, slow);
  const macdValues = fastEma.map((v, i) => v - slowEma[i]);
  const signalValues = emaSeries(macdValues, signal);

  const macdLine = candles.map((c, i) => toPoint(c, macdValues[i]));
  const signalLine = candles.map((c, i) => toPoint(c, signalValues[i]));
  const histogram = candles.map((c, i) => toPoint(c, macdValues[i] - signalValues[i]));
  return { macdLine, signalLine, histogram };
}

export interface BollingerResult {
  upper: LinePoint[];
  middle: LinePoint[];
  lower: LinePoint[];
}

export function bollinger(candles: Candle[], period: number, stdDev: number): BollingerResult {
  const upper: LinePoint[] = [];
  const middle: LinePoint[] = [];
  const lower: LinePoint[] = [];

  // Rolling sum/sum-of-squares instead of re-slicing + reducing the last
  // `period` candles at every bar — O(n) instead of O(n * period).
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < candles.length; i++) {
    const close = candles[i].close;
    sum += close;
    sumSq += close * close;
    if (i >= period) {
      const dropped = candles[i - period].close;
      sum -= dropped;
      sumSq -= dropped * dropped;
    }
    if (i >= period - 1) {
      const mean = sum / period;
      const variance = Math.max(0, sumSq / period - mean * mean);
      const sd = Math.sqrt(variance);
      middle.push(toPoint(candles[i], mean));
      upper.push(toPoint(candles[i], mean + stdDev * sd));
      lower.push(toPoint(candles[i], mean - stdDev * sd));
    }
  }
  return { upper, middle, lower };
}

/**
 * Cumulative VWAP over whatever candles are currently loaded (not a true
 * session VWAP reset at a fixed calendar boundary — matches how every other
 * indicator here is recomputed over the full in-memory window on each
 * `recomputeIndicators()` pass, see ChartPanel.tsx). Uses `tick_volume` as
 * the volume proxy, same convention as the volume histogram series.
 */
export function vwap(candles: Candle[]): LinePoint[] {
  const points: LinePoint[] = [];
  let cumulativePv = 0;
  let cumulativeVolume = 0;
  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    cumulativePv += typicalPrice * candle.tick_volume;
    cumulativeVolume += candle.tick_volume;
    points.push(toPoint(candle, cumulativeVolume === 0 ? typicalPrice : cumulativePv / cumulativeVolume));
  }
  return points;
}

export type StructureLabel = "HH" | "HL" | "LH" | "LL";

export interface StructurePoint {
  time: UTCTimestamp;
  price: number;
  label: StructureLabel;
}

/**
 * Swing-structure classification (HH/HL/LH/LL) over the whole loaded candle
 * series. Mirrors the backend vix75 strategy's `_swing_flags`/
 * `_classify_structure` (see `backend/src/strategies/generated/"pob_price_
 * action_snd for vix75_v1.py"`), but computed once over every candle here
 * instead of a single trade's ~100-bar pre-entry window — a real swing that
 * doesn't happen to fall inside some trade's specific lookback was
 * previously just never labeled, even when it's the most obvious peak on
 * screen. `marginAtrMult` requires a swing to clear the prior one of the
 * same type by more than `marginAtrMult * ATR` before it reads as "higher" —
 * without it, two swings a fraction of a point apart (a retest, not a real
 * break) flip unpredictably between HH/LH or HL/LL.
 */
export function swingStructure(
  candles: Candle[],
  lookback: number,
  atrPeriod: number,
  marginAtrMult: number,
): StructurePoint[] {
  const n = candles.length;
  if (n < 2 * lookback + 1) return [];

  const atrPoints = atr(candles, atrPeriod);
  const atrByTime = new Map(atrPoints.map((p) => [p.time as number, p.value]));
  let currentAtr = atrPoints.length > 0 ? atrPoints[0].value : 0;

  const isHigh = new Array<boolean>(n).fill(false);
  const isLow = new Array<boolean>(n).fill(false);
  for (let i = lookback; i < n - lookback; i++) {
    let maxHigh = -Infinity;
    let minLow = Infinity;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (candles[j].high > maxHigh) maxHigh = candles[j].high;
      if (candles[j].low < minLow) minLow = candles[j].low;
    }
    if (candles[i].high === maxHigh) isHigh[i] = true;
    if (candles[i].low === minLow) isLow[i] = true;
  }

  const points: StructurePoint[] = [];
  let lastHigh: number | null = null;
  let lastLow: number | null = null;
  for (let i = 0; i < n; i++) {
    const time = candles[i].time as UTCTimestamp;
    const atrHere = atrByTime.get(time as number);
    if (atrHere !== undefined) currentAtr = atrHere;
    const margin = currentAtr * marginAtrMult;

    if (isHigh[i]) {
      if (lastHigh !== null) {
        const label: StructureLabel = candles[i].high > lastHigh + margin ? "HH" : "LH";
        points.push({ time, price: candles[i].high, label });
      }
      lastHigh = candles[i].high;
    } else if (isLow[i]) {
      if (lastLow !== null) {
        const label: StructureLabel = candles[i].low > lastLow + margin ? "HL" : "LL";
        points.push({ time, price: candles[i].low, label });
      }
      lastLow = candles[i].low;
    }
  }
  return points;
}

export type QuasimodoKind = "QML" | "QML_INV";

export interface QuasimodoZone {
  /** Candle that confirmed the pattern — the close that broke the neckline. */
  time: UTCTimestamp;
  /** QML level: the left shoulder's extreme, where the retest entry sits. */
  price: number;
  kind: QuasimodoKind;
  /** The swing between shoulder and head whose break confirms the pattern. */
  necklinePrice: number;
  /** Head extreme — the "maximum pain level"; a move past it voids the level. */
  headPrice: number;
  /** Head swing's candle — where the QM zone rectangle starts on the chart. */
  headTime: UTCTimestamp;
  /** First candle that tags the QML level again after confirmation, if any —
   * the retest entry (sell for QML, buy for QML_INV). Undefined when price
   * hasn't come back yet or ran past the head first. */
  retestTime?: UTCTimestamp;
}

/**
 * Quasimodo (QML) levels from `swingStructure()`'s output plus raw candles —
 * a chart annotation only, not wired into any strategy's trading decision.
 *
 * QML (bearish, sell): an uptrend prints a high (left shoulder), a low
 * (neckline), then a higher high (head). The pattern confirms when a candle
 * CLOSES back below the neckline — the break of structure that turns the
 * up-sequence into a lower low. The QML level is the left shoulder's high:
 * price typically rallies back into it before continuing down, and that
 * retest is the sell signal. QML_INV (bullish, buy) is the exact mirror —
 * low shoulder, high neckline, lower-low head, confirmation on a close above
 * the neckline, buy on the drop back into the shoulder's low.
 *
 * The head is the "maximum pain level" (MPL): price trading beyond it —
 * before the neckline breaks, or after the break but before the retest —
 * voids the level (pattern dropped / no retest marker). Both quick and late
 * retests count; there is no bar-count expiry. Confirmation and retest are
 * checked against actual candles, not just labeled swings, so a break that
 * never printed a new swing pivot still confirms.
 */
export function quasimodoLevels(points: StructurePoint[], candles: Candle[]): QuasimodoZone[] {
  const zones: QuasimodoZone[] = [];
  const indexByTime = new Map<number, number>();
  for (let i = 0; i < candles.length; i++) indexByTime.set(candles[i].time as number, i);

  for (let i = 0; i + 2 < points.length; i++) {
    const shoulder = points[i];
    const neckline = points[i + 1];
    const head = points[i + 2];
    const headIdx = indexByTime.get(head.time as number);
    if (headIdx === undefined) continue;

    // The label margin (marginAtrMult) can call a high "HH" that is only a
    // hair above the shoulder, so re-check the head strictly beats it.
    const bearish =
      shoulder.label === "HH" &&
      neckline.label === "HL" &&
      head.label === "HH" &&
      head.price > shoulder.price;
    const bullish =
      shoulder.label === "LL" &&
      neckline.label === "LH" &&
      head.label === "LL" &&
      head.price < shoulder.price;
    if (!bearish && !bullish) continue;

    // Confirmation: after the head, price must close through the neckline
    // (break of structure) before extending past the head again.
    let confIdx = -1;
    for (let j = headIdx + 1; j < candles.length; j++) {
      const c = candles[j];
      if (bearish ? c.high > head.price : c.low < head.price) break;
      if (bearish ? c.close < neckline.price : c.close > neckline.price) {
        confIdx = j;
        break;
      }
    }
    if (confIdx === -1) continue;

    // Retest: first candle back at the QML level after the break. A move
    // beyond the head (maximum pain level) first voids the level instead.
    let retestTime: UTCTimestamp | undefined;
    for (let j = confIdx + 1; j < candles.length; j++) {
      const c = candles[j];
      if (bearish ? c.high >= shoulder.price : c.low <= shoulder.price) {
        retestTime = c.time as UTCTimestamp;
        break;
      }
      if (bearish ? c.high > head.price : c.low < head.price) break;
    }

    zones.push({
      time: candles[confIdx].time as UTCTimestamp,
      price: shoulder.price,
      kind: bearish ? "QML" : "QML_INV",
      necklinePrice: neckline.price,
      headPrice: head.price,
      headTime: head.time,
      retestTime,
    });
  }
  return zones;
}

export type SndPattern = "RBR" | "DBD" | "RBD" | "DBR";
export type SndKind = "demand" | "supply";

export interface SndZone {
  /** First leg-out candle whose close clears the base band — the pattern is
   * complete and the zone exists from here. */
  time: UTCTimestamp;
  pattern: SndPattern;
  /** demand (buy) for RBR/DBR, supply (sell) for DBD/RBD. */
  kind: SndKind;
  /** Zone band = the base candles' extremes. */
  priceHigh: number;
  priceLow: number;
  /** First base candle — where the zone rectangle starts on the chart. */
  baseStartTime: UTCTimestamp;
  /** First candle to trade back into the band after the leg-out — the TOUCH
   * that consumes the zone: it is the entry (buy demand / sell supply) AND
   * where the rectangle ends, because a touched zone is no longer valid.
   * Undefined while the zone is still fresh (untouched, extends to now). */
  touchedTime?: UTCTimestamp;
}

export interface SndParams {
  /** A leg run's NET travel (open of its first candle to close of its last)
   * must be ≥ this × ATR to count as a rally/drop. */
  legTravelAtrMult: number;
  /** Base candle body must be ≤ this × ATR to count as consolidation. */
  baseBodyAtrMult: number;
}

/** Same PoB doctrine as the backend vix75 strategy's zone params, tuned so
 * an M5/M15 chart shows the obvious bases without flagging every doji. */
export const DEFAULT_SND_PARAMS: SndParams = {
  legTravelAtrMult: 1.0,
  baseBodyAtrMult: 0.5,
};

/**
 * PoB "basing candle" height rule — the shared way both `sndZones` (v1) and
 * `sndZonesV2` (v2) size a zone's band. The tradable zone is NOT the full base
 * range (which draws far too tall on a multi-candle base) but the LAST candle
 * of the OPPOSITE colour to the departure:
 *   demand (up departure: RBR/DBR)   -> last RED (bearish) base candle;
 *   supply (down departure: DBD/RBD) -> last GREEN (bullish) base candle.
 * That candle's high/low is the band. Falls back to the full base high/low
 * when the base has no opposite-colour candle (all one colour / all doji).
 *
 * `demand` is the departure direction (true = up/rally, false = down/drop);
 * `baseStart`/`baseEnd` are inclusive candle indices of the base.
 */
export function basingCandleBand(
  candles: Candle[],
  baseStart: number,
  baseEnd: number,
  demand: boolean,
): { high: number; low: number } {
  for (let j = baseEnd; j >= baseStart; j--) {
    const bullish = candles[j].close > candles[j].open;
    const bearish = candles[j].close < candles[j].open;
    if (demand ? bearish : bullish) {
      return { high: candles[j].high, low: candles[j].low };
    }
  }
  // Fallback: no opposite-colour candle in the base — use its full extent.
  let high = -Infinity;
  let low = Infinity;
  for (let j = baseStart; j <= baseEnd; j++) {
    if (candles[j].high > high) high = candles[j].high;
    if (candles[j].low < low) low = candles[j].low;
  }
  return { high, low };
}

/**
 * First candle at/after `from` whose range enters the `[low, high]` band —
 * the TOUCH that consumes a fresh S&D zone. A zone is valid only until price
 * first trades back into it; on that touch it stops being drawn and is no
 * longer tradable (a demand/supply level is "used up" the first time it is
 * hit, well before price would fully break through it). Shared by both
 * `sndZones` (v1) and `sndZonesV2` (v2); returns undefined while the zone is
 * still untouched.
 */
export function firstBandTouch(
  candles: Candle[],
  from: number,
  low: number,
  high: number,
): UTCTimestamp | undefined {
  for (let j = from; j < candles.length; j++) {
    if (candles[j].high >= low && candles[j].low <= high) {
      return candles[j].time as UTCTimestamp;
    }
  }
  return undefined;
}

/**
 * PoB supply & demand zones — the "only 4 types of Entry Point" from the
 * Property of Bystra notes: RBR (Rally Base Rally) / DBR (Drop Base Rally)
 * demand zones to buy, DBD (Drop Base Drop) / RBD (Rally Base Drop) supply
 * zones to sell. A chart annotation like `quasimodoLevels`, not wired into
 * any strategy's trading decision.
 *
 * Detection finds the LEGS first, then reads the base as whatever sits
 * between them. Every candle is either base-class (body ≤ baseBodyAtrMult ×
 * ATR, any color) or a directional momentum bar; consecutive same-class
 * candles merge into runs, and a directional run is a *leg* when its net
 * travel reaches legTravelAtrMult × ATR — one 1.5-ATR candle and three
 * 0.6-ATR candles in a row are both rallies (the PDF draws the RBD/DBR arms
 * as multi-candle swings, not single bars). Two refinements, both from real
 * missed-zone reports:
 *   - weak same-direction runs split by a short pause merge into one run
 *     (a rally printing 0.7-ATR candles around a doji is one leg, not two
 *     non-legs) — but runs that BOTH already qualify stay separate, because
 *     the pause between them is a stacked-zone base, not leg interior;
 *   - the base between two legs is EVERY candle between them, including
 *     medium-bodied pullback bars that are neither base-class nor
 *     leg-strong (a lone 0.7-ATR red candle inside a rally used to break
 *     the pattern into up / junk / up and silently drop the zone).
 * A zone is then each adjacent pair of legs with 1..maxBaseCandles candles
 * between them, confirmed by the first leg-out candle whose close clears
 * the base extremes; the zone band is those between-candles' high/low. Leg
 * directions name the pattern: up-base-up = RBR, down-base-up = DBR,
 * down-base-down = DBD, up-base-down = RBD. Adjacent pairs share legs, so
 * stacked zones (rally → base → rally → base → rally) all detect.
 *
 * Validity: after the leg-out run, the first candle trading back into the
 * (refined) band is the TOUCH — the entry and the end of the zone, which is
 * consumed there (`firstBandTouch`, shared with v2); it does not wait for a
 * full break. `touchedTime` undefined means still fresh. Unlike QML there is
 * no separate confirmation step — the leg-out is itself the confirmation.
 */
export function sndZones(
  candles: Candle[],
  maxBaseCandles: number,
  atrPeriod: number,
  params: SndParams = DEFAULT_SND_PARAMS,
): SndZone[] {
  const n = candles.length;
  const atrPoints = atr(candles, atrPeriod);
  if (atrPoints.length === 0) return [];
  // atrPoints[k] is the ATR at candles[atrPeriod + k]; pad the warmup bars
  // with the first available value so early candles still classify.
  const atrAt = new Array<number>(n).fill(atrPoints[0].value);
  for (let k = 0; k < atrPoints.length; k++) atrAt[atrPeriod + k] = atrPoints[k].value;

  // 0 = base (small body, either color); +1/-1 = directional momentum bar.
  const classify = (i: number): -1 | 0 | 1 => {
    if (Math.abs(candles[i].close - candles[i].open) <= params.baseBodyAtrMult * atrAt[i]) return 0;
    return candles[i].close >= candles[i].open ? 1 : -1;
  };

  interface Run {
    cls: -1 | 0 | 1;
    start: number;
    end: number;
  }
  const runs: Run[] = [];
  for (let i = 0; i < n; i++) {
    const cls = classify(i);
    const last = runs[runs.length - 1];
    if (last && last.cls === cls) last.end = i;
    else runs.push({ cls, start: i, end: i });
  }

  const isLeg = (r: Run): boolean =>
    r.cls !== 0 &&
    Math.abs(candles[r.end].close - candles[r.start].open) >= params.legTravelAtrMult * atrAt[r.end];

  // Weak same-direction runs split by a short base run merge into one run:
  // a rally printing 0.7-ATR candles around a doji is one leg, not two
  // non-legs (which made the whole move — and its zones — invisible). Runs
  // that BOTH already qualify as legs stay separate: the pause between them
  // is a stacked-zone base (rally → base → rally), not leg interior.
  let mergedSomething = true;
  while (mergedSomething) {
    mergedSomething = false;
    for (let k = 0; k + 2 < runs.length; k++) {
      const d1 = runs[k];
      const pause = runs[k + 1];
      const d2 = runs[k + 2];
      if (d1.cls === 0 || pause.cls !== 0 || d2.cls !== d1.cls) continue;
      if (pause.end - pause.start + 1 > maxBaseCandles) continue;
      if (isLeg(d1) && isLeg(d2)) continue;
      runs.splice(k, 3, { cls: d1.cls, start: d1.start, end: d2.end });
      mergedSomething = true;
      break;
    }
  }

  // The base between two adjacent legs is EVERY candle between them —
  // base-class candles, but also medium-bodied pullback bars that are
  // neither base-class nor leg-strong (a lone 0.7-ATR counter candle inside
  // the pause used to break the pattern apart and drop the zone).
  const legs = runs.filter(isLeg);

  const zones: SndZone[] = [];
  for (let k = 0; k + 1 < legs.length; k++) {
    const legIn = legs[k];
    const legOut = legs[k + 1];
    const baseStart = legIn.end + 1;
    const baseEnd = legOut.start - 1;
    const baseCount = baseEnd - baseStart + 1;
    if (baseCount < 1 || baseCount > maxBaseCandles) continue;

    const legOutUp = legOut.cls === 1;
    // Zone height uses the shared PoB "basing candle" rule (same as
    // sndZonesV2): the last opposite-colour base candle, not the full base
    // range — so a multi-candle base still draws a tight, one-candle band.
    const { high: priceHigh, low: priceLow } = basingCandleBand(
      candles,
      baseStart,
      baseEnd,
      legOutUp,
    );

    // Confirmation: the first leg-out candle whose close actually departs
    // the base band — a momentum run that never clears the base is still
    // consolidation, not a zone.
    let confIdx = -1;
    for (let j = legOut.start; j <= legOut.end; j++) {
      if (legOutUp ? candles[j].close > priceHigh : candles[j].close < priceLow) {
        confIdx = j;
        break;
      }
    }
    if (confIdx === -1) continue;

    const pattern: SndPattern =
      legIn.cls === 1 ? (legOutUp ? "RBR" : "RBD") : legOutUp ? "DBR" : "DBD";
    const kind: SndKind = legOutUp ? "demand" : "supply";

    // Validity ends at the first TOUCH — price trading back into the band.
    // The scan starts after the whole leg-out run, whose own early candles'
    // wicks still overlap the base and are not a genuine return to it.
    const touchedTime = firstBandTouch(candles, legOut.end + 1, priceLow, priceHigh);

    zones.push({
      time: candles[confIdx].time as UTCTimestamp,
      pattern,
      kind,
      priceHigh,
      priceLow,
      baseStartTime: candles[baseStart].time as UTCTimestamp,
      touchedTime,
    });
  }
  return zones;
}

// A zone is "fresh" (valid, still drawn to the right edge) until price first
// touches it; that touch flips it to "touched" — consumed and no longer valid.
export type SndZoneStateV2 = "fresh" | "touched";

export interface SndZoneV2 {
  time: UTCTimestamp;            // confirmation candle
  kind: SndKind;                // "demand" | "supply" (SndKind already exported)
  priceHigh: number;            // drawn band top (after refinement)
  priceLow: number;             // drawn band bottom (after refinement)
  proximal: number;             // edge price returns to first
  distal: number;               // far break edge
  baseStartTime: UTCTimestamp;  // rectangle left edge
  baseCandles: number;
  hasLegIn: boolean;
  pattern: SndPattern | "DZ" | "SZ";   // SndPattern already exported ("RBR"|"DBD"|"RBD"|"DBR")
  /** First touch — the entry AND where the rectangle ends (the zone is
   * consumed here). Undefined while still fresh. */
  touchedTime?: UTCTimestamp;
  state: SndZoneStateV2;
}

export interface SndParamsV2 {
  impulseAtrMult: number;   // departure leg net travel >= this * ATR
  baseBodyAtrMult: number;  // body <= this * ATR => "quiet" (base-class)
  baseRangeAtrMult: number; // whole base band <= this * ATR (lets a range count)
}

export const DEFAULT_SND_PARAMS_V2: SndParamsV2 = {
  impulseAtrMult: 1.2,
  baseBodyAtrMult: 0.8,
  baseRangeAtrMult: 3.0,
};

/**
 * PoB supply & demand zones, v2 — a DEPARTURE-FIRST rewrite of `sndZones`.
 *
 * Where v1 requires TWO qualifying impulse legs (leg-in AND leg-out) with the
 * base read as whatever sits *between* them — and caps that base at ~3 candles
 * — this misses two whole classes of real zones: origin bases (the very first
 * consolidation of a move, which has no leg-in before it) and wide ranges
 * (a base that consolidated across more than a few candles). v2 fixes both by
 * inverting the search order:
 *
 *   1. Find the DEPARTURE impulse first — a single directional run whose net
 *      travel (open of its first candle to close of its last) reaches
 *      `impulseAtrMult × ATR`. This is the leg-out; it is the only leg the
 *      zone strictly needs.
 *   2. Read the base BACKWARDS from the candle just before the departure,
 *      extending through quiet/consolidation candles until either a leg-in
 *      impulse is hit (base starts there) or the accumulated band would grow
 *      wider than `baseRangeAtrMult × ATR` (past that it is trending, not
 *      basing). The leg-in is therefore OPTIONAL — when one exists the pattern
 *      names it (RBR/DBR/RBD/DBD), when none does the zone is an origin base
 *      (DZ for demand, SZ for supply).
 *
 * Weak same-direction runs split by a short pause still merge into one impulse
 * (same refinement as v1) so a rally printing sub-leg candles around a doji
 * reads as a single departure.
 *
 * HEIGHT REFINEMENT (the "basing candle" rule from PoB, and the key fix for
 * the "zone height too large" complaint): the drawn band is NOT the full base
 * range. It is the single LAST candle of the OPPOSITE colour to the departure
 * — the last red candle before an up departure (demand), the last green candle
 * before a down departure (supply) — which is the actual origin candle price
 * reacts from. Only when the base has no opposite-colour candle (all one
 * colour / all doji) does it fall back to the full base band. This keeps the
 * zone roughly one candle's range tall instead of the whole consolidation.
 *
 * VALIDITY: a zone is only good until it is first TOUCHED. After the departure
 * run, the first candle trading back into the (refined) band consumes the zone
 * — that touch is both the entry and the end of the drawing (see
 * `firstBandTouch`, shared with v1). There is no "counting retests" or waiting
 * for a full break: state is just `fresh` (untouched, still drawn to now) or
 * `touched` (consumed, invalid, rectangle ends at the touch).
 */
export function sndZonesV2(
  candles: Candle[],
  maxBaseCandles: number,
  atrPeriod: number,
  params: SndParamsV2 = DEFAULT_SND_PARAMS_V2,
): SndZoneV2[] {
  const n = candles.length;
  const atrPoints = atr(candles, atrPeriod);
  if (atrPoints.length === 0) return [];
  const atrAt = new Array<number>(n).fill(atrPoints[0].value);
  for (let k = 0; k < atrPoints.length; k++) atrAt[atrPeriod + k] = atrPoints[k].value;

  const classify = (i: number): -1 | 0 | 1 => {
    if (Math.abs(candles[i].close - candles[i].open) <= params.baseBodyAtrMult * atrAt[i]) return 0;
    return candles[i].close >= candles[i].open ? 1 : -1;
  };

  interface Run { cls: -1 | 0 | 1; start: number; end: number; }
  const runs: Run[] = [];
  for (let i = 0; i < n; i++) {
    const cls = classify(i);
    const last = runs[runs.length - 1];
    if (last && last.cls === cls) last.end = i;
    else runs.push({ cls, start: i, end: i });
  }

  const isLeg = (r: Run): boolean =>
    r.cls !== 0 &&
    Math.abs(candles[r.end].close - candles[r.start].open) >= params.impulseAtrMult * atrAt[r.end];

  // merge weak same-direction runs split by a short pause into one impulse
  let mergedSomething = true;
  while (mergedSomething) {
    mergedSomething = false;
    for (let k = 0; k + 2 < runs.length; k++) {
      const d1 = runs[k], pause = runs[k + 1], d2 = runs[k + 2];
      if (d1.cls === 0 || pause.cls !== 0 || d2.cls !== d1.cls) continue;
      if (pause.end - pause.start + 1 > maxBaseCandles) continue;
      if (isLeg(d1) && isLeg(d2)) continue;
      runs.splice(k, 3, { cls: d1.cls, start: d1.start, end: d2.end });
      mergedSomething = true;
      break;
    }
  }

  const runOf = new Array<number>(n).fill(-1);
  runs.forEach((r, ri) => { for (let i = r.start; i <= r.end; i++) runOf[i] = ri; });

  const zones: SndZoneV2[] = [];
  const seen = new Set<string>();

  for (let ri = 0; ri < runs.length; ri++) {
    const legOut = runs[ri];
    if (!isLeg(legOut)) continue;
    const departUp = legOut.cls === 1;

    const anchor = legOut.start - 1;
    if (anchor < 0 || isLeg(runs[runOf[anchor]])) continue; // impulse straight into impulse = no base

    const atrHere = atrAt[legOut.start];
    let hi = candles[anchor].high, lo = candles[anchor].low, baseStart = anchor;
    for (let j = anchor - 1; j >= 0 && anchor - j < maxBaseCandles; j--) {
      if (isLeg(runs[runOf[j]])) break;                    // reached the leg-in
      const nhi = Math.max(hi, candles[j].high), nlo = Math.min(lo, candles[j].low);
      if (nhi - nlo > params.baseRangeAtrMult * atrHere) break;  // wider than a base = trending
      hi = nhi; lo = nlo; baseStart = j;
    }
    const baseEnd = anchor;
    const baseCount = baseEnd - baseStart + 1;

    // Zone height uses the shared PoB "basing candle" rule — the last
    // opposite-colour base candle, not the full base range (which draws far
    // too tall). See basingCandleBand; sndZones (v1) uses the same helper.
    const { high: bandHi, low: bandLo } = basingCandleBand(
      candles,
      baseStart,
      baseEnd,
      departUp,
    );

    let confIdx = -1;
    for (let j = legOut.start; j <= legOut.end; j++) {
      if (departUp ? candles[j].close > bandHi : candles[j].close < bandLo) { confIdx = j; break; }
    }
    if (confIdx === -1) continue;

    const key = `${baseStart}:${baseEnd}:${departUp ? "d" : "s"}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const legInRun = baseStart > 0 && isLeg(runs[runOf[baseStart - 1]]) ? runs[runOf[baseStart - 1]] : undefined;
    const hasLegIn = legInRun !== undefined;
    const kind: SndKind = departUp ? "demand" : "supply";
    const pattern: SndPattern | "DZ" | "SZ" = legInRun
      ? legInRun.cls === 1 ? (departUp ? "RBR" : "RBD") : (departUp ? "DBR" : "DBD")
      : (departUp ? "DZ" : "SZ");

    const proximal = departUp ? bandHi : bandLo;
    const distal = departUp ? bandLo : bandHi;

    // Validity ends at the first TOUCH back into the band — not at a full
    // break. Once touched the zone is consumed and stops being drawn.
    const touchedTime = firstBandTouch(candles, legOut.end + 1, bandLo, bandHi);
    const state: SndZoneStateV2 = touchedTime ? "touched" : "fresh";

    zones.push({
      time: candles[confIdx].time as UTCTimestamp,
      kind, priceHigh: bandHi, priceLow: bandLo, proximal, distal,
      baseStartTime: candles[baseStart].time as UTCTimestamp,
      baseCandles: baseCount, hasLegIn, pattern, touchedTime, state,
    });
  }
  return zones;
}

export interface BaseRange {
  startTime: UTCTimestamp;   // first candle of the base
  endTime: UTCTimestamp;     // last candle of the base
  high: number;              // top boundary (range high)
  low: number;               // bottom boundary (range low)
  candles: number;
}

/**
 * Consolidation-base detector, independent of any departure impulse (unlike
 * `sndZones`/`sndZonesV2`, which only surface a base once price has left it).
 * A base is a maximal run of consecutive candles whose whole span
 * (max high − min low) stays within `rangeAtrMult × ATR` AND lasts at least
 * `minCandles` candles — i.e. price stayed range-bound rather than trending.
 *
 * The scan greedily extends the current window one candle at a time while the
 * range bound still holds; when a candle would blow the band wider than the
 * ATR bound, the window closes. It is emitted as a base only if it reached
 * `minCandles`, and the scan then resumes from the breaking candle either way,
 * so the returned bases are non-overlapping and ordered oldest-first. Reuses
 * the same `atr()` + warmup-padded `atrAt[]` pattern as `sndZonesV2`.
 */
export function detectBases(
  candles: Candle[],
  minCandles: number,
  atrPeriod: number,
  rangeAtrMult = 2.5,
): BaseRange[] {
  const n = candles.length;
  const atrPoints = atr(candles, atrPeriod);
  if (atrPoints.length === 0) return [];
  const atrAt = new Array<number>(n).fill(atrPoints[0].value);
  for (let k = 0; k < atrPoints.length; k++) atrAt[atrPeriod + k] = atrPoints[k].value;

  const bases: BaseRange[] = [];
  let i = 0;
  while (i < n) {
    let hi = candles[i].high;
    let lo = candles[i].low;
    let j = i;
    // Greedily extend while the window's whole range stays within the bound.
    while (j + 1 < n) {
      const nhi = Math.max(hi, candles[j + 1].high);
      const nlo = Math.min(lo, candles[j + 1].low);
      if (nhi - nlo > rangeAtrMult * atrAt[j + 1]) break;
      hi = nhi;
      lo = nlo;
      j += 1;
    }
    const count = j - i + 1;
    if (count >= minCandles) {
      bases.push({
        startTime: candles[i].time as UTCTimestamp,
        endTime: candles[j].time as UTCTimestamp,
        high: hi,
        low: lo,
        candles: count,
      });
    }
    // Resume from the breaking candle (j + 1) either way: non-overlapping.
    i = j + 1;
  }
  return bases;
}

export type PatternLabel =
  | "bullish_engulfing"
  | "bearish_engulfing"
  | "bullish_pin_bar"
  | "bearish_pin_bar";

export interface PatternPoint {
  time: UTCTimestamp;
  price: number;
  label: PatternLabel;
}

function isBullishEngulfing(candles: Candle[], i: number): boolean {
  if (i < 1) return false;
  const prevO = candles[i - 1].open;
  const prevC = candles[i - 1].close;
  const o = candles[i].open;
  const c = candles[i].close;
  if (!(prevC < prevO && c > o)) return false;
  return o <= prevC && c >= prevO;
}

function isBearishEngulfing(candles: Candle[], i: number): boolean {
  if (i < 1) return false;
  const prevO = candles[i - 1].open;
  const prevC = candles[i - 1].close;
  const o = candles[i].open;
  const c = candles[i].close;
  if (!(prevC > prevO && c < o)) return false;
  return o >= prevC && c <= prevO;
}

function isPinBar(
  candles: Candle[],
  i: number,
  maxBodyRatio: number,
  minWickBodyMult: number,
): "up" | "down" | null {
  const candle = candles[i];
  const range = candle.high - candle.low;
  if (range <= 0) return null;
  const body = Math.abs(candle.close - candle.open);
  if (body / range > maxBodyRatio) return null;
  const bodyFloor = Math.max(body, range * 0.05);
  const lowerWick = Math.min(candle.open, candle.close) - candle.low;
  const upperWick = candle.high - Math.max(candle.open, candle.close);
  if (lowerWick >= minWickBodyMult * bodyFloor && lowerWick > upperWick) return "up";
  if (upperWick >= minWickBodyMult * bodyFloor && upperWick > lowerWick) return "down";
  return null;
}

export interface PatternParams {
  pinBarMaxBodyRatio: number;
  pinBarMinWickMult: number;
}

/** Stricter than the backend strategy's own `pin_bar_*` params (0.35/2.0).
 * Those are calibrated as ONE OF SEVERAL gates on a specific breakout candle
 * inside `evaluate()`; applied to every candle on the whole chart, the
 * looser thresholds flagged ~35% of bars — unreadable clutter. Tightened
 * here (empirically, against real M5 data) to ~25%, matching engulfing's
 * own natural, non-tunable rate (~12%). */
export const DEFAULT_PATTERN_PARAMS: PatternParams = {
  pinBarMaxBodyRatio: 0.15,
  pinBarMinWickMult: 3.0,
};

/**
 * Candlestick pattern detection (engulfing, pin bar) over the whole loaded
 * candle series — the two patterns distinctive enough to be meaningful as an
 * always-on chart overlay. Mirrors the backend vix75 strategy's
 * `_is_bullish_engulfing`/`_is_bearish_engulfing`/`_is_pin_bar` (see
 * `backend/src/strategies/generated/"pob_price_action_snd for vix75_v1.py"`),
 * but evaluated at every candle instead of only at a strategy's own trade
 * entries. Deliberately excludes the strategy's third fallback pattern
 * ("body/momentum candle," any candle with a big-enough body) — that alone
 * matched ~33% of bars in testing, common enough to be noise rather than a
 * chart-worthy pattern; it stays a valid, useful gate inside the strategy's
 * own multi-filter `evaluate()`, just not here.
 */
export function detectPatterns(
  candles: Candle[],
  params: PatternParams = DEFAULT_PATTERN_PARAMS,
): PatternPoint[] {
  const points: PatternPoint[] = [];
  for (let i = 0; i < candles.length; i++) {
    let label: PatternLabel | null = null;
    if (isBullishEngulfing(candles, i)) {
      label = "bullish_engulfing";
    } else if (isBearishEngulfing(candles, i)) {
      label = "bearish_engulfing";
    } else {
      const pinSide = isPinBar(candles, i, params.pinBarMaxBodyRatio, params.pinBarMinWickMult);
      if (pinSide) {
        label = pinSide === "up" ? "bullish_pin_bar" : "bearish_pin_bar";
      }
    }
    if (label) {
      points.push({ time: candles[i].time as UTCTimestamp, price: candles[i].close, label });
    }
  }
  return points;
}

/**
 * Historical (realized) Volatility %, annualized — the stdev of log returns
 * over a rolling `period`-candle window, expressed as an annualized
 * percentage: `100 * stdev(logReturns) * sqrt(barsPerYear)`. Distinct from
 * `atr` (an average absolute range in price units, not annualized, not based
 * on returns) — this is the standard "HV" oscillator, plotted in its own
 * bottom pane like RSI. `barsPerYear` approximates trading-minutes-per-year
 * for intraday FX/CFD data (24/5 market, ~252 trading days); callers on a
 * coarser timeframe should scale it, but a fixed approximation is fine for a
 * relative/comparative oscillator like this.
 */
export function historicalVolatility(
  candles: Candle[],
  period: number,
  barsPerYear = 252 * 24 * 60,
): LinePoint[] {
  if (candles.length < period + 1) return [];
  const logReturns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const prevClose = candles[i - 1].close;
    const close = candles[i].close;
    logReturns.push(prevClose > 0 && close > 0 ? Math.log(close / prevClose) : 0);
  }

  const points: LinePoint[] = [];
  const annualize = Math.sqrt(barsPerYear);
  for (let i = period - 1; i < logReturns.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += logReturns[j];
    const mean = sum / period;
    let variance = 0;
    for (let j = i - period + 1; j <= i; j++) variance += (logReturns[j] - mean) ** 2;
    variance /= period;
    const stdev = Math.sqrt(variance);
    // logReturns[i] corresponds to candles[i + 1]
    points.push(toPoint(candles[i + 1], 100 * stdev * annualize));
  }
  return points;
}

/** Average True Range with Wilder smoothing (same scheme as `rsi` above). */
export function atr(candles: Candle[], period: number): LinePoint[] {
  if (candles.length < period + 1) return [];
  const trueRanges: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const prevClose = candles[i - 1].close;
    trueRanges.push(
      Math.max(
        candles[i].high - candles[i].low,
        Math.abs(candles[i].high - prevClose),
        Math.abs(candles[i].low - prevClose),
      ),
    );
  }

  const points: LinePoint[] = [];
  let value = trueRanges.slice(0, period).reduce((sum, tr) => sum + tr, 0) / period;
  points.push(toPoint(candles[period], value));
  for (let i = period; i < trueRanges.length; i++) {
    value = (value * (period - 1) + trueRanges[i]) / period;
    points.push(toPoint(candles[i + 1], value));
  }
  return points;
}

export interface SmcOb {
  time: UTCTimestamp;
  formedTime: UTCTimestamp;
  kind: 'bullish' | 'bearish';
  high: number;
  low: number;
  mitigatedTime?: UTCTimestamp;
}

export interface SmcFvg {
  time: UTCTimestamp;
  formedTime: UTCTimestamp;
  kind: 'bullish' | 'bearish';
  high: number;
  low: number;
  mitigatedTime?: UTCTimestamp;
}

export interface SmcStructure {
  time: UTCTimestamp;
  price: number;
  kind: 'BOS' | 'CHoCH';
  direction: 'bullish' | 'bearish';
}

export interface SmcResult {
  obs: SmcOb[];
  fvgs: SmcFvg[];
  structures: SmcStructure[];
}

export function smcConcepts(candles: Candle[], lookback: number, atrPeriod: number): SmcResult {
  const obs: SmcOb[] = [];
  const fvgs: SmcFvg[] = [];
  const structures: SmcStructure[] = [];
  const atrPoints = atr(candles, atrPeriod);
  if (atrPoints.length === 0) return { obs, fvgs, structures };

  const points = swingStructure(candles, lookback, atrPeriod, 0.1);
  
  // Detect BOS and CHoCH from points
  let trend: 'bullish' | 'bearish' | null = null;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];

    if (curr.label === 'HH' && (prev.label === 'HL' || prev.label === 'LH')) {
      const type = trend === 'bullish' ? 'BOS' : 'CHoCH';
      trend = 'bullish';
      structures.push({ time: curr.time, price: curr.price, kind: type, direction: 'bullish' });
    } else if (curr.label === 'LL' && (prev.label === 'LH' || prev.label === 'HL')) {
      const type = trend === 'bearish' ? 'BOS' : 'CHoCH';
      trend = 'bearish';
      structures.push({ time: curr.time, price: curr.price, kind: type, direction: 'bearish' });
    }
  }

  // Detect FVGs (Inefficiencies)
  // To avoid noise, only take FVGs that are visually significant (gap > 0.5 * atr)
  const atrByTime = new Map(atrPoints.map(p => [p.time as number, p.value]));
  for (let i = 2; i < candles.length; i++) {
    const c1 = candles[i - 2];
    const c3 = candles[i];
    const atrVal = atrByTime.get(candles[i].time as number) || 0;
    
    // Bullish FVG
    if (c1.high < c3.low && (c3.low - c1.high) > atrVal * 0.2) {
      fvgs.push({ time: candles[i - 1].time as UTCTimestamp, formedTime: c3.time as UTCTimestamp, kind: 'bullish', high: c3.low, low: c1.high });
      
      // Trace the Base (OB) just below the bullish FVG
      let obIdx = i - 2;
      // Find the last bearish (or doji) candle that forms the base before the impulse
      while (obIdx > 0 && candles[obIdx].close > candles[obIdx].open) {
        obIdx--;
      }
      if (obIdx >= 0) {
        obs.push({ time: candles[obIdx].time as UTCTimestamp, formedTime: c3.time as UTCTimestamp, kind: 'bullish', high: candles[obIdx].high, low: candles[obIdx].low });
      }
    }
    // Bearish FVG
    else if (c1.low > c3.high && (c1.low - c3.high) > atrVal * 0.2) {
      fvgs.push({ time: candles[i - 1].time as UTCTimestamp, formedTime: c3.time as UTCTimestamp, kind: 'bearish', high: c1.low, low: c3.high });
      
      // Trace the Base (OB) just above the bearish FVG
      let obIdx = i - 2;
      // Find the last bullish (or doji) candle that forms the base before the impulse
      while (obIdx > 0 && candles[obIdx].close < candles[obIdx].open) {
        obIdx--;
      }
      if (obIdx >= 0) {
        obs.push({ time: candles[obIdx].time as UTCTimestamp, formedTime: c3.time as UTCTimestamp, kind: 'bearish', high: candles[obIdx].high, low: candles[obIdx].low });
      }
    }
  }

  // Mitigations
  const mitigate = (list: any[], isFvg: boolean = false) => {
    for (let i = 0; i < list.length; i++) {
      const item = list[i];
      const startIdx = candles.findIndex(c => c.time === item.formedTime);
      if (startIdx === -1) continue;
      // FVGs mitigate when fully filled. OBs mitigate when touched.
      // We start checking strictly AFTER the candle that formed the FVG/OB.
      for (let j = startIdx + 1; j < candles.length; j++) {
        if (item.kind === 'bullish') {
          if (isFvg ? candles[j].low <= item.low : candles[j].low <= item.high) {
            item.mitigatedTime = candles[j].time as UTCTimestamp;
            break;
          }
        } else {
          if (isFvg ? candles[j].high >= item.high : candles[j].high >= item.low) {
            item.mitigatedTime = candles[j].time as UTCTimestamp;
            break;
          }
        }
      }
    }
  }

  mitigate(obs, false);
  mitigate(fvgs, true);

  return { obs, fvgs, structures };
}
