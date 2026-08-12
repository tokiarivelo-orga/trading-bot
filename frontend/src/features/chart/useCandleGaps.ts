'use client';

/**
 * Missing-candle detection and repair for the chart's currently loaded
 * window — the state behind ChartToolbar's "Fill gaps" button.
 *
 * A hole in stored history (stream outage, backend restart longer than the
 * candle stream's poll lookback, an interrupted backfill) is invisible to
 * `GET /market-data/candles`: it just returns fewer bars, and
 * lightweight-charts draws them side by side. The chart looks continuous
 * while every indicator, zone detector and strategy signal computed over that
 * window is quietly wrong, because bars hours apart are treated as adjacent.
 *
 * Detection deliberately lives on the backend (`GET /market-data/candle-gaps`)
 * rather than being recomputed from `candlesRef` here: what counts as a gap —
 * in particular which holes are just a weekend closure — is market-domain
 * logic that `backend/src/market_data/domain/gaps.py` owns, and the same scan
 * has to serve backtests, which never touch this UI.
 *
 * Two behaviors worth knowing about:
 *   - Weekend closures are reported by the backend but filtered out of
 *     `gaps` here, so the badge only ever counts holes worth acting on.
 *   - Holes the broker itself can't fill (holiday, halt, symbol listed later)
 *     come back in the repair's `remaining` list. They're remembered in
 *     `unfillableRef` for as long as this symbol/timeframe stays open, so a
 *     permanent hole reports itself once instead of nagging on every scan.
 */

import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import {
  getCandleGaps,
  repairCandleGaps,
  type Candle,
  type CandleGap,
} from '@/shared/api/client';
import type { GapRepairSummary } from './types';

/** How long a repair's outcome stays on the toolbar before fading out. */
const RESULT_TTL_MS = 8000;

const gapKey = (gap: CandleGap) => `${gap.start}-${gap.end}`;

export interface UseCandleGapsParams {
  /** Resolved active account id — null while GET /accounts is in flight, in
   * which case no scan is attempted. */
  accountId: string | null;
  symbol: string;
  timeframe: Candle['timeframe'];
  /** The loaded window, read at scan/repair time — see useCandleData.ts's
   * module doc for why this ref is owned by ChartPanel. */
  candlesRef: RefObject<Candle[]>;
  /** `useCandleData`'s window refetch, run after a repair so recovered bars
   * actually appear on the chart. */
  reloadWindow: () => Promise<void>;
  /** True while a symbol/timeframe/report switch is still loading its
   * history — scanning waits, since the loaded window is about to change. */
  switchingChart: boolean;
  /** True while "load more" is paging in older history, for the same
   * reason: the window's left edge is still moving. */
  loadingMore: boolean;
}

export function useCandleGaps(params: UseCandleGapsParams) {
  const { accountId, symbol, timeframe, candlesRef, reloadWindow, switchingChart, loadingMore } =
    params;

  const [gaps, setGaps] = useState<CandleGap[]>([]);
  const [scanning, setScanning] = useState(false);
  const [repairing, setRepairing] = useState(false);
  const [result, setResult] = useState<GapRepairSummary | null>(null);

  // Holes the broker has already told us it cannot fill, for this
  // symbol/timeframe. Cleared on every switch below, since a different
  // symbol/timeframe has entirely different history.
  const unfillableRef = useRef<Set<string>>(new Set());
  // Range of the last completed scan, so the repeated effect runs triggered
  // by `loadingMore` flipping don't refire an identical request.
  const lastScanRef = useRef<string>('');

  const scan = useCallback(async () => {
    if (!accountId) return;
    const bars = candlesRef.current;
    if (bars.length < 2) return;
    const from = bars[0].time;
    const to = bars[bars.length - 1].time;
    const key = `${symbol}:${timeframe}:${from}-${to}`;
    if (key === lastScanRef.current) return;
    setScanning(true);
    try {
      const scanned = await getCandleGaps(accountId, symbol, timeframe, from, to);
      lastScanRef.current = key;
      setGaps(
        scanned.gaps.filter((gap) => !gap.weekend && !unfillableRef.current.has(gapKey(gap))),
      );
    } catch {
      // The scan is advisory — a transient failure just leaves the badge as
      // it was; the next switch (or a repair) tries again.
    } finally {
      setScanning(false);
    }
  }, [accountId, symbol, timeframe, candlesRef]);

  // Reset per symbol/timeframe, then scan once this window's history has
  // finished loading (and again once any "load more" page has landed, since
  // that extends the range being judged).
  useEffect(() => {
    unfillableRef.current = new Set();
    lastScanRef.current = '';
    setGaps([]);
    setResult(null);
  }, [accountId, symbol, timeframe]);

  useEffect(() => {
    if (switchingChart || loadingMore) return;
    void scan();
  }, [scan, switchingChart, loadingMore]);

  useEffect(() => {
    if (!result) return;
    const timer = setTimeout(() => setResult(null), RESULT_TTL_MS);
    return () => clearTimeout(timer);
  }, [result]);

  /** Asks the backend to re-download whatever is missing across the loaded
   * window, then refetches and repaints it. Runs even when no gap is known:
   * the scan only sees the *local* database, so this doubles as a plain
   * "refetch these candles from the broker" for a window whose bars look
   * wrong for any other reason. */
  const repair = useCallback(async () => {
    if (!accountId || repairing) return;
    const bars = candlesRef.current;
    if (bars.length === 0) return;
    const iso = (epochSeconds: number) => new Date(epochSeconds * 1000).toISOString();
    setRepairing(true);
    try {
      const repaired = await repairCandleGaps(accountId, {
        symbol,
        timeframe,
        start: iso(bars[0].time),
        end: iso(bars[bars.length - 1].time),
      });
      for (const gap of repaired.remaining) unfillableRef.current.add(gapKey(gap));
      setResult({
        barsRecovered: repaired.bars_recovered,
        remaining: repaired.remaining.length,
        error: null,
      });
      await reloadWindow();
      // The reload changed the loaded window's contents, so the previous
      // scan's range no longer describes it — force the rescan through.
      lastScanRef.current = '';
      await scan();
    } catch (err) {
      setResult({
        barsRecovered: 0,
        remaining: 0,
        error:
          err instanceof Error && err.message
            ? err.message
            : 'could not reach the broker to refetch candles',
      });
    } finally {
      setRepairing(false);
    }
  }, [accountId, repairing, symbol, timeframe, candlesRef, reloadWindow, scan]);

  return {
    /** Actionable holes only — weekend closures and holes the broker has
     * already refused to fill are filtered out. */
    gaps,
    missingBars: gaps.reduce((total, gap) => total + gap.missing_bars, 0),
    scanning,
    repairing,
    /** Outcome of the last repair, auto-cleared after `RESULT_TTL_MS`. */
    result,
    repair,
  };
}

export type CandleGaps = ReturnType<typeof useCandleGaps>;
