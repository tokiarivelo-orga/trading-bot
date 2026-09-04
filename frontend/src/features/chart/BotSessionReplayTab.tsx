'use client';

/**
 * "Replay" tab of the bot dock (BOT_SESSION_REPLAY_PLAN phase 2).
 *
 * Shown inside `SignalsDock` whenever a bot's eye is on: lets the trader
 * replay that bot's own trading session bar-by-bar without hunting for the
 * period in the chart toolbar's generic session-replay picker.
 *
 * Owns nothing but its two From/To input strings — playback itself is the
 * existing `useReplayEngine` session replay, reached through the
 * `BotReplayControls` handlers ChartPanel.tsx builds. No fetching happens
 * here either: the bounds are derived from the signals/trades the dock is
 * already showing (polled by `useBacktestData`'s live-bot poll).
 *
 * Bounds auto-follow that poll only while the inputs still hold the previous
 * auto-derived value, so a manual edit is never stomped by the next tick.
 */

import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Pause, LogOut } from 'lucide-react';
import type { BacktestSignal, BacktestTrade } from '@/shared/api/client';
import type { BotReplayControls } from './types';

/** Breathing room either side of the bot's first/last activity, so replay
 * starts with context instead of exactly on the first signal's bar. */
const BOUNDS_PAD_SEC = 60 * 60;
/** Fallback window when the bot has no signals or trades yet. */
const EMPTY_FALLBACK_SEC = 24 * 60 * 60;

const SPEED_OPTIONS = [0.5, 1, 2, 4, 8, 16];

/** Epoch seconds → the `YYYY-MM-DDTHH:mm` string a `datetime-local` input
 * expects, in the browser's local zone (which is how `useReplayEngine`'s own
 * picker parses these strings back). */
function toDateTimeLocal(epochSeconds: number): string {
  const d = new Date(epochSeconds * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  );
}

function parseDateTimeLocal(value: string): number | null {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

function formatCursor(epochSeconds: number): string {
  return new Date(epochSeconds * 1000)
    .toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export interface BotSessionReplayTabProps {
  /** The eyed bot's signals — bounds source and "revealed so far" counter. */
  signals: BacktestSignal[];
  /** The eyed bot's trades — same two roles as `signals`. */
  trades: BacktestTrade[];
  /** Playback wiring over the shared replay engine. */
  replay: BotReplayControls;
}

function BotSessionReplayTabImpl({
  signals,
  trades,
  replay,
}: BotSessionReplayTabProps) {
  // The bot's own activity window, padded and clamped to "now" (a bot's last
  // trade can close in the future only through clock skew, but a period that
  // runs past now fetches nothing).
  const derived = useMemo(() => {
    const now = Math.floor(Date.now() / 1000);
    const starts: number[] = [];
    const ends: number[] = [];
    for (const s of signals) {
      starts.push(s.time);
      ends.push(s.time);
    }
    for (const t of trades) {
      starts.push(t.open_time);
      ends.push(t.close_time);
    }
    if (starts.length === 0) {
      return { from: now - EMPTY_FALLBACK_SEC, to: now, empty: true };
    }
    const from = Math.min(...starts) - BOUNDS_PAD_SEC;
    const to = Math.min(now, Math.max(...ends) + BOUNDS_PAD_SEC);
    return { from, to: Math.max(to, from + 60), empty: false };
  }, [signals, trades]);

  const derivedFrom = toDateTimeLocal(derived.from);
  const derivedTo = toDateTimeLocal(derived.to);

  const [fromInput, setFromInput] = useState(derivedFrom);
  const [toInput, setToInput] = useState(derivedTo);
  // Last value this component itself wrote into each input. The poll may only
  // overwrite an input that still holds exactly that — anything else is a
  // manual edit and is left alone.
  const lastAutoRef = useRef({ from: derivedFrom, to: derivedTo });

  useEffect(() => {
    setFromInput((current) =>
      current === lastAutoRef.current.from ? derivedFrom : current,
    );
    setToInput((current) =>
      current === lastAutoRef.current.to ? derivedTo : current,
    );
    lastAutoRef.current = { from: derivedFrom, to: derivedTo };
  }, [derivedFrom, derivedTo]);

  const fromSec = parseDateTimeLocal(fromInput);
  const toSec = parseDateTimeLocal(toInput);
  const rangeValid = fromSec !== null && toSec !== null && toSec > fromSec;

  const cursor = replay.cursorTime;
  const revealedSignals =
    cursor === null ? 0 : signals.filter((s) => s.time <= cursor).length;
  const revealedTrades =
    cursor === null ? 0 : trades.filter((t) => t.open_time <= cursor).length;

  const speedSelect = (
    <label className="flex items-center gap-1.5 text-[10px] text-ink-muted">
      Speed
      <select
        value={replay.speed}
        onChange={(e) => replay.onSpeedChange(Number(e.target.value))}
        className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
      >
        {SPEED_OPTIONS.map((s) => (
          <option key={s} value={s}>
            {s}x
          </option>
        ))}
      </select>
    </label>
  );

  if (replay.active) {
    return (
      <div className="flex flex-col gap-2.5 p-3">
        <div className="flex items-center gap-1.5">
          <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
          <span className="text-xs font-semibold text-ink">
            Replaying session
          </span>
        </div>

        <div className="rounded border border-line bg-panel-dark/30 p-2 font-mono text-[11px] text-ink">
          {cursor === null ? '—' : formatCursor(cursor)}
        </div>

        <dl className="grid grid-cols-2 gap-1.5 text-[10px] text-ink-muted">
          <div className="rounded border border-line bg-panel-dark/20 p-1.5">
            <dt className="uppercase tracking-wider">Signals</dt>
            <dd className="font-mono text-xs text-ink">
              {revealedSignals} / {signals.length}
            </dd>
          </div>
          <div className="rounded border border-line bg-panel-dark/20 p-1.5">
            <dt className="uppercase tracking-wider">Trades</dt>
            <dd className="font-mono text-xs text-ink">
              {revealedTrades} / {trades.length}
            </dd>
          </div>
        </dl>

        {speedSelect}

        <div className="flex gap-1.5">
          <button
            type="button"
            onClick={replay.onPlayPause}
            className="flex flex-1 cursor-pointer items-center justify-center gap-1 rounded border border-accent/40 bg-accent/10 px-2 py-1.5 text-xs font-semibold text-accent transition-colors hover:bg-accent/20"
          >
            {replay.playing ? <Pause size={12} /> : <Play size={12} />}
            {replay.playing ? 'Pause' : 'Resume'}
          </button>
          <button
            type="button"
            onClick={replay.onExit}
            className="flex cursor-pointer items-center justify-center gap-1 rounded border border-line bg-panel px-2 py-1.5 text-xs text-ink transition-colors hover:border-err hover:text-err"
          >
            <LogOut size={12} /> Exit
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 p-3">
      <p className="text-[11px] leading-relaxed text-ink-muted">
        Replay this bot&apos;s session bar by bar. The period below is derived
        from its own signals and trades — edit it to widen or narrow the
        window.
      </p>

      {derived.empty && (
        <p className="rounded border border-line bg-panel-dark/20 p-2 text-[10px] text-ink-muted">
          No signals or trades yet — defaulting to the last 24 hours.
        </p>
      )}

      <label className="flex flex-col gap-1 text-[10px] uppercase tracking-wider text-ink-muted">
        From
        <input
          type="datetime-local"
          value={fromInput}
          onChange={(e) => setFromInput(e.target.value)}
          className="rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
        />
      </label>
      <label className="flex flex-col gap-1 text-[10px] uppercase tracking-wider text-ink-muted">
        To
        <input
          type="datetime-local"
          value={toInput}
          onChange={(e) => setToInput(e.target.value)}
          className="rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink focus:border-accent focus:outline-none"
        />
      </label>

      {speedSelect}

      {!rangeValid && (
        <p className="text-[10px] text-err">
          &quot;To&quot; must be after &quot;From&quot;.
        </p>
      )}

      <button
        type="button"
        disabled={!rangeValid || replay.loading !== null}
        onClick={() => {
          if (rangeValid) replay.onStart(fromSec, toSec);
        }}
        className="flex cursor-pointer items-center justify-center gap-1 rounded border border-accent/40 bg-accent/10 px-2 py-1.5 text-xs font-semibold text-accent transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Play size={12} /> Play session
      </button>

      {replay.loading && (
        <p className="text-[10px] text-ink-muted">
          Loading… page {replay.loading.page} (
          {replay.loading.loaded.toLocaleString()} candles so far)
        </p>
      )}
    </div>
  );
}

export const BotSessionReplayTab = memo(BotSessionReplayTabImpl);
