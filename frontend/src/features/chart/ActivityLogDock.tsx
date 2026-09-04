"use client";

/**
 * ActivityLogDock — TradingView-style panel showing the bot's persisted
 * activity log (signals, HTF vetoes, risk gate blocks, spread vetoes,
 * fills, circuit breakers) for the chart's current symbol, so "why did it
 * just do that" is answerable without leaving the chart.
 *
 * Rendered inside ChartPanel below the chart header when the user clicks
 * the "Activity log" toggle button, same slot/style as IndicatorsDock.
 * Polls every 5s while open — there's no WS feed for this yet, and 5s is
 * plenty for a human reading log lines (see IMPLEMENTATION_PLAN's Socket.IO
 * note: only live candle streaming needs that path).
 */

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import { ApiError, getActivityLog, type LogEntry } from "@/shared/api/client";

const POLL_INTERVAL_MS = 5000;
const PAGE_SIZE = 50;

export function ActivityLogDock({
  symbol,
  replayEntries,
}: {
  symbol: string;
  /** When set (including an empty array), the dock renders exactly these
   * entries instead of polling the global activity log — driven by a
   * backtest report's own `activity_log`, filtered up to the replay cursor
   * (see ChartPanel's replay player, §F). `undefined` means "not replaying",
   * the normal live/global poll below. */
  replayEntries?: LogEntry[] | null;
}) {
  const accountId = useActiveAccount();
  const [entries, setEntries] = useState<LogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(true);
  const [levelFilter, setLevelFilter] = useState<string>("");
  const listRef = useRef<HTMLDivElement>(null);
  const replaying = replayEntries !== undefined;

  useEffect(() => {
    if (replaying || !accountId) return;
    const account = accountId;
    let cancelled = false;

    function fetchOnce() {
      getActivityLog(account, { q: symbol, level: levelFilter || undefined, limit: PAGE_SIZE })
        .then(({ items }) => {
          if (cancelled) return;
          setEntries(items);
          setError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setEntries([]);
          setError(e instanceof ApiError ? e.message : "failed to load activity log");
        });
    }

    fetchOnce();
    if (!live) return () => {
      cancelled = true;
    };
    const id = setInterval(fetchOnce, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [accountId, symbol, levelFilter, live, replaying]);

  // Keep the newest replayed entry in view as the replay cursor advances —
  // the whole point of showing the log during replay is to read it as it
  // happens, not to keep re-scrolling by hand.
  useEffect(() => {
    if (!replaying) return;
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [replaying, replayEntries]);

  const shown = replaying ? (replayEntries ?? []) : entries;

  return (
    <div className="border-b border-line bg-panel">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-1.5">
        <span className="text-xs font-medium text-ink">
          Activity log — {symbol}
          {replaying && " (replay)"}
        </span>
        {!replaying && (
          <>
            <select
              value={levelFilter}
              onChange={(e) => setLevelFilter(e.target.value)}
              className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
            >
              <option value="">All levels</option>
              <option value="INFO">Info</option>
              <option value="WARNING">Warning</option>
              <option value="ERROR">Error</option>
            </select>
            <label className="flex items-center gap-1.5 text-xs text-ink-muted">
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
              Watch live
            </label>
          </>
        )}
        <Link href="/logs" className="ml-auto text-xs text-ink-muted hover:text-accent">
          Full history →
        </Link>
      </div>
      <div ref={listRef} className="max-h-56 overflow-y-auto">
        {!replaying && error && <p className="px-3 py-2 text-xs text-err">{error}</p>}
        {!replaying && !error && shown === null && (
          <p className="px-3 py-2 text-xs text-ink-muted">Loading…</p>
        )}
        {!replaying && !error && shown !== null && shown.length === 0 && (
          <p className="px-3 py-2 text-xs text-ink-muted">
            No activity for {symbol} yet — signals, vetoes, and fills will show up here.
          </p>
        )}
        {replaying && (shown?.length ?? 0) === 0 && (
          <p className="px-3 py-2 text-xs text-ink-muted">
            No activity yet at this point in the replay.
          </p>
        )}
        {shown !== null && shown.length > 0 && (
          <ul className="divide-y divide-line text-xs">
            {shown.map((e) => (
              <li key={e.id} className="flex items-start gap-2 px-3 py-1.5">
                <span className="shrink-0 whitespace-nowrap text-ink-muted">
                  {formatTime(e.created_at)}
                </span>
                <span className={`shrink-0 font-medium ${levelTone(e.level)}`}>{e.level}</span>
                <span className="font-mono text-ink">{e.message}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function levelTone(level: string): string {
  if (level === "ERROR" || level === "CRITICAL") return "text-err";
  if (level === "WARNING") return "text-accent";
  return "text-ink-muted";
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
