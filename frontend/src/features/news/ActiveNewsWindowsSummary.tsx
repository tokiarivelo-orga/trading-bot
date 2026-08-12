"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getActiveNewsWindows, type NewsWindow } from "@/shared/api/client";
import { StatusBadge } from "@/features/strategies/StatusBadge";

const POLL_MS = 30_000;

/** Compact sidebar summary of `/news/active-windows` — full detail lives on
 * the `/news` page (§8 UI). */
export function ActiveNewsWindowsSummary() {
  const [windows, setWindows] = useState<NewsWindow[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      getActiveNewsWindows()
        .then((w) => {
          if (!cancelled) setWindows(w);
        })
        .catch(() => {
          // Ignore here — the /news page surfaces load errors in full.
        });
    }
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <>
      <Link href="/news" className="text-accent hover:underline">
        News →
      </Link>{" "}
      {windows === null ? (
        <span className="text-ink-muted">loading…</span>
      ) : windows.length === 0 ? (
        <span className="text-ink-muted">no active news windows</span>
      ) : (
        <ul className="mt-1 flex flex-col gap-1">
          {windows.map((w, i) => (
            <li key={`${w.event.name}-${w.window_start}-${i}`} className="flex items-center gap-1">
              <StatusBadge status={w.phase} />
              <span>{w.event.name}</span>
              <span className="text-ink-muted">({w.symbols.join(", ")})</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
