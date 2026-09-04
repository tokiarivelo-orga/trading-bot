"use client";

import { useEffect, useMemo, useState } from "react";
import {
  getActiveNewsWindows,
  getUpcomingNews,
  type ImpactLevel,
  type NewsEvent,
  type NewsWindow,
} from "@/shared/api/client";
import { StatusBadge } from "@/features/strategies/StatusBadge";

const POLL_MS = 30_000; // matches the backend's news-window transition-check cadence

const inputCls =
  "rounded border border-line bg-bg px-2 py-1 text-sm text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none";

interface EventFilters {
  search: string;
  currency: string;
  impact: ImpactLevel | "";
  hasSkill: "" | "yes" | "no";
}

const EMPTY_FILTERS: EventFilters = { search: "", currency: "", impact: "", hasSkill: "" };

/** Upcoming economic calendar + any currently active news window (§6.7, §8).
 * Read-only — the engine reacts to windows on its own via `NewsSkillSelector`
 * and the trade engine's pre-news flatten; this just shows what it's doing. */
export function UpcomingEventsPanel() {
  const [events, setEvents] = useState<NewsEvent[] | null>(null);
  const [windows, setWindows] = useState<NewsWindow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<EventFilters>(EMPTY_FILTERS);
  const [selected, setSelected] = useState<NewsEvent | null>(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      Promise.all([getUpcomingNews(), getActiveNewsWindows()])
        .then(([e, w]) => {
          if (cancelled) return;
          setEvents(e);
          setWindows(w);
          setError(null);
        })
        .catch(() => {
          if (!cancelled) setError("failed to load news calendar");
        });
    }
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const currencies = useMemo(
    () => Array.from(new Set((events ?? []).map((e) => e.currency))).sort(),
    [events],
  );

  const filteredEvents = useMemo(() => {
    if (events === null) return null;
    const search = filters.search.trim().toLowerCase();
    return events.filter((e) => {
      if (search && !e.name.toLowerCase().includes(search)) return false;
      if (filters.currency && e.currency !== filters.currency) return false;
      if (filters.impact && e.impact !== filters.impact) return false;
      if (filters.hasSkill === "yes" && e.skill === null) return false;
      if (filters.hasSkill === "no" && e.skill !== null) return false;
      return true;
    });
  }, [events, filters]);

  const hasActiveFilters =
    filters.search !== "" || filters.currency !== "" || filters.impact !== "" || filters.hasSkill !== "";

  return (
    <div className="flex flex-col gap-4 p-4">
      {error && <p className="text-sm text-err">{error}</p>}

      <section>
        <h2 className="mb-2 text-sm font-bold text-ink-muted">Active news windows</h2>
        {windows === null ? (
          <p className="text-sm text-ink-muted">Loading…</p>
        ) : windows.length === 0 ? (
          <p className="text-sm text-ink-muted">
            None active — trading follows the normal per-symbol skill.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {windows.map((w) => (
              <li
                key={`${w.event.name}-${w.window_start}`}
                className="flex items-center gap-2 rounded-md border border-line bg-panel p-3 text-sm"
              >
                <StatusBadge status={w.phase} />
                <span className="font-medium">{w.event.name}</span>
                <span className="text-ink-muted">skill: {w.skill}</span>
                <span className="ml-auto text-ink-muted">{formatTime(w.event.time)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-bold text-ink-muted">Upcoming events</h2>

        <div className="mb-3 flex flex-wrap items-end gap-2">
          <Field label="Search">
            <input
              className={`${inputCls} w-48`}
              placeholder="event name…"
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            />
          </Field>
          <Field label="Currency">
            <select
              className={inputCls}
              value={filters.currency}
              onChange={(e) => setFilters({ ...filters, currency: e.target.value })}
            >
              <option value="">Any</option>
              {currencies.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Impact">
            <select
              className={inputCls}
              value={filters.impact}
              onChange={(e) => setFilters({ ...filters, impact: e.target.value as ImpactLevel | "" })}
            >
              <option value="">Any</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </Field>
          <Field label="Skill">
            <select
              className={inputCls}
              value={filters.hasSkill}
              onChange={(e) =>
                setFilters({ ...filters, hasSkill: e.target.value as EventFilters["hasSkill"] })
              }
            >
              <option value="">Any</option>
              <option value="yes">Activates a skill</option>
              <option value="no">No skill</option>
            </select>
          </Field>
          {hasActiveFilters && (
            <button
              type="button"
              className="cursor-pointer rounded border border-line px-2 py-1 text-xs text-ink-muted hover:border-accent hover:text-accent"
              onClick={() => setFilters(EMPTY_FILTERS)}
            >
              Clear filters
            </button>
          )}
        </div>

        {filteredEvents === null ? (
          <p className="text-sm text-ink-muted">Loading…</p>
        ) : events !== null && events.length === 0 ? (
          <p className="text-sm text-ink-muted">Nothing in the next 7 days.</p>
        ) : filteredEvents.length === 0 ? (
          <p className="text-sm text-ink-muted">No events match the current filters.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border border-line bg-panel">
            <table className="w-full min-w-[760px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-muted">
                  <th className="px-3 py-2 font-medium">Event</th>
                  <th className="px-3 py-2 font-medium">Currency</th>
                  <th className="px-3 py-2 font-medium">Impact</th>
                  <th className="px-3 py-2 font-medium">Time</th>
                  <th className="px-3 py-2 font-medium">Forecast</th>
                  <th className="px-3 py-2 font-medium">Previous</th>
                  <th className="px-3 py-2 font-medium">Actual</th>
                  <th className="px-3 py-2 font-medium">Skill</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.map((e) => (
                  <tr
                    key={`${e.name}-${e.time}`}
                    className="cursor-pointer border-b border-line last:border-0 hover:bg-bg/40"
                    onClick={() => setSelected(e)}
                  >
                    <td className="px-3 py-2">{e.name}</td>
                    <td className="px-3 py-2 text-ink-muted">{e.currency}</td>
                    <td className="px-3 py-2">
                      <StatusBadge status={e.impact} />
                    </td>
                    <td className="px-3 py-2 text-ink-muted">{formatTime(e.time)}</td>
                    <td className="px-3 py-2 text-ink-muted">{e.forecast ?? "—"}</td>
                    <td className="px-3 py-2 text-ink-muted">{e.previous ?? "—"}</td>
                    <td className="px-3 py-2 text-ink-muted">{e.actual ?? "—"}</td>
                    <td className="px-3 py-2 text-ink-muted">{e.skill ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {selected && <EventDetailModal event={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function EventDetailModal({ event, onClose }: { event: NewsEvent; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-md border border-line bg-panel p-4 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-start justify-between gap-4">
          <h3 className="text-sm font-bold">{event.name}</h3>
          <button
            type="button"
            className="cursor-pointer text-ink-muted hover:text-ink"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <dl className="flex flex-col gap-2 text-sm">
          <Row label="Currency">{event.currency || "—"}</Row>
          <Row label="Impact">
            <StatusBadge status={event.impact} />
          </Row>
          <Row label="Time (UTC)">{formatFullTime(event.time)}</Row>
          <Row label="Forecast">{event.forecast ?? "—"}</Row>
          <Row label="Previous">{event.previous ?? "—"}</Row>
          <Row label="Actual">{event.actual ?? "not released yet"}</Row>
          <Row label="News skill">{event.skill ?? "None — this event never activates a window"}</Row>
        </dl>
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-ink-muted">{label}</dt>
      <dd className="font-medium">{children}</dd>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-ink-muted">
      {label}
      {children}
    </label>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatFullTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) + " UTC";
}
