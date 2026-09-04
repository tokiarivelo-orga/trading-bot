"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import {
  getStrategyVersion,
  getStrategyVersions,
  type ExtractedStrategySpec,
  type StrategyVersionSummary,
} from "@/shared/api/client";
import { downloadJson } from "@/shared/utils/download";
import { BulkVersionActions } from "./BulkVersionActions";
import { DuplicateVersionForm } from "./DuplicateVersionForm";
import { StatusBadge } from "./StatusBadge";
import { SymbolChipsEditor } from "./SymbolChipsEditor";
import { VersionLifecycleActions } from "./VersionLifecycleActions";

const FILTER_QUERY_KEY = "q";

/** Every recorded strategy version, newest first per name. Pass `name` to
 * restrict to one strategy family (used on the draft detail page once code
 * has been generated for it) — bulk selection and the URL-synced search box
 * only make sense on the full library table, so both are skipped when `name`
 * is set. */
export function StrategyVersionList({ name }: { name?: string }) {
  const accountId = useActiveAccount();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [versions, setVersions] = useState<StrategyVersionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState(() => (name ? "" : searchParams.get(FILTER_QUERY_KEY) ?? ""));
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  function updateFilter(value: string) {
    setFilter(value);
    if (name) return;
    const params = new URLSearchParams(searchParams.toString());
    if (value) params.set(FILTER_QUERY_KEY, value);
    else params.delete(FILTER_QUERY_KEY);
    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }

  function handleSpecUpdated(versionId: string, spec: ExtractedStrategySpec) {
    setVersions((prev) => prev?.map((v) => (v.id === versionId ? { ...v, spec } : v)) ?? prev);
  }

  function toggleSelected(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleExport(id: string, name: string, version: number) {
    if (!accountId) return;
    setExportingId(id);
    try {
      const detail = await getStrategyVersion(accountId, id);
      downloadJson(detail, `${name}_v${version}_${id}.json`);
    } catch (err) {
      console.error("Failed to export strategy version:", err);
    } finally {
      setExportingId(null);
    }
  }

  const reload = useCallback(() => {
    if (!accountId) return;
    getStrategyVersions(accountId, name)
      .then(setVersions)
      .catch(() => setError("failed to load strategy versions"));
  }, [accountId, name]);

  useEffect(reload, [reload]);

  if (error) {
    return (
      <div className="rounded-lg border border-err/30 bg-err/10 p-4 text-sm text-err font-medium m-4">
        ⚠️ {error}
      </div>
    );
  }

  if (versions === null) {
    return (
      <div className="flex items-center justify-center p-8 text-sm text-ink-muted">
        <svg className="animate-spin h-5 w-5 text-accent mr-2" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        Loading bots library...
      </div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center rounded-xl border border-line bg-panel/30 m-4">
        <svg className="h-10 w-10 text-ink-muted mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
        </svg>
        <p className="text-sm font-semibold text-ink">No strategies in library</p>
        <p className="text-xs text-ink-muted mt-1 max-w-xs">
          Generate code from an approved draft to deploy your first bot strategy.
        </p>
      </div>
    );
  }

  const query = filter.trim().toLowerCase();
  const filtered = query
    ? versions.filter(
        (v) =>
          v.name.toLowerCase().includes(query) ||
          (v.spec?.symbols ?? []).some((s) => s.toLowerCase().includes(query)),
      )
    : versions;

  const allFilteredSelected = filtered.length > 0 && filtered.every((v) => selectedIds.has(v.id));

  function toggleSelectAll() {
    setSelectedIds(allFilteredSelected ? new Set() : new Set(filtered.map((v) => v.id)));
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      {!name && (
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative w-full max-w-sm">
            <span className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-ink-muted">
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </span>
            <input
              className="w-full rounded-lg border border-line bg-bg/60 pl-9 pr-4 py-2 text-sm text-ink placeholder:text-ink-muted focus:border-accent focus:ring-1 focus:ring-accent focus:outline-none transition-all duration-200"
              placeholder="Filter by strategy name or symbol..."
              value={filter}
              onChange={(e) => updateFilter(e.target.value)}
            />
          </div>
          <BulkVersionActions
            selectedIds={Array.from(selectedIds)}
            onDone={() => {
              setSelectedIds(new Set());
              reload();
            }}
          />
        </div>
      )}

      <div className="overflow-x-auto rounded-xl border border-line bg-panel/40 shadow-sm">
        <table className="w-full min-w-[720px] border-collapse text-sm text-left">
          <thead>
            <tr className="border-b border-line bg-panel/50 text-[10px] font-semibold text-ink-muted uppercase tracking-wider">
              {!name && (
                <th className="px-4 py-3 font-semibold w-8">
                  <input
                    type="checkbox"
                    className="cursor-pointer accent-accent"
                    checked={allFilteredSelected}
                    onChange={toggleSelectAll}
                    aria-label="Select all strategy versions"
                  />
                </th>
              )}
              <th className="px-4 py-3 font-semibold">Strategy Name</th>
              <th className="px-4 py-3 font-semibold text-right">Version</th>
              <th className="px-4 py-3 font-semibold">Symbols</th>
              <th className="px-4 py-3 font-semibold">Source</th>
              <th className="px-4 py-3 font-semibold">Status</th>
              <th className="px-4 py-3 font-semibold">Created On</th>
              <th className="px-4 py-3 font-semibold text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/40">
            {filtered.map((v) => (
              <tr key={v.id} className="hover:bg-panel/40 transition-colors duration-150">
                {!name && (
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      className="cursor-pointer accent-accent"
                      checked={selectedIds.has(v.id)}
                      onChange={() => toggleSelected(v.id)}
                      aria-label={`Select ${v.name} v${v.version}`}
                    />
                  </td>
                )}
                <td className="px-4 py-3 font-medium">
                  <Link
                    href={`/strategies/versions/${v.id}`}
                    className="text-accent hover:underline font-semibold"
                  >
                    {v.name}
                  </Link>
                </td>
                <td className="px-4 py-3 text-right font-mono text-ink/80">v{v.version}</td>
                <td className="px-4 py-3 text-ink/80">
                  {v.spec ? (
                    <SymbolChipsEditor
                      versionId={v.id}
                      spec={v.spec}
                      onUpdated={(spec) => handleSpecUpdated(v.id, spec)}
                    />
                  ) : (
                    <span className="text-ink-muted">—</span>
                  )}
                </td>
                <td className="px-4 py-3 text-ink-muted">
                  <span className={`text-2xs font-semibold px-2 py-0.5 rounded-full border ${
                    v.source === "ai_generated"
                      ? "border-accent/30 text-accent bg-accent/5"
                      : "border-ink-muted/30 text-ink bg-bg"
                  }`}>
                    {v.source === "ai_generated" ? "🤖 AI Generated" : "✍️ Manual"}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1.5">
                    <StatusBadge status={v.status} />
                    {v.status === "active" && v.paused && <StatusBadge status="paused" />}
                  </div>
                </td>
                <td className="px-4 py-3 text-ink-muted text-xs">{formatTime(v.created_at)}</td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-2 flex-wrap">
                    <DuplicateVersionForm versionId={v.id} sourceSymbols={v.spec?.symbols ?? []} />
                    <VersionLifecycleActions
                      version={v}
                      onChanged={reload}
                      onDeleted={reload}
                    />
                    <button
                      type="button"
                      className="cursor-pointer rounded-lg border border-line bg-bg/50 px-2.5 py-1 text-xs text-ink hover:border-accent hover:text-accent transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                      disabled={exportingId !== null}
                      onClick={() => handleExport(v.id, v.name, v.version)}
                    >
                      {exportingId === v.id ? "Exporting…" : "Export JSON"}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="p-8 text-center text-sm text-ink-muted">
            No versions matching &quot;{filter}&quot; found in library.
          </div>
        )}
      </div>
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
