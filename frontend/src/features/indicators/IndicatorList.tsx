"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ApiError, deleteIndicator, getIndicator, listIndicators, type IndicatorSummary } from "@/shared/api/client";
import { DuplicateIndicatorForm } from "./DuplicateIndicatorForm";
import { IndicatorCreateForm } from "./IndicatorCreateForm";

/** Every saved custom indicator, alphabetical by name — create/duplicate/
 * delete/copy-code from here, or click through for the full code editor. */
export function IndicatorList() {
  const [indicators, setIndicators] = useState<IndicatorSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const reload = useCallback(() => {
    listIndicators()
      .then(setIndicators)
      .catch(() => setError("failed to load indicators"));
  }, []);

  useEffect(reload, [reload]);

  async function handleCopy(id: string) {
    try {
      const detail = await getIndicator(id);
      await navigator.clipboard.writeText(detail.code);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      setError("failed to copy code");
    }
  }

  async function handleDelete(id: string, name: string) {
    if (!confirm(`Delete indicator "${name}"? This cannot be undone.`)) return;
    setBusyId(id);
    try {
      await deleteIndicator(id);
      reload();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "delete failed");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      <IndicatorCreateForm />
      {error && <p className="text-sm text-err">{error}</p>}

      {indicators === null ? (
        <p className="text-sm text-ink-muted">Loading…</p>
      ) : indicators.length === 0 ? (
        <p className="text-sm text-ink-muted">
          No custom indicators yet — create one above to use it on the chart.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-muted">
                <th className="px-3 py-2 font-medium">Name</th>
                <th className="px-3 py-2 font-medium">Default params</th>
                <th className="px-3 py-2 font-medium">Updated</th>
                <th className="px-3 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {indicators.map((ind) => (
                <tr key={ind.id} className="border-b border-line last:border-0 hover:bg-bg/40">
                  <td className="px-3 py-2">
                    <Link href={`/indicators/${ind.id}`} className="text-accent hover:underline">
                      {ind.name}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-ink-muted">
                    {Object.keys(ind.default_params).length > 0
                      ? Object.entries(ind.default_params)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-ink-muted">{formatTime(ind.updated_at)}</td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <DuplicateIndicatorForm indicatorId={ind.id} onDuplicated={reload} />
                      <button
                        type="button"
                        className={`${btnCls} transition-colors ${
                          copiedId === ind.id ? "border-ok text-ok hover:border-ok hover:text-ok" : ""
                        }`}
                        onClick={() => handleCopy(ind.id)}
                      >
                        {copiedId === ind.id ? "Copied!" : "Copy code"}
                      </button>
                      <button
                        type="button"
                        className="cursor-pointer rounded border border-err px-2 py-1 text-xs text-err hover:bg-err hover:text-bg disabled:cursor-not-allowed disabled:opacity-50"
                        disabled={busyId === ind.id}
                        onClick={() => handleDelete(ind.id, ind.name)}
                      >
                        {busyId === ind.id ? "Deleting…" : "Delete"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

const btnCls =
  "cursor-pointer rounded border border-line px-2 py-1 text-xs hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50";
