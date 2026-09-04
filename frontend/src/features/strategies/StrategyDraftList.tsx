"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import { getStrategyDrafts, type StrategyDraft } from "@/shared/api/client";
import { downloadJson } from "@/shared/utils/download";
import { StatusBadge } from "./StatusBadge";
import { StrategyUploadForm } from "./StrategyUploadForm";

/** Every PDF-derived draft, newest first, with the upload form above it —
 * pass `showUploadForm={false}` when an embedding page (e.g. the Bots hub)
 * already renders its own generation form above this list. */
export function StrategyDraftList({ showUploadForm = true }: { showUploadForm?: boolean } = {}) {
  const accountId = useActiveAccount();
  const [drafts, setDrafts] = useState<StrategyDraft[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accountId) return;
    getStrategyDrafts(accountId)
      .then(setDrafts)
      .catch(() => setError("failed to load strategy drafts"));
  }, [accountId]);

  return (
    <div className="flex flex-col gap-4 p-4">
      {showUploadForm && <StrategyUploadForm />}
      {error && (
        <div className="rounded-lg border border-err/30 bg-err/10 p-3 text-sm text-err font-medium">
          ⚠️ {error}
        </div>
      )}
      {drafts === null ? (
        <div className="flex items-center justify-center py-8 text-sm text-ink-muted">
          <svg className="animate-spin h-5 w-5 text-accent mr-2" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Loading drafts...
        </div>
      ) : drafts.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center rounded-xl border border-line bg-panel/30">
          <svg className="h-10 w-10 text-ink-muted mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-sm font-semibold text-ink">No drafts pending review</p>
          <p className="text-xs text-ink-muted mt-1 max-w-xs">
            Upload a strategy PDF spec or prompt the AI to generate a draft.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-line bg-panel/40 shadow-sm">
          <table className="w-full min-w-[560px] border-collapse text-sm text-left">
            <thead>
              <tr className="border-b border-line bg-panel/50 text-[10px] font-semibold text-ink-muted uppercase tracking-wider">
                <th className="px-4 py-3 font-semibold">Strategy Name</th>
                <th className="px-4 py-3 font-semibold">Target Symbols</th>
                <th className="px-4 py-3 font-semibold">Source File</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Created On</th>
                <th className="px-4 py-3 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/40">
              {drafts.map((d) => (
                <tr key={d.id} className="hover:bg-panel/40 transition-colors duration-150">
                  <td className="px-4 py-3.5 font-medium">
                    <Link
                      href={`/strategies/drafts/${d.id}`}
                      className="text-accent hover:underline font-semibold"
                    >
                      {d.effective_spec.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3.5 text-ink/80">
                    {d.effective_spec.symbols.length ? (
                      <span className="flex flex-wrap gap-1">
                        {d.effective_spec.symbols.map((sym) => (
                          <span key={sym} className="text-2xs bg-bg px-1.5 py-0.5 rounded border border-line font-mono">
                            {sym}
                          </span>
                        ))}
                      </span>
                    ) : (
                      <span className="text-ink-muted">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3.5 text-ink-muted max-w-[150px] truncate" title={d.source_filename}>
                    {d.source_filename || "Natural Language Prompt"}
                  </td>
                  <td className="px-4 py-3.5">
                    <StatusBadge status={d.status} />
                  </td>
                  <td className="px-4 py-3.5 text-ink-muted text-xs">
                    {formatTime(d.created_at)}
                  </td>
                  <td className="px-4 py-3.5 text-right">
                    <button
                      type="button"
                      onClick={() => downloadJson(d, `draft_${d.id}_${d.effective_spec.name}.json`)}
                      className="cursor-pointer rounded-lg border border-line bg-bg/50 px-2.5 py-1 text-xs text-ink hover:border-accent hover:text-accent transition-all duration-200"
                    >
                      Export JSON
                    </button>
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
