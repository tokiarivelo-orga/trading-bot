"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import { getAnalysisReports, type AnalysisReport } from "@/shared/api/client";
import { StatusBadge } from "@/features/strategies/StatusBadge";

/** Every 10-trade AI review, newest first — including reviews that found
 * nothing worth changing, kept for audit (§8.2). */
export function AnalysisReportList() {
  const accountId = useActiveAccount();
  const [reports, setReports] = useState<AnalysisReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accountId) return;
    getAnalysisReports(accountId)
      .then(setReports)
      .catch(() => setError("failed to load analysis reports"));
  }, [accountId]);

  return (
    <div className="flex flex-col gap-3 p-4">
      {error && <p className="text-sm text-err">{error}</p>}
      {reports === null ? (
        <p className="text-sm text-ink-muted">Loading…</p>
      ) : reports.length === 0 ? (
        <p className="text-sm text-ink-muted">
          No reviews yet — one runs automatically every 10 closed trades per symbol.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-md border border-line bg-panel">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-muted">
                <th className="px-3 py-2 font-medium">Strategy</th>
                <th className="px-3 py-2 font-medium">Symbol</th>
                <th className="px-3 py-2 font-medium">Win rate</th>
                <th className="px-3 py-2 font-medium">Avg R</th>
                <th className="px-3 py-2 font-medium">Verdict</th>
                <th className="px-3 py-2 font-medium">Reviewed</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.id} className="border-b border-line last:border-0 hover:bg-bg/40">
                  <td className="px-3 py-2">
                    <Link href={`/ai-reports/${r.id}`} className="text-accent hover:underline">
                      {r.strategy_name}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-ink-muted">{r.symbol}</td>
                  <td className="px-3 py-2 text-ink-muted">{(r.win_rate * 100).toFixed(0)}%</td>
                  <td className="px-3 py-2 text-ink-muted">{r.avg_r.toFixed(2)}</td>
                  <td className="px-3 py-2">
                    <StatusBadge status={r.verdict} />
                  </td>
                  <td className="px-3 py-2 text-ink-muted">{formatTime(r.created_at)}</td>
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
