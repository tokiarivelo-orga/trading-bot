"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useActiveAccount } from "@/shared/api/account-context";
import { getAnalysisReport, type AnalysisReport } from "@/shared/api/client";
import { StatusBadge } from "@/features/strategies/StatusBadge";

/** Full review detail: findings, verdict, and — if the AI proposed a
 * refinement — a link to that proposal's diff/backtest comparison. */
export function AnalysisReportDetail({ reportId }: { reportId: string }) {
  const accountId = useActiveAccount();
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!accountId) return;
    getAnalysisReport(accountId, reportId)
      .then(setReport)
      .catch(() => setError("report not found"));
  }, [accountId, reportId]);

  if (error) return <p className="p-4 text-sm text-err">{error}</p>;
  if (report === null) return <p className="p-4 text-sm text-ink-muted">Loading…</p>;

  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <Link href="/ai-reports" className="text-xs text-ink-muted hover:text-accent">
          ← All reviews
        </Link>
        <div className="mt-1 flex items-center gap-2">
          <h2 className="text-lg font-semibold">
            {report.strategy_name} · {report.symbol}
          </h2>
          <StatusBadge status={report.verdict} />
        </div>
        <p className="text-sm text-ink-muted">
          Reviewed {report.trade_ids.length} trades, {formatTime(report.created_at)}
        </p>
      </div>

      <dl className="grid gap-x-4 gap-y-1 rounded-md border border-line bg-panel p-3 text-sm sm:grid-cols-2">
        <Row label="Win rate" value={`${(report.win_rate * 100).toFixed(1)}%`} />
        <Row label="Avg R" value={report.avg_r.toFixed(2)} />
      </dl>

      <section className="rounded-md border border-line bg-panel p-3 text-sm">
        <header className="mb-2 text-ink-muted">Findings</header>
        <p className="whitespace-pre-wrap">
          <strong className="text-ink">Common failure pattern:</strong>{" "}
          {report.common_failure_pattern || "none identified"}
        </p>
        <p className="mt-1 whitespace-pre-wrap">
          <strong className="text-ink">Session/news correlation:</strong>{" "}
          {report.session_or_news_correlation || "none identified"}
        </p>
      </section>

      {report.proposal_id && (
        <Link
          href={`/ai-reports/proposals/${report.proposal_id}`}
          className="w-fit rounded border border-accent px-3 py-1 text-sm text-accent hover:bg-accent hover:text-bg"
        >
          View proposed refinement →
        </Link>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2 sm:block">
      <dt className="text-xs text-ink-muted">{label}</dt>
      <dd className="truncate">{value}</dd>
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
