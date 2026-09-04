"use client";

/**
 * "Why did the bot take this trade" detail — the full decision context
 * behind one journaled trade (reason, confidence, S&D zone, pattern,
 * swing structure), shown as a click-through modal from a table cell.
 * Shared by the Active Orders table (`AllOrdersPanel`) and the Trade
 * History table (`TradeHistoryTable`) so both read the same journal fields
 * the same way. Modeled on `EventDetailModal` in
 * `features/news/UpcomingEventsPanel.tsx`.
 */

import { useEffect, useState } from "react";

import { useActiveAccount } from "@/shared/api/account-context";
import { getStrategyVersions, type TradeHistoryItem } from "@/shared/api/client";
import { DecisionChartSnippet } from "@/shared/ui/DecisionChartSnippet";

function formatFullTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) + " UTC";
}

// `strategy_version` is always written as `"{name}:v{version}"` (see
// `trade_loop.py`'s `open_position` call) — parsed back out here to look
// up that exact version's spec snapshot for its timeframes.
function parseStrategyVersion(value: string | null): { name: string; version: number } | null {
  if (value === null) return null;
  const match = /^(.+):v(\d+)$/.exec(value);
  if (!match) return null;
  return { name: match[1], version: Number(match[2]) };
}

/** The entry/confirmation timeframes a trade was decided on — looked up
 * live from the strategy version's spec snapshot rather than stored on the
 * trade itself, since no such field exists on `TradeHistoryItem` (see
 * `backend/src/journal/domain/models.py`). Not available for manual/API
 * trades (no `strategy_version`) or if the lookup fails. */
function useDecisionTimeframes(strategyVersion: string | null): {
  entryTimeframe: string | null;
  confirmationTimeframes: string[];
} {
  const accountId = useActiveAccount();
  const [entryTimeframe, setEntryTimeframe] = useState<string | null>(null);
  const [confirmationTimeframes, setConfirmationTimeframes] = useState<string[]>([]);

  useEffect(() => {
    setEntryTimeframe(null);
    setConfirmationTimeframes([]);
    const parsed = parseStrategyVersion(strategyVersion);
    if (!accountId || !parsed) return;
    let cancelled = false;
    getStrategyVersions(accountId, parsed.name)
      .then((versions) => {
        if (cancelled) return;
        const match = versions.find((v) => v.version === parsed.version);
        if (match?.spec) {
          setEntryTimeframe(match.spec.entry_timeframe);
          setConfirmationTimeframes(match.spec.confirmation_timeframes);
        }
      })
      .catch(() => {
        // Best-effort enrichment — the rest of the "Why" modal still works
        // without it.
      });
    return () => {
      cancelled = true;
    };
  }, [accountId, strategyVersion]);

  return { entryTimeframe, confirmationTimeframes };
}

export function TradeDecisionModal({
  trade,
  onClose,
}: {
  trade: TradeHistoryItem;
  onClose: () => void;
}) {
  const { entryTimeframe, confirmationTimeframes } = useDecisionTimeframes(trade.strategy_version);
  const hasDecisionContext =
    trade.reason !== "" ||
    trade.confidence !== null ||
    trade.zone !== null ||
    trade.pattern !== null ||
    trade.structure.length > 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl rounded-md border border-line bg-panel p-4 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-start justify-between gap-4">
          <h3 className="text-sm font-bold">
            Why #{trade.id} — {trade.symbol}{" "}
            <span className={trade.side === "buy" ? "text-ok" : "text-err"}>{trade.side}</span>
          </h3>
          <div className="flex items-center gap-3">
            <button
              type="button"
              className="flex cursor-pointer items-center gap-1 rounded border border-line px-2 py-1 text-xs text-ink-muted hover:text-ink"
              onClick={() => {
                const dataStr =
                  "data:text/json;charset=utf-8," +
                  encodeURIComponent(JSON.stringify(trade, null, 2));
                const node = document.createElement("a");
                node.setAttribute("href", dataStr);
                node.setAttribute("download", `trade_${trade.id}_metadata.json`);
                document.body.appendChild(node);
                node.click();
                node.remove();
              }}
              title="Download all metadata for analysis"
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download JSON
            </button>
            <button
              type="button"
              className="cursor-pointer text-ink-muted hover:text-ink"
              onClick={onClose}
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="mb-3">
          <DecisionChartSnippet tradeId={trade.id} />
        </div>

        {!hasDecisionContext ? (
          <p className="text-sm text-ink-muted">
            Placed manually or via the API — no bot decision to explain.
          </p>
        ) : (
          <dl className="flex flex-col gap-2 text-sm">
            {trade.reason !== "" && (
              <Row label="Reason">
                <span className="whitespace-pre-wrap">{trade.reason}</span>
              </Row>
            )}
            {trade.confidence !== null && (
              <Row label="Confidence">{(trade.confidence * 100).toFixed(0)}%</Row>
            )}
            {trade.indicators.length > 0 && (
              <Row label="Confluence checklist">
                <ul className="flex flex-col gap-0.5">
                  {trade.indicators.map((ind, i) => (
                    <li key={i} className={`text-xs ${ind.passed ? "text-ok" : "text-err"}`}>
                      {ind.passed ? "✓" : "✗"} {ind.name}: {ind.value.toFixed(2)} {ind.comparison}{" "}
                      {ind.threshold.toFixed(2)}
                    </li>
                  ))}
                </ul>
              </Row>
            )}
            <Row label="Strategy">{trade.strategy_version ?? "—"}</Row>
            <Row label="Bot / skill">{trade.skill ?? "Manual"}</Row>
            {entryTimeframe !== null && <Row label="Position taken on">{entryTimeframe}</Row>}
            {confirmationTimeframes.length > 0 && (
              <Row label="Decision confirmed on">{confirmationTimeframes.join(", ")}</Row>
            )}
            {trade.pattern !== null && (
              <Row label="Confirming pattern">{trade.pattern.replace(/_/g, " ")}</Row>
            )}
            {trade.zone !== null && (
              <Row label="Supply/demand zone">
                <span className={trade.zone.kind === "demand" ? "text-ok" : "text-err"}>
                  {trade.zone.kind}
                </span>{" "}
                {trade.zone.price_low.toFixed(5)} – {trade.zone.price_high.toFixed(5)}
                <br />
                <span className="text-xs text-ink-muted">
                  {formatFullTime(trade.zone.time_start)} → {formatFullTime(trade.zone.time_end)}
                </span>
              </Row>
            )}
            {trade.structure.length > 0 && (
              <Row label="Swing structure">
                <ul className="flex flex-col gap-0.5">
                  {trade.structure.map((p, i) => (
                    <li key={i} className="text-xs">
                      <span className="font-medium">{p.label}</span> @ {p.price.toFixed(5)}{" "}
                      <span className="text-ink-muted">({formatFullTime(p.time)})</span>
                    </li>
                  ))}
                </ul>
              </Row>
            )}
          </dl>
        )}
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 border-b border-line pb-2 last:border-0 last:pb-0">
      <dt className="text-xs text-ink-muted">{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}
