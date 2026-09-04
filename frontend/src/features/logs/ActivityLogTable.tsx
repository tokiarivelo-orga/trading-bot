"use client";

import type { LogEntry } from "@/shared/api/client";

export function ActivityLogTable({
  entries,
  selectedIds,
  onToggleSelect,
  onToggleSelectAll,
  onDeleteOne,
}: {
  entries: LogEntry[];
  /** Selection/delete controls are omitted when these are left unset — used
   * by the read-only backtest replay view (BacktestReportDetail), which has
   * no persisted rows to delete. */
  selectedIds?: Set<number>;
  onToggleSelect?: (id: number) => void;
  onToggleSelectAll?: (checked: boolean) => void;
  onDeleteOne?: (id: number) => void;
}) {
  const selectable = selectedIds !== undefined && onToggleSelect !== undefined;
  const allSelected =
    selectable && entries.length > 0 && entries.every((e) => selectedIds.has(e.id));

  return (
    <div className="overflow-x-auto p-4">
      <table className="w-full min-w-[760px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-muted">
            {selectable && (
              <Th className="w-8">
                <input
                  type="checkbox"
                  aria-label="Select all entries on this page"
                  checked={allSelected}
                  onChange={(e) => onToggleSelectAll?.(e.target.checked)}
                />
              </Th>
            )}
            <Th>Time</Th>
            <Th>Level</Th>
            <Th>Module</Th>
            <Th>Message</Th>
            {onDeleteOne && <Th className="w-16" />}
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.id} className="border-b border-line last:border-0 hover:bg-panel/40">
              {selectable && (
                <Td>
                  <input
                    type="checkbox"
                    aria-label={`Select entry ${e.id}`}
                    checked={selectedIds.has(e.id)}
                    onChange={() => onToggleSelect(e.id)}
                  />
                </Td>
              )}
              <Td className="whitespace-nowrap text-ink-muted">{formatTime(e.created_at)}</Td>
              <Td className={`whitespace-nowrap font-medium ${levelTone(e.level)}`}>{e.level}</Td>
              <Td className="whitespace-nowrap text-ink-muted">{shortLogger(e.logger)}</Td>
              <Td className="font-mono text-xs">{e.message}</Td>
              {onDeleteOne && (
                <Td className="whitespace-nowrap">
                  <button
                    type="button"
                    className="cursor-pointer text-xs text-ink-muted hover:text-err"
                    onClick={() => onDeleteOne(e.id)}
                    title="Delete this entry"
                  >
                    Delete
                  </button>
                </Td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function levelTone(level: string): string {
  if (level === "ERROR" || level === "CRITICAL") return "text-err";
  if (level === "WARNING") return "text-accent";
  return "text-ink-muted";
}

/** "src.engine.application.trade_loop" -> "engine.trade_loop" — drops the
 * "src"/"application"/"adapters"/"api" boilerplate segments shared by every
 * module so the module column stays scannable. */
function shortLogger(logger: string): string {
  const parts = logger
    .split(".")
    .filter((p) => !["src", "application", "adapters", "api", "domain", "ports"].includes(p));
  return parts.join(".");
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function Th({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return <th className={`px-3 py-2 font-medium ${className}`}>{children}</th>;
}

function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2 align-top ${className}`}>{children}</td>;
}
