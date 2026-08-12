"use client";

import { useState } from "react";
import type { TradeHistoryItem } from "@/shared/api/client";
import { StatusBadge } from "@/features/strategies/StatusBadge";
import { useSortableRows } from "@/shared/hooks/useSortableRows";
import { DecisionBadge } from "@/shared/ui/DecisionBadge";
import { SortTh } from "@/shared/ui/SortTh";
import { TradeDecisionModal } from "@/shared/ui/TradeDecisionModal";
import { type GroupBy, type TradeGroup, groupTrades, outcomeOf } from "./groupTrades";

type TradeSortKey =
  | "symbol"
  | "side"
  | "volume"
  | "open_time"
  | "close_time"
  | "open_price"
  | "close_price"
  | "profit"
  | "strategy_version"
  | "skill"
  | "outcome";

function tradeSortValue(t: TradeHistoryItem, key: TradeSortKey): string | number | null {
  switch (key) {
    case "outcome":
      return outcomeOf(t);
    case "strategy_version":
      return t.strategy_version;
    case "skill":
      return t.skill;
    default:
      return t[key];
  }
}

export function TradeHistoryTable({
  trades,
  groupBy,
  selectedTicket = null,
  onSelectTicket,
}: {
  trades: TradeHistoryItem[];
  groupBy: GroupBy;
  /** Ticket currently highlighted on the chart (see page.tsx's
   * `selectedOrderTicket`) — used to mark the matching row, same convention
   * as AllOrdersPanel's active-orders tables. */
  selectedTicket?: string | number | null;
  /** Called with a row's ticket + symbol when clicked. The caller owns
   * toggling selection off on a repeat click and switching the chart to that
   * symbol if it isn't already on screen. */
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
}) {
  const groups = groupTrades(trades, groupBy);

  if (groupBy === "none") {
    return (
      <div className="overflow-x-auto p-4">
        <TradesTable trades={trades} selectedTicket={selectedTicket} onSelectTicket={onSelectTicket} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      {groups.map((group) => (
        <GroupSection
          key={group.key}
          group={group}
          selectedTicket={selectedTicket}
          onSelectTicket={onSelectTicket}
        />
      ))}
    </div>
  );
}

function GroupSection({
  group,
  selectedTicket = null,
  onSelectTicket,
}: {
  group: TradeGroup;
  selectedTicket?: string | number | null;
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="rounded border border-line">
      <button
        type="button"
        className="flex w-full cursor-pointer items-center justify-between gap-4 bg-panel px-3 py-2 text-left text-sm hover:bg-panel/60"
        onClick={() => setCollapsed((v) => !v)}
      >
        <span className="flex items-center gap-2 font-medium">
          <span className="text-ink-muted">{collapsed ? "▸" : "▾"}</span>
          {group.label}
          <span className="text-xs font-normal text-ink-muted">
            ({group.count} trade{group.count === 1 ? "" : "s"})
          </span>
        </span>
        <span className="flex items-center gap-4 text-xs text-ink-muted">
          {group.winRate !== null && <span>win rate {(group.winRate * 100).toFixed(0)}%</span>}
          <span className={group.netProfit >= 0 ? "text-ok" : "text-err"}>
            net {group.netProfit >= 0 ? "+" : ""}
            {group.netProfit.toFixed(2)}
          </span>
        </span>
      </button>
      {!collapsed && (
        <div className="overflow-x-auto">
          <TradesTable trades={group.trades} selectedTicket={selectedTicket} onSelectTicket={onSelectTicket} />
        </div>
      )}
    </div>
  );
}

function TradesTable({
  trades,
  selectedTicket = null,
  onSelectTicket,
}: {
  trades: TradeHistoryItem[];
  selectedTicket?: string | number | null;
  onSelectTicket?: (ticket: string | number, symbol: string) => void;
}) {
  const { sorted, sort, toggle } = useSortableRows<TradeHistoryItem, TradeSortKey>(
    trades,
    tradeSortValue,
    { key: "open_time", dir: "desc" },
  );
  const [whyTrade, setWhyTrade] = useState<TradeHistoryItem | null>(null);

  if (trades.length === 0) {
    return <p className="px-3 py-2 text-sm text-ink-muted">No trades.</p>;
  }
  return (
    <>
    <table className="w-full min-w-[1020px] border-collapse text-sm">
      <thead>
        <tr className="border-b border-line text-left text-xs text-ink-muted">
          <SortTh className="px-3 py-2 font-medium" label="Symbol" sortKey="symbol" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Side" sortKey="side" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Volume" sortKey="volume" sort={sort} onSort={toggle} align="right" />
          <SortTh className="px-3 py-2 font-medium" label="Opened" sortKey="open_time" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Closed" sortKey="close_time" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Open price" sortKey="open_price" sort={sort} onSort={toggle} align="right" />
          <SortTh className="px-3 py-2 font-medium" label="Close price" sortKey="close_price" sort={sort} onSort={toggle} align="right" />
          <th className="px-3 py-2 font-medium">Close reason</th>
          <SortTh className="px-3 py-2 font-medium" label="P/L" sortKey="profit" sort={sort} onSort={toggle} align="right" />
          <th className="px-3 py-2 font-medium text-right">Max Profit (MFE)</th>
          <th className="px-3 py-2 font-medium">MFE Axis</th>
          <th className="px-3 py-2 font-medium text-right">Max Drawdown (MAE)</th>
          <th className="px-3 py-2 font-medium">MAE Axis</th>
          <SortTh className="px-3 py-2 font-medium" label="Strategy" sortKey="strategy_version" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Skill" sortKey="skill" sort={sort} onSort={toggle} />
          <SortTh className="px-3 py-2 font-medium" label="Outcome" sortKey="outcome" sort={sort} onSort={toggle} />
          <th className="px-3 py-2 font-medium">Why</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((t) => {
          const ticket = isNaN(Number(t.id)) ? t.id : Number(t.id);
          const selected = selectedTicket === ticket;
          return (
          <tr
            key={t.id}
            onClick={() => onSelectTicket?.(ticket, t.symbol)}
            className={`cursor-pointer border-b border-line last:border-0 ${
              selected ? "bg-accent/10 ring-1 ring-inset ring-accent" : "hover:bg-panel/40"
            }`}
            title={`Highlight #${t.id} on the chart`}
          >
            <Td className={getBannerClass(t)}>{t.symbol}</Td>
            <Td className={t.side === "buy" ? "text-ok" : "text-err"}>{t.side}</Td>
            <Td align="right">{t.volume}</Td>
            <Td>{formatTime(t.open_time)}</Td>
            <Td>{t.close_time !== null ? formatTime(t.close_time) : "—"}</Td>
            <Td align="right">{t.open_price}</Td>
            <Td align="right">{t.close_price ?? "—"}</Td>
            <Td className="text-ink-muted">{t.close_reason ?? "—"}</Td>
            <Td align="right" className={plTone(t.profit)}>
              {t.profit != null ? t.profit.toFixed(2) : "—"}
            </Td>
            <Td align="right" className="text-ok">
              {t.mfe != null ? t.mfe.toFixed(5) : "—"}
            </Td>
            <Td className="text-ink-muted text-xs">
              {t.mfe_time != null ? formatTime(t.mfe_time) : "—"}
            </Td>
            <Td align="right" className="text-err">
              {t.mae != null ? t.mae.toFixed(5) : "—"}
            </Td>
            <Td className="text-ink-muted text-xs">
              {t.mae_time != null ? formatTime(t.mae_time) : "—"}
            </Td>
            <Td className="text-ink-muted">{t.strategy_version ?? "—"}</Td>
            <Td className="text-ink-muted">{t.skill ?? "—"}</Td>
            <Td>
              <StatusBadge status={outcomeOf(t)} />
            </Td>
            <Td align="right">
              <DecisionBadge trade={t} onClick={() => setWhyTrade(t)} />
            </Td>
          </tr>
          );
        })}
      </tbody>
    </table>
    {whyTrade && <TradeDecisionModal trade={whyTrade} onClose={() => setWhyTrade(null)} />}
    </>
  );
}

function plTone(profit: number | null): string {
  if (profit === null) return "";
  return profit >= 0 ? "text-ok" : "text-err";
}

function getBannerClass(t: TradeHistoryItem): string {
  if (t.profit === null) return "border-l-4 border-l-transparent";
  if (t.profit < 0) return "border-l-4 border-l-err";

  let isSl = false;
  const comment = (t.comment || "").toLowerCase();
  const reason = (t.close_reason || "").toLowerCase();
  
  if (comment.includes("[sl]") || reason.includes("sl")) {
    isSl = true;
  } else if (comment.includes("[tp]") || reason.includes("tp")) {
    isSl = false;
  } else if (t.close_price !== null && t.sl !== null) {
    if (t.tp !== null) {
      const distTp = Math.abs(t.close_price - t.tp);
      const distSl = Math.abs(t.close_price - t.sl);
      if (distSl < distTp) isSl = true;
    } else {
      const distSl = Math.abs(t.close_price - t.sl);
      const tradeSize = Math.abs(t.open_price - t.close_price);
      if (tradeSize > 0 && distSl / tradeSize < 0.2) isSl = true;
    }
  }

  return isSl ? "border-l-4 border-l-[#ff9800]" : "border-l-4 border-l-ok";
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toISOString().replace("T", " ").slice(0, 16);
}

function Td({
  children,
  align = "left",
  className = "",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
}) {
  return (
    <td className={`px-3 py-2 ${align === "right" ? "text-right" : ""} ${className}`}>{children}</td>
  );
}
