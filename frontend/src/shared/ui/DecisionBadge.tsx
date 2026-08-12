import type { TradeHistoryItem } from "@/shared/api/client";

export interface DecisionBadgeTrade
  extends Pick<TradeHistoryItem, "indicators" | "zone" | "pattern" | "structure" | "reason" | "confidence"> {}

interface DecisionBadgeProps {
  trade: DecisionBadgeTrade;
  onClick?: () => void;
}

/** Small glanceable pill hinting at *how* a trade was decided (confluence
 * score, zone kind, pattern, or structure point), shown in the "Why" column
 * of `AllOrdersPanel` and `TradeHistoryTable`. Clicking it opens
 * `TradeDecisionModal` for the full breakdown. Falls back to a plain "—"
 * when the trade has no decision context (manual/API trades). */
export function DecisionBadge({ trade, onClick }: DecisionBadgeProps) {
  const { indicators, zone, pattern, structure, reason, confidence } = trade;

  let label: string;
  let tone: string;

  if (indicators.length > 0) {
    const passedCount = indicators.filter((i) => i.passed).length;
    label = `${passedCount}/${indicators.length} confluence`;
    tone =
      passedCount === indicators.length
        ? "border-ok text-ok"
        : passedCount === 0
          ? "border-err text-err"
          : "border-ink-muted text-ink-muted";
  } else if (zone !== null) {
    label = zone.kind === "demand" ? "Demand zone" : "Supply zone";
    tone = zone.kind === "demand" ? "border-ok text-ok" : "border-err text-err";
  } else if (pattern !== null) {
    label = pattern.replace(/_/g, " ");
    tone = "border-ink-muted text-ink-muted";
  } else if (structure.length > 0) {
    label = `${structure[structure.length - 1].label} breakout`;
    tone = "border-ink-muted text-ink-muted";
  } else if (reason && reason.startsWith("DL ")) {
    label = `AI Conf: ${confidence ? Math.round(confidence * 100) : '?'}%`;
    tone = "border-accent text-accent shadow-[0_0_8px_rgba(var(--accent-rgb),0.3)]";
  } else if (reason) {
    label = "View reason";
    tone = "border-ink-muted text-ink-muted";
  } else {
    return <span className="cursor-default text-ink-muted">—</span>;
  }

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      className="cursor-pointer"
      title="Why the bot took this trade"
    >
      <span className={`rounded-full border px-2 py-0.5 text-xs whitespace-nowrap ${tone}`}>
        {label}
      </span>
    </button>
  );
}
