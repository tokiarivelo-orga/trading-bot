"use client";

import { memo } from "react";
import type { BotFunnel, FunnelDrop } from "@/shared/api/client";
import { SIGNAL_OUTCOME_META } from "@/features/backtest/signalOutcome";

/** The funnel stages, in the order the engine actually evaluates its gates:
 * HTF confirmation and the pre-trade risk gate, then the open-position cap /
 * lot sizing, then the broker's spread cap and
 * spread-adjusted risk-reward floor, then the fill. */
const STAGES = [
  {
    key: "fired" as const,
    label: "Fired",
    hint: "Signals the strategy emitted",
  },
  {
    key: "passed_htf" as const,
    label: "Passed HTF",
    hint: "Cleared the higher-timeframe confirmation and the pre-trade risk gate",
  },
  {
    key: "sized_ok" as const,
    label: "Sized OK",
    hint: "Cleared the position cap and produced a tradable lot size",
  },
  {
    key: "passed_spread" as const,
    label: "Passed spread/RR",
    hint: "Cleared the broker spread cap and the spread-adjusted risk-reward floor",
  },
  { key: "filled" as const, label: "Filled", hint: "The broker actually filled it" },
];

function outcomeLabel(outcome: string): string {
  return SIGNAL_OUTCOME_META[outcome as keyof typeof SIGNAL_OUTCOME_META]?.label ?? outcome;
}

function outcomeTone(outcome: string): string {
  return SIGNAL_OUTCOME_META[outcome as keyof typeof SIGNAL_OUTCOME_META]?.className ?? "text-ink";
}

function pctOfFired(count: number, fired: number): string {
  if (fired === 0) return "—";
  return `${Math.round((count / fired) * 100)}%`;
}

const DropRow = memo(function DropRow({ drop, fired }: { drop: FunnelDrop; fired: number }) {
  return (
    <li className="flex items-baseline gap-2 py-0.5">
      <span className="w-10 shrink-0 text-right font-mono text-xs text-ink">{drop.count}</span>
      <span className={`shrink-0 text-xs font-medium ${outcomeTone(drop.outcome)}`}>
        {outcomeLabel(drop.outcome)}
      </span>
      <span className="shrink-0 font-mono text-[10px] text-ink-muted">
        {pctOfFired(drop.count, fired)}
      </span>
      <span className="truncate text-[11px] text-ink-muted" title={drop.example_reason}>
        {drop.example_reason}
      </span>
    </li>
  );
});

const BotFunnelCard = memo(function BotFunnelCard({ funnel }: { funnel: BotFunnel }) {
  const { fired } = funnel;
  return (
    <li className="border-b border-line px-4 py-3 last:border-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-sm font-medium text-ink">{funnel.bot}</h3>
        <p className="text-xs text-ink-muted">
          {funnel.symbols.join(", ")} · {fired} signal{fired === 1 ? "" : "s"} →{" "}
          <span className="text-ok">{funnel.filled} filled</span> ({pctOfFired(funnel.filled, fired)})
        </p>
      </div>

      <div className="mt-2 flex flex-wrap items-stretch gap-1.5">
        {STAGES.map((stage) => {
          const count = funnel[stage.key];
          const width = fired === 0 ? 0 : Math.round((count / fired) * 100);
          return (
            <div
              key={stage.key}
              title={stage.hint}
              className="min-w-[7rem] flex-1 overflow-hidden rounded-md border border-line bg-panel/40"
            >
              <div className="px-2 pt-1.5">
                <p className="truncate text-[10px] uppercase tracking-wide text-ink-muted">
                  {stage.label}
                </p>
                <p className="font-mono text-sm text-ink">
                  {count}
                  <span className="ml-1 text-[10px] text-ink-muted">
                    {pctOfFired(count, fired)}
                  </span>
                </p>
              </div>
              <div className="mt-1.5 h-1 w-full bg-line/40">
                <div
                  className={`h-full ${stage.key === "filled" ? "bg-ok" : "bg-accent"}`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {funnel.drops.length > 0 && (
        <ul className="mt-2">
          {funnel.drops.map((drop) => (
            <DropRow key={`${drop.stage}:${drop.outcome}`} drop={drop} fired={fired} />
          ))}
        </ul>
      )}
    </li>
  );
});

interface Props {
  funnels: BotFunnel[];
  loading: boolean;
  error: string | null;
}

/** Per-bot answer to "of 120 signals, why did only 14 trade?" — each gate's
 * survivor count plus the grouped reasons everything else dropped out
 * (OBSERVABILITY_PLAN.md Phase 2). Fed by the typed `signal_decisions` trail,
 * so a period predating that table is legitimately empty rather than
 * reconstructed from log text. */
export function SignalFunnelPanel({ funnels, loading, error }: Props) {
  if (error) {
    return <p className="p-4 text-sm text-err">{error}</p>;
  }
  if (loading) {
    return <p className="p-4 text-sm text-ink-muted">Loading signal funnel…</p>;
  }
  if (funnels.length === 0) {
    return (
      <p className="p-4 text-sm text-ink-muted">
        No recorded signal decisions in this period. Decisions are recorded from the moment a bot
        fires a signal; periods before this was tracked show nothing here.
      </p>
    );
  }
  return (
    <ul>
      {funnels.map((funnel) => (
        <BotFunnelCard key={funnel.bot} funnel={funnel} />
      ))}
    </ul>
  );
}
