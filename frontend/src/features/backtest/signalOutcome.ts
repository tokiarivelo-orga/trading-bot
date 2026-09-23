import type { BacktestSignal } from "@/shared/api/client";

/** Display metadata per signal outcome, shared by the report detail page's
 * signals table and the chart's signals dock/markers so the same outcome
 * always reads the same everywhere. `token` is the design-token CSS variable
 * (from globals.css `@theme`) for canvas rendering (chart markers), where
 * Tailwind classes can't reach. */
export const SIGNAL_OUTCOME_META: Record<
  BacktestSignal["outcome"],
  { label: string; className: string; token: string }
> = {
  opened: { label: "opened", className: "text-ok", token: "--color-ok" },
  htf_veto: { label: "HTF veto", className: "text-sell", token: "--color-sell" },
  // Pre-Phase-2 catch-all: every named bucket below used to collapse into
  // this one, and historical rows still carry it, so it must stay.
  risk_rejected: { label: "risk rejected", className: "text-err", token: "--color-err" },
  spread_veto: { label: "spread veto", className: "text-err", token: "--color-err" },
  broker_rejected: { label: "broker rejected", className: "text-err", token: "--color-err" },
  skipped: { label: "skipped", className: "text-ink-muted", token: "--color-ink-muted" },
  // Split out of `risk_rejected` by the veto funnel (OBSERVABILITY_PLAN.md
  // Phase 2). Every consumer indexes this map unguarded, so a new backend
  // outcome MUST land here in the same change or the chart crashes.
  rr_gate: { label: "RR gate", className: "text-err", token: "--color-err" },
  // Legacy: the engine's volatility guard was removed, but rows recorded
  // while it existed still carry this outcome.
  volatility_guard: { label: "volatility guard (legacy)", className: "text-sell", token: "--color-sell" },
  max_positions: { label: "max positions", className: "text-err", token: "--color-err" },
  risk_sizing: { label: "risk sizing", className: "text-err", token: "--color-err" },
  daily_loss_breaker: { label: "circuit breaker", className: "text-err", token: "--color-err" },
};
