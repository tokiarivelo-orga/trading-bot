/** Shared number/time formatting for the analytics tables and charts —
 * kept in one place so "∞ vs null" and "positive vs negative" conventions
 * stay consistent across the symbol table, bot table, and overview strip. */

export function pct(ratio: number): string {
  return `${(ratio * 100).toFixed(1)}%`;
}

export function money(value: number, opts: { sign?: boolean } = {}): string {
  const sign = opts.sign && value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

export function profitFactor(value: number | null): string {
  return value === null ? "∞" : value.toFixed(2);
}

export function duration(seconds: number | null): string {
  if (seconds === null) return "—";
  const abs = Math.abs(seconds);
  if (abs < 60) return `${Math.round(abs)}s`;
  if (abs < 3600) return `${Math.round(abs / 60)}m`;
  if (abs < 86400) return `${(abs / 3600).toFixed(1)}h`;
  return `${(abs / 86400).toFixed(1)}d`;
}

export function timeAgo(epochSeconds: number | null): string {
  if (epochSeconds === null) return "—";
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function plTone(value: number): string {
  return value >= 0 ? "text-ok" : "text-err";
}

/** A price-units figure (slippage, MFE/MAE). Unlike `money`, the scale varies
 * wildly by symbol — 0.20 on XAUUSD is the same kind of number as 0.00020 on
 * EURUSD — so this keeps enough significant digits for both instead of
 * rounding the FX case away to "0.00". */
export function priceDelta(value: number | null, opts: { sign?: boolean } = {}): string {
  if (value === null) return "—";
  const abs = Math.abs(value);
  const digits = abs === 0 ? 2 : abs >= 1 ? 2 : abs >= 0.01 ? 3 : 5;
  const sign = opts.sign && value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

/** Milliseconds, promoted to seconds once the number stops being readable. */
export function millis(value: number | null): string {
  if (value === null) return "—";
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
}

/** Slippage is signed so POSITIVE means the fills cost the trader — the
 * opposite tone convention to P/L, which is exactly why it isn't `plTone`. */
export function slippageTone(value: number | null): string {
  if (value === null) return "text-ink-muted";
  return value > 0 ? "text-err" : "text-ok";
}

/** Cost-as-%-of-gross-edge (OBSERVABILITY_PLAN.md Phase 6): flags a bot
 * spending most or all of its edge on spread + slippage — the plan's
 * headline concern for M1 scalps, whose edge per trade is small relative to
 * what they pay in cost. */
export function costPctTone(value: number | null): string {
  if (value === null) return "text-ink-muted";
  return value >= 0.75 ? "text-err" : "text-ink";
}
