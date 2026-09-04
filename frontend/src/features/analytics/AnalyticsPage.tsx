"use client";

import { useEffect, useMemo, useState } from "react";
import { MenuButton } from "@/shared/ui/NavigationDrawer";
import { AnalyticsFilters } from "./AnalyticsFilters";
import { AnalyticsOverview } from "./AnalyticsOverview";
import { BotDrawdownChart } from "./BotDrawdownChart";
import { BotEquityChart, seriesColor } from "./BotEquityChart";
import { BotLegend } from "./BotLegend";
import { BotPerformanceTable } from "./BotPerformanceTable";
import { DailyPnlPanel } from "./DailyPnlPanel";
import { RegimeAnalyticsPanel } from "./RegimeAnalyticsPanel";
import { SignalFunnelPanel } from "./SignalFunnelPanel";
import { SymbolAnalyticsTable } from "./SymbolAnalyticsTable";
import { TradePnLHistogram } from "./TradePnLHistogram";
import { useAnalytics } from "./useAnalytics";
import { useDailyPnl } from "./useDailyPnl";
import { useRegimeAnalytics } from "./useRegimeAnalytics";
import { useSignalFunnel } from "./useSignalFunnel";
import { useAnalyticsExport } from "./useAnalyticsExport";
import { downloadCsv, downloadJson } from "@/shared/utils/download";

const MAX_CHARTED_BOTS = 6;
const DEFAULT_CHARTED_BOTS = 3;

const QUERY_KEY_SYMBOLS = "symbols";
const QUERY_KEY_BOTS = "bots";
const QUERY_KEY_CHART = "chart";
const QUERY_KEY_FROM = "from";
const QUERY_KEY_TO = "to";
const LS_KEY_SYMBOLS = "tb.analytics.symbols";
const LS_KEY_BOTS = "tb.analytics.bots";
const LS_KEY_CHART = "tb.analytics.chart";
const LS_KEY_FROM = "tb.analytics.from";
const LS_KEY_TO = "tb.analytics.to";

// Symbol/bot filters: empty set already means "everything," so the query
// param and localStorage entry are simply omitted/absent in that case —
// no unset-vs-empty distinction needed.
function loadFilterSet(queryKey: string, lsKey: string): Set<string> {
  try {
    const raw = new URLSearchParams(window.location.search).get(queryKey);
    if (raw !== null) return raw === "" ? new Set() : new Set(raw.split(","));
    const stored = localStorage.getItem(lsKey);
    if (!stored) return new Set();
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? new Set(parsed) : new Set();
  } catch {
    return new Set();
  }
}

function loadFilterString(queryKey: string, lsKey: string): string {
  try {
    const raw = new URLSearchParams(window.location.search).get(queryKey);
    if (raw !== null) return raw;
    return localStorage.getItem(lsKey) ?? "";
  } catch {
    return "";
  }
}

// Chart selection is different: `null` (never touched → default top-N bots)
// is a distinct state from an explicit empty set (user unchecked them all),
// so the query param's mere presence — not just its contents — matters.
function loadChartSelection(): Set<string> | null {
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.has(QUERY_KEY_CHART)) {
      const raw = params.get(QUERY_KEY_CHART) ?? "";
      return raw === "" ? new Set() : new Set(raw.split(","));
    }
    const stored = localStorage.getItem(LS_KEY_CHART);
    if (stored === null) return null;
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? new Set(parsed) : null;
  } catch {
    return null;
  }
}

export function AnalyticsPage() {
  const [dateFrom, setDateFrom] = useState<string>(() =>
    loadFilterString(QUERY_KEY_FROM, LS_KEY_FROM),
  );
  const [dateTo, setDateTo] = useState<string>(() =>
    loadFilterString(QUERY_KEY_TO, LS_KEY_TO),
  );

  const apiFilters = useMemo(
    () => ({
      open_from: dateFrom ? Math.floor(Date.parse(`${dateFrom}T00:00:00`) / 1000) : undefined,
      open_to: dateTo ? Math.floor(Date.parse(`${dateTo}T23:59:59`) / 1000) : undefined,
    }),
    [dateFrom, dateTo],
  );

  const { symbols, bots, loading, error, refresh } = useAnalytics(apiFilters);
  // Same window as the tables above, but a different source: the typed
  // signal-decision trail rather than the journal, so this answers "why
  // didn't it trade" where everything else answers "how did the trades go".
  const {
    funnels,
    loading: funnelLoading,
    error: funnelError,
  } = useSignalFunnel(apiFilters.open_from, apiFilters.open_to);
  // Same window again, but sliced by market regime at entry rather than by
  // bot alone (OBSERVABILITY_PLAN.md Phase 6) — a breakdown of the same
  // underlying trade table `useAnalytics` above already loaded.
  const {
    regimes,
    loading: regimeLoading,
    error: regimeError,
  } = useRegimeAnalytics(apiFilters.open_from, apiFilters.open_to);
  // Same window again, at daily granularity rather than per-bot (Phase 7) —
  // realized P&L grouped by calendar date, tagged with whether a HIGH-impact
  // news event fell on that day.
  const {
    rows: dailyPnlRows,
    loading: dailyPnlLoading,
    error: dailyPnlError,
  } = useDailyPnl(apiFilters.open_from, apiFilters.open_to);
  const [selected, setSelected] = useState<Set<string> | null>(loadChartSelection);
  const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(() =>
    loadFilterSet(QUERY_KEY_SYMBOLS, LS_KEY_SYMBOLS),
  );
  const [selectedBots, setSelectedBots] = useState<Set<string>>(() =>
    loadFilterSet(QUERY_KEY_BOTS, LS_KEY_BOTS),
  );

  // Keep the URL and localStorage in sync with filters/chart selection —
  // the URL makes the current view shareable/bookmarkable, localStorage
  // means a plain refresh (or a nav back into /analytics without params)
  // restores the last state instead of resetting to defaults.
  useEffect(() => {
    try {
      const url = new URL(window.location.href);
      if (selectedSymbols.size > 0) {
        url.searchParams.set(QUERY_KEY_SYMBOLS, [...selectedSymbols].join(","));
      } else {
        url.searchParams.delete(QUERY_KEY_SYMBOLS);
      }
      if (selectedBots.size > 0) {
        url.searchParams.set(QUERY_KEY_BOTS, [...selectedBots].join(","));
      } else {
        url.searchParams.delete(QUERY_KEY_BOTS);
      }
      if (selected === null) {
        url.searchParams.delete(QUERY_KEY_CHART);
      } else {
        url.searchParams.set(QUERY_KEY_CHART, [...selected].join(","));
      }
      if (dateFrom) {
        url.searchParams.set(QUERY_KEY_FROM, dateFrom);
        localStorage.setItem(LS_KEY_FROM, dateFrom);
      } else {
        url.searchParams.delete(QUERY_KEY_FROM);
        localStorage.removeItem(LS_KEY_FROM);
      }
      if (dateTo) {
        url.searchParams.set(QUERY_KEY_TO, dateTo);
        localStorage.setItem(LS_KEY_TO, dateTo);
      } else {
        url.searchParams.delete(QUERY_KEY_TO);
        localStorage.removeItem(LS_KEY_TO);
      }
      window.history.replaceState(null, "", url);

      localStorage.setItem(LS_KEY_SYMBOLS, JSON.stringify([...selectedSymbols]));
      localStorage.setItem(LS_KEY_BOTS, JSON.stringify([...selectedBots]));
      localStorage.setItem(LS_KEY_CHART, JSON.stringify(selected === null ? null : [...selected]));
    } catch {
      // Ignore URL/storage errors during SSR or edge environments.
    }
  }, [selectedSymbols, selectedBots, selected, dateFrom, dateTo]);

  const availableSymbols = useMemo(
    () =>
      Array.from(new Set([...symbols.map((s) => s.symbol), ...bots.map((b) => b.symbol)])).sort(),
    [symbols, bots],
  );

  // Symbol filter narrows which bots are even offered in the bot filter —
  // picking XAUUSD hides bots that only trade BTCUSD, etc.
  const symbolFilteredBots = useMemo(
    () => (selectedSymbols.size === 0 ? bots : bots.filter((b) => selectedSymbols.has(b.symbol))),
    [bots, selectedSymbols],
  );

  const availableBots = useMemo(
    () => symbolFilteredBots.map((b) => ({ skill: b.skill, bot_name: b.bot_name, symbol: b.symbol })),
    [symbolFilteredBots],
  );

  const filteredBots = useMemo(
    () =>
      selectedBots.size === 0
        ? symbolFilteredBots
        : symbolFilteredBots.filter((b) => selectedBots.has(b.skill)),
    [symbolFilteredBots, selectedBots],
  );

  const filteredSymbols = useMemo(
    () => (selectedSymbols.size === 0 ? symbols : symbols.filter((s) => selectedSymbols.has(s.symbol))),
    [symbols, selectedSymbols],
  );

  function toggleSymbol(symbol: string) {
    const next = new Set(selectedSymbols);
    if (next.has(symbol)) {
      next.delete(symbol);
    } else {
      next.add(symbol);
    }
    // A bot no longer offered by the new symbol selection shouldn't stay
    // silently pinned in the bot filter.
    setSelectedBots((prevBots) => {
      const stillAvailable = new Set(
        (next.size === 0 ? bots : bots.filter((b) => next.has(b.symbol))).map((b) => b.skill),
      );
      const pruned = new Set([...prevBots].filter((skill) => stillAvailable.has(skill)));
      return pruned.size === prevBots.size ? prevBots : pruned;
    });
    setSelectedSymbols(next);
  }

  function toggleBot(skill: string) {
    const next = new Set(selectedBots);
    if (next.has(skill)) {
      next.delete(skill);
    } else {
      next.add(skill);
    }
    setSelectedBots(next);
  }

  function clearFilters() {
    setSelectedSymbols(new Set());
    setSelectedBots(new Set());
    setDateFrom("");
    setDateTo("");
  }

  // Ranked (for stable chart color assignment) within the filtered set —
  // filters are an explicit user action, so it's fine for colors to shift
  // when the filter changes, unlike the chart checkboxes below.
  const rankedBots = useMemo(
    () => filteredBots.map((b, i) => ({ ...b, color: seriesColor(i) })),
    [filteredBots],
  );

  // A charted skill that the current symbol/bot filters exclude isn't drawn
  // anywhere, so it must not eat one of the MAX_CHARTED_BOTS slots — otherwise
  // stale selections restored from the URL/localStorage silently make every
  // remaining checkbox a no-op. Skip pruning while bots are still loading so a
  // transiently empty list doesn't wipe the selection.
  const activeSelection = useMemo(() => {
    if (selected === null) {
      return new Set(rankedBots.slice(0, DEFAULT_CHARTED_BOTS).map((b) => b.skill));
    }
    if (rankedBots.length === 0) return selected;
    const visible = new Set(rankedBots.map((b) => b.skill));
    return new Set([...selected].filter((skill) => visible.has(skill)));
  }, [selected, rankedBots]);

  const chartAtCapacity = activeSelection.size >= MAX_CHARTED_BOTS;

  function toggle(skill: string) {
    const next = new Set(activeSelection);
    if (next.has(skill)) {
      next.delete(skill);
    } else {
      if (next.size >= MAX_CHARTED_BOTS) return;
      next.add(skill);
    }
    setSelected(next);
  }

  const chartedBots = rankedBots.filter((b) => activeSelection.has(b.skill));

  // Trivial synchronous export of the already-loaded daily-P&L rows — unlike
  // useAnalyticsExport.ts's builder (N paginated /journal/history fetches
  // per bot), there's nothing async to await here, so no dedicated hook.
  function exportDailyPnlJson() {
    downloadJson(dailyPnlRows, `daily_pnl_${new Date().toISOString().slice(0, 10)}.json`);
  }
  function exportDailyPnlCsv() {
    downloadCsv(
      dailyPnlRows.map((r) => ({ ...r, highImpactEvents: r.highImpactEvents.join("; ") })),
      `daily_pnl_${new Date().toISOString().slice(0, 10)}.csv`,
    );
  }

  const {
    exportJson,
    exportCsv,
    exporting,
    error: exportError,
    disabled: exportDisabled,
  } = useAnalyticsExport(
    filteredSymbols,
    filteredBots,
    [...selectedSymbols],
    [...selectedBots],
    dateFrom,
    dateTo,
    apiFilters.open_from,
    apiFilters.open_to,
  );

  return (
    <div className="flex h-screen flex-col bg-bg text-ink">
      <header className="flex items-center gap-3 border-b border-line px-6 py-3 bg-panel/30 backdrop-blur-md">
        <MenuButton />
        <div className="flex flex-col">
          <h1 className="text-lg font-bold tracking-wide text-ink">Analytics</h1>
          <p className="text-3xs font-semibold uppercase tracking-wider text-ink-muted">
            Symbols &amp; Bot Performance
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={exportJson}
            disabled={exportDisabled || exporting}
            title={
              exportDisabled
                ? "No bots in the current filter to export"
                : "Export filtered bots + full trade history as JSON"
            }
            className="cursor-pointer rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs font-semibold text-ink-muted transition-all duration-200 hover:bg-bg hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            {exporting ? "Exporting…" : "Export JSON"}
          </button>
          <button
            onClick={exportCsv}
            disabled={exportDisabled || exporting}
            title={
              exportDisabled
                ? "No bots in the current filter to export"
                : "Export filtered bots' trades as a flat CSV, for AI analysis"
            }
            className="cursor-pointer rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs font-semibold text-ink-muted transition-all duration-200 hover:bg-bg hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            {exporting ? "Exporting…" : "Export CSV"}
          </button>
          <button
            onClick={refresh}
            title="Refresh data"
            className="cursor-pointer rounded-lg border border-line bg-panel p-1.5 text-ink-muted transition-all duration-200 hover:bg-bg hover:text-ink"
          >
            <svg className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.228 10H18.228" />
            </svg>
          </button>
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto p-6 space-y-6">
        {error && <p className="text-sm text-err">{error}</p>}
        {exportError && <p className="text-sm text-err">{exportError}</p>}
        {loading && symbols.length === 0 && bots.length === 0 ? (
          <p className="text-sm text-ink-muted">Loading analytics…</p>
        ) : (
          <>
            <AnalyticsOverview symbols={filteredSymbols} bots={filteredBots} />

            <AnalyticsFilters
              availableSymbols={availableSymbols}
              selectedSymbols={selectedSymbols}
              onToggleSymbol={toggleSymbol}
              availableBots={availableBots}
              selectedBots={selectedBots}
              onToggleBot={toggleBot}
              dateFrom={dateFrom}
              onDateFromChange={setDateFrom}
              dateTo={dateTo}
              onDateToChange={setDateTo}
              onClear={clearFilters}
            />

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="border-b border-line px-4 py-2.5">
                <h2 className="text-sm font-bold text-ink">Bot comparison charts</h2>
                <p className="text-xs text-ink-muted">
                  Check bots in the table below to overlay them here — up to {MAX_CHARTED_BOTS} at once, in a
                  fixed color per bot across all three charts. Use the filters above to narrow which bots are
                  even offered.
                </p>
              </header>
              <BotLegend bots={chartedBots} />

              <div className="border-b border-line px-4 pt-3 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Equity curve — cumulative realized profit
              </div>
              <BotEquityChart bots={chartedBots} />

              <div className="border-y border-line px-4 pt-3 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Trade P/L — per-trade wins/losses over time
              </div>
              <TradePnLHistogram bots={chartedBots} />

              <div className="border-y border-line px-4 pt-3 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Drawdown — distance below running peak
              </div>
              <BotDrawdownChart bots={chartedBots} />
            </section>

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="border-b border-line px-4 py-2.5">
                <h2 className="text-sm font-bold text-ink">Bot performance</h2>
                <p className="text-xs text-ink-muted">
                  Every bot ranked by realized profit — check the box to overlay its equity curve above.
                </p>
              </header>
              <BotPerformanceTable
                bots={filteredBots}
                selected={activeSelection}
                onToggle={toggle}
                atCapacity={chartAtCapacity}
                maxCharted={MAX_CHARTED_BOTS}
              />
            </section>

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-2.5">
                <div>
                  <h2 className="text-sm font-bold text-ink">Daily P&amp;L</h2>
                  <p className="text-xs text-ink-muted">
                    Realized profit grouped by calendar day, across every bot and symbol in the
                    current filter. Days marked with a newspaper icon had a HIGH-impact economic
                    event release.
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    onClick={exportDailyPnlJson}
                    disabled={dailyPnlRows.length === 0}
                    title="Export daily P&L as JSON"
                    className="cursor-pointer rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs font-semibold text-ink-muted transition-all duration-200 hover:bg-bg hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Export JSON
                  </button>
                  <button
                    onClick={exportDailyPnlCsv}
                    disabled={dailyPnlRows.length === 0}
                    title="Export daily P&L as CSV"
                    className="cursor-pointer rounded-lg border border-line bg-panel px-2.5 py-1.5 text-xs font-semibold text-ink-muted transition-all duration-200 hover:bg-bg hover:text-ink disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Export CSV
                  </button>
                </div>
              </header>
              <DailyPnlPanel rows={dailyPnlRows} loading={dailyPnlLoading} error={dailyPnlError} />
            </section>

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="border-b border-line px-4 py-2.5">
                <h2 className="text-sm font-bold text-ink">Performance by market regime</h2>
                <p className="text-xs text-ink-muted">
                  Each bot&apos;s win rate, profit factor, and expectancy split by the volatility,
                  trend, or trading session it traded in at entry — is this bot&apos;s edge
                  regime-dependent?
                </p>
              </header>
              <RegimeAnalyticsPanel regimes={regimes} loading={regimeLoading} error={regimeError} />
            </section>

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="border-b border-line px-4 py-2.5">
                <h2 className="text-sm font-bold text-ink">Symbols traded</h2>
                <p className="text-xs text-ink-muted">
                  Every trade on each symbol, any bot or manual — which instruments are actually profitable.
                </p>
              </header>
              <SymbolAnalyticsTable symbols={filteredSymbols} />
            </section>

            <section className="rounded-xl border border-line bg-panel/30 shadow-inner overflow-hidden">
              <header className="border-b border-line px-4 py-2.5">
                <h2 className="text-sm font-bold text-ink">Signal funnel</h2>
                <p className="text-xs text-ink-muted">
                  Of every signal each bot fired, how many survived each gate — and why the rest
                  never became trades.
                </p>
              </header>
              <SignalFunnelPanel funnels={funnels} loading={funnelLoading} error={funnelError} />
            </section>
          </>
        )}
      </main>
    </div>
  );
}
