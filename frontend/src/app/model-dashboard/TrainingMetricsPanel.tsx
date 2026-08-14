"use client";

/**
 * Surfaces the `model-training` API (`backend/src/strategies/api/
 * routes_training.py`) — a well-typed backend that, before this panel, no
 * frontend code consumed at all: training-run history, each trained model's
 * honest out-of-sample verdict, and per-fold walk-forward economics — plus
 * download triggers for the two new bulk-export endpoints
 * (`market-data/candles/export`, `order-book/export`) and the corrected
 * "Export Dataset" trigger for `journal/export/dataset`.
 *
 * Uses TanStack Query directly (this page's surrounding components still
 * poll by hand — see DashboardClient.tsx — but new code here follows the
 * app-wide convention documented in `shared/api/queryKeys.ts`).
 */

import { useMemo, useState } from "react";
import { Download, FlaskConical, Loader2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import {
  getTrainingRuns,
  getTrainedModels,
  getWalkForward,
  type TrainedModel,
} from "@/shared/api/client";
import { useActiveAccount } from "@/shared/api/account-context";
import { queryKeys } from "@/shared/api/queryKeys";
import { downloadCsv, downloadFileFromApi, downloadJson } from "@/shared/utils/download";

const TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"] as const;

const cardCls = "bg-panel/40 backdrop-blur-xl rounded-2xl border border-line/50 shadow-xl overflow-hidden";
const headerCls = "p-5 border-b border-line/50 bg-panel/80 flex items-center justify-between gap-4 flex-wrap";
const thCls = "px-4 py-3 font-semibold text-ink-muted uppercase tracking-wider text-xs text-left";
const tdCls = "px-4 py-3 text-ink";
const inputCls =
  "bg-background/80 border border-line rounded-lg px-3 py-1.5 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/50";
const btnCls =
  "flex items-center gap-1.5 px-3 py-1.5 bg-accent/20 hover:bg-accent/30 text-accent border border-accent/40 rounded-lg text-xs font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed";

function fmtNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function fmtDate(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

function toEpochSeconds(dateStr: string): number | null {
  if (!dateStr) return null;
  const ms = new Date(dateStr).getTime();
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

export default function TrainingMetricsPanel({ symbol }: { symbol: string }) {
  const accountId = useActiveAccount();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const runsQuery = useQuery({
    queryKey: queryKeys.modelTraining.runs(),
    queryFn: getTrainingRuns,
  });
  const modelsQuery = useQuery({
    queryKey: queryKeys.modelTraining.models(),
    queryFn: getTrainedModels,
  });

  const models: TrainedModel[] = modelsQuery.data ?? [];
  const effectiveModel = selectedModel ?? models[0]?.model_name ?? null;

  const walkForwardQuery = useQuery({
    queryKey: queryKeys.modelTraining.walkForward(effectiveModel ?? ""),
    queryFn: () => getWalkForward(effectiveModel as string),
    enabled: effectiveModel !== null,
  });

  const loading = runsQuery.isLoading || modelsQuery.isLoading;
  const error = runsQuery.isError || modelsQuery.isError;

  const exportPayload = useMemo(
    () => ({
      runs: runsQuery.data ?? [],
      models: modelsQuery.data ?? [],
      walk_forward: effectiveModel ? { model_name: effectiveModel, folds: walkForwardQuery.data ?? [] } : null,
    }),
    [runsQuery.data, modelsQuery.data, effectiveModel, walkForwardQuery.data],
  );

  const handleExportJson = () => {
    downloadJson(exportPayload, `model_training_metrics_${new Date().toISOString().slice(0, 10)}.json`);
  };

  const handleExportCsv = () => {
    // Flatten to one row per model — the metric this panel is really about —
    // with its walk-forward folds appended as JSON-string cells (folds are a
    // nested list per model, not tabular on their own).
    const rows = models.map((m) => ({
      model_name: m.model_name,
      symbol: m.symbol,
      trained_at: m.trained_at,
      n_bars: m.n_bars,
      feature_count: m.feature_count,
      folds: m.summary.folds,
      is_avg_r_mean: m.summary.is_avg_r_mean,
      oos_avg_r_mean: m.summary.oos_avg_r_mean,
      oos_avg_r_min: m.summary.oos_avg_r_min,
      oos_pf_mean: m.summary.oos_pf_mean,
      oos_trades_total: m.summary.oos_trades_total,
      folds_positive_oos: m.summary.folds_positive_oos,
      brier_secure_is_mean: m.summary.brier_secure_is_mean,
      brier_secure_oos_mean: m.summary.brier_secure_oos_mean,
      threshold_tunable: m.summary.threshold_tunable,
    }));
    downloadCsv(rows, `model_training_metrics_${new Date().toISOString().slice(0, 10)}.csv`);
  };

  return (
    <div className="space-y-6">
      <div className={cardCls}>
        <div className={headerCls}>
          <h2 className="font-bold text-lg text-ink flex items-center gap-2">
            <FlaskConical size={18} className="text-accent" />
            Training Metrics
          </h2>
          <div className="flex items-center gap-2">
            <button onClick={handleExportJson} className={btnCls} disabled={loading || models.length === 0}>
              <Download size={14} /> Export JSON
            </button>
            <button onClick={handleExportCsv} className={btnCls} disabled={loading || models.length === 0}>
              <Download size={14} /> Export CSV
            </button>
          </div>
        </div>

        {loading ? (
          <div className="p-8 flex items-center justify-center text-ink-muted gap-2 text-sm">
            <Loader2 size={16} className="animate-spin" /> Loading training metrics…
          </div>
        ) : error ? (
          <div className="p-6 text-error text-sm">Failed to load model-training data.</div>
        ) : (
          <div className="p-5 space-y-8">
            {/* Training run history */}
            <section>
              <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider mb-3">
                Training Runs
              </h3>
              <div className="overflow-x-auto rounded-lg border border-line/50">
                <table className="w-full text-sm">
                  <thead className="bg-background/50">
                    <tr>
                      <th className={thCls}>Symbol</th>
                      <th className={thCls}>Model</th>
                      <th className={thCls}>State</th>
                      <th className={thCls}>Started</th>
                      <th className={thCls}>Finished</th>
                      <th className={thCls}>Exit Code</th>
                      <th className={thCls}>Error</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line/30">
                    {(runsQuery.data ?? []).map((run) => (
                      <tr key={run.id}>
                        <td className={tdCls}>{run.symbol}</td>
                        <td className={tdCls}>{run.model_name}</td>
                        <td className={tdCls}>
                          <span
                            className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                              run.state === "succeeded"
                                ? "bg-ok/10 text-ok"
                                : run.state === "failed"
                                  ? "bg-error/10 text-error"
                                  : "bg-accent/10 text-accent"
                            }`}
                          >
                            {run.state}
                          </span>
                        </td>
                        <td className={`${tdCls} font-mono text-xs`}>{fmtDate(run.started_at)}</td>
                        <td className={`${tdCls} font-mono text-xs`}>{fmtDate(run.finished_at)}</td>
                        <td className={tdCls}>{run.exit_code ?? "—"}</td>
                        <td className={`${tdCls} text-error text-xs max-w-xs truncate`} title={run.error ?? undefined}>
                          {run.error ?? "—"}
                        </td>
                      </tr>
                    ))}
                    {(runsQuery.data ?? []).length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-4 py-6 text-center text-ink-muted text-sm">
                          No training runs recorded in this backend process yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Trained models */}
            <section>
              <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider mb-3">
                Trained Models (Out-of-Sample Verdict)
              </h3>
              <div className="overflow-x-auto rounded-lg border border-line/50">
                <table className="w-full text-sm whitespace-nowrap">
                  <thead className="bg-background/50">
                    <tr>
                      <th className={thCls}>Model</th>
                      <th className={thCls}>Symbol</th>
                      <th className={thCls}>Trained At</th>
                      <th className={thCls}>Bars</th>
                      <th className={thCls}>Features</th>
                      <th className={thCls}>Folds</th>
                      <th className={thCls}>IS Avg R</th>
                      <th className={thCls}>OOS Avg R</th>
                      <th className={thCls}>OOS Avg R (min fold)</th>
                      <th className={thCls}>OOS PF</th>
                      <th className={thCls}>OOS Trades</th>
                      <th className={thCls}>Folds Positive OOS</th>
                      <th className={thCls}>Brier IS</th>
                      <th className={thCls}>Brier OOS</th>
                      <th className={thCls}>Threshold Tunable</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line/30">
                    {models.map((m) => (
                      <tr
                        key={m.model_name}
                        onClick={() => setSelectedModel(m.model_name)}
                        className={`cursor-pointer transition-colors ${
                          effectiveModel === m.model_name ? "bg-accent/10" : "hover:bg-line/20"
                        }`}
                      >
                        <td className={`${tdCls} font-semibold`}>{m.model_name}</td>
                        <td className={tdCls}>{m.symbol}</td>
                        <td className={`${tdCls} font-mono text-xs`}>{fmtDate(m.trained_at)}</td>
                        <td className={tdCls}>{m.n_bars.toLocaleString()}</td>
                        <td className={tdCls}>{m.feature_count}</td>
                        <td className={tdCls}>{m.summary.folds}</td>
                        <td className={tdCls}>{fmtNum(m.summary.is_avg_r_mean)}</td>
                        <td
                          className={`${tdCls} font-bold ${
                            (m.summary.oos_avg_r_mean ?? 0) >= 0 ? "text-ok" : "text-error"
                          }`}
                        >
                          {fmtNum(m.summary.oos_avg_r_mean)}
                        </td>
                        <td className={tdCls}>{fmtNum(m.summary.oos_avg_r_min)}</td>
                        <td className={tdCls}>{fmtNum(m.summary.oos_pf_mean, 2)}</td>
                        <td className={tdCls}>{m.summary.oos_trades_total ?? "—"}</td>
                        <td className={tdCls}>
                          {m.summary.folds_positive_oos ?? "—"}
                          {m.summary.folds ? ` / ${m.summary.folds}` : ""}
                        </td>
                        <td className={tdCls}>{fmtNum(m.summary.brier_secure_is_mean)}</td>
                        <td className={tdCls}>{fmtNum(m.summary.brier_secure_oos_mean)}</td>
                        <td className={tdCls}>{m.summary.threshold_tunable ?? "—"}</td>
                      </tr>
                    ))}
                    {models.length === 0 && (
                      <tr>
                        <td colSpan={14} className="px-4 py-6 text-center text-ink-muted text-sm">
                          No trained models on disk.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              {models.length > 0 && (
                <p className="text-xs text-ink-muted mt-2">
                  Click a row to load its walk-forward fold detail below.
                </p>
              )}
            </section>

            {/* Walk-forward folds for the selected model */}
            {effectiveModel && (
              <section>
                <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider mb-3">
                  Walk-Forward Folds — {effectiveModel}
                </h3>
                <div className="overflow-x-auto rounded-lg border border-line/50">
                  <table className="w-full text-sm whitespace-nowrap">
                    <thead className="bg-background/50">
                      <tr>
                        <th className={thCls}>Fold</th>
                        <th className={thCls}>OOS Start</th>
                        <th className={thCls}>OOS End</th>
                        <th className={thCls}>Temp.</th>
                        <th className={thCls}>Brier IS</th>
                        <th className={thCls}>Brier OOS</th>
                        <th className={thCls}>Revert Acc OOS</th>
                        <th className={thCls}>Base Rate Secure OOS</th>
                        <th className={thCls}>IS Trades</th>
                        <th className={thCls}>IS Sum R</th>
                        <th className={thCls}>IS Avg R</th>
                        <th className={thCls}>IS PF</th>
                        <th className={thCls}>IS Max DD (R)</th>
                        <th className={thCls}>OOS Trades</th>
                        <th className={thCls}>OOS Sum R</th>
                        <th className={thCls}>OOS Avg R</th>
                        <th className={thCls}>OOS PF</th>
                        <th className={thCls}>OOS Max DD (R)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line/30">
                      {walkForwardQuery.isLoading ? (
                        <tr>
                          <td colSpan={18} className="px-4 py-6 text-center text-ink-muted text-sm">
                            Loading fold detail…
                          </td>
                        </tr>
                      ) : walkForwardQuery.isError ? (
                        <tr>
                          <td colSpan={18} className="px-4 py-6 text-center text-error text-sm">
                            No walk-forward metadata for this model.
                          </td>
                        </tr>
                      ) : (
                        (walkForwardQuery.data ?? []).map((fold) => (
                          <tr key={fold.fold}>
                            <td className={tdCls}>{fold.fold}</td>
                            <td className={`${tdCls} font-mono text-xs`}>{fold.oos_start}</td>
                            <td className={`${tdCls} font-mono text-xs`}>{fold.oos_end}</td>
                            <td className={tdCls}>{fmtNum(fold.temperature, 2)}</td>
                            <td className={tdCls}>{fmtNum(fold.brier_secure_is)}</td>
                            <td className={tdCls}>{fmtNum(fold.brier_secure_oos)}</td>
                            <td className={tdCls}>{fmtNum(fold.revert_acc_oos, 2)}</td>
                            <td className={tdCls}>{fmtNum(fold.base_rate_secure_oos, 2)}</td>
                            <td className={tdCls}>{fold.economics_is.trades}</td>
                            <td className={tdCls}>{fmtNum(fold.economics_is.sum_r, 2)}</td>
                            <td className={tdCls}>{fmtNum(fold.economics_is.avg_r)}</td>
                            <td className={tdCls}>{fmtNum(fold.economics_is.profit_factor, 2)}</td>
                            <td className={tdCls}>{fmtNum(fold.economics_is.max_drawdown_r, 2)}</td>
                            <td className={tdCls}>{fold.economics_oos.trades}</td>
                            <td className={`${tdCls} font-bold`}>{fmtNum(fold.economics_oos.sum_r, 2)}</td>
                            <td
                              className={`${tdCls} font-bold ${
                                fold.economics_oos.avg_r >= 0 ? "text-ok" : "text-error"
                              }`}
                            >
                              {fmtNum(fold.economics_oos.avg_r)}
                            </td>
                            <td className={tdCls}>{fmtNum(fold.economics_oos.profit_factor, 2)}</td>
                            <td className={tdCls}>{fmtNum(fold.economics_oos.max_drawdown_r, 2)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </div>
        )}
      </div>

      <RawDataExportSection symbol={symbol} accountId={accountId} />
    </div>
  );
}

/** "Export raw data" — download triggers for the two new bulk-export
 * endpoints (`market-data/candles/export`, `order-book/export`). Both are
 * server-generated files (their own `Content-Disposition`), so both go
 * through `downloadFileFromApi` rather than `downloadJson`/`downloadCsv`. */
function RawDataExportSection({ symbol, accountId }: { symbol: string; accountId: string | null }) {
  const [candleSymbol, setCandleSymbol] = useState(symbol);
  const [timeframe, setTimeframe] = useState<(typeof TIMEFRAMES)[number]>("M5");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [candleFormat, setCandleFormat] = useState<"csv" | "json">("csv");
  const [candleBusy, setCandleBusy] = useState(false);
  const [candleError, setCandleError] = useState<string | null>(null);

  const [obSymbol, setObSymbol] = useState(symbol);
  const [sinceDate, setSinceDate] = useState("");
  const [untilDate, setUntilDate] = useState("");
  const [obFormat, setObFormat] = useState<"csv" | "json">("csv");
  const [obBusy, setObBusy] = useState(false);
  const [obError, setObError] = useState<string | null>(null);

  const fromTime = toEpochSeconds(fromDate);
  const toTime = toEpochSeconds(toDate);
  const candleRangeValid = fromTime !== null && toTime !== null && toTime > fromTime;

  const handleCandleExport = async () => {
    if (!accountId || !candleRangeValid) return;
    setCandleBusy(true);
    setCandleError(null);
    try {
      const params = new URLSearchParams({
        symbol: candleSymbol,
        timeframe,
        from_time: String(fromTime),
        to_time: String(toTime),
        format: candleFormat,
      });
      await downloadFileFromApi(
        `/accounts/${encodeURIComponent(accountId)}/market-data/candles/export?${params}`,
        `${candleSymbol}_${timeframe}_candles.${candleFormat}`,
      );
    } catch {
      setCandleError("Export failed — check the range has stored candles.");
    } finally {
      setCandleBusy(false);
    }
  };

  const handleOrderBookExport = async () => {
    if (!accountId) return;
    setObBusy(true);
    setObError(null);
    try {
      const params = new URLSearchParams({ format: obFormat });
      if (obSymbol) params.set("symbol", obSymbol);
      const since = toEpochSeconds(sinceDate);
      const until = toEpochSeconds(untilDate);
      if (since !== null) params.set("since", String(since));
      if (until !== null) params.set("until", String(until));
      await downloadFileFromApi(
        `/accounts/${encodeURIComponent(accountId)}/order-book/export?${params}`,
        `order_book_export.${obFormat}`,
      );
    } catch {
      setObError("Export failed.");
    } finally {
      setObBusy(false);
    }
  };

  return (
    <div className={cardCls}>
      <div className={headerCls}>
        <h2 className="font-bold text-lg text-ink">Export Raw Data</h2>
        <p className="text-xs text-ink-muted">
          Bulk downloads for AI-training pipelines — reads local storage only, never the live gateway.
        </p>
      </div>
      <div className="p-5 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Candle history export */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider">Candle History</h3>
          <div className="flex flex-wrap gap-3 items-end">
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Symbol
              <input
                type="text"
                value={candleSymbol}
                onChange={(e) => setCandleSymbol(e.target.value)}
                className={inputCls}
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Timeframe
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value as (typeof TIMEFRAMES)[number])}
                className={inputCls}
              >
                {TIMEFRAMES.map((tf) => (
                  <option key={tf} value={tf}>
                    {tf}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              From
              <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} className={inputCls} />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              To
              <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} className={inputCls} />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Format
              <select
                value={candleFormat}
                onChange={(e) => setCandleFormat(e.target.value as "csv" | "json")}
                className={inputCls}
              >
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
              </select>
            </label>
            <button
              onClick={handleCandleExport}
              disabled={!accountId || !candleRangeValid || candleBusy}
              className={btnCls}
            >
              {candleBusy ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
              Download
            </button>
          </div>
          <p className="text-xs text-ink-muted">
            `from_time`/`to_time` are required (epoch seconds under the hood). `format=json` is capped server-side —
            narrow the range or use CSV for a wide export.
          </p>
          {candleError && <p className="text-xs text-error">{candleError}</p>}
        </div>

        {/* Order-book export */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider">Order Book Snapshots</h3>
          <div className="flex flex-wrap gap-3 items-end">
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Symbol (optional)
              <input
                type="text"
                value={obSymbol}
                onChange={(e) => setObSymbol(e.target.value)}
                placeholder="All symbols"
                className={inputCls}
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Since (optional)
              <input type="date" value={sinceDate} onChange={(e) => setSinceDate(e.target.value)} className={inputCls} />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Until (optional)
              <input type="date" value={untilDate} onChange={(e) => setUntilDate(e.target.value)} className={inputCls} />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-muted">
              Format
              <select
                value={obFormat}
                onChange={(e) => setObFormat(e.target.value as "csv" | "json")}
                className={inputCls}
              >
                <option value="csv">CSV (flattened)</option>
                <option value="json">JSON (nested)</option>
              </select>
            </label>
            <button onClick={handleOrderBookExport} disabled={!accountId || obBusy} className={btnCls}>
              {obBusy ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
              Download
            </button>
          </div>
          <p className="text-xs text-ink-muted">
            Sparse by design — most symbols/brokers never report market depth, so an empty result is expected, not
            an error.
          </p>
          {obError && <p className="text-xs text-error">{obError}</p>}
        </div>
      </div>
    </div>
  );
}
