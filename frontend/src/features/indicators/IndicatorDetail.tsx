"use client";

/** Full indicator detail: code view/edit toggle, duplicate, delete, copy,
 * and a small "Preview" panel that runs the (possibly unsaved) code against
 * real candle history via POST /indicators/preview so a trader can see its
 * output before committing an edit. Mirrors
 * features/strategies/CodeEditorPanel.tsx's view/edit shape, minus the
 * AI-regenerate flow and spec snapshot indicators don't have. */

import { python } from "@codemirror/lang-python";
import { githubDarkInit } from "@uiw/codemirror-theme-github";
import CodeMirror from "@uiw/react-codemirror";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  ApiError,
  deleteIndicator,
  editIndicatorCode,
  getIndicator,
  previewIndicatorCode,
  type ComputeIndicatorResponse,
  type IndicatorDetail as IndicatorDetailType,
} from "@/shared/api/client";
import { DuplicateIndicatorForm } from "./DuplicateIndicatorForm";

const cmTheme = githubDarkInit({
  settings: {
    background: "var(--color-bg)",
    gutterBackground: "var(--color-bg)",
    lineHighlight: "var(--color-panel)",
    foreground: "var(--color-ink)",
    caret: "var(--color-accent)",
    selection: "color-mix(in srgb, var(--color-accent) 30%, transparent)",
  },
});

const TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

export function IndicatorDetail({ indicatorId }: { indicatorId: string }) {
  const router = useRouter();
  const [indicator, setIndicator] = useState<IndicatorDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const [previewSymbol, setPreviewSymbol] = useState("XAUUSD");
  const [previewTimeframe, setPreviewTimeframe] = useState("M5");
  const [previewPeriod, setPreviewPeriod] = useState("2026-06:2026-07");
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewResult, setPreviewResult] = useState<ComputeIndicatorResponse | null>(null);

  function load() {
    getIndicator(indicatorId)
      .then(setIndicator)
      .catch(() => setError("indicator not found"));
  }

  useEffect(load, [indicatorId]); // eslint-disable-line react-hooks/exhaustive-deps

  function startEdit() {
    if (!indicator) return;
    setDraft(indicator.code);
    setError(null);
    setEditing(true);
  }

  async function handleCopy() {
    if (!indicator) return;
    await navigator.clipboard.writeText(editing ? draft : indicator.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  async function onSave() {
    setBusy(true);
    setError(null);
    try {
      const saved = await editIndicatorCode(indicatorId, draft);
      setIndicator(saved);
      setEditing(false);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "save failed");
    } finally {
      setBusy(false);
    }
  }

  async function onDelete() {
    if (!indicator || !confirm(`Delete indicator "${indicator.name}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteIndicator(indicatorId);
      router.push("/indicators");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "delete failed");
      setBusy(false);
    }
  }

  async function onPreview() {
    setPreviewBusy(true);
    setPreviewResult(null);
    try {
      const result = await previewIndicatorCode({
        code: editing ? draft : indicator?.code ?? "",
        symbol: previewSymbol,
        timeframe: previewTimeframe,
        period: previewPeriod,
      });
      setPreviewResult(result);
    } catch (e) {
      setPreviewResult({ times: [], series: {}, error: e instanceof ApiError ? e.message : "preview failed" });
    } finally {
      setPreviewBusy(false);
    }
  }

  if (error && indicator === null) return <p className="p-4 text-sm text-err">{error}</p>;
  if (indicator === null) return <p className="p-4 text-sm text-ink-muted">Loading…</p>;

  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <Link href="/indicators" className="text-xs text-ink-muted hover:text-accent">
          ← All indicators
        </Link>
        <h2 className="mt-1 text-lg font-semibold">{indicator.name}</h2>
        <p className="text-sm text-ink-muted">
          Updated {formatTime(indicator.updated_at)} · code hash {indicator.code_hash.slice(0, 12)}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 text-sm">
        <DuplicateIndicatorForm indicatorId={indicator.id} />
        <button
          type="button"
          className="cursor-pointer rounded border border-err px-3 py-1 text-err hover:bg-err hover:text-bg disabled:cursor-not-allowed disabled:opacity-50"
          disabled={busy}
          onClick={onDelete}
        >
          Delete
        </button>
        {error && <span className="text-err">{error}</span>}
      </div>

      <section className="rounded-md border border-line bg-panel">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2 text-sm text-ink-muted">
          <span>Source code</span>
          <div className="flex gap-2">
            <button
              type="button"
              className={`${btnCls} transition-colors ${
                copied ? "border-ok text-ok hover:border-ok hover:text-ok" : ""
              }`}
              onClick={handleCopy}
            >
              {copied ? "Copied!" : "Copy"}
            </button>
            {!editing ? (
              <button type="button" className={btnCls} onClick={startEdit}>
                Edit
              </button>
            ) : (
              <button
                type="button"
                className={btnCls}
                disabled={busy}
                onClick={() => {
                  setEditing(false);
                  setError(null);
                }}
              >
                Cancel
              </button>
            )}
          </div>
        </header>

        {editing ? (
          <>
            <CodeMirror value={draft} height="28rem" theme={cmTheme} extensions={[python()]} onChange={setDraft} />
            <div className="flex gap-2 border-t border-line p-3">
              <button type="button" className={btnAccentCls} disabled={busy} onClick={onSave}>
                {busy ? "Saving…" : "Save"}
              </button>
            </div>
          </>
        ) : (
          <pre className="max-h-[28rem] overflow-auto p-3 text-xs">{indicator.code}</pre>
        )}
      </section>

      <section className="rounded-md border border-line bg-panel p-3 text-sm">
        <header className="mb-2 text-ink-muted">
          Preview — run {editing ? "the draft above" : "this indicator"} against real candle history
        </header>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-ink-muted">Symbol</span>
            <input
              className={inputCls}
              value={previewSymbol}
              onChange={(e) => setPreviewSymbol(e.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-ink-muted">Timeframe</span>
            <select
              className={inputCls}
              value={previewTimeframe}
              onChange={(e) => setPreviewTimeframe(e.target.value)}
            >
              {TIMEFRAMES.map((tf) => (
                <option key={tf} value={tf}>
                  {tf}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-ink-muted">Period (YYYY-MM:YYYY-MM)</span>
            <input
              className={inputCls}
              value={previewPeriod}
              onChange={(e) => setPreviewPeriod(e.target.value)}
            />
          </label>
          <button type="button" className={btnAccentCls} disabled={previewBusy} onClick={onPreview}>
            {previewBusy ? "Running…" : "Preview"}
          </button>
        </div>
        {previewResult && (
          <div className="mt-3">
            {previewResult.error ? (
              <p className="text-xs text-err">{previewResult.error}</p>
            ) : (
              <div className="flex flex-col gap-1 text-xs text-ink-muted">
                <p>{previewResult.times.length} bars computed.</p>
                {Object.entries(previewResult.series).map(([name, values]) => {
                  const tail = values.slice(-8).map((v) => (v === null ? "—" : v.toFixed(4)));
                  return (
                    <p key={name} className="font-mono">
                      {name}: … {tail.join(", ")}
                    </p>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

const inputCls =
  "rounded border border-line bg-bg px-2 py-1 text-ink placeholder:text-ink-muted focus:border-accent focus:outline-none";
const btnCls =
  "cursor-pointer rounded border border-line px-2 py-1 text-xs hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50";
const btnAccentCls =
  "cursor-pointer rounded border border-accent px-3 py-1 text-xs text-accent hover:bg-accent hover:text-bg disabled:cursor-not-allowed disabled:opacity-50";
