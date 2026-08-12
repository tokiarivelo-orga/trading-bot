"use client";

/**
 * IndicatorsDock — TradingView-style panel for manually adding indicator
 * overlays to the chart, independent of whatever the active strategy's
 * PDF-derived spec already draws (see ChartPanel.tsx's `recomputeIndicators`,
 * which plots strategy + manual indicators together).
 *
 * Rendered inside ChartPanel below the chart header when the user clicks the
 * "Indicators (N)" toggle button, same slot/style as DrawingsList.
 *
 * The "custom" indicator type doubles as an ad-hoc code workbench: besides
 * picking a saved indicator (with View/Run shortcuts right on the picker),
 * "Write new code…" opens an inline CodeMirror editor to run unsaved code
 * against the live chart and, once happy with it, save it as a normal
 * reusable saved indicator (POST /indicators) — bridging the old separate
 * "Run Custom Code" script drawer and the /indicators CRUD system.
 */

import { python } from '@codemirror/lang-python';
import { githubDarkInit } from '@uiw/codemirror-theme-github';
import CodeMirror from '@uiw/react-codemirror';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import {
  ApiError,
  createIndicator,
  listIndicators,
  type IndicatorSummary,
} from '@/shared/api/client';
import type {
  IndicatorLineStyle,
  IndicatorLineWidth,
  ManualIndicator,
  ManualIndicatorType,
} from './types';
import { IndicatorCodePeek } from './IndicatorCodePeek';
import { listBottomPanes, MAIN_PANE_OPTION, NEW_PANE_OPTION } from './paneTargets';

const PRESET_COLORS = [
  "#42a5f5", // Blue
  "#ffa726", // Orange
  "#ab47bc", // Purple
  "#26a69a", // Green
  "#ef5350", // Red
  "#78909c", // Grey
];

const TYPE_LABELS: Record<ManualIndicatorType, string> = {
  ema: "EMA",
  sma: "SMA",
  rsi: "RSI",
  macd: "MACD",
  bollinger: "Bollinger Bands",
  vwap: "VWAP",
  atr: "ATR",
  volatility: "Volatility (Historical %)",
  volume_profile: "Volume Profile",
  structure: "Structure (HH/HL/LH/LL)",
  qml: "Quasimodo (QML / inversed)",
  snd: "S&D zones (RBR/DBD/RBD/DBR)",
  snd_v2: "S&D zones v2 (bases + ranges)",
  base: "Base ranges (2 lines)",
  patterns: "Candlestick patterns",
  smc: "Smart Money Concept (SMC)",
  custom: "Custom (saved indicator)",
};

/** Default period per type, and whether the period is even user-editable. */
const TYPE_DEFAULTS: Record<ManualIndicatorType, { period: number; editablePeriod: boolean }> = {
  ema: { period: 20, editablePeriod: true },
  sma: { period: 20, editablePeriod: true },
  rsi: { period: 14, editablePeriod: true },
  atr: { period: 14, editablePeriod: true },
  bollinger: { period: 20, editablePeriod: true },
  // period here is the rolling stdev window, same meaning as other periods.
  volatility: { period: 20, editablePeriod: true },
  // period here doubles as bucketCount's fallback display; actual bucket
  // count/lookback are edited via the dedicated volume-profile controls.
  volume_profile: { period: 24, editablePeriod: true },
  macd: { period: 12, editablePeriod: false }, // fixed 12/26/9, see indicatorLabel()
  vwap: { period: 0, editablePeriod: false }, // cumulative, no period
  // period here is the swing-detection lookback (bars each side), same
  // meaning as the backend vix75 strategy's `swing_lookback` param.
  structure: { period: 3, editablePeriod: true },
  qml: { period: 3, editablePeriod: true },
  // period here is the max base-candle count of a zone, same meaning as
  // maxBaseCandles in sndZones() (indicators.ts).
  snd: { period: 3, editablePeriod: true },
  // period is maxBaseCandles for sndZonesV2 — larger so wide low-timeframe
  // range bases are captured whole (bounded by ATR, not by count alone).
  snd_v2: { period: 20, editablePeriod: true },
  // period is the base's minimum candle count for detectBases().
  base: { period: 5, editablePeriod: true },
  patterns: { period: 0, editablePeriod: false }, // fixed thresholds, no period
  smc: { period: 20, editablePeriod: true },
  // Params come from the saved indicator's own default_params (edit them on
  // /indicators, or from the code-peek panel below) rather than this dock.
  custom: { period: 0, editablePeriod: false },
};

/** Builds the display label shown in the chip list and on the chart series. */
function indicatorLabel(type: ManualIndicatorType, period: number): string {
  switch (type) {
    case "macd":
      return "MACD (12/26/9)";
    case "vwap":
      return "VWAP";
    case "bollinger":
      return `Bollinger (${period}, 2σ)`;
    case "volatility":
      return `Volatility (${period}, HV%)`;
    case "volume_profile":
      return `Volume Profile (${period} buckets)`;
    case "structure":
      return `Structure (lookback ${period})`;
    case "qml":
      return `Quasimodo (lookback ${period})`;
    case "snd":
      return `S&D zones (base ≤ ${period})`;
    case "snd_v2":
      return `S&D zones v2 (base ≤ ${period})`;
    case "base":
      return `Base ranges (min ${period}c)`;
    case "patterns":
      return "Candlestick patterns";
    case "smc":
      return `SMC (lookback ${period})`;
    default:
      return `${TYPE_LABELS[type]} (${period})`;
  }
}

const cmTheme = githubDarkInit({
  settings: {
    background: 'var(--color-bg)',
    gutterBackground: 'var(--color-bg)',
    lineHighlight: 'var(--color-panel)',
    foreground: 'var(--color-ink)',
    caret: 'var(--color-accent)',
    selection: 'color-mix(in srgb, var(--color-accent) 30%, transparent)',
  },
});

/** Sentinel `<option>` value that switches the custom-indicator picker into
 * "write ad-hoc code" mode instead of selecting a saved indicator. */
const NEW_CODE_OPTION = '__new__';

interface Props {
  indicators: ManualIndicator[];
  onAdd: (indicator: ManualIndicator) => void;
  onRemove: (id: string) => void;
  onUpdate?: (id: string, patch: Partial<ManualIndicator>) => void;
  /** Called after the code-peek panel saves an edit to a saved indicator's
   * code, so the chart can recompute every chip currently using it. */
  onCustomIndicatorCodeSaved: () => void;
}

export function IndicatorsDock({ indicators, onAdd, onRemove, onUpdate, onCustomIndicatorCodeSaved }: Props) {
  const [type, setType] = useState<ManualIndicatorType>("ema");
  const [period, setPeriod] = useState<number>(TYPE_DEFAULTS.ema.period);
  const [color, setColor] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('chart-default-indicator-color') || PRESET_COLORS[0];
    }
    return PRESET_COLORS[0];
  });
  const [lineStyle, setLineStyle] = useState<IndicatorLineStyle>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('chart-default-indicator-style') as IndicatorLineStyle) || "solid";
    }
    return "solid";
  });
  const [lineWidth, setLineWidth] = useState<IndicatorLineWidth>(() => {
    if (typeof window !== 'undefined') {
      const val = Number(localStorage.getItem('chart-default-indicator-width'));
      return ([1, 2, 3, 4].includes(val) ? val : 1) as IndicatorLineWidth;
    }
    return 1;
  });

  useEffect(() => {
    localStorage.setItem('chart-default-indicator-color', color);
  }, [color]);

  useEffect(() => {
    localStorage.setItem('chart-default-indicator-style', lineStyle);
  }, [lineStyle]);

  useEffect(() => {
    localStorage.setItem('chart-default-indicator-width', String(lineWidth));
  }, [lineWidth]);

  // Which "screen" (pane) the next-added indicator targets — 'main' the
  // price pane, an existing bottom pane's key (combine), or the
  // NEW_PANE_OPTION sentinel (always creates a fresh pane at add time). See
  // paneTargets.ts for the full contract this feeds `ManualIndicator.paneTarget`.
  const [paneChoice, setPaneChoice] = useState<string>(MAIN_PANE_OPTION);
  const bottomPanes = listBottomPanes(indicators);

  /** Resolves the current pane picker choice to a `ManualIndicator.paneTarget`
   * value — a fresh `paneKey` for "New split pane" so the choice is fixed
   * from add time on, not re-resolved every render. */
  function resolvePaneTarget(): ManualIndicator['paneTarget'] {
    if (paneChoice === MAIN_PANE_OPTION) return 'main';
    if (paneChoice === NEW_PANE_OPTION) return { paneKey: crypto.randomUUID() };
    return { paneKey: paneChoice };
  }

  const [customIndicators, setCustomIndicators] = useState<IndicatorSummary[]>([]);
  const [selectedCustomId, setSelectedCustomId] = useState<string | null>(null);
  const [peekIndicator, setPeekIndicator] = useState<{ id: string; name: string } | null>(null);

  // Ad-hoc "write new code" workbench state — a single unsaved preview chip
  // (stable id below) drawn via ManualIndicator.previewCode instead of a
  // saved indicatorId, computed by ChartPanel's existing custom-indicator
  // compute effect through POST /indicators/preview.
  const [previewChipId] = useState(() => `preview-${crypto.randomUUID()}`);
  const [newCode, setNewCode] = useState('');
  const [newCodeRan, setNewCodeRan] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveBusy, setSaveBusy] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  function refreshCustomIndicators() {
    return listIndicators()
      .then(setCustomIndicators)
      .catch(() => setCustomIndicators([]));
  }

  useEffect(() => {
    refreshCustomIndicators();
  }, []);

  const defaults = TYPE_DEFAULTS[type];
  const writingNewCode = type === 'custom' && selectedCustomId === NEW_CODE_OPTION;

  function handleTypeChange(next: ManualIndicatorType) {
    setType(next);
    setPeriod(TYPE_DEFAULTS[next].period);
    if (next === "custom") setSelectedCustomId(customIndicators[0]?.id ?? null);
  }

  function handleAdd() {
    if (type === "custom") {
      const chosen = customIndicators.find((c) => c.id === selectedCustomId);
      if (!chosen) return;
      onAdd({
        id: crypto.randomUUID(),
        type,
        period: 0,
        color,
        lineStyle,
        lineWidth,
        label: chosen.name,
        indicatorId: chosen.id,
        paneTarget: resolvePaneTarget(),
      });
      return;
    }
    const resolvedPeriod = defaults.editablePeriod ? period : defaults.period;
    onAdd({
      id: crypto.randomUUID(),
      type,
      period: resolvedPeriod,
      color,
      lineStyle,
      lineWidth,
      label: indicatorLabel(type, resolvedPeriod),
      paneTarget: resolvePaneTarget(),
    });
  }

  function handleViewSelectedCode() {
    const chosen = customIndicators.find((c) => c.id === selectedCustomId);
    if (chosen) setPeekIndicator({ id: chosen.id, name: chosen.name });
  }

  function handleRunPreview() {
    if (!newCode.trim()) return;
    onRemove(previewChipId);
    onAdd({
      id: previewChipId,
      type: 'custom',
      period: 0,
      color,
      lineStyle,
      lineWidth,
      label: 'Preview (unsaved)',
      previewCode: newCode,
      paneTarget: resolvePaneTarget(),
    });
    setNewCodeRan(true);
    setSaveError(null);
  }

  async function handleSaveAsIndicator() {
    if (!saveName.trim()) return;
    setSaveBusy(true);
    setSaveError(null);
    try {
      const created = await createIndicator({ name: saveName.trim(), code: newCode });
      await refreshCustomIndicators();
      onRemove(previewChipId);
      onAdd({
        id: crypto.randomUUID(),
        type: 'custom',
        period: 0,
        color,
        lineStyle,
        lineWidth,
        label: created.name,
        indicatorId: created.id,
        paneTarget: resolvePaneTarget(),
      });
      setSelectedCustomId(created.id);
      setNewCode('');
      setNewCodeRan(false);
      setSaveName('');
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'save failed');
    } finally {
      setSaveBusy(false);
    }
  }

  function handleCancelNewCode() {
    if (newCodeRan) onRemove(previewChipId);
    setNewCode('');
    setNewCodeRan(false);
    setSaveName('');
    setSaveError(null);
    setSelectedCustomId(customIndicators[0]?.id ?? null);
  }

  return (
    <div
      style={{
        borderBottom: "1px solid var(--color-line)",
        background: "var(--color-panel)",
      }}
    >
      {/* Add-indicator form */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 8,
          padding: "6px 12px",
          borderBottom: indicators.length > 0 || writingNewCode ? "1px solid var(--color-line)" : "none",
        }}
      >
        <select
          value={type}
          onChange={(e) => handleTypeChange(e.target.value as ManualIndicatorType)}
          className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
        >
          {(Object.keys(TYPE_LABELS) as ManualIndicatorType[]).map((t) => (
            <option key={t} value={t}>
              {TYPE_LABELS[t]}
            </option>
          ))}
        </select>

        {type === "custom" && (
          <>
            <select
              value={selectedCustomId ?? ""}
              onChange={(e) => setSelectedCustomId(e.target.value || null)}
              className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
            >
              <option value={NEW_CODE_OPTION}>✏️ Write new code…</option>
              {customIndicators.length === 0 && <option value="">No saved indicators yet</option>}
              {customIndicators.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            {!writingNewCode && (
              <>
                <button
                  type="button"
                  title="View this indicator's code"
                  onClick={handleViewSelectedCode}
                  disabled={!selectedCustomId}
                  className="cursor-pointer rounded border border-line px-1.5 py-0.5 text-xs text-ink-muted hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
                >
                  👁 View code
                </button>
                <button
                  type="button"
                  title="Run this indicator now and draw it on the chart"
                  onClick={handleAdd}
                  disabled={!selectedCustomId}
                  className="cursor-pointer rounded border border-accent px-1.5 py-0.5 text-xs text-accent disabled:cursor-not-allowed disabled:opacity-50"
                >
                  ▶ Run
                </button>
              </>
            )}
          </>
        )}

        {defaults.editablePeriod && type !== 'custom' && (
          <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--color-ink-muted)" }}>
            Period
            <input
              type="number"
              min={1}
              max={500}
              value={period}
              onChange={(e) => setPeriod(Math.max(1, Number(e.target.value) || 1))}
              className="rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
              style={{ width: 56 }}
            />
          </label>
        )}

        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {PRESET_COLORS.map((c) => (
            <button
              key={c}
              title={c}
              onClick={() => setColor(c)}
              className="cursor-pointer rounded-full transition-transform hover:scale-110"
              style={{
                width: 14,
                height: 14,
                backgroundColor: c,
                border: color === c ? "2px solid var(--color-ink)" : "1px solid var(--color-line)",
              }}
            />
          ))}
          <input
            type="color"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            className="color-picker-input"
            style={{ width: 14, height: 14 }}
            title="Custom color"
          />
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--color-ink-muted)" }}>
          Type
          <select
            value={lineStyle}
            onChange={(e) => setLineStyle(e.target.value as IndicatorLineStyle)}
            className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
          >
            <option value="solid">Solid</option>
            <option value="dashed">Dashed</option>
            <option value="dotted">Dotted</option>
          </select>
        </label>

        <label style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--color-ink-muted)" }}>
          Size
          <select
            value={lineWidth}
            onChange={(e) => setLineWidth(Number(e.target.value) as IndicatorLineWidth)}
            className="cursor-pointer rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
          >
            <option value={1}>1px</option>
            <option value={2}>2px</option>
            <option value={3}>3px</option>
            <option value={4}>4px</option>
          </select>
        </label>

        {/* Screen (pane) picker — which chart pane the next-added indicator
            renders on. Selected screen is highlighted like an active tab,
            matching the accent/panel/line tokens used everywhere else in
            this dock. See paneTargets.ts for what each option resolves to. */}
        <div
          role="group"
          aria-label="Target screen"
          style={{ display: "flex", alignItems: "center", gap: 4 }}
        >
          <span style={{ fontSize: 11, color: "var(--color-ink-muted)" }}>Screen</span>
          <button
            type="button"
            onClick={() => setPaneChoice(MAIN_PANE_OPTION)}
            title="Render on the main price chart"
            className={`cursor-pointer rounded border px-1.5 py-0.5 text-xs ${
              paneChoice === MAIN_PANE_OPTION
                ? "border-accent bg-accent/15 text-accent"
                : "border-line text-ink-muted hover:border-accent hover:text-accent"
            }`}
          >
            Main chart
          </button>
          {bottomPanes.map((pane) => (
            <button
              key={pane.key}
              type="button"
              onClick={() => setPaneChoice(pane.key)}
              title={`Combine into the same pane as: ${pane.label}`}
              className={`cursor-pointer rounded border px-1.5 py-0.5 text-xs ${
                paneChoice === pane.key
                  ? "border-accent bg-accent/15 text-accent"
                  : "border-line text-ink-muted hover:border-accent hover:text-accent"
              }`}
            >
              {pane.label.length > 24 ? `${pane.label.slice(0, 24)}…` : pane.label}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setPaneChoice(NEW_PANE_OPTION)}
            title="Create a new split pane below the chart"
            className={`cursor-pointer rounded border px-1.5 py-0.5 text-xs ${
              paneChoice === NEW_PANE_OPTION
                ? "border-accent bg-accent/15 text-accent"
                : "border-line text-ink-muted hover:border-accent hover:text-accent"
            }`}
          >
            + New split pane
          </button>
        </div>

        {!writingNewCode && (
          <button
            onClick={handleAdd}
            disabled={type === "custom" && !selectedCustomId}
            className="cursor-pointer rounded border border-accent px-2 py-0.5 text-xs text-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            + Add
          </button>
        )}

        <Link
          href="/indicators"
          className="ml-auto text-xs text-ink-muted hover:text-accent"
        >
          Manage indicators →
        </Link>
      </div>

      {/* Ad-hoc "write new code" workbench */}
      {writingNewCode && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            padding: "8px 12px",
            borderBottom: indicators.length > 0 ? "1px solid var(--color-line)" : "none",
          }}
        >
          <div className="overflow-hidden rounded border border-line">
            <CodeMirror
              value={newCode}
              height="10rem"
              theme={cmTheme}
              extensions={[python()]}
              onChange={setNewCode}
              placeholder={
                'class MyIndicator:\n    def compute(self, candles, params):\n        return {"value": [...]}'
              }
            />
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
            <button
              type="button"
              onClick={handleRunPreview}
              disabled={!newCode.trim()}
              className="cursor-pointer rounded border border-accent px-2 py-0.5 text-xs text-accent disabled:cursor-not-allowed disabled:opacity-50"
            >
              ▶ Run & Draw
            </button>
            {newCodeRan && (
              <>
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="Name to save as…"
                  className="rounded border border-line bg-panel px-1.5 py-1 text-xs text-ink"
                  style={{ width: 160 }}
                />
                <button
                  type="button"
                  onClick={handleSaveAsIndicator}
                  disabled={saveBusy || !saveName.trim()}
                  className="cursor-pointer rounded border border-ok px-2 py-0.5 text-xs text-ok disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {saveBusy ? "Saving…" : "💾 Save as indicator"}
                </button>
              </>
            )}
            <button
              type="button"
              onClick={handleCancelNewCode}
              className="cursor-pointer rounded border border-line px-2 py-0.5 text-xs text-ink-muted hover:border-err hover:text-err"
            >
              Cancel
            </button>
          </div>
          {saveError && <p style={{ fontSize: 11, color: "var(--color-err)" }}>{saveError}</p>}
          {!newCodeRan && (
            <p style={{ fontSize: 11, color: "var(--color-ink-muted)" }}>
              Define a class with a <code>compute(candles, params)</code> method returning a dict
              of named series. Run it to preview on the chart before saving.
            </p>
          )}
        </div>
      )}

      {/* Active manual indicators */}
      {indicators.length === 0 ? (
        <div style={{ padding: "8px 16px", fontSize: 12, color: "var(--color-ink-muted)" }}>
          No manual indicators yet. Pick a type above and click Add.
        </div>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "6px 12px" }}>
          {indicators.map((ind) => (
            <span
              key={ind.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                borderRadius: 4,
                border: "1px solid var(--color-line)",
                padding: "2px 6px",
                fontSize: 12,
                color: "var(--color-ink)",
              }}
            >
              <input
                type="color"
                value={ind.color}
                onChange={(e) => onUpdate?.(ind.id, { color: e.target.value })}
                title="Change color"
                className="cursor-pointer"
                style={{
                  width: 14,
                  height: 14,
                  border: "none",
                  padding: 0,
                  background: "transparent",
                  borderRadius: "50%",
                  flexShrink: 0,
                }}
              />
              <span>{ind.label}</span>
              <select
                value={ind.lineStyle ?? 'solid'}
                onChange={(e) => onUpdate?.(ind.id, { lineStyle: e.target.value as IndicatorLineStyle })}
                className="cursor-pointer rounded border border-line bg-panel px-1 py-0 text-[10px] text-ink-muted hover:text-ink"
                title="Line style"
              >
                <option value="solid">Solid</option>
                <option value="dashed">Dashed</option>
                <option value="dotted">Dotted</option>
              </select>
              <select
                value={ind.lineWidth ?? 1}
                onChange={(e) => onUpdate?.(ind.id, { lineWidth: Number(e.target.value) as IndicatorLineWidth })}
                className="cursor-pointer rounded border border-line bg-panel px-1 py-0 text-[10px] text-ink-muted hover:text-ink"
                title="Line thickness"
              >
                <option value={1}>1px</option>
                <option value={2}>2px</option>
                <option value={3}>3px</option>
                <option value={4}>4px</option>
              </select>
              {ind.type === "custom" && ind.indicatorId && (
                <button
                  title="View/edit this indicator's code"
                  onClick={() =>
                    setPeekIndicator({ id: ind.indicatorId as string, name: ind.label })
                  }
                  style={{
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    color: "var(--color-ink-muted)",
                    fontSize: 11,
                    padding: 0,
                    lineHeight: 1,
                  }}
                >
                  ✏️
                </button>
              )}
              <button
                title={ind.hidden ? "Show this indicator" : "Hide this indicator"}
                onClick={() => onUpdate?.(ind.id, { hidden: !ind.hidden })}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: ind.hidden ? "var(--color-ink-muted)" : "var(--color-ink)",
                  fontSize: 11,
                  padding: 0,
                  lineHeight: 1,
                  opacity: ind.hidden ? 0.5 : 1,
                }}
              >
                👁
              </button>
              <button
                title="Remove this indicator"
                onClick={() => onRemove(ind.id)}
                style={{
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--color-err)",
                  fontSize: 11,
                  padding: 0,
                  lineHeight: 1,
                }}
              >
                ✕
              </button>
            </span>
          ))}
        </div>
      )}

      {peekIndicator && (
        <IndicatorCodePeek
          indicatorId={peekIndicator.id}
          indicatorName={peekIndicator.name}
          onClose={() => setPeekIndicator(null)}
          onSaved={onCustomIndicatorCodeSaved}
        />
      )}
    </div>
  );
}
