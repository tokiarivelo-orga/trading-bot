'use client';

/**
 * Chart engine: owns the lightweight-charts instance itself — candlestick +
 * volume series, the two series-markers plugin instances (trade markers and
 * the separate structure/QML marker layer), the `DrawingManager` attachment,
 * drawing-tool mouse/context-menu wiring, the resize observer, and teardown.
 * This is a straight structural move of ChartPanel's original single `[]`-
 * dependency chart-creation effect — same one-time setup/teardown, same
 * event wiring — with exactly one thing carved out: indicator-series
 * creation (`recomputeIndicators`), which used to live inline in this same
 * effect and now lives in `useIndicators.ts`'s own effect instead. That
 * fusion (indicators tied to a `[]`-effect that never reacts to indicator
 * state) was the bug this split exists to fix; nothing else about this
 * effect's behavior changes.
 *
 * Everything the effect still does beyond bare chart/series/manager setup
 * (drawing highlight/select/save-and-sync, the drag-to-move-a-drawing
 * handler, the right-click context-menu router between drawing/order
 * popovers, cursor-tracking) deliberately stays inline here even after phase
 * 8 landed `useDrawingTools` — splitting the listener wiring itself out
 * would mean either duplicating the setup/teardown pairing across two
 * effects (real risk: a right-click landing in the gap between one effect's
 * setup and the other's) or leaving this hook's surface this wide anyway, so
 * this hook still accepts the handful of external setters/refs that code
 * needs (now sourced from `useDrawingTools`/ChartPanel.tsx's own state
 * rather than plain ChartPanel.tsx locals, but the same shapes as before).
 * `DrawingMenuState` itself has moved to `useDrawingTools.ts`, which is the
 * concern it actually belongs to — this hook just imports the type to
 * populate it. Later phases can still peel the listener wiring itself off
 * as its own hook if that setup/teardown risk is ever worth taking on.
 *
 * Every other place in ChartPanel.tsx that reaches into `chartRef.current`,
 * `candleSeriesRef.current`, etc. keeps doing so unchanged — this hook
 * returns the raw refs alongside the `ChartEngineController` so this phase's
 * blast radius stays contained to construction/teardown + the controller
 * surface, not every read site (~80 of them) across the file. Later phases
 * migrate call sites to the controller incrementally as they extract the
 * hooks that own them.
 */

import { useEffect, useMemo, useRef, useState, type Dispatch, type RefObject, type SetStateAction } from 'react';
import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
} from 'lightweight-charts';
import { DrawingManager, type IDrawing } from 'lightweight-charts-drawing';
import type { Candle } from '@/shared/api/client';
import { cssVar, hexToRgba, getLocalTimeZoneOptions } from './chartFormat';
import { isProgrammaticDrawingId, loadDrawingsFromStorage } from './chartStorage';
import { FUTURE_RIGHT_OFFSET } from './useCandleData';
import type { ChartEngineController, DrawingToolType, ZoneMeta } from './types';
import type { ContextMenuState, OrderPopoverState } from './useOrderPopovers';
import type { DrawingMenuState } from './useDrawingTools';

// How many of the most-recent visible bars the replay-following autoscale
// override considers — bounded so a long replay window doesn't scan its
// entire revealed history on every autoscale call.
const AUTOSCALE_RECENT_BARS = 100;

export interface UseChartEngineParams {
  /** The full loaded window normally, or a prefix up to the replay cursor
   * while replaying — see ChartPanel's `visibleCandles()`. Read once at
   * mount (this effect has `[]` deps, same as before the split); safe
   * because it only ever reads through refs internally, never closure
   * state, so a "stale" function reference behaves identically to a fresh
   * one. Used only by the candlestick series' `autoscaleInfoProvider`. */
  visibleCandles: () => Candle[];
  replayActiveRef: RefObject<boolean>;
  followCursorRef: RefObject<boolean>;
  symbolRef: RefObject<string>;
  isSwitchingSymbolRef: RefObject<boolean>;
  drawingToolRef: RefObject<DrawingToolType | null>;
  setDrawingTool: Dispatch<SetStateAction<DrawingToolType | null>>;
  setDrawingsList: Dispatch<SetStateAction<IDrawing[]>>;
  originalStylesRef: RefObject<Record<string, any>>;
  setActiveColor: Dispatch<SetStateAction<string>>;
  /** Written once (`saveAndSyncRef.current = saveAndSync`) so effects
   * elsewhere in ChartPanel.tsx that mutate drawings outside this hook can
   * still trigger a persist + drawings-list resync. */
  saveAndSyncRef: RefObject<() => void>;
  setContextMenu: Dispatch<SetStateAction<ContextMenuState | null>>;
  setOrderPopover: Dispatch<SetStateAction<OrderPopoverState | null>>;
  setDrawingContextMenu: Dispatch<SetStateAction<DrawingMenuState | null>>;
  setDrawingEditPopover: Dispatch<SetStateAction<DrawingMenuState | null>>;
  bumpLines: Dispatch<SetStateAction<number>>;
  showVolume: boolean;
}

export function useChartEngine(params: UseChartEngineParams) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const whitespaceSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const seriesMarkersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  // Separate marker-plugin instance for the manual 'structure'/'qml'
  // indicators (see useIndicators' recomputeIndicators) — kept independent
  // of seriesMarkersRef's trade-entry/exit markers so toggling structure
  // on/off never touches the live/backtest/custom-code marker-setting
  // effects.
  const structureMarkersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(
    null,
  );
  // Drawing tools: one manager instance, alive for the lifetime of the chart.
  const drawingManagerRef = useRef<DrawingManager | null>(null);
  // Zone-rectangle metadata side map (see `ZoneMeta`) — one instance for the
  // component's lifetime, mutated in place by useIndicators/useBacktestData
  // rather than recreated, so a stable reference survives their rebuilds.
  const zoneMetaMapRef = useRef<Map<string, ZoneMeta>>(new Map());
  const [isReady, setIsReady] = useState(false);

  // Create the chart once; destroy on unmount.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const line = cssVar('--color-line');
    const timeZoneOpts = getLocalTimeZoneOptions();
    const chart = createChart(container, {
      layout: {
        background: { color: cssVar('--color-panel') },
        textColor: cssVar('--color-ink'),
        // Required by lightweight-charts' free-tier license — do not hide or
        // replace this mark. Explicit `true` (not the implicit default) so
        // the license condition is visible here in code.
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: line },
        horzLines: { color: line },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: line,
        // Reserves empty space past the last candle so there's somewhere to
        // draw into — paired with the future whitespace bars useCandleData's
        // `render()` appends to the series, which is what actually makes
        // that space's coordinates resolvable by the drawing tools.
        rightOffset: FUTURE_RIGHT_OFFSET,
        tickMarkFormatter: timeZoneOpts.timeScale.tickMarkFormatter,
      },
      localization: timeZoneOpts.localization,
      rightPriceScale: { borderColor: line },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: cssVar('--color-ok'),
      downColor: cssVar('--color-err'),
      borderVisible: false,
      wickUpColor: cssVar('--color-ok'),
      wickDownColor: cssVar('--color-err'),
      autoscaleInfoProvider: (originalProvider: any) => {
        const res = originalProvider ? originalProvider() : null;
        if (params.replayActiveRef.current && params.followCursorRef.current) {
          const bars = params.visibleCandles();
          if (bars.length > 0) {
            // Derive the range from the visible bars' own high/low instead of
            // blending with `originalProvider()`'s result: that provider
            // reflects the full loaded dataset (or a wider pre-replay
            // window), and on the very first replay tick it hasn't caught up
            // to the just-narrowed `bars` yet — using its span centered on
            // one bar's close produced a wildly oversized range (the whole
            // history's high/low squeezed around a single price).
            const recent = bars.slice(-AUTOSCALE_RECENT_BARS);
            let minValue = Infinity;
            let maxValue = -Infinity;
            for (const bar of recent) {
              if (bar.low < minValue) minValue = bar.low;
              if (bar.high > maxValue) maxValue = bar.high;
            }
            if (minValue <= maxValue) {
              const span = maxValue - minValue;
              const padding = span > 0 ? span * 0.1 : recent[recent.length - 1].close * 0.02;
              return {
                priceRange: {
                  minValue: minValue - padding,
                  maxValue: maxValue + padding,
                }
              };
            }
          }
        }
        return res;
      }
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      visible: params.showVolume,
    });
    volumeSeries
      .priceScale()
      .applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    const whitespaceSeries = chart.addSeries(LineSeries, {
      color: 'transparent',
      priceScaleId: '',
      lastValueVisible: false,
      priceLineVisible: false,
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    whitespaceSeriesRef.current = whitespaceSeries;
    seriesMarkersRef.current = createSeriesMarkers(candleSeries, []);
    structureMarkersRef.current = createSeriesMarkers(candleSeries, []);

    // Attach the drawing manager to the chart and its primary series so the
    // drawing tools can convert pixel ↔ price/time coordinates and register
    // their mouse-event handlers on the container element.
    const zoneMetaMap = zoneMetaMapRef.current;
    const manager = new DrawingManager();
    manager.attach(chart, candleSeries, container);
    drawingManagerRef.current = manager;
    setIsReady(true);

    const highlightDrawing = (drawing: IDrawing) => {
      if (!params.originalStylesRef.current[drawing.id]) {
        params.originalStylesRef.current[drawing.id] = {
          lineColor: drawing.style.lineColor,
          lineWidth: drawing.style.lineWidth,
          lineDash: drawing.style.lineDash || [],
          fillColor: drawing.style.fillColor,
          showLabels: drawing.style.showLabels,
          labelColor: drawing.style.labelColor,
        };
      }
      drawing.updateStyle({
        lineWidth: 4,
        lineColor: '#00f0ff',
        labelColor: '#00f0ff',
        fillColor: hexToRgba('#00f0ff', 0.25),
      });
    };

    const restoreDrawing = (drawingId: string) => {
      const orig = params.originalStylesRef.current[drawingId];
      if (orig) {
        const drawing = manager.getDrawing(drawingId);
        if (drawing) {
          drawing.updateStyle(orig);
        }
        delete params.originalStylesRef.current[drawingId];
      }
    };

    // Persist drawings + keep the drawings-list panel in sync whenever any
    // drawing mutation happens.
    // Strategy-derived drawings are recomputed from the active spec on every
    // candle tick — they're never user data, so they're excluded from both
    // the persisted localStorage snapshot and the drawings-list panel.
    const syncList = () =>
      params.setDrawingsList((prev) => {
        const next = manager.getAllDrawings().filter((d) => !isProgrammaticDrawingId(d.id));
        if (
          prev.length === next.length &&
          prev.every((d, i) => d.id === next[i].id && d === next[i])
        ) {
          return prev;
        }
        return next;
      });
    const saveAndSync = () => {
      if (params.isSwitchingSymbolRef.current) {
        syncList();
        return;
      }
      try {
        const selected = manager.getSelectedDrawing();
        let backup: any = null;
        if (selected && params.originalStylesRef.current[selected.id]) {
          backup = { ...selected.style };
          selected.updateStyle(params.originalStylesRef.current[selected.id]);
        }

        const data = manager
          .exportDrawings()
          .filter((d) => !isProgrammaticDrawingId(d.id));
        localStorage.setItem(
          `chart-drawings:${params.symbolRef.current}`,
          JSON.stringify(data),
        );

        if (selected && backup) {
          selected.updateStyle(backup);
        }
      } catch {
        // localStorage quota or serialisation errors are non-fatal.
      }
      syncList();
    };
    params.saveAndSyncRef.current = saveAndSync;
    const syncSelectedColor = () => {
      const selected = manager.getSelectedDrawing();
      if (selected) {
        const orig = params.originalStylesRef.current[selected.id];
        if (orig && orig.lineColor) {
          params.setActiveColor(orig.lineColor);
        } else if (selected.style?.lineColor) {
          params.setActiveColor(selected.style.lineColor);
        }
      }
    };
    const isProgrammaticEvent = (e: any): boolean => {
      if (!e) return false;
      const id = typeof e === 'string' ? e : e.drawingId || e.drawing?.id || e.id;
      return typeof id === 'string' ? isProgrammaticDrawingId(id) : false;
    };
    const unsubAdd = manager.on('drawing:added', (e: any) => {
      if (!isProgrammaticEvent(e)) saveAndSync();
    });
    const unsubRemove = manager.on('drawing:removed', (e) => {
      if (e?.drawingId) {
        delete params.originalStylesRef.current[e.drawingId];
      }
      if (!isProgrammaticEvent(e)) saveAndSync();
    });
    const unsubClear = manager.on('drawing:cleared', () => {
      params.originalStylesRef.current = {};
      saveAndSync();
    });
    const unsubUpdate = manager.on('drawing:updated', (e: any) => {
      if (!isProgrammaticEvent(e)) saveAndSync();
    });
    const unsubSelect = manager.on('drawing:selected', (e) => {
      if (e.drawing) {
        highlightDrawing(e.drawing);
      }
      syncSelectedColor();
      if (!isProgrammaticEvent(e)) saveAndSync();
    });
    const unsubDeselect = manager.on('drawing:deselected', (e) => {
      if (e.drawingId) {
        restoreDrawing(e.drawingId);
      }
      saveAndSync();
    });

    // Restore any previously saved drawings for the initial symbol.
    loadDrawingsFromStorage(manager, params.symbolRef.current);
    // Initialise the drawings-list panel state.
    syncList();

    // Guard against 0×0 measurements — the sidebar-resize drag (and its
    // collapse animation) can momentarily report a zero-size container
    // mid-reflow; applying that to the chart blanks the canvas and it never
    // recovers once the container settles back to a real size.
    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) return;
      chart.applyOptions({ width, height });
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    let isDraggingAnchor = false;
    let dragStartPoint: { x: number; y: number } | null = null;
    let dragDrawing: IDrawing | null = null;
    let dragInitialPixels: Array<{ x: number; y: number } | null> = [];

    // Event listener to lock chart panning/scaling when dragging/resizing a drawing
    const handleDrawingDragStart = (e: MouseEvent) => {
      if (!manager || !chart) return;
      if (params.drawingToolRef.current) return;

      const rect = container.getBoundingClientRect();
      const point = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };

      const anchorIndex = manager.hitTestAnchor(point);
      if (anchorIndex !== null) {
        isDraggingAnchor = true;
        container.style.cursor = 'grabbing';
        // Disable chart panning and zooming so the chart doesn't move during drag.
        chart.applyOptions({ handleScroll: false, handleScale: false });

        // Listen to mouseup on window to re-enable chart scrolling/scaling.
        const handleDragEnd = () => {
          isDraggingAnchor = false;
          container.style.cursor = '';
          chart.applyOptions({ handleScroll: true, handleScale: true });
          window.removeEventListener('mouseup', handleDragEnd);
        };
        window.addEventListener('mouseup', handleDragEnd);
        return;
      }

      const hoveredDrawing = manager.hitTest(point);
      if (hoveredDrawing !== null && !hoveredDrawing.options.locked) {
        // Automatically select the hovered drawing if it wasn't selected
        if (manager.getSelectedDrawing()?.id !== hoveredDrawing.id) {
          manager.selectDrawing(hoveredDrawing.id);
        }

        dragStartPoint = { x: e.clientX, y: e.clientY };
        dragDrawing = hoveredDrawing;

        const viewport = hoveredDrawing.getViewport();
        if (viewport) {
          dragInitialPixels = hoveredDrawing.anchors.map((a) =>
            (hoveredDrawing as any).anchorToPixel(a, viewport),
          );
        }

        container.style.cursor = 'grabbing';
        // Disable chart panning and zooming so the chart doesn't move during drag.
        chart.applyOptions({ handleScroll: false, handleScale: false });

        const handleBodyDrag = (moveEvent: MouseEvent) => {
          if (!dragStartPoint || !dragDrawing || !viewport) return;
          const dx = moveEvent.clientX - dragStartPoint.x;
          const dy = moveEvent.clientY - dragStartPoint.y;

          const newAnchors = dragDrawing.anchors.map((anchor, idx) => {
            const pixel = dragInitialPixels[idx];
            if (!pixel) return anchor;
            const newPixel = { x: pixel.x + dx, y: pixel.y + dy };
            const newAnchor = (dragDrawing as any).pixelToAnchor(
              newPixel,
              viewport,
            );
            return newAnchor || anchor;
          });

          dragDrawing.anchors = newAnchors;
          (manager as any).emit('drawing:updated', {
            drawingId: dragDrawing.id,
            drawing: dragDrawing,
          });
        };

        const handleBodyDragEnd = () => {
          window.removeEventListener('mousemove', handleBodyDrag);
          window.removeEventListener('mouseup', handleBodyDragEnd);

          dragStartPoint = null;
          dragDrawing = null;
          dragInitialPixels = [];

          container.style.cursor = '';
          chart.applyOptions({ handleScroll: true, handleScale: true });
          saveAndSync();
        };

        window.addEventListener('mousemove', handleBodyDrag);
        window.addEventListener('mouseup', handleBodyDragEnd);
      }
    };
    container.addEventListener('mousedown', handleDrawingDragStart, {
      capture: true,
    });

    const handleContextMenu = (e: MouseEvent) => {
      if (params.drawingToolRef.current) {
        params.setDrawingTool(null);
        e.preventDefault();
        return;
      }

      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const hoveredDrawing = manager.hitTest({ x, y });
      if (hoveredDrawing !== null) {
        e.preventDefault();
        e.stopPropagation();

        manager.selectDrawing(hoveredDrawing.id);

        params.setOrderPopover(null);
        params.setContextMenu(null);
        params.setDrawingEditPopover(null);
        params.setDrawingContextMenu({
          x,
          y,
          drawingId: hoveredDrawing.id,
          drawingType: hoveredDrawing.type,
          containerWidth: container.clientWidth,
          containerHeight: container.clientHeight,
        });
        return;
      }

      const candleSeries = candleSeriesRef.current;
      if (!candleSeries) return;

      const price = candleSeries.coordinateToPrice(y);
      if (price === null) return;

      e.preventDefault();

      params.setOrderPopover(null);
      params.setDrawingContextMenu(null);
      params.setDrawingEditPopover(null);
      params.setContextMenu({
        x,
        y,
        price,
        containerWidth: container.clientWidth,
        containerHeight: container.clientHeight,
      });
    };
    container.addEventListener('contextmenu', handleContextMenu);

    const handleMouseMoveCursor = (e: MouseEvent) => {
      // Re-trigger layout updates for HTML overlays
      params.bumpLines((t) => t + 1);

      if (!manager || isDraggingAnchor) return;

      // If we are currently in drawing tool placement mode, let that cursor (crosshair) stay.
      if (params.drawingToolRef.current) {
        container.style.cursor = 'crosshair';
        return;
      }

      const rect = container.getBoundingClientRect();
      const point = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };

      // Check if hovering over an anchor point of the selected drawing
      const anchorIndex = manager.hitTestAnchor(point);
      if (anchorIndex !== null) {
        container.style.cursor = 'nwse-resize';
        return;
      }

      // Check if hovering over any drawing body
      const hoveredDrawing = manager.hitTest(point);
      if (hoveredDrawing !== null) {
        container.style.cursor = 'pointer';
        return;
      }

      // Default: let chart cursor rule
      container.style.cursor = '';
    };
    container.addEventListener('mousemove', handleMouseMoveCursor, {
      capture: true,
    });

    return () => {
      observer.disconnect();
      container.removeEventListener('contextmenu', handleContextMenu);
      container.removeEventListener('mousemove', handleMouseMoveCursor, {
        capture: true,
      });
      container.removeEventListener('mousedown', handleDrawingDragStart, {
        capture: true,
      });
      unsubAdd();
      unsubRemove();
      unsubClear();
      unsubUpdate();
      unsubSelect();
      unsubDeselect();
      seriesMarkersRef.current?.detach();
      structureMarkersRef.current?.detach();
      manager.detach();
      drawingManagerRef.current = null;
      zoneMetaMap.clear();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      whitespaceSeriesRef.current = null;
      seriesMarkersRef.current = null;
      structureMarkersRef.current = null;
      setIsReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.applyOptions({ visible: params.showVolume });
    }
  }, [params.showVolume]);

  // Stable except when `isReady` flips (mount, and unmount if this instance
  // is ever torn down) — safe as a consumer effect dependency; it won't
  // change identity on every ChartPanel re-render the way a fresh object
  // literal would.
  const chartController: ChartEngineController = useMemo(
    () => ({
      containerRef,
      getChart: () => chartRef.current,
      getCandleSeries: () => candleSeriesRef.current,
      getVolumeSeries: () => volumeSeriesRef.current,
      getWhitespaceSeries: () => whitespaceSeriesRef.current,
      getDrawingManager: () => drawingManagerRef.current,
      getZoneMetaMap: () => zoneMetaMapRef.current,
      getSeriesMarkersPrimitive: () => seriesMarkersRef.current,
      getStructureMarkersPrimitive: () => structureMarkersRef.current,
      isReady,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [isReady],
  );

  return {
    chartController,
    containerRef,
    chartRef,
    candleSeriesRef,
    volumeSeriesRef,
    whitespaceSeriesRef,
    seriesMarkersRef,
    structureMarkersRef,
    drawingManagerRef,
  };
}

export type ChartEngine = ReturnType<typeof useChartEngine>;
