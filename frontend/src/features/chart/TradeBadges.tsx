import { useEffect, useState, memo } from 'react';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';
import type { BacktestTrade, TradeMarker, Candle } from '@/shared/api/client';
import { groupByKey, nearestCandleTime } from './chartData';

export interface TradeBadgesProps {
  chart: IChartApi | null;
  series: ISeriesApi<'Candlestick'> | null;
  trades: (TradeMarker | BacktestTrade)[];
  candles: Candle[];
  onClick: (x: number, y: number, trades: (TradeMarker | BacktestTrade)[]) => void;
  width: number;
  height: number;
}

export const TradeBadges = memo(function TradeBadges({ chart, series, trades, candles, onClick, width, height }: TradeBadgesProps) {
  const [positions, setPositions] = useState<{ x: number, y: number, group: (TradeMarker | BacktestTrade)[], side: string }[]>([]);

  useEffect(() => {
    if (!chart || !series || trades.length === 0 || candles.length === 0) {
      setPositions([]);
      return;
    }

    // Group by nearest candle time AND side
    const groups = groupByKey(trades, (t) => {
      const barTime = nearestCandleTime(candles, t.open_time);
      return `${barTime}:${t.side}`;
    });

    const timeScale = chart.timeScale();
    const priceScale = chart.priceScale('right'); // default series price scale
    
    const updatePositions = () => {
      const newPositions = [];
      for (const group of groups) {
        if (!group[0]) continue;
        const barTime = nearestCandleTime(candles, group[0].open_time);
        if (barTime === null) continue;
        
        const candle = candles.find(c => c.time === barTime);
        if (!candle) continue;

        const x = timeScale.timeToCoordinate(barTime as any);
        if (x === null || x < -50 || x > width + 50) continue;

        const side = group[0].side;
        // place above high for sell, below low for buy
        const y = series.priceToCoordinate(side === 'buy' ? candle.low : candle.high);
        
        if (y === null || y < -50 || y > height + 50) continue;

        newPositions.push({ x, y, group, side });
      }
      setPositions(newPositions);
    };

    updatePositions();
    
    // Subscribe to both time scale and price scale changes so badges stick to the candles
    timeScale.subscribeVisibleTimeRangeChange(updatePositions);
    timeScale.subscribeVisibleLogicalRangeChange(updatePositions);
    // lightweight-charts doesn't have a direct subscribe on price scale changes, but resizing or time changes cover most.
    // If the chart pan/zooms vertically, a generic mouse/touch event on the chart container would be needed for perfect sync.
    // However, time range changes cover most panning.
    
    // To handle vertical dragging, we can attach a passive listener to the chart container:
    const container = chart.chartElement();
    const handleMove = () => updatePositions();
    container.addEventListener('mousemove', handleMove, { passive: true });
    container.addEventListener('touchmove', handleMove, { passive: true });
    container.addEventListener('wheel', handleMove, { passive: true });

    return () => {
      timeScale.unsubscribeVisibleTimeRangeChange(updatePositions);
      timeScale.unsubscribeVisibleLogicalRangeChange(updatePositions);
      container.removeEventListener('mousemove', handleMove);
      container.removeEventListener('touchmove', handleMove);
      container.removeEventListener('wheel', handleMove);
    };
  }, [chart, series, trades, candles, width, height]);

  return (
    <>
      {positions.map((p, i) => {
        const count = p.group.length;
        const displayCount = count > 9 ? '+9' : String(count);
        const yOffset = p.side === 'buy' ? 24 : -44; // Position below low or above high

        return (
          <div
            key={`${p.x}-${p.y}-${i}`}
            onClick={(e) => {
              e.stopPropagation();
              onClick(p.x, e.clientY, p.group);
            }}
            className={`absolute z-20 flex items-center justify-center rounded-full text-[11px] font-bold shadow-md cursor-pointer hover:scale-110 transition-transform ${
              p.side === 'buy' ? 'bg-buy text-panel border border-buy' : 'bg-sell text-panel border border-sell'
            }`}
            style={{
              left: p.x - 10, // Center horizontally
              top: p.y + yOffset,
              width: 20,
              height: 20,
            }}
            title={`${count} ${p.side.toUpperCase()} order(s)`}
          >
            {displayCount}
          </div>
        );
      })}
    </>
  );
});
