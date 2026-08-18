import { useEffect, useState, memo, useMemo } from 'react';
import type { IChartApi } from 'lightweight-charts';
import type { NewsEventRecord } from '@/shared/api/client';
import { groupByKey, nearestCandleTime } from './chartData';

export interface NewsAxisMarkersProps {
  chart: IChartApi | null;
  newsEvents: NewsEventRecord[];
  candles: any[]; // using Candle type
  onClick: (x: number, y: number, events: NewsEventRecord[]) => void;
  width: number;
}

const currencyFlags: Record<string, string> = {
  USD: '🇺🇸', EUR: '🇪🇺', GBP: '🇬🇧', JPY: '🇯🇵',
  CAD: '🇨🇦', AUD: '🇦🇺', CHF: '🇨🇭', NZD: '🇳🇿', CNY: '🇨🇳'
};

const impactColors: Record<string, string> = {
  high: 'bg-err text-panel',
  medium: 'bg-sell text-panel',
  low: 'bg-ink-muted text-panel',
};

export const NewsAxisMarkers = memo(function NewsAxisMarkers({ chart, newsEvents, candles, onClick, width }: NewsAxisMarkersProps) {
  const [positions, setPositions] = useState<{ x: number, group: NewsEventRecord[], flag: string, impact: string }[]>([]);

  useEffect(() => {
    if (!chart || newsEvents.length === 0 || candles.length === 0) {
      setPositions([]);
      return;
    }

    const groups = groupByKey(newsEvents, (e) => String(nearestCandleTime(candles, e.time)));
    const timeScale = chart.timeScale();
    
    const updatePositions = () => {
      const newPositions = [];
      for (const group of groups) {
        if (!group[0]) continue;
        const barTime = nearestCandleTime(candles, group[0].time);
        if (barTime === null) continue;
        const x = timeScale.timeToCoordinate(barTime as any);
        if (x !== null && x >= 0 && x <= width) {
          const hasHigh = group.some(e => e.impact === 'high');
          const hasMedium = group.some(e => e.impact === 'medium');
          const impact = hasHigh ? 'high' : (hasMedium ? 'medium' : 'low');
          const curr = group[0].currency;
          const flag = currencyFlags[curr] || curr || 'NEWS';
          newPositions.push({ x, group, flag, impact });
        }
      }
      setPositions(newPositions);
    };

    updatePositions();
    timeScale.subscribeVisibleTimeRangeChange(updatePositions);
    timeScale.subscribeVisibleLogicalRangeChange(updatePositions);
    
    return () => {
      timeScale.unsubscribeVisibleTimeRangeChange(updatePositions);
      timeScale.unsubscribeVisibleLogicalRangeChange(updatePositions);
    };
  }, [chart, newsEvents, candles, width]);

  return (
    <>
      {positions.map((p, i) => (
        <div
          key={`${p.x}-${i}`}
          onClick={(e) => onClick(p.x, e.clientY, p.group)}
          className={`absolute z-20 flex items-center justify-center rounded-full text-base font-bold shadow-md cursor-pointer hover:scale-110 transition-transform ${impactColors[p.impact]}`}
          style={{
            left: p.x - 16,
            bottom: 2,
            width: 32,
            height: 32,
          }}
          title={`${p.group.length} news event(s)`}
        >
          {p.flag}
        </div>
      ))}
    </>
  );
});
