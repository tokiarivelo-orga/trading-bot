import { memo, useEffect, useRef } from 'react';
import type { BacktestTrade, TradeMarker } from '@/shared/api/client';

export interface TradeInfoPopoverProps {
  x: number;
  y: number;
  trades: (TradeMarker | BacktestTrade)[];
  containerWidth: number;
  containerHeight: number;
  onClose: () => void;
  onSelectTrade: (idOrIndex: string | number) => void;
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function isLiveTrade(t: TradeMarker | BacktestTrade): t is TradeMarker {
  return 'id' in t;
}

export const TradeInfoPopover = memo(function TradeInfoPopover({
  x,
  y,
  trades,
  containerWidth,
  containerHeight,
  onClose,
  onSelectTrade,
}: TradeInfoPopoverProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseDownOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    window.addEventListener('mousedown', handleMouseDownOutside);
    return () => window.removeEventListener('mousedown', handleMouseDownOutside);
  }, [onClose]);

  const popoverWidth = 288;
  const popoverHeight = 240;
  const left = x + popoverWidth > containerWidth ? x - popoverWidth : x;
  const top = y + popoverHeight > containerHeight ? y - popoverHeight : y;

  return (
    <div
      ref={ref}
      className='pointer-events-auto absolute z-30 flex w-72 flex-col gap-2 rounded border border-line bg-panel p-3 text-xs shadow-xl backdrop-blur-sm bg-opacity-95'
      style={{ left: `${left}px`, top: `${top}px` }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className='flex items-center justify-between border-b border-line pb-1'>
        <span className='font-bold text-ink'>
          {trades.length > 1 ? `Trades (${trades.length})` : 'Trade'}
        </span>
        <button
          onClick={onClose}
          className='cursor-pointer text-ink-muted hover:text-ink text-sm font-bold'
          title='Close'
        >
          ×
        </button>
      </div>

      <div className='flex max-h-64 flex-col gap-3 overflow-y-auto'>
        {trades.map((t, i) => {
          const buy = t.side === 'buy';
          // Determine ID for highlighting (live trades have id, backtest trades need index from ChartPanel if possible,
          // but if not possible, we use index. Wait, for backtestTrades it's the originalIndex which is lost here.
          // Since onSelectTicket handles live trades for highlighting, we pass id if it exists.
          const handleSelect = () => {
            if (isLiveTrade(t)) {
              onSelectTrade(t.id);
            } else {
              // For backtestTrades, highlighting might not work perfectly without originalIndex, 
              // but we pass a generic value to trigger potential behavior if supported.
              // We'll leave it as a no-op or pass the loop index just in case.
              onSelectTrade(i); 
            }
          };

          return (
            <div 
              key={isLiveTrade(t) ? t.id : `bt-${t.open_time}-${i}`} 
              className='flex flex-col gap-1 cursor-pointer hover:bg-line/20 p-1 -mx-1 rounded transition-colors'
              onClick={handleSelect}
              title="Click to highlight trade"
            >
              <div className='flex items-center gap-1.5'>
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                    buy ? 'bg-buy/15 text-buy' : 'bg-sell/15 text-sell'
                  }`}
                >
                  {buy ? 'BUY' : 'SELL'}
                </span>
                <span className='font-bold text-ink'>
                  Vol: {t.volume}
                </span>
              </div>
              <div className='flex justify-between gap-2 text-ink-muted'>
                <span>Open Time</span>
                <span className='text-ink text-right'>{formatTime(t.open_time)}</span>
              </div>
              <div className='flex justify-between gap-2 text-ink-muted'>
                <span>Open Price</span>
                <span className='text-ink text-right'>{t.open_price}</span>
              </div>
              {t.close_price !== null && (
                 <div className='flex justify-between gap-2 text-ink-muted'>
                   <span>Close Price</span>
                   <span className='text-ink text-right'>{t.close_price}</span>
                 </div>
              )}
              {t.profit !== null && t.profit !== undefined && (
                <div className='flex justify-between gap-2 font-bold'>
                  <span className='text-ink-muted'>Profit</span>
                  <span className={t.profit >= 0 ? 'text-ok' : 'text-err'}>
                    {t.profit >= 0 ? '+' : ''}{t.profit.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
});
