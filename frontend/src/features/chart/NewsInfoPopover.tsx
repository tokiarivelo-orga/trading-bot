import { memo, useEffect, useRef } from 'react';
import type { NewsEventRecord } from '@/shared/api/client';

export interface NewsInfoPopoverProps {
  x: number;
  y: number;
  events: NewsEventRecord[];
  containerWidth: number;
  containerHeight: number;
  onClose: () => void;
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export const NewsInfoPopover = memo(function NewsInfoPopover({
  x,
  y,
  events,
  containerWidth,
  containerHeight,
  onClose,
}: NewsInfoPopoverProps) {
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

  const colorByImpact: Record<string, string> = {
    high: 'bg-err/15 text-err',
    medium: 'bg-sell/15 text-sell',
    low: 'bg-ink-muted/15 text-ink-muted',
  };

  return (
    <div
      ref={ref}
      className='pointer-events-auto absolute z-30 flex w-72 flex-col gap-2 rounded border border-line bg-panel p-3 text-xs shadow-xl backdrop-blur-sm bg-opacity-95'
      style={{ left: `${left}px`, top: `${top}px` }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className='flex items-center justify-between border-b border-line pb-1'>
        <span className='font-bold text-ink'>
          {events.length > 1 ? `Economic News (${events.length})` : 'Economic News'}
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
        {events.map((e, i) => {
          const impactClass = colorByImpact[e.impact] || colorByImpact['low'];
          return (
            <div key={`${e.time}:${i}`} className='flex flex-col gap-1 border-b border-line/30 pb-2 last:border-0'>
              <div className='flex items-center justify-between'>
                <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${impactClass}`}>
                  {e.impact.toUpperCase()}
                </span>
                <span className='text-ink-muted'>{formatTime(e.time)}</span>
              </div>
              <span className='font-semibold text-ink'>{e.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
