'use client';

import { useRouter } from 'next/navigation';

export default function ModelSelector({ currentSymbol }: { currentSymbol: string }) {
  const router = useRouter();

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="symbol" className="text-sm font-medium text-ink-muted">Model / Symbol:</label>
      <select 
        name="symbol" 
        id="symbol" 
        value={currentSymbol}
        onChange={(e) => {
          const newSymbol = e.target.value;
          router.push(`/model-dashboard?symbol=${encodeURIComponent(newSymbol)}`);
        }}
        className="bg-surface border border-line rounded px-3 py-1.5 text-sm outline-none focus:border-accent"
      >
        <option value="XAUUSD">smc_dl_m5 (XAUUSD)</option>
        <option value="Step Index 200">smc_dl_m5_step200 (Step Index 200)</option>
      </select>
    </div>
  );
}
