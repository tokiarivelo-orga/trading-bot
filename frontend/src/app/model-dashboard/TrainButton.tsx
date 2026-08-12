"use client";

import { useState, useTransition } from "react";
import { Play } from "lucide-react";
import { triggerManualTraining } from "./actions";

export default function TrainButton() {
  const [isPending, startTransition] = useTransition();
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  
  // Format dates as YYYY-MM
  const [startDate, setStartDate] = useState("2026-06");
  const [endDate, setEndDate] = useState("2026-08");
  const [symbol, setSymbol] = useState("XAUUSD");

  const handleTrain = () => {
    startTransition(async () => {
      setStatusMsg(null);
      const res = await triggerManualTraining(startDate, endDate, symbol);
      setStatusMsg(res.message);
      
      // Auto-hide the message after 5 seconds
      setTimeout(() => {
        setStatusMsg(null);
      }, 5000);
    });
  };

  return (
    <div className="flex flex-col sm:flex-row items-end sm:items-center gap-3">
      {statusMsg && (
        <span className={`text-xs ${statusMsg.includes("Erreur") ? "text-error" : "text-ok"} animate-pulse whitespace-nowrap`}>
          {statusMsg}
        </span>
      )}
      
      <div className="flex items-center gap-2 bg-panel/50 border border-line rounded-lg p-1 shadow-inner">
        <div className="flex flex-col px-2 border-r border-line">
          <label className="text-[10px] text-ink-muted uppercase font-bold tracking-wider mb-0.5">Asset</label>
          <select 
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-transparent border-none text-sm text-ink focus:outline-none w-32 cursor-pointer appearance-none"
          >
            <option value="XAUUSD">XAUUSD</option>
            <option value="Step Index 200">Step Index 200</option>
          </select>
        </div>
        <div className="flex flex-col px-2">
          <label className="text-[10px] text-ink-muted uppercase font-bold tracking-wider mb-0.5">From</label>
          <input 
            type="month" 
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="bg-transparent border-none text-sm text-ink focus:outline-none w-28 cursor-pointer"
          />
        </div>
        <div className="w-px h-8 bg-line"></div>
        <div className="flex flex-col px-2">
          <label className="text-[10px] text-ink-muted uppercase font-bold tracking-wider mb-0.5">To</label>
          <input 
            type="month" 
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="bg-transparent border-none text-sm text-ink focus:outline-none w-28 cursor-pointer"
          />
        </div>
      </div>

      <button
        onClick={handleTrain}
        disabled={isPending}
        className="flex items-center gap-2 px-4 py-2.5 h-[42px] bg-accent/10 hover:bg-accent/20 text-accent font-semibold rounded-lg border border-accent/30 transition-all shadow-[0_0_15px_rgba(var(--accent-rgb),0.15)] disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
      >
        <Play size={16} className={isPending ? "animate-pulse" : ""} />
        {isPending ? "Lancement..." : "Entraîner"}
      </button>
    </div>
  );
}
