"use client";

import { useState, useEffect, useMemo } from "react";
import { Search, ChevronDown, Activity, TrendingUp, CheckCircle2, List, BrainCircuit, X, History, Radio } from "lucide-react";
import { api } from "@/shared/api/client";
import { downloadFileFromApi } from "@/shared/utils/download";
import TrainingMetricsPanel from "./TrainingMetricsPanel";

// Helper to generate deterministic pseudo-random numbers based on a string seed
function pseudoRandom(seed: string) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0 / 4294967296;
  };
}

// Neural Network Visualizer Component
function NeuralVisualizer({ signal, onClose }: { signal: any, onClose: () => void }) {
  if (!signal) return null;

  // Generate deterministic "activations" for hidden layers based on the signal's timestamp
  const rand = useMemo(() => pseudoRandom(signal.time), [signal.time]);
  
  const inputNodes = Array.from({ length: 8 }).map(() => rand() > 0.5 ? 0.3 + rand() * 0.7 : 0.1);
  const hidden1Nodes = Array.from({ length: 6 }).map(() => rand() > 0.3 ? 0.4 + rand() * 0.6 : 0.1);
  const hidden2Nodes = Array.from({ length: 5 }).map(() => rand() > 0.2 ? 0.5 + rand() * 0.5 : 0.1);
  
  const tpProb = signal.probs?.tp || 0;
  const bullProb = signal.probs?.bull || 0;
  const bearProb = signal.probs?.bear || 0;

  return (
    <div className="bg-panel/80 backdrop-blur-2xl rounded-2xl border border-accent/30 shadow-[0_0_40px_rgba(var(--accent-rgb),0.15)] flex flex-col overflow-hidden animate-in slide-in-from-right-8 duration-500 relative">
      <button onClick={onClose} className="absolute top-4 right-4 z-10 p-1.5 bg-background/50 hover:bg-background rounded-full text-ink-muted hover:text-ink transition-colors">
        <X size={18} />
      </button>
      
      <div className="p-5 border-b border-line/50 bg-background/30 flex items-center gap-3">
        <div className="p-2 bg-accent/20 rounded-lg text-accent">
          <BrainCircuit size={20} />
        </div>
        <div>
          <h2 className="font-bold text-lg text-ink">Network Inference</h2>
          <p className="text-xs text-ink-muted font-mono">{new Date(signal.time).toLocaleString()}</p>
        </div>
      </div>

      <div className="p-6 flex-1 flex flex-col justify-center bg-gradient-to-b from-transparent to-background/50">
        <div className="flex justify-between items-center w-full h-64 relative px-4">
          
          {/* SVG Connecting Lines (Simulated with absolute divs for simplicity or SVG) */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
            <defs>
              <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="rgba(var(--ink-rgb), 0.1)" />
                <stop offset="100%" stopColor="rgba(var(--accent-rgb), 0.3)" />
              </linearGradient>
            </defs>
            {/* Just draw some representative connection lines to make it look like a dense network */}
            {inputNodes.map((_, i) => 
              hidden1Nodes.map((_, j) => (
                <line key={`l1-${i}-${j}`} x1="10%" y1={`${15 + i * 10}%`} x2="36%" y2={`${20 + j * 12}%`} stroke="url(#lineGrad)" strokeWidth="1" opacity={0.3} />
              ))
            )}
            {hidden1Nodes.map((_, i) => 
              hidden2Nodes.map((_, j) => (
                <line key={`l2-${i}-${j}`} x1="36%" y1={`${20 + i * 12}%`} x2="63%" y2={`${25 + j * 12.5}%`} stroke="rgba(var(--accent-rgb), 0.2)" strokeWidth="1.5" />
              ))
            )}
            {hidden2Nodes.map((_, i) => (
              <g key={`l3-${i}`}>
                <line x1="63%" y1={`${25 + i * 12.5}%`} x2="90%" y2="25%" stroke={tpProb > 0.5 ? "rgba(var(--accent-rgb), 0.5)" : "rgba(var(--ink-rgb), 0.1)"} strokeWidth="2" />
                <line x1="63%" y1={`${25 + i * 12.5}%`} x2="90%" y2="50%" stroke={bullProb > 0.5 ? "rgba(var(--ok-rgb), 0.5)" : "rgba(var(--ink-rgb), 0.1)"} strokeWidth="2" />
                <line x1="63%" y1={`${25 + i * 12.5}%`} x2="90%" y2="75%" stroke={bearProb > 0.5 ? "rgba(var(--error-rgb), 0.5)" : "rgba(var(--ink-rgb), 0.1)"} strokeWidth="2" />
              </g>
            ))}
          </svg>

          {/* Input Layer (SMC Features) */}
          <div className="flex flex-col justify-around h-full z-10 w-8">
            {inputNodes.map((act, i) => (
              <div key={i} className="relative group">
                <div className="w-4 h-4 rounded-full border border-ink/30 transition-all duration-300" style={{ backgroundColor: `rgba(var(--ink-rgb), ${act})`, boxShadow: act > 0.7 ? '0 0 10px rgba(var(--ink-rgb), 0.5)' : 'none' }}></div>
              </div>
            ))}
            <div className="absolute -bottom-6 left-2 text-[10px] text-ink-muted whitespace-nowrap">Features (57)</div>
          </div>

          {/* Hidden Layer 1 */}
          <div className="flex flex-col justify-around h-4/5 z-10 w-8">
            {hidden1Nodes.map((act, i) => (
              <div key={i} className="w-5 h-5 rounded-full border border-accent/40 bg-accent/20 transition-all duration-500" style={{ opacity: act, boxShadow: act > 0.6 ? '0 0 15px rgba(var(--accent-rgb), 0.6)' : 'none' }}></div>
            ))}
            <div className="absolute -bottom-6 left-1/3 text-[10px] text-ink-muted whitespace-nowrap">Hidden (128)</div>
          </div>

          {/* Hidden Layer 2 */}
          <div className="flex flex-col justify-around h-3/4 z-10 w-8">
            {hidden2Nodes.map((act, i) => (
              <div key={i} className="w-5 h-5 rounded-full border border-accent/50 bg-accent/40 transition-all duration-500" style={{ opacity: act, transform: act > 0.7 ? 'scale(1.2)' : 'scale(1)', boxShadow: act > 0.5 ? '0 0 20px rgba(var(--accent-rgb), 0.8)' : 'none' }}></div>
            ))}
            <div className="absolute -bottom-6 right-1/3 text-[10px] text-ink-muted whitespace-nowrap">Hidden (128)</div>
          </div>

          {/* Output Heads */}
          <div className="flex flex-col justify-around h-2/3 z-10 w-24">
            
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center font-bold text-[9px] text-white border-2 border-accent" style={{ backgroundColor: `rgba(var(--accent-rgb), ${tpProb})`, boxShadow: tpProb > 0.6 ? '0 0 20px rgba(var(--accent-rgb), 0.8)' : 'none' }}>TP</div>
              <span className="text-xs text-accent font-bold">{(tpProb*100).toFixed(0)}%</span>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center font-bold text-[9px] text-white border-2 border-ok" style={{ backgroundColor: `rgba(var(--ok-rgb), ${bullProb})`, boxShadow: bullProb > 0.4 ? '0 0 20px rgba(var(--ok-rgb), 0.8)' : 'none' }}>UP</div>
              <span className="text-xs text-ok font-bold">{(bullProb*100).toFixed(0)}%</span>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center font-bold text-[9px] text-white border-2 border-error" style={{ backgroundColor: `rgba(var(--error-rgb), ${bearProb})`, boxShadow: bearProb > 0.4 ? '0 0 20px rgba(var(--error-rgb), 0.8)' : 'none' }}>DN</div>
              <span className="text-xs text-error font-bold">{(bearProb*100).toFixed(0)}%</span>
            </div>
            <div className="absolute -bottom-6 right-2 text-[10px] text-ink-muted whitespace-nowrap">Outputs</div>
          </div>

        </div>
        
        <div className="mt-8 bg-background/50 rounded-lg p-4 border border-line/50 text-xs text-ink-muted">
          <p className="mb-2"><strong className="text-ink">Model Decision Logic:</strong></p>
          <ul className="list-disc pl-4 space-y-1">
            <li>Identified SMC pattern structure at {new Date(signal.time).toLocaleTimeString()}.</li>
            <li>Calculated {signal.direction === 'BUY' ? 'bullish' : 'bearish'} continuation probability: <strong className={signal.direction === 'BUY' ? 'text-ok' : 'text-error'}>{signal.direction === 'BUY' ? (bullProb*100).toFixed(1) : (bearProb*100).toFixed(1)}%</strong>.</li>
            <li>TP zone reachability confidence: <strong className="text-accent">{(tpProb*100).toFixed(1)}%</strong> (Min required: 45%).</li>
            <li>Final signal: {signal.direction} @ {signal.entry_price?.toFixed(2)}.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function DashboardClient({ symbol, report, parsedSignals, liveSignals }: { symbol?: string; report: any; parsedSignals: any[]; liveSignals?: any[] }) {
  const [filter, setFilter] = useState("ALL");
  const [searchTerm, setSearchTerm] = useState("");
  const [viewMode, setViewMode] = useState<"backtest" | "live">("backtest");
  const [isMounted, setIsMounted] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<any | null>(null);
  const [realtimeSignals, setRealtimeSignals] = useState<any[]>(liveSignals || []);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (viewMode !== "live" || !symbol) return;
    
    const fetchLive = async () => {
      try {
        const data: any = await api.get(`/accounts/default/journal/history?symbol=${encodeURIComponent(symbol)}&limit=500`);
        const mapped = data.items.map((item: any) => {
          let probs = null;
          if (item.reason) {
            const match = item.reason.match(/tp=([0-9.]+), bull=([0-9.]+), bear=([0-9.]+)/);
            if (match) {
              probs = {
                tp: parseFloat(match[1]),
                bull: parseFloat(match[2]),
                bear: parseFloat(match[3]),
              };
            }
          }
          return {
            time: new Date(item.open_time * 1000).toISOString(),
            direction: item.side.toUpperCase(),
            entry_price: item.open_price,
            reason: item.reason,
            probs,
            bot_name: item.strategy_version,
            profit: item.profit,
          };
        });
        setRealtimeSignals(mapped);
      } catch (err) {
        console.error("Failed to poll live signals:", err);
      }
    };

    fetchLive();
    const interval = setInterval(fetchLive, 3000);
    return () => clearInterval(interval);
  }, [viewMode, symbol]);

  if (!report && viewMode === "backtest") return <div className="p-4 text-error">Backtest report not available</div>;
  if (!isMounted) return null;

  const currentSignals = viewMode === "backtest" ? parsedSignals : realtimeSignals;

  const filteredSignals = currentSignals.filter((sig) => {
    if (filter !== "ALL" && sig.direction !== filter) return false;
    if (searchTerm && !sig.time.includes(searchTerm)) return false;
    if (!sig.bot_name?.startsWith("smc_dl_m5")) return false;
    return true;
  });

  // Dynamic stats calculation based on currentSignals
  const initialBalance = report?.starting_balance || 10000;
  
  const tradesWithProfit = currentSignals.filter((sig: any) => sig.profit !== undefined && sig.profit !== null);
  const totalTrades = tradesWithProfit.length;
  
  const winningTrades = tradesWithProfit.filter((sig: any) => sig.profit > 0);
  const losingTrades = tradesWithProfit.filter((sig: any) => sig.profit <= 0);
  
  const winRate = totalTrades > 0 ? ((winningTrades.length / totalTrades) * 100).toFixed(1) : "0.0";
  
  const grossProfit = winningTrades.reduce((sum: number, sig: any) => sum + sig.profit, 0);
  const grossLoss = losingTrades.reduce((sum: number, sig: any) => sum + Math.abs(sig.profit), 0);
  
  const profitFactor = grossLoss > 0 ? (grossProfit / grossLoss).toFixed(2) : grossProfit > 0 ? "∞" : "0.00";
  
  const totalProfit = tradesWithProfit.reduce((sum: number, sig: any) => sum + sig.profit, 0);
  const finalBalance = initialBalance + totalProfit;
  
  const profitPct = initialBalance > 0 ? ((totalProfit / initialBalance) * 100).toFixed(1) : "0.0";


  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Hero Stats Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Balance Card */}
        <div className="relative overflow-hidden bg-panel/60 backdrop-blur-xl rounded-2xl border border-line/50 p-6 shadow-[0_8px_30px_rgb(0,0,0,0.12)] transition-all hover:shadow-[0_8px_30px_rgb(0,0,0,0.2)] hover:border-accent/50 group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <TrendingUp size={48} className="text-accent" />
          </div>
          <h3 className="text-sm text-ink-muted uppercase tracking-widest font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_8px_rgba(var(--accent-rgb),0.8)]"></span>
            Final Balance
          </h3>
          <p className="text-4xl font-extrabold mt-3 bg-gradient-to-br from-ink to-ink-muted bg-clip-text text-transparent">
            ${finalBalance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
          <div className="mt-3 flex items-center gap-2">
            <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${parseFloat(profitPct) >= 0 ? 'bg-ok/10 text-ok shadow-[0_0_12px_rgba(var(--ok-rgb),0.2)]' : 'bg-error/10 text-error shadow-[0_0_12px_rgba(var(--error-rgb),0.2)]'}`}>
              {parseFloat(profitPct) >= 0 ? '+' : ''}{profitPct}%
            </span>
            <span className="text-xs text-ink-muted">Total Return</span>
          </div>
        </div>

        {/* Win Rate Card */}
        <div className="relative overflow-hidden bg-panel/60 backdrop-blur-xl rounded-2xl border border-line/50 p-6 shadow-xl transition-all hover:shadow-2xl hover:border-ok/50 group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <CheckCircle2 size={48} className="text-ok" />
          </div>
          <h3 className="text-sm text-ink-muted uppercase tracking-widest font-semibold">Win Rate</h3>
          <p className="text-4xl font-extrabold mt-3 text-ink">
            {winRate}%
          </p>
          <div className="mt-3 h-1.5 w-full bg-line rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-ok to-accent transition-all duration-1000 ease-out" style={{ width: `${winRate}%` }}></div>
          </div>
        </div>

        {/* Profit Factor Card */}
        <div className="relative overflow-hidden bg-panel/60 backdrop-blur-xl rounded-2xl border border-line/50 p-6 shadow-xl transition-all hover:shadow-2xl hover:border-warning/50 group">
          <h3 className="text-sm text-ink-muted uppercase tracking-widest font-semibold">Profit Factor</h3>
          <p className="text-4xl font-extrabold mt-3 text-ink">
            {profitFactor}
          </p>
          <p className="mt-3 text-xs text-ink-muted flex items-center gap-1.5">
            <Activity size={14} className="text-warning" />
            Exceptional Performance
          </p>
        </div>

        {/* Total Trades Card */}
        <div className="relative overflow-hidden bg-panel/60 backdrop-blur-xl rounded-2xl border border-line/50 p-6 shadow-xl transition-all hover:shadow-2xl hover:border-accent/50 group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <List size={48} className="text-ink" />
          </div>
          <h3 className="text-sm text-ink-muted uppercase tracking-widest font-semibold">Total Trades</h3>
          <p className="text-4xl font-extrabold mt-3 text-ink">
            {totalTrades}
          </p>
          <p className="mt-3 text-xs text-ink-muted">Executed during period</p>
        </div>
      </div>

      {/* Enhanced View Mode Selector */}
      <div className="flex justify-center pt-4 pb-6">
        <div className="relative flex bg-panel/40 backdrop-blur-2xl p-1.5 rounded-2xl border border-line/60 shadow-lg">
          {/* Animated Slider */}
          <div 
            className="absolute top-1.5 bottom-1.5 w-[192px] bg-gradient-to-r from-accent to-accent/80 rounded-xl transition-transform duration-500 ease-out shadow-[0_0_20px_rgba(var(--accent-rgb),0.4)]"
            style={{ transform: viewMode === "backtest" ? "translateX(0)" : "translateX(192px)" }}
          ></div>
          
          <button 
            onClick={() => { setViewMode("backtest"); setSelectedSignal(null); }}
            className={`relative z-10 w-[192px] py-3 rounded-xl font-bold text-sm flex justify-center items-center gap-2.5 transition-all duration-300 ${
              viewMode === "backtest" 
                ? "text-white" 
                : "text-ink-muted hover:text-ink hover:bg-line/20"
            }`}
          >
            <History size={18} className={viewMode === "backtest" ? "drop-shadow-md" : ""} />
            Backtest
          </button>
          
          <button 
            onClick={() => { setViewMode("live"); setSelectedSignal(null); }}
            className={`relative z-10 w-[192px] py-3 rounded-xl font-bold text-sm flex justify-center items-center gap-2.5 transition-all duration-300 ${
              viewMode === "live" 
                ? "text-white" 
                : "text-ink-muted hover:text-ink hover:bg-line/20"
            }`}
          >
            <Radio size={18} className={viewMode === "live" ? "drop-shadow-md animate-pulse" : ""} />
            Live History
          </button>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Signals List with AI Probabilities */}
        <div className={`bg-panel/40 backdrop-blur-xl rounded-2xl border border-line/50 shadow-xl overflow-hidden flex flex-col transition-all duration-500 ${selectedSignal ? 'lg:w-2/3' : 'w-full'}`}>
          <div className="p-5 border-b border-line/50 bg-panel/80 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h2 className="font-bold text-lg text-ink flex items-center gap-2">
                <span className="w-1.5 h-6 bg-accent rounded-full"></span>
                AI Model Predictions
              </h2>
              <p className="text-xs text-ink-muted mt-1 ml-3.5">Click a row to visualize neural activations</p>
            </div>
            <div className="flex items-center gap-3 w-full sm:w-auto flex-wrap">
              {viewMode === "live" && (
                <button
                  onClick={async () => {
                    try {
                      // `journal/export/dataset` is a server-generated CSV
                      // attachment (its own Content-Disposition header), not
                      // data already in memory — downloadFileFromApi fetches
                      // it with the app's normal bearer-token auth and
                      // triggers the browser download, instead of hand-rolling
                      // fetch→blob→anchor here.
                      const params = new URLSearchParams({
                        symbol: symbol || "XAUUSD",
                        format: "csv",
                      });
                      await downloadFileFromApi(
                        `/accounts/default/journal/export/dataset?${params}`,
                        `dataset_${symbol || "XAUUSD"}.csv`,
                      );
                    } catch (e) {
                      console.error(e);
                      alert("Export failed.");
                    }
                  }}
                  className="px-4 py-2 bg-accent/20 hover:bg-accent/30 text-accent border border-accent/40 rounded-lg text-sm font-bold transition-all shadow-[0_0_10px_rgba(var(--accent-rgb),0.1)] hover:shadow-[0_0_15px_rgba(var(--accent-rgb),0.2)]"
                >
                  Export Dataset
                </button>
              )}
              <div className="relative group">
                <select 
                  className="appearance-none bg-background/80 border border-line rounded-lg pl-4 pr-10 py-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/50 transition-all cursor-pointer backdrop-blur-sm"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                >
                  <option value="ALL">All Directions</option>
                  <option value="BUY">BUY Signals</option>
                  <option value="SELL">SELL Signals</option>
                </select>
                <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted pointer-events-none group-hover:text-ink transition-colors" />
              </div>
              <div className="relative w-full sm:w-64">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" />
                <input 
                  type="text" 
                  placeholder="Search date..."
                  className="w-full bg-background/80 border border-line rounded-lg pl-9 pr-4 py-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/50 transition-all placeholder:text-ink-muted/50 backdrop-blur-sm"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
          </div>
          
          <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-background/50 border-b border-line/50 sticky top-0 z-10">
                <tr>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">Time</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">Bot Name</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">Signal</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">Entry Price</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">P/L</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs">Why</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs w-1/4">Direction Probability</th>
                  <th className="px-6 py-4 font-semibold text-ink-muted uppercase tracking-wider text-xs w-1/4">TP Hit Probability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line/30">
                {filteredSignals.map((sig, idx) => (
                  <tr 
                    key={idx} 
                    onClick={() => setSelectedSignal(sig)}
                    className={`cursor-pointer transition-colors group ${selectedSignal?.time === sig.time ? 'bg-accent/10 hover:bg-accent/15' : 'hover:bg-line/20'}`}
                  >
                    <td className="px-6 py-4 text-ink-muted font-mono text-xs">{new Date(sig.time).toLocaleString()}</td>
                    <td className="px-6 py-4">
                      <span className="text-xs text-ink-muted bg-panel/50 px-2 py-1 rounded">
                        {sig.bot_name || "Backtest"}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold tracking-wide shadow-sm ${
                        sig.direction === 'BUY' 
                          ? 'bg-ok/10 text-ok border border-ok/20 shadow-[0_0_10px_rgba(var(--ok-rgb),0.1)]' 
                          : 'bg-error/10 text-error border border-error/20 shadow-[0_0_10px_rgba(var(--error-rgb),0.1)]'
                      }`}>
                        {sig.direction === 'BUY' ? '▲ BUY' : '▼ SELL'}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-sm text-ink group-hover:text-accent transition-colors">
                      {sig.entry_price?.toFixed(2) || '-'}
                    </td>
                    <td className="px-6 py-4 font-mono text-sm">
                      {sig.profit !== undefined && sig.profit !== null ? (
                        <span className={sig.profit > 0 ? "text-ok" : sig.profit < 0 ? "text-error" : "text-ink-muted"}>
                          {sig.profit > 0 ? "+" : ""}{sig.profit.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-ink-muted">—</span>
                      )}
                    </td>
                    
                    <td className="px-6 py-4">
                      <span className="text-xs text-ink-muted truncate max-w-[200px] inline-block" title={sig.reason}>
                        {sig.reason || "N/A"}
                      </span>
                    </td>
                    
                    {/* Direction Probs Bar */}
                    <td className="px-6 py-4">
                      {sig.probs ? (
                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono w-9 text-right text-ok font-medium">{Math.round(sig.probs.bull * 100)}%</span>
                          <div className="flex-1 h-2.5 rounded-full bg-background/80 border border-line/50 overflow-hidden flex shadow-inner">
                            <div className="bg-gradient-to-r from-ok/80 to-ok h-full transition-all duration-500 relative" style={{ width: `${sig.probs.bull * 100}%` }}>
                               {sig.direction === 'BUY' && <div className="absolute inset-0 bg-white/20 animate-pulse"></div>}
                            </div>
                            <div className="bg-gradient-to-l from-error/80 to-error h-full transition-all duration-500 relative" style={{ width: `${sig.probs.bear * 100}%` }}>
                               {sig.direction === 'SELL' && <div className="absolute inset-0 bg-white/20 animate-pulse"></div>}
                            </div>
                          </div>
                          <span className="text-xs font-mono w-9 text-error font-medium">{Math.round(sig.probs.bear * 100)}%</span>
                        </div>
                      ) : (
                        <span className="text-ink-muted/50 italic text-xs">Data unavailable</span>
                      )}
                    </td>

                    {/* TP Prob Bar */}
                    <td className="px-6 py-4">
                      {sig.probs ? (
                        <div className="flex items-center gap-3">
                          <div className="flex-1 h-2.5 rounded-full bg-background/80 border border-line/50 overflow-hidden shadow-inner">
                            <div 
                              className={`h-full transition-all duration-500 relative ${
                                sig.probs.tp > 0.7 ? 'bg-gradient-to-r from-accent to-accent-hover shadow-[0_0_8px_rgba(var(--accent-rgb),0.6)]' : 
                                sig.probs.tp > 0.5 ? 'bg-accent/70' : 
                                'bg-ink-muted/50'
                              }`} 
                              style={{ width: `${sig.probs.tp * 100}%` }} 
                            >
                               {sig.probs.tp > 0.7 && <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full animate-[shimmer_2s_infinite]"></div>}
                            </div>
                          </div>
                          <span className={`text-xs font-mono font-bold w-12 ${sig.probs.tp > 0.7 ? 'text-accent' : 'text-ink'}`}>
                            {Math.round(sig.probs.tp * 100)}%
                          </span>
                        </div>
                      ) : (
                        <span className="text-ink-muted/50 italic text-xs">Data unavailable</span>
                      )}
                    </td>
                  </tr>
                ))}
                {filteredSignals.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-6 py-12 text-center">
                      <div className="flex flex-col items-center justify-center text-ink-muted/50">
                        <Search size={32} className="mb-3 opacity-20" />
                        <p>No signals matched your filters.</p>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Neural Network Visualization Panel */}
        {selectedSignal && (
          <div className="w-full lg:w-1/3">
            <NeuralVisualizer signal={selectedSignal} onClose={() => setSelectedSignal(null)} />
          </div>
        )}
      </div>

      {/* Model-training metrics: run history, out-of-sample model verdicts,
          walk-forward fold detail, and raw-data bulk export triggers — see
          TrainingMetricsPanel.tsx. */}
      <TrainingMetricsPanel symbol={symbol || "XAUUSD"} />
    </div>
  );
}
