import { promises as fs } from "fs";
import path from "path";
import DashboardClient from "./DashboardClient";
import { MenuButton } from "@/shared/ui/NavigationDrawer";
import TrainButton from "./TrainButton";
import ModelSelector from "./ModelSelector";
import ToggleBotButton from "./ToggleBotButton";

// Ensure this runs dynamically so it picks up the latest file changes
export const dynamic = "force-dynamic";

export default async function ModelDashboardPage({ searchParams }: { searchParams: Promise<{ symbol?: string }> }) {
  const params = await searchParams;
  const symbol = params.symbol || "XAUUSD";
  const modelName = symbol === "XAUUSD" ? "smc_dl_m5" : "smc_dl_m5_step200";

  // Read the JSON file from the backend directory
  const reportPath = path.join(
    process.cwd(),
    `../backend/src/backtest/reports/${modelName}_${symbol}_2026-06_2026-08.json`
  );

  let report = null;
  let error = null;

  try {
    const fileContent = await fs.readFile(reportPath, "utf-8");
    report = JSON.parse(fileContent);
  } catch (e: any) {
    error = e.message;
  }

  // Parse probabilities out of the signals
  const parsedSignals = [];
  if (report && report.signals) {
    // Create a map of trades by time and direction to quickly look up profit
    const tradesMap = new Map();
    if (report.trades) {
      for (const trade of report.trades) {
        // trade.open_time is an ISO string like "2026-06-02T15:25:00Z"
        // key it by time + side
        tradesMap.set(`${trade.open_time}_${trade.side.toUpperCase()}`, trade);
      }
    }

    for (const sig of report.signals.reverse().slice(0, 500)) { // Top 500 recent
      // Example reason: "DL Buy (tp=0.77, bull=0.41, bear=0.37)"
      const match = sig.reason.match(/tp=([0-9.]+), bull=([0-9.]+), bear=([0-9.]+)/);
      
      const tradeKey = `${sig.time}_${sig.direction.toUpperCase()}`;
      const matchingTrade = tradesMap.get(tradeKey);

      parsedSignals.push({
        ...sig,
        entry_price: matchingTrade ? matchingTrade.open_price : sig.price,
        profit: matchingTrade ? matchingTrade.profit : null,
        direction: sig.direction.toUpperCase(),
        probs: match
          ? {
              tp: parseFloat(match[1]),
              bull: parseFloat(match[2]),
              bear: parseFloat(match[3]),
            }
          : null,
      });
    }
  }

  // Fetch live history from API
  let liveSignals: any[] = [];
  try {
    const res = await fetch(`http://localhost:8000/accounts/default/journal/history?symbol=${encodeURIComponent(symbol)}&limit=500`, {
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      liveSignals = data.items.map((item: any) => {
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
    }
  } catch (err) {
    console.error("Failed to fetch live signals:", err);
  }

  return (
    <div className="flex h-full flex-col bg-background text-ink">
      <header className="border-b border-line px-6 py-4 flex items-center gap-4 justify-between">
        <div className="flex items-center gap-4">
          <MenuButton />
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              SMC Deep Learning Dashboard <span className="text-accent text-sm font-normal ml-2">beta</span>
            </h1>
            <p className="text-xs text-ink-muted mt-1">Live visualization of Neural Network decisions during backtest</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <ToggleBotButton symbol={symbol} />
          <ModelSelector currentSymbol={symbol} />
          <TrainButton />
        </div>
      </header>

      <main className="flex-1 overflow-auto p-6">
        {error ? (
          <div className="rounded border border-error/50 bg-error/10 p-4 text-error">
            <h2 className="font-bold">Error loading report</h2>
            <p className="text-sm">{error}</p>
            <p className="text-xs mt-2 text-ink-muted">Waiting for backtest to complete...</p>
          </div>
        ) : (
          <DashboardClient symbol={symbol} report={report} parsedSignals={parsedSignals} liveSignals={liveSignals} />
        )}
      </main>
    </div>
  );
}
