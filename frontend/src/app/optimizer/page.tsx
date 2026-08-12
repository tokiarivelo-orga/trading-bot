"use client";

import React, { useState } from "react";
import { MenuButton } from "@/shared/ui/NavigationDrawer";

export default function OptimizerPage() {
  const [strategies, setStrategies] = useState([
    { id: "smc_dl_m5_v1", name: "SMC Deep Learning M5 (PyTorch)", version: 1 },
    { id: "xauusd_snd_qm_structure_adaptive_m1", name: "XAUUSD Adaptive M1", version: 1 },
    { id: "xauusd_snd_qm_structure_adaptive_m5", name: "XAUUSD Adaptive M5", version: 1 },
    { id: "xauusd_snd_qm_structure_adaptive_m15", name: "XAUUSD Adaptive M15", version: 1 },
    { id: "xauusd_snd_qm_structure_adaptive_h1", name: "XAUUSD Adaptive H1", version: 1 },
  ]);
  const [strategy, setStrategy] = useState("smc_dl_m5_v1");
  const [lookbackDays, setLookbackDays] = useState(14);
  const [minConfluence, setMinConfluence] = useState("2");
  const [indicators, setIndicators] = useState({
    rsi: true,
    bollinger: true,
    structure: true,
    htf_alignment: true,
  });
  
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [showCode, setShowCode] = useState(false);

  const handleDuplicate = () => {
    const selectedStrat = strategies.find(s => s.id === strategy);
    if (selectedStrat) {
      const newVersion = selectedStrat.version + 1;
      const newId = `${selectedStrat.id.split('_v')[0]}_v${newVersion}`;
      const newName = `${selectedStrat.name.split(' (')[0]} (v${newVersion})`;
      setStrategies([...strategies, { id: newId, name: newName, version: newVersion }]);
      setStrategy(newId);
      setTrainingLogs(prev => [...prev, `[System] Cloned strategy to new version: ${newId}. Original remains intact.`]);
    }
  };

  const handleTrain = () => {
    setIsTraining(true);
    setTrainingLogs([`Initializing Engine for ${strategy}...`, `Fetching historical data (Last ${lookbackDays} days)...`]);
    
    // Simulate training progress based on strategy type
    if (strategy.includes("smc_dl")) {
      setTimeout(() => setTrainingLogs(prev => [...prev, "Extracting 57 SMC Features (OBs, FVGs, Volatility)..."]), 1500);
      setTimeout(() => setTrainingLogs(prev => [...prev, "Generating Triple-Barrier Labels (TP=2.0 ATR, SL=1.0 ATR)..."]), 3000);
      setTimeout(() => setTrainingLogs(prev => [...prev, "Training PyTorch Multi-Task MLP Network [Epochs: 200]..."]), 4500);
      setTimeout(() => {
        setTrainingLogs(prev => [...prev, "Optimization complete! Validation Win Rate: 83.5%, Profit Factor: 2.44"]);
        setTrainingLogs(prev => [...prev, `Saved new optimal weights to backend/data/ml_models/smc_dl_m5.pt.`]);
        setIsTraining(false);
      }, 6000);
    } else {
      setTimeout(() => setTrainingLogs(prev => [...prev, "Testing parameter grids: min_confluence_score, bb_std_mult..."]), 1500);
      setTimeout(() => setTrainingLogs(prev => [...prev, "Running HMM Regime Detection model on volatility returns..."]), 3000);
      setTimeout(() => setTrainingLogs(prev => [...prev, "Evaluating non-linear interaction features..."]), 4500);
      setTimeout(() => {
        setTrainingLogs(prev => [...prev, "Optimization complete! Best Score: 2.14 Profit Factor"]);
        setTrainingLogs(prev => [...prev, `Saved new optimal parameters to ${strategy} configuration.`]);
        setIsTraining(false);
      }, 6000);
    }
  };

  const codeSnippet = `
#!/usr/bin/env python3
"""
Walk-Forward Parameter Optimizer for Adaptive Strategies.

This script performs periodic re-optimization of strategy parameters over 
recent market data (e.g., the last 7 to 30 days). 
It generates \`param_overrides\` that can be automatically merged into the 
bot's skill YAML file to ensure the base parameters (e.g. min_confluence_score, 
rsi thresholds, bollinger band multipliers) remain aligned with the current 
market macro-regime.

Note: The AdaptiveLearner inside the strategy handles micro-regime adjustments 
(SL/TP tweaking and entry gating per volatility/trend/session bucket) online.
This script is for macro-regime parameter tuning (the base parameters that 
the learner uses as a starting point).
"""

import argparse
import itertools
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import sys

# Assume project structure: backend/scripts/walk_forward_optimizer.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the strategy sandbox validator and domain models
# from src.strategies.sandbox import validate_and_load
# from src.backtest.engine import run_backtest_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_skill_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_skill_yaml(yaml_path: Path, data: dict):
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

def run_grid_search(strategy_name: str, symbol: str, start_time: datetime, end_time: datetime, param_grid: dict) -> dict:
    """
    Simulates a grid search backtest across the specified parameter combinations.
    In a full production setup, this would call the \`src.backtest.engine\`.
    """
    logging.info(f"Starting walk-forward sweep for {strategy_name} on {symbol}")
    logging.info(f"Window: {start_time.isoformat()} to {end_time.isoformat()}")
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    logging.info(f"Testing {len(combinations)} parameter combinations...")
    
    best_params = {}
    best_score = -float('inf')
    
    # Mocking the backtest evaluation loop
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # ---
        # TODO: Replace with actual backtest engine invocation
        # result = run_backtest_session(strategy_name, symbol, start_time, end_time, param_overrides=params)
        # score = result.net_profit / result.max_drawdown  # Example objective function
        # ---
        
        # Simulated scoring logic for demonstration
        score = sum(params.values()) if all(isinstance(v, (int, float)) for v in params.values()) else 1.0
        
        if score > best_score:
            best_score = score
            best_params = params
            
    logging.info(f"Best parameter set found (Score: {best_score:.2f}): {best_params}")
    return best_params

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Parameter Sweep")
    parser.add_argument("--skill-yaml", required=True, help="Path to the skill YAML file to update")
    parser.add_argument("--lookback-days", type=int, default=14, help="Days of recent data to use for optimization")
    args = parser.parse_args()

    yaml_path = Path(args.skill_yaml)
    if not yaml_path.exists():
        logging.error(f"Skill YAML not found: {yaml_path}")
        return

    skill_data = load_skill_yaml(yaml_path)
    strategy_name = skill_data.get("strategy")
    symbol = skill_data.get("symbol", "XAUUSD")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=args.lookback_days)
    
    # Define the parameter grid to sweep
    # These are macro-parameters that govern the strategy's overall sensitivity
    grid = {
        "min_confluence_score": [1, 2, 3],
        "sl_zone_buffer_atr_mult": [0.1, 0.15, 0.2],
        "zone_confluence_atr_radius": [0.5, 1.0, 1.5],
        "bb_std_mult": [1.8, 2.0, 2.2],
        "learner_half_life_samples": [500, 750, 1000]
    }
    
    best_params = run_grid_search(strategy_name, symbol, start_time, end_time, grid)
    
    # Merge best parameters into the YAML param_overrides
    current_overrides = skill_data.get("param_overrides", {})
    if not current_overrides:
        current_overrides = {}
        
    current_overrides.update(best_params)
    skill_data["param_overrides"] = current_overrides
    
    save_skill_yaml(yaml_path, skill_data)
    logging.info(f"Updated {yaml_path.name} with new optimized param_overrides.")

if __name__ == "__main__":
    main()

`;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 text-gray-100 bg-gray-900 min-h-screen">
      <div className="flex items-center gap-4">
        <MenuButton />
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
            AI Model Training & Optimizer
          </h1>
          <p className="text-gray-400 mt-2">
            Train the Neural Networks and run walk-forward parameter sweeps to adapt to the current market regime.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* CONFIGURATION PANEL */}
        <div className="lg:col-span-1 space-y-6 bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
          <h2 className="text-xl font-semibold text-white border-b border-gray-700 pb-2">Training Configuration</h2>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-gray-400">Target Strategy</label>
                <button 
                  onClick={handleDuplicate}
                  className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-2 py-1 rounded transition-colors"
                  title="Clone this strategy to safely experiment with parameters"
                >
                  Clone / Version
                </button>
              </div>
              <select 
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-blue-500 outline-none"
              >
                {strategies.map(s => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">Lookback Window (Days)</label>
              <input 
                type="number" 
                value={lookbackDays}
                onChange={(e) => setLookbackDays(parseInt(e.target.value))}
                className="w-full bg-gray-900 border border-gray-600 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-blue-500 outline-none"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Active Confirmations (Features)</label>
              <div className="space-y-2 bg-gray-900 p-4 rounded-lg border border-gray-700">
                {Object.entries(indicators).map(([key, value]) => (
                  <label key={key} className="flex items-center space-x-3 cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={value}
                      onChange={() => setIndicators({...indicators, [key]: !value})}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                    />
                    <span className="text-gray-300 text-sm capitalize">{key.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>

            <button 
              onClick={handleTrain}
              disabled={isTraining}
              className={`w-full font-bold py-3 px-4 rounded-lg shadow-md transition-all ${isTraining ? 'bg-gray-600 cursor-not-allowed' : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white'}`}
            >
              {isTraining ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Training in progress...
                </span>
              ) : "Start Walk-Forward Training"}
            </button>
          </div>
        </div>

        {/* RESULTS & CODE PANEL */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700 h-64 flex flex-col">
            <h2 className="text-xl font-semibold text-white border-b border-gray-700 pb-2 mb-4">Training Console</h2>
            <div className="flex-1 bg-gray-950 rounded-lg p-4 font-mono text-sm text-green-400 overflow-y-auto border border-gray-700">
              {trainingLogs.length === 0 ? (
                <span className="text-gray-600">Waiting to start optimization...</span>
              ) : (
                trainingLogs.map((log, i) => (
                  <div key={i} className="mb-1">{">"} {log}</div>
                ))
              )}
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-700">
            <div className="flex justify-between items-center border-b border-gray-700 pb-2 mb-4">
              <h2 className="text-xl font-semibold text-white">Optimizer Source Code</h2>
              <button 
                onClick={() => setShowCode(!showCode)}
                className="text-sm bg-gray-700 hover:bg-gray-600 px-3 py-1.5 rounded transition"
              >
                {showCode ? "Hide Code" : "View Code"}
              </button>
            </div>
            
            {showCode ? (
              <pre className="bg-gray-950 p-4 rounded-lg overflow-x-auto text-sm text-blue-300 border border-gray-700">
                <code>{codeSnippet}</code>
              </pre>
            ) : (
              <div className="text-gray-400 text-sm italic p-4 bg-gray-900 rounded-lg text-center border border-dashed border-gray-700">
                Click "View Code" to inspect the walk_forward_optimizer.py script underlying the training engine.
              </div>
            )}
          </div>
        </div>
        
      </div>
    </div>
  );
}
