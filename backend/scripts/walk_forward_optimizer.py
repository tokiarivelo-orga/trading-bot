#!/usr/bin/env python3
"""
Walk-Forward Parameter Optimizer for Adaptive Strategies.

This script performs periodic re-optimization of strategy parameters over 
recent market data (e.g., the last 7 to 30 days). 
It generates `param_overrides` that can be automatically merged into the 
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
    In a full production setup, this would call the `src.backtest.engine`.
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
