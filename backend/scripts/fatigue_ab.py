"""A/B sweep harness for parameter variants of a single strategy file.

    uv run python scripts/fatigue_ab.py <strategy> <symbol> <period>

Run from `backend/`. Loads the strategy's source through the real sandbox
(so a variant that would not load live cannot score here either), then runs
one backtest per named variant with `spec.params` patched, and prints the
comparison table.

Why patch params rather than run the skill YAMLs: `src.backtest.cli` selects
skills through `FixedSkillSelector`, which reports `skill=backtest` and never
applies a skill's `param_overrides`. The turbo variant *is* nothing but a set
of `param_overrides` on the flagship's strategy file, so the only way to
backtest turbo at all is to apply that dict here.

Reported metrics deliberately lead with profit factor, avg R and drawdown.
Win rate on this system is a trailing artifact — the engine's secure-base
rule (`position_manager.DEFAULT_SECURE_BUFFER_R_MULT = 0.2`) closes ~94% of
trades at exactly +0.20R, so win rate is close to constant across variants
and tells you nothing about which one is better.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from src.backtest.application.run_backtest import run_backtest
from src.shared.config.settings import Settings
from src.strategies.registry import StrategyRegistry
from src.strategies.sandbox import validate_and_load

# The turbo skill (`skills/normal/xauusd/xauusd_apex_turbo_m1.yaml`) is a
# pure param_overrides layer over the flagship strategy file. Mirrored here
# so turbo can be backtested; keep in sync with that YAML by hand.
TURBO = {
    "confirm_mode": "off",
    "max_signals_per_bar": 2,
    "max_touch_episode": 20,
    "max_entry_depth": 1.00,
    "zone_height_atr_min": 0.20,
    "zone_height_atr_max": 2.40,
    "min_impulse_atr": 0.55,
    "allow_counter_trend": True,
    "zri_min": 0.35,
    "neutral_min_score": 0.0,
    "tp2_min_rr": 1.8,
}

FILE_KEY = "__file__"
"""Variant key naming the strategy *file* to load, when a variant compares
two generated versions rather than two parameter sets (e.g. the active v3
against a validated-but-unactivated v4). Value is a filename inside
`src/strategies/generated/`. Absent means "use the file the CLI argument
resolves to"."""

# name -> params patched onto the file's own defaults.
VARIANTS: dict[str, dict] = {
    "flagship-baseline": {},
    "flagship-fatigue0.50": {"fatigue_max": 0.50},
    "flagship-fatigue0.40": {"fatigue_max": 0.40},
    "flagship-fatigue0.30": {"fatigue_max": 0.30},
    "turbo-baseline": dict(TURBO),
    "turbo-fatigue0.40": {**TURBO, "fatigue_max": 0.40},
    # Turbo is the only variant that enables counter-trend entries, so it is
    # the only place the "fade needs a tired move" permission can bite.
    "turbo-fade0.45": {**TURBO, "fatigue_fade_min": 0.45},
    # Threshold robustness. A real filter shows a *plateau* — neighbouring
    # thresholds all beat baseline. A single spike at 0.45 with 0.35/0.55
    # back at baseline means the 0.45 number was fitted to this window and
    # will not survive out of sample.
    "turbo-fade0.35": {**TURBO, "fatigue_fade_min": 0.35},
    "turbo-fade0.55": {**TURBO, "fatigue_fade_min": 0.55},
    "turbo-fade0.65": {**TURBO, "fatigue_fade_min": 0.65},
    # Control for the fade result. fade0.45/0.55/0.65 returned byte-identical
    # numbers, which means the threshold had stopped binding — no
    # counter-trend candidate ever scores that tired, so the "permission"
    # degenerates into "reject every counter-trend entry". If this variant
    # matches them exactly, the fade gain was never the fatigue measure
    # working; it was just turning counter-trend off, which is what the
    # flagship already does by default.
    "turbo-no-counter": {**TURBO, "allow_counter_trend": False},
    # ── xauusd_snd_qm_structure: the superseded family, still live ──
    # v3 is the ACTIVE version; v4 is marked `validated` but was never
    # activated. v4's only change is tp1_target_rr 1.2 -> 1.8, because at
    # 1.2 the Scalp leg could never clear XAUUSD's min_rr=1.5 spread gate
    # and so never filled in production. Run with
    #   `... xauusd_snd_qm_structure_m1 XAUUSD <period> sndqm-v3 sndqm-v4`
    "sndqm-v3": {FILE_KEY: "xauusd_snd_qm_structure_m1_v3.py"},
    "sndqm-v4": {FILE_KEY: "xauusd_snd_qm_structure_m1_v4.py"},
}


def _load(path: Path):
    strategy, errors = validate_and_load(path.read_text())
    if strategy is None:
        raise SystemExit(f"strategy failed sandbox validation: {errors}")
    return strategy


async def main(strategy_name: str, symbol: str, period: str, only: list[str]) -> None:
    settings = Settings()
    generated = Path("src/strategies/generated")
    default_path = generated / f"{strategy_name}_v5.py"

    rows = []
    for name, overrides in VARIANTS.items():
        if only and name not in only:
            continue
        overrides = dict(overrides)
        path = generated / overrides.pop(FILE_KEY, default_path.name)
        if not path.exists():
            raise SystemExit(f"no such strategy file: {path}")
        # A fresh instance per variant: these strategies memoise zone
        # detection on self, so reusing one across runs would leak state
        # from the previous variant's replay into the next.
        strategy = _load(path)
        strategy.spec.params.update(overrides)
        registry = StrategyRegistry()
        registry.register(strategy_name, strategy)

        report = await run_backtest(
            strategy_name,
            symbol,
            period,
            strategy_source=registry,
            database_url=settings.database_url,
        )
        rs = [t.r_multiple for t in report.trades if t.r_multiple is not None]
        scratch = sum(1 for r in rs if 0.0 < r < 0.25)
        real_tp = sum(1 for r in rs if r >= 1.0)
        rows.append(
            {
                "variant": name,
                "trades": len(report.trades),
                "signals": len(report.signals),
                "PF": round(report.profit_factor, 3),
                "avgR": round(report.avg_r, 4),
                "sumR": round(sum(rs), 1),
                "maxDD%": round(report.max_drawdown_pct, 2),
                "end": round(report.ending_balance, 2),
                "scratch%": round(100.0 * scratch / len(rs), 1) if rs else 0.0,
                "reachedTP": real_tp,
                "vetoes": dict(Counter(s.outcome for s in report.signals)),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    print("\n" + "=" * 118)
    print(
        f"{'variant':<22}{'trades':>7}{'PF':>8}{'avgR':>8}{'sumR':>9}"
        f"{'maxDD%':>8}{'end':>11}{'scratch%':>10}{'reachedTP':>11}"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row['variant']:<22}{row['trades']:>7}{row['PF']:>8.3f}{row['avgR']:>8.4f}"
            f"{row['sumR']:>9.1f}{row['maxDD%']:>8.2f}{row['end']:>11.2f}"
            f"{row['scratch%']:>10.1f}{row['reachedTP']:>11}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python scripts/fatigue_ab.py <strategy> <symbol> <period> [variant ...]"
        )
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
