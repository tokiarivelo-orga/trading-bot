"""Bot-level backtest CLI — replays a specific deployed bot (its skill's
session windows, `risk_multiplier` and `param_overrides`), not just its raw
strategy.

    uv run python -m src.backtest.bot_cli <bot_slug> <symbol> <period>
    uv run python -m src.backtest.bot_cli xauusd_structure_shift_m1 XAUUSD 2026-05:2026-09

`src.backtest.cli` (`make backtest strategy=...`) answers "how would this
strategy have performed" — it ignores every per-bot override
(`skills/normal/<symbol>/<bot_slug>.yaml`'s `sessions`, `risk_multiplier`,
`param_overrides`, `htf_veto_override`) via `FixedSkillSelector`. That's fine
for a strategy that has no bot with real overrides, but for one that does
(a session-gated bot, a non-1.0 risk_multiplier, or any `param_overrides`)
its numbers can diverge sharply from what the live bot actually experiences.
This entry point loads the bot's actual `NormalSkill` from disk and passes it
to `run_backtest(bot_skill=...)`, which routes through the same
`SkillSelector`/`_effective_strategy` the live engine uses — "how would this
bot have performed."

`<bot_slug>` is the skill YAML's filename without `.yaml`
(`skills/normal/<symbol>/<bot_slug>.yaml`'s `name` field's last path segment)
— e.g. `xauusd_structure_shift_m1` for
`skills/normal/xauusd/xauusd_structure_shift_m1.yaml`.

Same history/backfill requirements as `src.backtest.cli` — see that module's
docstring and `.claude/skills/backtest/SKILL.md`. The report is written to
`backend/src/backtest/reports/<bot_slug>_<symbol>_<period_with_underscores>.json`
(distinct from the plain strategy-name filename `cli.py` writes, so the two
never collide even when run against the same strategy/symbol/period).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
from pathlib import Path

from src.backtest.application.period import InvalidPeriod
from src.backtest.application.run_backtest import NoHistoryError, NoSymbolSpecError, run_backtest
from src.backtest.reports.writer import render_summary, write_report
from src.shared.config.settings import Settings
from src.skills.adapters.normal_skill_repository import NormalSkillRepository

# Mirrors container.py's `_SKILLS_DIR` — `skills/normal/<symbol>/<bot_slug>.yaml`
# lives under `backend/src/skills/`, not under `configs/` (`CONFIGS_DIR`).
_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills" / "normal"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m src.backtest.bot_cli <bot_slug> <symbol> <period>", file=sys.stderr)
        print(
            "e.g.:  python -m src.backtest.bot_cli xauusd_structure_shift_m1 XAUUSD "
            "2026-05:2026-09",
            file=sys.stderr,
        )
        return 2

    bot_slug, symbol, period = argv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()

    repository = NormalSkillRepository(_SKILLS_DIR)
    skill = repository.get(symbol, bot_slug)
    if skill is None:
        print(
            f"error: no skills/normal/{symbol.lower()}/{bot_slug}.yaml on disk",
            file=sys.stderr,
        )
        return 1

    try:
        report = asyncio.run(
            run_backtest(
                skill.strategy,
                symbol,
                period,
                bot_skill=skill,
                database_url=settings.database_url,
            )
        )
    except (InvalidPeriod, NoHistoryError, NoSymbolSpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = dataclasses.replace(report, strategy=bot_slug)
    path = write_report(report)
    print(render_summary(report))
    print(f"\nFull report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
