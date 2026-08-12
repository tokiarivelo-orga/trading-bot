"""Seed the XAUUSD APEX strategy family into the StrategyVersion DB table
so the Apex bots appear in the Bots UI and can be backtested.

Run from `backend/`:

    uv run python -m scripts.seed_xauusd_apex_strategies

Safely re-runnable: a family whose active version already has this exact
code hash is skipped rather than re-saved.

There is one strategy family per *entry timeframe*, because
`StrategySpec.entry_timeframe` is not something a skill's `param_overrides`
can reach — only params are overridable. Everything else that separates one
Apex bot from another (trendguard / turbo / precision) *is* a param, so those
bots share a family: `xauusd_apex_trendguard_m1`, `..._precision_m1` and
`..._turbo_m1` are all the M1 family with different overrides.

The M5/M15/H1/D1 files are exact-substitution copies of the M1 one — same
algorithm, shifted timeframe ladder. Re-run this script after editing any of
them or the DB's active version silently goes stale against the source.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from src.shared.config.settings import Settings
from src.shared.db.base import make_session_factory
from src.strategies.adapters.repository import StrategyVersionRepository
from src.strategies.application.versioning import StrategyVersionService
from src.strategies.domain.versioning import CodeSource
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

_GENERATED_DIR = Path(__file__).resolve().parent.parent / "src" / "strategies" / "generated"

_STRATEGIES: tuple[tuple[str, str], ...] = tuple(
    (f"xauusd_snd_apex_trendguard_{tf}", f"xauusd_snd_apex_trendguard_{tf}_v1")
    for tf in ("m1", "m5", "m15", "h1", "d1")
)


def seed(service: StrategyVersionService, repository: StrategyVersionRepository) -> None:
    for name, file_stem in _STRATEGIES:
        code = (_GENERATED_DIR / f"{file_stem}.py").read_text()
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        existing = repository.list_all(name)
        if any(v.code_hash == code_hash for v in existing):
            logger.info("skipping %r — version with this exact code hash already in DB", name)
            continue
        version = service.save_generated_code(name=name, code=code, source=CodeSource.MANUAL)
        service.activate_version(version.id)
        logger.info("seeded/activated %r (version=%d, id=%s)", name, version.version, version.id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings()
    session_factory = make_session_factory(settings.database_url)
    repository = StrategyVersionRepository(session_factory)
    service = StrategyVersionService(
        repository=repository, registry=StrategyRegistry(), generated_dir=_GENERATED_DIR
    )
    seed(service, repository)


if __name__ == "__main__":
    main()
