"""Seed the XAUUSD pure market-structure (CHoCH weakness + continuation)
strategies into the StrategyVersion DB table so they appear in the Bots UI.

Run from `backend/`:

    uv run python -m scripts.seed_xauusd_structure_shift_strategies

Safely re-runnable: a family whose code hash already exists is skipped.

Deliberately does **not** call `activate_version` — both versions are
saved as `VALIDATED`, not `ACTIVE`. This is a brand-new setup type (no
zone dependency, unlike every other bot this seed script's siblings
register) with zero live track record, so activation is left as an
explicit human decision after reviewing a backtest — the same pattern
already used for `structure_pivot_config` and the adaptive_m1 v1->v2
switch elsewhere in this fleet.
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

_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("xauusd_structure_shift_m1", "xauusd_structure_shift_m1_v1"),
    ("xauusd_structure_shift_m5", "xauusd_structure_shift_m5_v1"),
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
        logger.info(
            "seeded %r as VALIDATED (version=%d, id=%s) — not activated, "
            "review a backtest before turning it on",
            name,
            version.version,
            version.id,
        )


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
