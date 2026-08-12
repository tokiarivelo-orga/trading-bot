"""Seed the XAUUSD S&D + Quasimodo + Market Structure strategies into the
StrategyVersion DB table so they appear in the Bots UI.

Run from `backend/`:

    uv run python -m scripts.seed_xauusd_snd_qm_strategies

Safely re-runnable: a family that already has any recorded version is
skipped rather than erroring.
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
    ("xauusd_snd_qm_structure_m1", "xauusd_snd_qm_structure_m1_v1"),
    ("xauusd_snd_qm_structure_m5", "xauusd_snd_qm_structure_m5_v1"),
    ("xauusd_snd_qm_structure_m15", "xauusd_snd_qm_structure_m15_v1"),
    ("xauusd_snd_qm_structure_h1", "xauusd_snd_qm_structure_h1_v1"),
    ("xauusd_snd_qm_structure_adaptive_m1", "xauusd_snd_qm_structure_adaptive_m1_v1"),
    ("xauusd_snd_qm_structure_adaptive_m5", "xauusd_snd_qm_structure_adaptive_m5_v1"),
    ("xauusd_snd_qm_structure_adaptive_m15", "xauusd_snd_qm_structure_adaptive_m15_v1"),
    ("xauusd_snd_qm_structure_adaptive_h1", "xauusd_snd_qm_structure_adaptive_h1_v1"),
)


def seed(service: StrategyVersionService, repository: StrategyVersionRepository) -> None:
    for name, file_stem in _STRATEGIES:
        code = (_GENERATED_DIR / f"{file_stem}.py").read_text()
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        existing = repository.list_all(name)
        if any(v.code_hash == code_hash for v in existing):
            logger.info("skipping %r — version with this exact code hash already in DB", name)
            continue
        if existing:
            logger.info("updating %r — saving new version for upgraded multi-TP strategy", name)
        version = service.save_generated_code(name=name, code=code, source=CodeSource.MANUAL)
        service.activate_version(version.id)
        logger.info(
            "seeded/activated %r (version=%d, id=%s)", name, version.version, version.id
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
