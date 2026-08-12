"""Seed the SMC DL Step 200 strategy."""

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
    ("smc_dl_m5_step200", "smc_dl_m5_step200_v1"),
    ("smc_dl_m5", "smc_dl_m5_v1"),
)

def seed(service: StrategyVersionService, repository: StrategyVersionRepository) -> None:
    for name, file_stem in _STRATEGIES:
        file_path = _GENERATED_DIR / f"{file_stem}.py"
        if not file_path.exists():
            continue
        code = file_path.read_text()
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        existing = repository.list_all(name)
        if any(v.code_hash == code_hash for v in existing):
            logger.info("skipping %r — version with this exact code hash already in DB", name)
            continue
        if existing:
            logger.info("updating %r — saving new version", name)
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
