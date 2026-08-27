"""Regression test for `scripts/seed_xauusd_snd_qm_strategies.py`: it must
seed and activate the `_v4` file for each of the 4 timeframe presets, not
`_v1`/`_v2`/`_v3`. Those earlier versions hardcode `tp1_target_rr=1.2`,
below XAUUSD's `min_rr=1.5` spread-gate floor, so the Scalp leg is vetoed on
every single order — confirmed via 2026-08 production backtest activity
logs (see each generated file's own v4 changelog entry). Seeding v1 would
silently reactivate that dead leg on any fresh/reset environment."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.seed_xauusd_snd_qm_strategies import _STRATEGIES, seed
from src.shared.db.base import Base
from src.strategies.adapters.repository import StrategyVersionRepository
from src.strategies.application.versioning import StrategyVersionService
from src.strategies.registry import StrategyRegistry


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    repository = StrategyVersionRepository(session_factory)
    svc = StrategyVersionService(repository, StrategyRegistry(), generated_dir)
    return svc, repository


def test_all_four_presets_point_at_v4() -> None:
    for name, file_stem in _STRATEGIES:
        assert file_stem == f"{name}_v4", (
            f"{name} seeds {file_stem!r} instead of the v4 tp1_target_rr fix"
        )


def test_seeds_and_activates_the_tp1_target_rr_fix(service) -> None:
    svc, repository = service
    seed(svc, repository)

    for name, _file_stem in _STRATEGIES:
        active = repository.get_active(name, "default")
        assert active is not None, f"{name} was not seeded/activated"
        code = svc.get_code(active)
        assert '"tp1_target_rr": 1.8' in code, (
            f"{name}'s active version doesn't carry the tp1_target_rr=1.8 fix"
        )
        assert '"tp1_target_rr": 1.2' not in code, (
            f"{name}'s active version still hardcodes the broken tp1_target_rr=1.2"
        )
