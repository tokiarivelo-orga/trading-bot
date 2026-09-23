"""Symbol -> bots routing (§6.6): activating a bot must persist to disk, take
effect in the live SkillSelector immediately, and refuse to route to a
symbol or strategy family that doesn't actually exist/isn't live. A symbol
not yet in configs/app.yaml must be durably and immediately activated for
live trading on its first bot — the real "Apply" action, not just a routing
reassignment (§ dynamic symbol activation). Several bots may be
concurrently active on one symbol, each independently addressable."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from src.broker.application.spread_gate import SpreadGate
from src.market_data.application.candle_stream import CandleStreamService
from src.market_data.domain.models import Timeframe
from src.shared.events.bus import EventBus
from src.skills.adapters.normal_skill_repository import NormalSkillRepository
from src.skills.application.skill_assignment import (
    DuplicateBotError,
    InvalidBotNameError,
    InvalidParamValueError,
    InvalidSessionError,
    SkillAssignmentService,
    UnknownBotError,
    UnknownParamError,
    UnknownStrategyError,
    UnknownSymbolError,
)
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import NormalSkill, SessionWindow
from src.strategies.domain.models import StrategySpec
from src.strategies.registry import StrategyRegistry


class FakeStrategy:
    def __init__(self, params: dict | None = None, htf_veto: bool = True) -> None:
        self.spec = StrategySpec(
            name="fake",
            version=1,
            symbols=("XAUUSD",),
            entry_timeframe="M5",
            confirmation_timeframes=(),
            params=params if params is not None else {"lookback": 20, "use_filter": False},
            htf_veto=htf_veto,
        )

    def evaluate(self, ctx):
        return None


class FakeMarketData:
    async def get_candles(self, symbol, timeframe, count):
        return []


class FakeRepository:
    def upsert_many(self, candles):
        return 0


class FakeBroadcaster:
    async def broadcast(self, message):
        pass


def _make_candle_stream(symbols: list[str]) -> CandleStreamService:
    return CandleStreamService(
        market_data=FakeMarketData(),
        repository=FakeRepository(),
        event_bus=EventBus(),
        broadcaster=FakeBroadcaster(),
        symbols=symbols,
        timeframes=[Timeframe.M5],
    )


def _write_symbol_config(configs_dir: Path, symbol: str) -> None:
    (configs_dir / "symbols").mkdir(parents=True, exist_ok=True)
    (configs_dir / "symbols" / f"{symbol.lower()}.yaml").write_text(
        f"symbol: {symbol}\n"
        "max_spread_points: 35\n"
        "min_rr: 1.5\n"
        "contract_size: 100\n"
        "point: 0.01\n"
        "digits: 2\n"
        "stops_level: 0\n"
        "volume_min: 0.01\n"
        "volume_max: 50\n"
        "volume_step: 0.01\n"
    )


def _write_skill_yaml(skills_dir: Path, symbol: str, bot_slug: str, strategy: str) -> None:
    symbol_dir = skills_dir / symbol.lower()
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / f"{bot_slug}.yaml").write_text(
        f"name: normal/{symbol.lower()}/{bot_slug}\n"
        f"symbol: {symbol}\n"
        f"strategy: {strategy}\n"
        "risk_multiplier: 0.8\n"
        'sessions:\n  - { start: "09:00", end: "12:00" }\n'
    )


def _write_app_config(configs_dir: Path, symbols: list[str]) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    symbols_literal = ", ".join(f'"{s}"' if " " in s else s for s in symbols)
    (configs_dir / "app.yaml").write_text(
        "# Global app configuration. Hot-reloadable.\n"
        "mode: live              # paper | live  — NEVER switch to live before Phase 9 criteria\n"
        f"symbols: [{symbols_literal}]\n"
    )


@pytest.fixture
def setup(tmp_path):
    configs_dir = tmp_path / "configs"
    skills_dir = tmp_path / "skills" / "normal"
    _write_symbol_config(configs_dir, "XAUUSD")
    _write_app_config(configs_dir, ["XAUUSD"])
    _write_skill_yaml(skills_dir, "XAUUSD", "breakout_v1", "breakout_v1")

    repository = NormalSkillRepository(skills_dir)
    selector = SkillSelector(repository.load_all(["XAUUSD"]), timezone="UTC")
    registry = StrategyRegistry()
    registry.register("breakout_v1", FakeStrategy())
    registry.register("gold_ema_pullback", FakeStrategy())
    registry.register("mean_reversion_v1", FakeStrategy())
    candle_stream = _make_candle_stream(["XAUUSD"])
    spread_gate = SpreadGate({})
    service = SkillAssignmentService(
        repository=repository,
        selector=selector,
        strategy_registry=registry,
        candle_streams=[candle_stream],
        spread_gate=spread_gate,
        configs_dir=configs_dir,
    )
    return service, repository, selector, skills_dir, candle_stream, spread_gate, configs_dir


async def test_update_bot_persists_and_hot_swaps(setup):
    service, repository, selector, skills_dir, *_ = setup

    skill, strategy = await service.update_bot("XAUUSD", "breakout_v1", "gold_ema_pullback")

    assert skill.strategy == "gold_ema_pullback"
    assert strategy is service._strategy_registry.get("gold_ema_pullback")
    # Existing sessions/risk_multiplier/name are preserved, not reset.
    assert skill.risk_multiplier == 0.8
    assert skill.sessions == (SessionWindow.parse("09:00", "12:00"),)
    assert skill.name == "normal/xauusd/breakout_v1"

    # Persisted to disk.
    on_disk = repository.get("XAUUSD", "breakout_v1")
    assert on_disk.strategy == "gold_ema_pullback"

    # Live selector reflects the change without any restart.
    (decision,) = selector.select_all("XAUUSD")
    assert decision.strategy_name == "gold_ema_pullback"


async def test_update_bot_resets_overrides_on_strategy_reassignment(setup):
    service, repository, *_ = setup
    await service.update_config(
        "XAUUSD",
        "breakout_v1",
        risk_multiplier=0.8,
        sessions=[("09:00", "12:00")],
        param_overrides={"lookback": 30},
        htf_veto_override=False,
    )

    skill, _strategy = await service.update_bot("XAUUSD", "breakout_v1", "gold_ema_pullback")

    assert skill.param_overrides == {}
    assert skill.htf_veto_override is None
    on_disk = repository.get("XAUUSD", "breakout_v1")
    assert on_disk.param_overrides == {}
    assert on_disk.htf_veto_override is None


async def test_update_bot_unknown_bot_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownBotError):
        await service.update_bot("XAUUSD", "does_not_exist", "gold_ema_pullback")


async def test_update_bot_unknown_strategy_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownStrategyError):
        await service.update_bot("XAUUSD", "breakout_v1", "does_not_exist")


async def test_add_bot_unknown_symbol_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownSymbolError):
        await service.add_bot("EURUSD", "gold_ema_pullback")


async def test_add_bot_unknown_strategy_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownStrategyError):
        await service.add_bot("XAUUSD", "does_not_exist")


async def test_add_bot_paused_strategy_rejected(setup):
    service, *_ = setup
    registry: StrategyRegistry = service._strategy_registry
    registry.pause("gold_ema_pullback")
    with pytest.raises(UnknownStrategyError):
        await service.add_bot("XAUUSD", "gold_ema_pullback")


async def test_add_bot_alongside_existing_bot_does_not_replace_it(setup):
    service, repository, selector, *_ = setup

    result = await service.add_bot("XAUUSD", "mean_reversion_v1")

    assert result.skill.strategy == "mean_reversion_v1"
    assert result.newly_activated is False  # XAUUSD already live via the fixture's bot
    # Both bots now active on disk and live.
    assert {s.strategy for s in repository.list_for_symbol("XAUUSD")} == {
        "breakout_v1",
        "mean_reversion_v1",
    }
    decisions = selector.select_all("XAUUSD")
    assert {d.strategy_name for d in decisions} == {"breakout_v1", "mean_reversion_v1"}


async def test_add_bot_duplicate_name_rejected(setup):
    service, *_ = setup
    with pytest.raises(DuplicateBotError):
        await service.add_bot("XAUUSD", "gold_ema_pullback", bot_name="breakout_v1")


async def test_add_bot_invalid_name_rejected(setup):
    service, *_ = setup
    with pytest.raises(InvalidBotNameError):
        await service.add_bot("XAUUSD", "gold_ema_pullback", bot_name="***")


async def test_rename_bot_persists_and_hot_swaps(setup):
    service, repository, selector, *_ = setup

    skill, strategy = await service.rename_bot("XAUUSD", "breakout_v1", "my_breakout")

    assert skill.name == "normal/xauusd/my_breakout"
    assert strategy is service._strategy_registry.get("breakout_v1")
    # Strategy/risk/sessions are preserved, not reset.
    assert skill.strategy == "breakout_v1"
    assert skill.risk_multiplier == 0.8
    assert skill.sessions == (SessionWindow.parse("09:00", "12:00"),)

    # Old slug is gone, new slug is on disk.
    assert repository.get("XAUUSD", "breakout_v1") is None
    on_disk = repository.get("XAUUSD", "my_breakout")
    assert on_disk.name == "normal/xauusd/my_breakout"

    # Live selector reflects the rename without any restart.
    (decision,) = selector.select_all("XAUUSD")
    assert decision.skill_name == "normal/xauusd/my_breakout"


async def test_rename_bot_slugifies_the_new_name(setup):
    service, repository, *_ = setup

    skill, _strategy = await service.rename_bot("XAUUSD", "breakout_v1", "My Breakout Bot!")

    assert skill.name == "normal/xauusd/my_breakout_bot"
    assert repository.get("XAUUSD", "my_breakout_bot") is not None


async def test_rename_bot_to_same_slug_is_a_no_op(setup):
    service, repository, *_ = setup

    skill, _strategy = await service.rename_bot("XAUUSD", "breakout_v1", "breakout_v1")

    assert skill.name == "normal/xauusd/breakout_v1"
    assert repository.get("XAUUSD", "breakout_v1") is not None


async def test_rename_bot_unknown_bot_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownBotError):
        await service.rename_bot("XAUUSD", "does_not_exist", "new_name")


async def test_rename_bot_invalid_name_rejected(setup):
    service, *_ = setup
    with pytest.raises(InvalidBotNameError):
        await service.rename_bot("XAUUSD", "breakout_v1", "***")


async def test_rename_bot_duplicate_name_rejected(setup):
    service, *_ = setup
    await service.add_bot("XAUUSD", "mean_reversion_v1")
    with pytest.raises(DuplicateBotError):
        await service.rename_bot("XAUUSD", "breakout_v1", "mean_reversion_v1")


async def test_remove_bot_stops_it_without_touching_others(setup):
    service, repository, selector, *_ = setup
    await service.add_bot("XAUUSD", "mean_reversion_v1")

    await service.remove_bot("XAUUSD", "breakout_v1")

    assert repository.get("XAUUSD", "breakout_v1") is None
    assert repository.get("XAUUSD", "mean_reversion_v1") is not None
    decisions = selector.select_all("XAUUSD")
    assert [d.strategy_name for d in decisions] == ["mean_reversion_v1"]


async def test_remove_last_bot_deactivates_the_symbol(setup):
    # Mirror of add_bot's activation: removing a symbol's last bot takes the
    # symbol out of app.yaml and stops streaming its candles.
    service, repository, selector, _skills_dir, candle_stream, _spread_gate, configs_dir = setup

    await service.remove_bot("XAUUSD", "breakout_v1")

    assert repository.list_for_symbol("XAUUSD") == []
    assert selector.select_all("XAUUSD") == []
    app_config = yaml.safe_load((configs_dir / "app.yaml").read_text())
    assert "XAUUSD" not in app_config["symbols"]
    assert "XAUUSD" not in candle_stream.active_symbols


async def test_remove_bot_keeps_the_symbol_while_other_bots_remain(setup):
    service, _repository, _selector, _skills_dir, candle_stream, _spread_gate, configs_dir = setup
    await service.add_bot("XAUUSD", "mean_reversion_v1")

    await service.remove_bot("XAUUSD", "breakout_v1")

    app_config = yaml.safe_load((configs_dir / "app.yaml").read_text())
    assert "XAUUSD" in app_config["symbols"]
    assert "XAUUSD" in candle_stream.active_symbols


async def test_remove_bot_unknown_bot_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownBotError):
        await service.remove_bot("XAUUSD", "does_not_exist")


async def test_list_assignments_returns_one_per_bot(setup):
    service, *_ = setup
    await service.add_bot("XAUUSD", "mean_reversion_v1")

    assignments = await service.list_assignments()

    assert sorted(skill.strategy for skill, _strategy in assignments) == [
        "breakout_v1",
        "mean_reversion_v1",
    ]
    assert all(skill.symbol == "XAUUSD" for skill, _strategy in assignments)
    assert all(strategy is not None for _skill, strategy in assignments)


async def test_add_bot_to_new_symbol_activates_it_live(setup):
    # The actual "Apply to <symbol>" flow for a symbol not yet in
    # configs/app.yaml — this must durably and immediately activate it, with
    # no restart, matching what TradeEngine._try_enter and the spread gate
    # need to actually trade it.
    service, repository, selector, skills_dir, candle_stream, spread_gate, configs_dir = setup
    _write_symbol_config(configs_dir, "Volatility 75 Index")
    registry: StrategyRegistry = service._strategy_registry
    registry.register("pob_price_action_snd_for_vix75", FakeStrategy())

    result = await service.add_bot("Volatility 75 Index", "pob_price_action_snd_for_vix75")

    assert result.newly_activated is True
    # Durable: persisted to configs/app.yaml.
    app_config = yaml.safe_load((configs_dir / "app.yaml").read_text())
    assert "Volatility 75 Index" in app_config["symbols"]
    # The load-bearing safety comment survives the write untouched.
    assert "NEVER switch to live" in (configs_dir / "app.yaml").read_text()
    # Immediate: hot-added to candle streaming and the spread gate, no
    # restart needed.
    assert "Volatility 75 Index" in candle_stream.active_symbols
    # The real configs/symbols/volatility 75 index.yaml cap (35pts, per
    # _write_symbol_config) is enforced immediately — not the unconfigured
    # fallback of no cap at all.
    veto = spread_gate.check(
        "Volatility 75 Index", spread_points=40, point=0.01, sl_distance=None, tp_distance=None
    )
    assert veto is not None
    assert "max 35pts" in veto.reason
    # And routed live via the selector, same as any other assignment.
    (decision,) = selector.select_all("Volatility 75 Index")
    assert decision.allowed is True
    assert decision.strategy_name == "pob_price_action_snd_for_vix75"
    # And reported by list_assignments() without needing a restart.
    assignments = await service.list_assignments()
    assert "Volatility 75 Index" in [skill.symbol for skill, _strategy in assignments]


async def test_add_bot_to_new_symbol_twice_does_not_duplicate_or_re_report_activation(setup):
    service, _repository, _selector, _skills_dir, _candle_stream, _spread_gate, configs_dir = setup
    _write_symbol_config(configs_dir, "Volatility 75 Index")
    registry: StrategyRegistry = service._strategy_registry
    registry.register("pob_price_action_snd_for_vix75", FakeStrategy())
    registry.register("mean_reversion_v1", FakeStrategy())

    first = await service.add_bot("Volatility 75 Index", "pob_price_action_snd_for_vix75")
    second = await service.add_bot("Volatility 75 Index", "mean_reversion_v1")

    assert first.newly_activated is True
    assert second.newly_activated is False  # symbol already active from the first bot
    app_config = yaml.safe_load((configs_dir / "app.yaml").read_text())
    assert app_config["symbols"].count("Volatility 75 Index") == 1


async def test_update_config_persists_risk_sessions_and_overrides(setup):
    service, repository, selector, *_ = setup

    skill, strategy = await service.update_config(
        "XAUUSD",
        "breakout_v1",
        risk_multiplier=1.5,
        sessions=[("08:00", "10:00"), ("13:00", "17:00")],
        param_overrides={"lookback": 30, "use_filter": True},
        htf_veto_override=False,
    )

    assert skill.risk_multiplier == 1.5
    assert skill.sessions == (
        SessionWindow.parse("08:00", "10:00"),
        SessionWindow.parse("13:00", "17:00"),
    )
    assert skill.param_overrides == {"lookback": 30, "use_filter": True}
    assert skill.htf_veto_override is False
    assert strategy is service._strategy_registry.get("breakout_v1")

    on_disk = repository.get("XAUUSD", "breakout_v1")
    assert on_disk == skill

    # 09:00 UTC falls inside the 08:00-10:00 session window just saved,
    # regardless of the real wall-clock time this test runs at.
    (decision,) = selector.select_all("XAUUSD", datetime(2026, 7, 20, 9, 0, tzinfo=UTC))
    assert decision.param_overrides == {"lookback": 30, "use_filter": True}
    assert decision.htf_veto_override is False


async def test_update_config_unknown_bot_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownBotError):
        await service.update_config(
            "XAUUSD",
            "does_not_exist",
            risk_multiplier=1.0,
            sessions=[],
            param_overrides={},
            htf_veto_override=None,
        )


async def test_update_config_unknown_param_rejected(setup):
    service, *_ = setup
    with pytest.raises(UnknownParamError):
        await service.update_config(
            "XAUUSD",
            "breakout_v1",
            risk_multiplier=1.0,
            sessions=[],
            param_overrides={"not_a_real_param": 1},
            htf_veto_override=None,
        )


async def test_update_config_wrong_type_rejected(setup):
    service, *_ = setup
    with pytest.raises(InvalidParamValueError):
        await service.update_config(
            "XAUUSD",
            "breakout_v1",
            risk_multiplier=1.0,
            sessions=[],
            param_overrides={"lookback": "thirty"},  # default is an int
            htf_veto_override=None,
        )


async def test_update_config_bool_does_not_pass_as_numeric(setup):
    service, *_ = setup
    with pytest.raises(InvalidParamValueError):
        await service.update_config(
            "XAUUSD",
            "breakout_v1",
            risk_multiplier=1.0,
            sessions=[],
            param_overrides={"lookback": True},  # default is an int, not bool
            htf_veto_override=None,
        )


async def test_update_config_invalid_session_rejected(setup):
    service, *_ = setup
    with pytest.raises(InvalidSessionError):
        await service.update_config(
            "XAUUSD",
            "breakout_v1",
            risk_multiplier=1.0,
            sessions=[("not-a-time", "12:00")],
            param_overrides={},
            htf_veto_override=None,
        )


async def test_update_config_allows_risk_edit_when_strategy_unregistered(setup):
    service, *_ = setup
    service._strategy_registry.pause("breakout_v1")

    skill, strategy = await service.update_config(
        "XAUUSD",
        "breakout_v1",
        risk_multiplier=2.0,
        sessions=[],
        param_overrides={},
        htf_veto_override=None,
    )

    assert skill.risk_multiplier == 2.0
    assert strategy is None


def test_normal_skill_repository_save_round_trips(tmp_path):
    skills_dir = tmp_path / "skills"
    repository = NormalSkillRepository(skills_dir)
    skill = NormalSkill(
        name="normal/eurusd/breakout_v1",
        symbol="EURUSD",
        strategy="breakout_v1",
        risk_multiplier=1.2,
        sessions=(SessionWindow.parse("08:00", "17:00"),),
        param_overrides={"lookback": 30, "label": "aggressive"},
        htf_veto_override=False,
    )
    repository.save(skill)

    loaded = repository.get("EURUSD", "breakout_v1")
    assert loaded == skill


def test_normal_skill_repository_get_missing_returns_none(tmp_path):
    repository = NormalSkillRepository(tmp_path / "skills")
    assert repository.get("EURUSD", "breakout_v1") is None


def test_normal_skill_repository_list_for_symbol_empty_when_no_directory(tmp_path):
    repository = NormalSkillRepository(tmp_path / "skills")
    assert repository.list_for_symbol("EURUSD") == []


def test_normal_skill_repository_delete_removes_file(tmp_path):
    skills_dir = tmp_path / "skills"
    repository = NormalSkillRepository(skills_dir)
    skill = NormalSkill(
        name="normal/eurusd/breakout_v1",
        symbol="EURUSD",
        strategy="breakout_v1",
        risk_multiplier=1.0,
        sessions=(),
    )
    repository.save(skill)

    repository.delete("EURUSD", "breakout_v1")

    assert repository.get("EURUSD", "breakout_v1") is None


def test_normal_skill_repository_delete_missing_is_a_no_op(tmp_path):
    repository = NormalSkillRepository(tmp_path / "skills")
    repository.delete("EURUSD", "breakout_v1")  # must not raise
