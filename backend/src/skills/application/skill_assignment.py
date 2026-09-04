"""Symbol -> bots routing (§6.6) — the real "apply a bot to a symbol"
action. Distinct from `strategies.application.versioning.activate_version`,
which decides which *version* of a family is live; this decides which
*bots* (each pinned to a strategy family) are concurrently active on a
symbol, by writing `skills/normal/<symbol>/<bot_slug>.yaml` files and
hot-swapping the running `SkillSelector`.

`add_bot()` is also where a symbol not yet in the automated-trading universe
gets activated: it persists the symbol into `configs/app.yaml` and hot-adds
it to candle streaming and the spread gate, so adding the symbol's first bot
is the one deliberate action that fully turns on live trading for a symbol —
no manual YAML edit, no restart. See the module docstring in
`market_data.application.candle_stream` for why the engine only ever
evaluates a symbol it's actively polling.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from src.broker.application.spread_gate import SpreadGate
from src.market_data.application.candle_stream import CandleStreamService
from src.shared.config.app_config_writer import add_symbol_to_app_config
from src.shared.config.loaders import load_symbol_trading_config_if_exists
from src.skills.adapters.normal_skill_repository import NormalSkillRepository
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import NormalSkill, SessionWindow, slugify
from src.strategies.domain.models import Strategy
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


class UnknownSymbolError(Exception):
    pass


class UnknownStrategyError(Exception):
    pass


class UnknownBotError(Exception):
    pass


class DuplicateBotError(Exception):
    pass


class InvalidBotNameError(Exception):
    pass


class UnknownParamError(Exception):
    pass


class InvalidParamValueError(Exception):
    pass


class InvalidSessionError(Exception):
    pass


def _validate_overrides(strategy: Strategy, overrides: dict[str, float | int | str | bool]) -> None:
    """Rejects override keys the strategy doesn't declare, and values whose
    type doesn't match that param's default — a bad override would otherwise
    only surface as a `TypeError` deep inside the strategy's `evaluate()` on
    the next candle close."""
    for key, value in overrides.items():
        if key not in strategy.spec.params:
            raise UnknownParamError(f"{key!r} is not a param of strategy {strategy.spec.name!r}")
        default = strategy.spec.params[key]
        if isinstance(default, bool) or isinstance(value, bool):
            # bool is an int subclass in Python — handled separately so a
            # JSON `true`/`false` never silently passes as a numeric param.
            ok = isinstance(default, bool) and isinstance(value, bool)
        elif isinstance(default, (int, float)) and isinstance(value, (int, float)):
            ok = True
        else:
            ok = type(value) is type(default)
        if not ok:
            raise InvalidParamValueError(
                f"{key!r} expects a {type(default).__name__}, got {type(value).__name__}"
            )


@dataclass(frozen=True)
class AssignmentResult:
    """`add_bot()`'s outcome — `newly_activated` is a call outcome, not part
    of the persisted `NormalSkill`, so it doesn't belong on that dataclass."""

    skill: NormalSkill
    newly_activated: bool
    strategy: Strategy


class SkillAssignmentService:
    def __init__(
        self,
        repository: NormalSkillRepository,
        selector: SkillSelector,
        strategy_registry: StrategyRegistry,
        candle_streams: Sequence[CandleStreamService],
        spread_gate: SpreadGate,
        configs_dir: Path,
    ) -> None:
        self._repository = repository
        self._selector = selector
        self._strategy_registry = strategy_registry
        self._candle_streams = candle_streams
        self._spread_gate = spread_gate
        self._configs_dir = configs_dir

    async def list_assignments(self) -> list[tuple[NormalSkill, Strategy | None]]:
        """Every bot currently routed on every symbol, i.e. every
        skills/normal/<symbol>/*.yaml file on disk — independent of
        configs/app.yaml, so a bot added via `add_bot()` shows up here
        immediately. Paired with each bot's currently registered `Strategy`
        (`None` if unregistered/paused) so callers can read the strategy's
        own default params/htf_veto alongside this bot's overrides."""
        skills = await asyncio.to_thread(self._repository.list_all)
        return [(skill, self._strategy_registry.get(skill.strategy)) for skill in skills]

    async def add_bot(
        self,
        symbol: str,
        strategy_name: str,
        bot_name: str | None = None,
        risk_multiplier: float = 1.0,
    ) -> AssignmentResult:
        """Activates a new, independent bot on `symbol` — added alongside
        any bots already trading it, never replacing one. `bot_name`
        defaults to `strategy_name` (slugified) and must be unique among
        this symbol's currently active bots."""
        config = await asyncio.to_thread(
            load_symbol_trading_config_if_exists, symbol, self._configs_dir
        )
        if config is None:
            raise UnknownSymbolError(f"no configs/symbols/{symbol.lower()}.yaml for {symbol!r}")
        strategy = self._strategy_registry.get(strategy_name)
        if strategy is None:
            raise UnknownStrategyError(
                f"{strategy_name!r} has no currently active, non-paused strategy version"
            )

        bot_slug = slugify(bot_name or strategy_name)
        if not bot_slug:
            raise InvalidBotNameError(f"{bot_name or strategy_name!r} has no valid slug characters")
        if await asyncio.to_thread(self._repository.get, symbol, bot_slug) is not None:
            raise DuplicateBotError(f"{symbol!r} already has a bot named {bot_slug!r}")

        was_active = bool(await asyncio.to_thread(self._repository.list_for_symbol, symbol))

        # Durable writes first, in-memory hot-swaps after: every step below
        # is individually idempotent, so a retry (or a restart, which just
        # re-reads the now-updated app.yaml) always converges to the same
        # state with no partial-activation window to worry about.
        newly_activated = await asyncio.to_thread(
            add_symbol_to_app_config, symbol, self._configs_dir
        )

        skill = NormalSkill(
            name=f"normal/{symbol.lower()}/{bot_slug}",
            symbol=symbol,
            strategy=strategy_name,
            risk_multiplier=risk_multiplier,
            sessions=(),
        )
        await asyncio.to_thread(self._repository.save, skill)

        if not was_active:
            for candle_stream in self._candle_streams:
                candle_stream.add_symbol(symbol)
            self._spread_gate.set_config(symbol, config)
        self._selector.set(skill)

        if newly_activated:
            logger.info(
                "symbol %s newly activated for live automated trading via UI apply "
                "(strategy=%s, bot=%s)",
                symbol,
                strategy_name,
                bot_slug,
            )
        logger.info("bot %s added to %s (strategy=%s)", bot_slug, symbol, strategy_name)
        return AssignmentResult(skill=skill, newly_activated=newly_activated, strategy=strategy)

    async def update_bot(
        self, symbol: str, bot_name: str, strategy_name: str
    ) -> tuple[NormalSkill, Strategy]:
        """Reassigns an existing bot's strategy family in place, keeping its
        `sessions`/`risk_multiplier` — writes the YAML and hot-swaps the
        live selector immediately, no restart needed. Resets
        `param_overrides`/`htf_veto_override` to defaults: the old overrides
        may name params the new strategy doesn't have, so carrying them over
        silently would risk an unknown/mistyped override surfacing only as a
        runtime error inside the new strategy's `evaluate()`."""
        bot_slug = slugify(bot_name)
        existing = await asyncio.to_thread(self._repository.get, symbol, bot_slug)
        if existing is None:
            raise UnknownBotError(f"{symbol!r} has no bot named {bot_slug!r}")
        strategy = self._strategy_registry.get(strategy_name)
        if strategy is None:
            raise UnknownStrategyError(
                f"{strategy_name!r} has no currently active, non-paused strategy version"
            )

        skill = replace(
            existing,
            strategy=strategy_name,
            param_overrides={},
            htf_veto_override=None,
        )
        await asyncio.to_thread(self._repository.save, skill)
        self._selector.set(skill)
        logger.info("bot %s on %s reassigned to strategy=%s", bot_slug, symbol, strategy_name)
        return skill, strategy

    async def update_config(
        self,
        symbol: str,
        bot_name: str,
        *,
        risk_multiplier: float,
        sessions: list[tuple[str, str]],
        param_overrides: dict[str, float | int | str | bool],
        htf_veto_override: bool | None,
    ) -> tuple[NormalSkill, Strategy | None]:
        """Replaces `bot_name`'s risk_multiplier, sessions, and per-bot
        strategy overrides all at once — writes the YAML and hot-swaps the
        live selector immediately, no restart needed. Does not touch which
        strategy family the bot trades (see `update_bot`). `sessions` is
        `(start, end)` HH:MM pairs; `param_overrides` keys are validated
        against the bot's current strategy's own `StrategySpec.params` only
        when that strategy is currently registered — a paused/unregistered
        strategy shouldn't block editing risk/sessions."""
        bot_slug = slugify(bot_name)
        existing = await asyncio.to_thread(self._repository.get, symbol, bot_slug)
        if existing is None:
            raise UnknownBotError(f"{symbol!r} has no bot named {bot_slug!r}")

        try:
            parsed_sessions = tuple(SessionWindow.parse(start, end) for start, end in sessions)
        except ValueError as exc:
            raise InvalidSessionError(f"invalid session window: {exc}") from exc

        strategy = self._strategy_registry.get(existing.strategy)
        if strategy is not None:
            _validate_overrides(strategy, param_overrides)

        skill = replace(
            existing,
            risk_multiplier=risk_multiplier,
            sessions=parsed_sessions,
            param_overrides=dict(param_overrides),
            htf_veto_override=htf_veto_override,
        )
        await asyncio.to_thread(self._repository.save, skill)
        self._selector.set(skill)
        logger.info(
            "bot %s on %s config updated (risk_multiplier=%s, sessions=%d, "
            "param_overrides=%s, htf_veto_override=%s)",
            bot_slug,
            symbol,
            risk_multiplier,
            len(parsed_sessions),
            sorted(param_overrides),
            htf_veto_override,
        )
        return skill, strategy

    async def rename_bot(
        self, symbol: str, bot_name: str, new_bot_name: str
    ) -> tuple[NormalSkill, Strategy | None]:
        """Renames an existing bot on `symbol` in place, keeping its
        strategy/risk_multiplier/sessions/overrides — writes the YAML under
        the new slug, deletes the old file, and hot-swaps the running
        SkillSelector, no restart needed. Renaming changes this bot's MT5
        magic number (`magic_number()` is derived from the full skill name,
        which includes the bot slug) — any position the broker already has
        open under the old magic number will no longer be recognized as
        this bot's on the next `_try_enter` check, so avoid renaming a bot
        while it holds an open position."""
        old_slug = slugify(bot_name)
        existing = await asyncio.to_thread(self._repository.get, symbol, old_slug)
        if existing is None:
            raise UnknownBotError(f"{symbol!r} has no bot named {old_slug!r}")

        new_slug = slugify(new_bot_name)
        if not new_slug:
            raise InvalidBotNameError(f"{new_bot_name!r} has no valid slug characters")
        strategy = self._strategy_registry.get(existing.strategy)
        if new_slug == old_slug:
            return existing, strategy
        if await asyncio.to_thread(self._repository.get, symbol, new_slug) is not None:
            raise DuplicateBotError(f"{symbol!r} already has a bot named {new_slug!r}")

        skill = replace(existing, name=f"normal/{symbol.lower()}/{new_slug}")
        await asyncio.to_thread(self._repository.save, skill)
        await asyncio.to_thread(self._repository.delete, symbol, old_slug)
        self._selector.remove(symbol, old_slug)
        self._selector.set(skill)
        logger.info("bot %s on %s renamed to %s", old_slug, symbol, new_slug)
        return skill, strategy

    async def remove_bot(self, symbol: str, bot_name: str) -> None:
        """Deactivates one bot on `symbol`. If this was the last bot on the symbol,
        removes the symbol from configs/app.yaml and hot-removes it from candle streaming
        so the backend stops fetching unnecessarily."""
        bot_slug = slugify(bot_name)
        existing = await asyncio.to_thread(self._repository.get, symbol, bot_slug)
        if existing is None:
            raise UnknownBotError(f"{symbol!r} has no bot named {bot_slug!r}")
        await asyncio.to_thread(self._repository.delete, symbol, bot_slug)
        self._selector.remove(symbol, bot_slug)
        
        # Check if it was the last bot
        remaining_bots = await asyncio.to_thread(self._repository.list_for_symbol, symbol)
        if not remaining_bots:
            from src.shared.config.app_config_writer import remove_symbol_from_app_config
            await asyncio.to_thread(remove_symbol_from_app_config, symbol, self._configs_dir)
            for candle_stream in self._candle_streams:
                candle_stream.remove_symbol(symbol)
            logger.info("symbol %s removed from automated trading because its last bot was deleted", symbol)
            
        logger.info("bot %s removed from %s", bot_slug, symbol)
