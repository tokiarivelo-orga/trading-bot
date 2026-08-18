"""Composition root.

The only place where concrete adapters are chosen and wired to ports.
Modules receive their dependencies from here — they never construct
adapters themselves.

MULTI_ACCOUNT_PLAN.md Phase 5: one `AccountRuntime` per enabled entry in
`configs/accounts.yaml` — its own gateway connection, event bus, trade
engine, journal, and strategy registry, so N accounts trade concurrently and
in isolation. `Container` keeps the genuinely process-wide singletons plus
the `accounts` registry, and exposes every field the old single-account
`Container` had as a read-only property delegating to the *primary* account
(the first enabled `accounts.yaml` entry) — this is what keeps every
existing route/test working unchanged until Phase 6 rewires routes to
`get_account_runtime(account_id)` and deletes these properties one at a
time.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import httpx
import yaml
from dotenv import dotenv_values

from src.activity.adapters.repository import ActivityLogRepository
from src.activity.adapters.signal_decision_repository import SignalDecisionRepository
from src.activity.application.activity_log_service import ActivityLogService
from src.activity.application.retention_service import ActivityLogRetentionService
from src.activity.application.signal_decision_service import SignalDecisionService
from src.activity.application.silence_monitor import SilenceMonitor
from src.ai.adapters.claude import ClaudeAdapter
from src.ai.adapters.claude_code import ClaudeCodeAdapter
from src.ai.adapters.gemini import GeminiAdapter
from src.ai.adapters.ollama import OllamaAdapter
from src.ai.adapters.openai_compatible import OpenAICompatibleAdapter
from src.ai.adapters.openclaw import OpenClawAdapter
from src.ai.adapters.provider_config_repository import ProviderConfigRepository
from src.ai.adapters.provider_secret_store import ProviderSecretStore
from src.ai.adapters.report_repository import AnalysisReportRepository, RefinementProposalRepository
from src.ai.adapters.repository import DraftRepository
from src.ai.application.code_regeneration import CodeRegenerationService
from src.ai.application.llm_router import LLMProviderNotConfiguredError, LLMRouter, ProviderFactory
from src.ai.application.pdf_to_strategy import PdfToStrategyService
from src.ai.application.provider_settings import ProviderSettingsService
from src.ai.application.refinement_loop import RefinementLoopService
from src.ai.ports.llm import ProviderSpec
from src.alerting.adapters.composite import CompositeAlertAdapter
from src.alerting.adapters.email import EmailAlertAdapter
from src.alerting.adapters.noop import NoopAlertAdapter
from src.alerting.adapters.telegram import TelegramAlertAdapter
from src.alerting.application.alert_service import AlertService
from src.alerting.domain.models import SilenceConfig
from src.alerting.ports.alert import AlertPort
from src.broker.adapters.credential_store import FernetCredentialStore, credentials_path_for
from src.broker.adapters.mt5_gateway import GatewayAccount, GatewayBroker
from src.broker.adapters.paper import PaperBroker
from src.broker.application.account_service import AccountService
from src.broker.application.health_monitor import GatewayHealthMonitor
from src.broker.application.order_service import OrderService
from src.broker.application.reconciliation import ReconciliationService
from src.broker.application.reconciliation_poller import ReconciliationPoller
from src.broker.application.spread_gate import SpreadGate
from src.broker.domain.account import AccountConfig
from src.broker.ports.trading import BrokerPort
from src.engine.application.manual_trading import ManualTradeGate
from src.engine.application.position_manager import PositionManager
from src.engine.application.risk_manager import RiskManager, apply_risk_override
from src.engine.application.trade_loop import TradeEngine
from src.engine.domain.exit_policy import ExitPolicySettings
from src.engine.domain.models import RiskCaps
from src.engine.domain.regime import RegimeConfig
from src.engine.domain.volatility import VolatilityConfig
from src.indicators.adapters.repository import IndicatorRepository
from src.indicators.application.service import IndicatorService
from src.journal.adapters.market_context import CandleRepositoryMarketContext
from src.journal.adapters.repository import JournalRepository
from src.journal.application.trade_journal import TradeJournalService
from src.market_data.adapters.candle_repository import CandleRepository
from src.market_data.adapters.mt5_gateway import GatewayMarketData
from src.market_data.adapters.symbol_spec_repository import SymbolSpecRepository
from src.market_data.api.ws import WsBroadcaster
from src.market_data.application.candle_stream import CandleStreamService
from src.market_data.application.history import CandleHistoryService
from src.market_data.application.live_candle import LiveCandleService
from src.market_data.domain.models import Candle, Timeframe
from src.news.adapters.finnhub import FinnhubCalendar
from src.news.adapters.forexfactory import ForexFactoryCalendar
from src.news.adapters.repository import NewsEventRepository
from src.news.application.news_window_service import NewsWindowService
from src.news.domain.models import WindowSpec
from src.news.ports.calendar import NewsCalendarPort
from src.order_book.adapters.gateway_client import GatewayOrderBookClient
from src.order_book.adapters.repository import OrderBookSnapshotRepository
from src.order_book.application.capture import OrderBookCaptureService
from src.shared.auth.session import SessionTokenIssuer
from src.shared.config.loaders import (
    load_accounts_config,
    load_alerting_config,
    load_exit_policy_settings,
    load_llm_provider_config,
    load_maintenance_config,
    load_news_config,
    load_refinement_config,
    load_regime_config,
    load_risk_caps,
    load_symbol_trading_config,
    load_volatility_config,
)
from src.shared.config.settings import REPO_ROOT, Settings, load_yaml_config
from src.shared.db.base import make_session_factory
from src.shared.db.checkpoint import WalCheckpointService
from src.shared.events.bus import EventBus
from src.shared.events.definitions import (
    BotWentSilent,
    CandleClosed,
    CircuitBreakerTripped,
    Event,
    GatewayHealthChanged,
    NewsWindowEntered,
    PositionClosed,
    PositionOpened,
    RefinementCompleted,
    TenTradesCompleted,
)
from src.shared.metrics.registry import observe_gateway_rtt, position_closed, position_opened
from src.skills.adapters.normal_skill_repository import NormalSkillRepository
from src.skills.application.news_skill_selector import NewsSkillSelector
from src.skills.application.skill_assignment import SkillAssignmentService
from src.skills.application.skill_selector import SkillSelector
from src.skills.domain.models import (
    NewsActivation,
    NewsActivationWindow,
    NewsSkill,
    PostEventRules,
    PreEventRules,
    magic_number,
)
from src.skills.ports.skill_selector import SkillSelectorPort
from src.strategies.adapters.repository import StrategyVersionRepository
from src.strategies.application.model_training import ModelTrainingService
from src.strategies.application.training_scheduler import TrainingScheduler
from src.strategies.application.versioning import StrategyVersionService
from src.strategies.domain.models import Strategy
from src.strategies.generated.breakout_v1 import BreakoutV1
from src.strategies.generated.breakout_v2 import BreakoutV2
from src.strategies.generated.mean_reversion_v1 import MeanReversionV1
from src.strategies.generated.trend_structure_v1 import TrendStructureV1
from src.strategies.generated.trend_structure_v2 import TrendStructureV2
from src.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills" / "normal"
_NEWS_SKILLS_DIR = Path(__file__).resolve().parent / "skills" / "news"
_STRATEGIES_GENERATED_DIR = Path(__file__).resolve().parent / "strategies" / "generated"


class _FanOutEventBus:
    """Duck-typed `EventBus.publish` fan-out to every account's own bus.

    `NewsWindowService` is one process-wide instance (calendar polling has
    no account dimension), but each account's `TradeEngine` subscribes to
    `NewsWindowEntered` on its *own* bus (Phase 5: buses are per-account).
    Passing this instead of a real `EventBus` lets `NewsWindowService` stay
    unchanged — it only ever calls `.publish()`, never `.subscribe()`.
    """

    def __init__(self, buses: Sequence[EventBus]) -> None:
        self._buses = buses

    async def publish(self, event: Event) -> None:
        for bus in self._buses:
            await bus.publish(event)


def _gateway_rtt_hooks(
    account_id: str,
) -> dict[str, list]:
    """`httpx.AsyncClient(event_hooks=...)` pair that times every request to
    this account's own gateway (OBSERVABILITY_PLAN.md Phase 5's "gateway
    RTT" metric). `account_id` is bound at construction time (closure)
    rather than read from `current_account_id` at call time, because
    `gateway_client` is also reachable from request-handling tasks that
    never set that ContextVar (e.g. a manual order placed via
    `/accounts/{account_id}/broker/orders`) — binding it here is strictly
    more accurate than the ContextVar fallback other Phase 5 metrics use.
    Uses `request.extensions` (httpx's documented per-request scratch dict)
    rather than `response.elapsed`, since `elapsed` is only guaranteed set
    once the response body is closed/read, which is later than this hook
    needs to be correct for a streamed response.
    """

    async def on_request(request: httpx.Request) -> None:
        request.extensions["tb_metrics_start"] = time.monotonic()

    async def on_response(response: httpx.Response) -> None:
        start = response.request.extensions.get("tb_metrics_start")
        if start is None:
            return
        observe_gateway_rtt(
            account_id=account_id,
            method=response.request.method,
            path=response.request.url.path,
            seconds=time.monotonic() - start,
        )

    return {"request": [on_request], "response": [on_response]}


def _resolve_gateway_secret(env_var: str) -> str:
    """Looks up `env_var` (an `accounts.yaml` entry's
    `gateway_shared_secret_env`) the same way `Settings` resolves its own
    fields: a real environment variable wins, falling back to `.env` — so
    each account's gateway can name a differently-secreted `.env` variable
    without needing its own `Settings` field."""
    if env_var in os.environ:
        return os.environ[env_var]
    return dotenv_values(REPO_ROOT / ".env").get(env_var) or ""


def _resolve_account_risk_caps(
    global_caps: RiskCaps, account_cfg: AccountConfig, configs_dir: Path
) -> RiskCaps:
    """MULTI_ACCOUNT_PLAN.md Phase 7: `configs/risk.yaml` stays the global,
    user-owned floor for every account. An account only gets a different
    (always stricter) `RiskCaps` if its `accounts.yaml` entry sets
    `risk_override_file`, naming a `configs/risk/{file}.yaml` — merged on top
    of `global_caps` via `apply_risk_override`, which rejects any attempt to
    loosen a cap."""
    if not account_cfg.risk_override_file:
        return global_caps
    override_data = load_yaml_config(f"risk/{account_cfg.risk_override_file}", configs_dir)
    return apply_risk_override(global_caps, override_data)


def _baseline_strategies() -> list[tuple[str, Strategy]]:
    """Same baseline instances registered into every account's own
    `StrategyRegistry` — stateless (`evaluate()` is a pure function), so
    sharing one instance across N registries is safe; only the registry
    bookkeeping (active/paused) is per-account."""
    breakout_v1 = BreakoutV1()
    breakout_v2 = BreakoutV2()
    trend_structure_v1 = TrendStructureV1()
    trend_structure_v2 = TrendStructureV2()
    mean_reversion_v1 = MeanReversionV1()
    strategies = [
        (breakout_v1.spec.name, breakout_v1),
        (breakout_v2.spec.name, breakout_v2),
        (trend_structure_v1.spec.name, trend_structure_v1),
        (trend_structure_v2.spec.name, trend_structure_v2),
        (mean_reversion_v1.spec.name, mean_reversion_v1),
    ]
    try:
        from src.strategies.generated.smc_dl_m5_step200_v1 import SmcDlM5Step200
        smc_step200 = SmcDlM5Step200()
        strategies.append((smc_step200.spec.name, smc_step200))
    except Exception:
        logger.warning("smc_dl_m5_step200 skipped (model weights missing or torch error)")

    try:
        from src.strategies.generated.smc_dl_m5_v1 import SmcDlM5V1
        smc_v1 = SmcDlM5V1()
        strategies.append((smc_v1.spec.name, smc_v1))
    except Exception:
        logger.warning("smc_dl_m5 skipped (model weights missing or torch error)")

    return strategies


@dataclass
class AccountRuntime:
    """Everything scoped to one `configs/accounts.yaml` entry — its own
    gateway connection, event bus, trade engine, and strategy registry, so
    an AI refinement promoted on this account never affects another's
    active code (MULTI_ACCOUNT_PLAN.md Phase 5)."""

    id: str
    config: AccountConfig
    symbols: list[str]
    gateway_client: httpx.AsyncClient
    event_bus: EventBus
    market_data: GatewayMarketData
    candle_history: CandleHistoryService
    # Direct repository handles (rather than only the higher-level
    # `candle_history`/symbol-spec-consuming services above) — for read paths
    # that need *stored* history/broker facts without any live-gateway
    # round trip, e.g. `journal/api/export.py`'s training-dataset export,
    # which reads a bounded historical window and must never depend on the
    # gateway being reachable. Both are the same shared, account-agnostic
    # instances `build_container` already constructs and passes into this
    # function; this just also exposes them on the runtime.
    candle_repository: CandleRepository
    symbol_spec_repository: SymbolSpecRepository
    candle_stream: CandleStreamService
    live_candle: LiveCandleService
    ws_broadcaster: WsBroadcaster
    account: AccountService
    broker: BrokerPort
    order_service: OrderService
    manual_trade_gate: ManualTradeGate
    reconciliation: ReconciliationService
    reconciliation_poller: ReconciliationPoller
    health_monitor: GatewayHealthMonitor
    silence_monitor: SilenceMonitor
    position_manager: PositionManager
    trade_journal: TradeJournalService
    order_book_capture: OrderBookCaptureService
    activity_log: ActivityLogService
    risk_manager: RiskManager
    trade_engine: TradeEngine
    strategy_registry: StrategyRegistry
    strategy_versions: StrategyVersionService
    pdf_to_strategy: PdfToStrategyService
    code_regeneration: CodeRegenerationService
    refinement_loop: RefinementLoopService

    async def aclose(self) -> None:
        await self.candle_stream.stop()
        await self.live_candle.stop()
        await self.health_monitor.stop()
        await self.silence_monitor.stop()
        await self.reconciliation_poller.stop()
        await self.gateway_client.aclose()


@dataclass
class Container:
    settings: Settings
    session_issuer: SessionTokenIssuer
    symbols: list[str]
    spread_gate: SpreadGate
    indicators: IndicatorService
    skill_selector: SkillSelectorPort
    skill_assignment: SkillAssignmentService
    provider_settings: ProviderSettingsService
    news_client: httpx.AsyncClient
    news_window_service: NewsWindowService
    activity_log_retention_service: ActivityLogRetentionService
    wal_checkpoint_service: WalCheckpointService
    model_training: ModelTrainingService
    training_scheduler: TrainingScheduler
    accounts: dict[str, AccountRuntime]
    primary_account_id: str

    _closers: list = field(default_factory=list)
    alert_telegram_client: httpx.AsyncClient | None = None

    @property
    def _primary(self) -> AccountRuntime:
        return self.accounts[self.primary_account_id]

    # --- backward-compat single-account accessors ---
    # Every field the pre-Phase-5 `Container` had, delegating to the primary
    # account (the first enabled `accounts.yaml` entry) — this is why every
    # existing route's `request.app.state.container.<x>` accessor and every
    # `SimpleNamespace`-stubbed route test keep working unchanged. Phase 6
    # deletes these one at a time as each route switches to
    # `get_account_runtime(account_id)`.
    @property
    def gateway_client(self) -> httpx.AsyncClient:
        return self._primary.gateway_client

    @property
    def event_bus(self) -> EventBus:
        return self._primary.event_bus

    @property
    def market_data(self) -> GatewayMarketData:
        return self._primary.market_data

    @property
    def candle_history(self) -> CandleHistoryService:
        return self._primary.candle_history

    @property
    def candle_stream(self) -> CandleStreamService:
        return self._primary.candle_stream

    @property
    def live_candle(self) -> LiveCandleService:
        return self._primary.live_candle

    @property
    def ws_broadcaster(self) -> WsBroadcaster:
        return self._primary.ws_broadcaster

    @property
    def account(self) -> AccountService:
        return self._primary.account

    @property
    def broker(self) -> BrokerPort:
        return self._primary.broker

    @property
    def order_service(self) -> OrderService:
        return self._primary.order_service

    @property
    def manual_trade_gate(self) -> ManualTradeGate:
        return self._primary.manual_trade_gate

    @property
    def reconciliation(self) -> ReconciliationService:
        return self._primary.reconciliation

    @property
    def health_monitor(self) -> GatewayHealthMonitor:
        return self._primary.health_monitor

    @property
    def trade_journal(self) -> TradeJournalService:
        return self._primary.trade_journal

    @property
    def activity_log(self) -> ActivityLogService:
        return self._primary.activity_log

    @property
    def risk_manager(self) -> RiskManager:
        return self._primary.risk_manager

    @property
    def trade_engine(self) -> TradeEngine:
        return self._primary.trade_engine

    @property
    def strategy_registry(self) -> StrategyRegistry:
        return self._primary.strategy_registry

    @property
    def strategy_versions(self) -> StrategyVersionService:
        return self._primary.strategy_versions

    @property
    def pdf_to_strategy(self) -> PdfToStrategyService:
        return self._primary.pdf_to_strategy

    @property
    def code_regeneration(self) -> CodeRegenerationService:
        return self._primary.code_regeneration

    @property
    def refinement_loop(self) -> RefinementLoopService:
        return self._primary.refinement_loop

    async def aclose(self) -> None:
        for runtime in self.accounts.values():
            await runtime.aclose()
        await self.news_window_service.stop()
        await self.activity_log_retention_service.stop()
        await self.wal_checkpoint_service.stop()
        await self.news_client.aclose()
        if self.alert_telegram_client is not None:
            await self.alert_telegram_client.aclose()


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings()
    session_issuer = SessionTokenIssuer()
    app_config = load_yaml_config("app", settings.configs_dir)
    symbols = list(app_config["symbols"])
    timezone = app_config.get("timezone", "UTC")
    engine_config = app_config.get("engine", {})

    account_configs = [a for a in load_accounts_config(settings.configs_dir) if a.enabled]
    if not account_configs:
        raise ValueError("configs/accounts.yaml has no enabled account")
    primary_account_id = account_configs[0].id

    session_factory = make_session_factory(settings.database_url)

    # --- shared, account-agnostic DB repositories ---
    # Thin wrappers over one shared SQLite DB — the `account_id` param each
    # method already accepts (MULTI_ACCOUNT_PLAN.md Phase 4) is enough to
    # isolate rows; only the *application services* built per account below
    # need a distinct instance (to bake in which account_id to pass).
    candle_repository = CandleRepository(session_factory)
    symbol_spec_repository = SymbolSpecRepository(session_factory)
    journal_repository = JournalRepository(session_factory)
    order_book_repository = OrderBookSnapshotRepository(session_factory)
    news_event_repository = NewsEventRepository(session_factory)
    activity_log_repository = ActivityLogRepository(session_factory)
    signal_decision_repository = SignalDecisionRepository(session_factory)
    strategy_version_repository = StrategyVersionRepository(session_factory)

    # --- housekeeping: activity_logs retention + SQLite WAL checkpoint ---
    # Both grow/accumulate unbounded otherwise (configs/maintenance.yaml) —
    # process-wide, not per-account, since there's one shared SQLite db.
    maintenance_config = load_maintenance_config(settings.configs_dir)
    activity_log_retention_service = ActivityLogRetentionService(
        activity_log_repository,
        retention_days=maintenance_config.activity_log_retention_days,
        check_interval_s=maintenance_config.activity_log_check_interval_hours * 3600,
    )
    wal_checkpoint_service = WalCheckpointService(
        session_factory,
        interval_s=maintenance_config.wal_checkpoint_interval_minutes * 60,
        enabled=maintenance_config.wal_checkpoint_enabled,
    )

    # Deep-learning retraining: one service shared by the API's "Train"
    # button and the bi-weekly scheduler, so a manual run and a scheduled run
    # are the same object with the same history and cannot race each other.
    model_training = ModelTrainingService(repo_root=Path(__file__).resolve().parent.parent)
    training_scheduler = TrainingScheduler(
        model_training,
        list(maintenance_config.model_training_symbols) or symbols,
        interval_days=maintenance_config.model_training_interval_days,
        enabled=maintenance_config.model_training_enabled,
        train_on_startup=maintenance_config.model_training_on_startup,
    )

    global_risk_caps = load_risk_caps(settings.configs_dir)
    volatility_config = load_volatility_config(settings.configs_dir)
    exit_policy_settings = load_exit_policy_settings(settings.configs_dir)
    regime_config = load_regime_config(settings.configs_dir)
    account_risk_caps = {
        cfg.id: _resolve_account_risk_caps(global_risk_caps, cfg, settings.configs_dir)
        for cfg in account_configs
    }
    symbol_configs = {
        symbol: load_symbol_trading_config(symbol, settings.configs_dir) for symbol in symbols
    }
    spread_gate = SpreadGate(symbol_configs)

    review_every_n_trades = load_yaml_config("ai", settings.configs_dir).get(
        "review_every_n_trades", 10
    )

    provider_config_repository = ProviderConfigRepository(session_factory)
    provider_secrets = ProviderSecretStore(Path("data/ai_provider_keys.enc"))
    llm_router = LLMRouter(
        load_llm_provider_config(settings.configs_dir),
        _build_provider_factories(settings, provider_secrets),
        overrides={
            task: ProviderSpec(provider=override.provider, model=override.model)
            for task, override in provider_config_repository.get_all().items()
        },
    )
    provider_settings = ProviderSettingsService(
        repository=provider_config_repository,
        llm_router=llm_router,
        provider_secrets=provider_secrets,
    )
    draft_repository = DraftRepository(session_factory)
    report_repository = AnalysisReportRepository(session_factory)
    proposal_repository = RefinementProposalRepository(session_factory)
    refinement_config = load_refinement_config(settings.configs_dir)

    indicator_repository = IndicatorRepository(session_factory)
    indicators = IndicatorService(
        repository=indicator_repository, candle_repository=candle_repository
    )

    normal_skill_repository = NormalSkillRepository(_SKILLS_DIR)
    normal_skill_selector = SkillSelector(
        skills=normal_skill_repository.load_all(symbols), timezone=timezone
    )

    # magic -> strategy, so PositionManager can resolve the per-bot exit
    # policy from the only bot identity an open position carries.
    magic_to_strategy = {
        magic_number(skill.symbol, skill.name): skill.strategy
        for skills in normal_skill_repository.load_all(symbols).values()
        for skill in skills
    }

    news_config = load_news_config(settings.configs_dir)
    news_skills = _load_news_skills()
    window_specs = {
        name: WindowSpec(
            skill_name=name,
            before_min=skill.activation.window.before_min,
            after_min=skill.activation.window.after_min,
            symbols=skill.activation.symbols,
            close_all=skill.pre_event.close_all,
        )
        for name, skill in news_skills.items()
    }
    news_client = httpx.AsyncClient(
        base_url=(
            settings.finnhub_calendar_url
            if news_config.calendar_source == "finnhub"
            else settings.forexfactory_calendar_url
        ),
        timeout=15.0,
    )
    news_calendar: NewsCalendarPort = (
        FinnhubCalendar(news_client, settings.finnhub_api_key)
        if news_config.calendar_source == "finnhub"
        else ForexFactoryCalendar(news_client)
    )

    alerting_config = load_alerting_config(settings.configs_dir)
    alert_adapters: list[AlertPort] = []
    alert_telegram_client: httpx.AsyncClient | None = None
    if alerting_config.telegram_enabled and settings.telegram_bot_token:
        alert_telegram_client = httpx.AsyncClient(base_url="https://api.telegram.org", timeout=10.0)
        alert_adapters.append(
            TelegramAlertAdapter(
                alert_telegram_client, settings.telegram_bot_token, settings.telegram_chat_id
            )
        )
    if alerting_config.email_enabled and alerting_config.smtp_host:
        alert_adapters.append(
            EmailAlertAdapter(
                smtp_host=alerting_config.smtp_host,
                smtp_port=alerting_config.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                from_address=alerting_config.from_address,
                to_address=alerting_config.to_address,
            )
        )
    alert_port: AlertPort = (
        CompositeAlertAdapter(alert_adapters) if alert_adapters else NoopAlertAdapter()
    )
    alert_service = AlertService(port=alert_port, config=alerting_config)

    baseline_strategies = _baseline_strategies()

    # Event buses are created up front, one per account, so `NewsWindowService`
    # (one process-wide instance) and `skill_selector` (which every account's
    # `TradeEngine` needs at construction) can both be wired before any
    # `AccountRuntime` itself is built — avoids constructing engines with a
    # placeholder and patching them after the fact.
    event_buses = {cfg.id: EventBus() for cfg in account_configs}

    news_window_service = NewsWindowService(
        calendar=news_calendar,
        config=news_config,
        window_specs=window_specs,
        event_bus=_FanOutEventBus(list(event_buses.values())),
        repository=news_event_repository,
    )
    skill_selector: SkillSelectorPort = NewsSkillSelector(
        normal_selector=normal_skill_selector,
        news_skills=news_skills,
        window_source=news_window_service,
    )

    accounts: dict[str, AccountRuntime] = {}
    for account_cfg in account_configs:
        runtime = build_account_runtime(
            settings,
            account_cfg,
            event_bus=event_buses[account_cfg.id],
            symbols=list(symbols),
            timezone=timezone,
            engine_config=engine_config,
            candle_repository=candle_repository,
            symbol_spec_repository=symbol_spec_repository,
            journal_repository=journal_repository,
            order_book_repository=order_book_repository,
            activity_log_repository=activity_log_repository,
            signal_decision_repository=signal_decision_repository,
            strategy_version_repository=strategy_version_repository,
            baseline_strategies=baseline_strategies,
            risk_caps=account_risk_caps[account_cfg.id],
            volatility_config=volatility_config,
            exit_policy_settings=exit_policy_settings,
            magic_to_strategy=magic_to_strategy,
            regime_config=regime_config,
            spread_gate=spread_gate,
            review_every_n_trades=review_every_n_trades,
            skill_selector=skill_selector,
            llm_router=llm_router,
            draft_repository=draft_repository,
            report_repository=report_repository,
            proposal_repository=proposal_repository,
            refinement_config=refinement_config,
            silence_config=alerting_config.silence,
        )
        accounts[account_cfg.id] = runtime

    # Phase 6 Part C: `NewsWindowService` is process-wide and constructed
    # before any `AccountRuntime` exists (see the comment above
    # `event_buses`), so its balance source for `balance_before`/
    # `balance_after` stamping can only be wired in after the accounts loop
    # above. Uses the primary account's balance — `news_events` has no
    # per-account column (one calendar, shared across every account; see
    # `NewsEventRepository`'s module docstring), so there is only room for
    # one account's balance per row, and the primary account is this
    # process's existing "one canonical account" convention (see
    # `Container._primary`).
    news_window_service.set_account_service(accounts[primary_account_id].account)

    for runtime in accounts.values():
        event_bus = runtime.event_bus
        event_bus.subscribe(PositionOpened, alert_service.on_position_opened)
        event_bus.subscribe(PositionClosed, alert_service.on_position_closed)
        event_bus.subscribe(CircuitBreakerTripped, alert_service.on_circuit_breaker_tripped)
        event_bus.subscribe(RefinementCompleted, alert_service.on_refinement_completed)
        event_bus.subscribe(GatewayHealthChanged, alert_service.on_gateway_health_changed)
        event_bus.subscribe(BotWentSilent, alert_service.on_bot_went_silent)
        event_bus.subscribe(NewsWindowEntered, runtime.trade_engine.on_news_window_entered)

    skill_assignment = SkillAssignmentService(
        repository=normal_skill_repository,
        selector=normal_skill_selector,
        strategy_registry=accounts[primary_account_id].strategy_registry,
        candle_streams=[rt.candle_stream for rt in accounts.values()],
        spread_gate=spread_gate,
        configs_dir=settings.configs_dir,
    )

    return Container(
        settings=settings,
        session_issuer=session_issuer,
        symbols=symbols,
        spread_gate=spread_gate,
        indicators=indicators,
        skill_selector=skill_selector,
        skill_assignment=skill_assignment,
        provider_settings=provider_settings,
        news_client=news_client,
        news_window_service=news_window_service,
        activity_log_retention_service=activity_log_retention_service,
        wal_checkpoint_service=wal_checkpoint_service,
        model_training=model_training,
        training_scheduler=training_scheduler,
        accounts=accounts,
        primary_account_id=primary_account_id,
        alert_telegram_client=alert_telegram_client,
    )


def build_account_runtime(
    settings: Settings,
    account_cfg: AccountConfig,
    *,
    event_bus: EventBus,
    symbols: list[str],
    timezone: str,
    engine_config: dict,
    candle_repository: CandleRepository,
    symbol_spec_repository: SymbolSpecRepository,
    journal_repository: JournalRepository,
    order_book_repository: OrderBookSnapshotRepository,
    activity_log_repository: ActivityLogRepository,
    signal_decision_repository: SignalDecisionRepository,
    strategy_version_repository: StrategyVersionRepository,
    baseline_strategies: list[tuple[str, Strategy]],
    risk_caps: RiskCaps,
    volatility_config: VolatilityConfig,
    exit_policy_settings: ExitPolicySettings,
    magic_to_strategy: Mapping[int, str],
    regime_config: RegimeConfig,
    spread_gate: SpreadGate,
    review_every_n_trades: int,
    skill_selector: SkillSelectorPort,
    llm_router: LLMRouter,
    draft_repository: DraftRepository,
    report_repository: AnalysisReportRepository,
    proposal_repository: RefinementProposalRepository,
    refinement_config,
    silence_config: SilenceConfig,
) -> AccountRuntime:
    """Builds one account's full, isolated trading runtime — its own gateway
    connection, event bus, journal, and strategy registry — mirroring what
    the pre-Phase-5 `build_container` wired once for the whole process."""
    account_id = account_cfg.id

    gateway_secret = _resolve_gateway_secret(account_cfg.gateway_shared_secret_env)
    gateway_client = httpx.AsyncClient(
        base_url=account_cfg.gateway_url,
        headers={"X-Gateway-Secret": gateway_secret},
        # 30 s: the Wine-hosted gateway can take 15-25 s on the first
        # mt5.initialize() call while the terminal completes its IPC handshake.
        timeout=30.0,
        event_hooks=_gateway_rtt_hooks(account_id),
    )

    market_data = GatewayMarketData(gateway_client)
    ws_broadcaster = WsBroadcaster(account_id)
    candle_history = CandleHistoryService(
        market_data, candle_repository, symbol_spec_repository, account_id=account_id
    )
    recent_candle_cache: dict[tuple[str, Timeframe], tuple[float, Candle]] = {}
    candle_stream = CandleStreamService(
        market_data=market_data,
        repository=candle_repository,
        event_bus=event_bus,
        broadcaster=ws_broadcaster,
        symbols=symbols,
        timeframes=list(Timeframe),
        candle_history=candle_history,
        recent_candle_cache=recent_candle_cache,
        account_id=account_id,
    )
    live_candle = LiveCandleService(
        market_data=market_data,
        broadcaster=ws_broadcaster,
        recent_candle_cache=recent_candle_cache,
        account_id=account_id,
    )

    account = AccountService(
        gateway=GatewayAccount(gateway_client),
        store=FernetCredentialStore(credentials_path_for(account_id)),
    )

    # mode: paper (always safe) | live (real orders via this account's gateway).
    broker: BrokerPort = (
        GatewayBroker(gateway_client) if account_cfg.mode == "live" else PaperBroker(market_data)
    )
    # Typed signal-decision trail (OBSERVABILITY_PLAN.md Phase 1): the engine
    # records a decision per signal, the order service stamps its fill/reject
    # outcome, and the chart's signal trail reads the table back instead of
    # regex-scraping log lines.
    signal_decisions = SignalDecisionService(signal_decision_repository, account_id=account_id)
    order_service = OrderService(
        broker=broker,
        market_data=market_data,
        spread_gate=spread_gate,
        event_bus=event_bus,
        signal_decisions=signal_decisions,
    )

    trade_journal = TradeJournalService(
        repository=journal_repository,
        market_context=CandleRepositoryMarketContext(candle_repository),
        event_bus=event_bus,
        review_every_n_trades=review_every_n_trades,
        account_id=account_id,
    )
    # Order-book (market depth) capture — an independently-testable vertical
    # slice, not wired into the trade loop yet (a later phase does that).
    # Gracefully degrades to storing nothing for symbols/brokers that report
    # no depth (common for OTC CFD/synthetic symbols); see
    # `order_book/application/capture.py`.
    order_book_capture = OrderBookCaptureService(
        gateway=GatewayOrderBookClient(gateway_client),
        repository=order_book_repository,
        account_id=account_id,
    )

    event_bus.subscribe(PositionOpened, trade_journal.on_position_opened)
    event_bus.subscribe(PositionClosed, trade_journal.on_position_closed)
    # Accumulates MFE/MAE on open trades bar by bar (OBSERVABILITY_PLAN.md
    # Phase 3); the handler ignores every timeframe but M5 itself.
    event_bus.subscribe(CandleClosed, trade_journal.on_candle_closed)

    # `tradingbot_open_positions` gauge (OBSERVABILITY_PLAN.md Phase 5) — driven
    # off the event bus, not counted at scrape time, so it also picks up
    # broker-side closes/opens `ReconciliationService` republishes for
    # tickets the app didn't see change directly (see reconciliation.py).
    async def _on_position_opened_metric(_event: PositionOpened) -> None:
        position_opened(account_id=account_id)

    async def _on_position_closed_metric(_event: PositionClosed) -> None:
        position_closed(account_id=account_id)

    event_bus.subscribe(PositionOpened, _on_position_opened_metric)
    event_bus.subscribe(PositionClosed, _on_position_closed_metric)

    activity_log = ActivityLogService(
        activity_log_repository,
        account_id=account_id,
        signal_decisions=signal_decision_repository,
    )

    risk_manager = RiskManager(caps=risk_caps, timezone=timezone)
    manual_trade_gate = ManualTradeGate(order_service=order_service, risk_manager=risk_manager)

    reconciliation = ReconciliationService(
        broker=broker, journal=trade_journal, event_bus=event_bus
    )
    reconciliation_poller = ReconciliationPoller(
        reconciliation=reconciliation,
        poll_interval_s=engine_config.get("reconciliation_poll_interval_s", 5.0),
        account_id=account_id,
    )
    health_monitor = GatewayHealthMonitor(
        account=account, reconciliation=reconciliation, event_bus=event_bus, account_id=account_id
    )
    # Dead-bot silence alerting (OBSERVABILITY_PLAN.md Phase 5) — reads the
    # same `signal_decision_repository` the decision trail/veto funnel do,
    # scoped to this account.
    silence_monitor = SilenceMonitor(
        signal_decision_repository,
        event_bus,
        account_id=account_id,
        poll_interval_s=silence_config.poll_interval_s,
        lookback=timedelta(days=silence_config.lookback_days),
        multiplier=silence_config.multiplier,
        min_signals=silence_config.min_signals,
    )
    position_manager = PositionManager(
        order_service=order_service,
        market_data=market_data,
        reconciliation=reconciliation,
        risk_manager=risk_manager,
        volatility_config=volatility_config,
        exit_policy_settings=exit_policy_settings,
        magic_to_strategy=magic_to_strategy,
    )

    strategy_registry = StrategyRegistry()
    for name, instance in baseline_strategies:
        strategy_registry.register(name, instance)
    strategy_versions = StrategyVersionService(
        repository=strategy_version_repository,
        registry=strategy_registry,
        generated_dir=_STRATEGIES_GENERATED_DIR,
        account_id=account_id,
    )
    # Restores whichever AI-generated versions were active on *this* account
    # before a restart — separate per account, since Phase 4's repository
    # already keys "active version" by (name, account_id).
    strategy_versions.load_active_into_registry()

    pdf_to_strategy = PdfToStrategyService(
        draft_repository=draft_repository,
        strategy_versions=strategy_versions,
        llm_router=llm_router,
    )
    code_regeneration = CodeRegenerationService(
        strategy_versions=strategy_versions,
        llm_router=llm_router,
    )
    refinement_loop = RefinementLoopService(
        report_repository=report_repository,
        proposal_repository=proposal_repository,
        journal_repository=journal_repository,
        strategy_versions=strategy_versions,
        strategy_registry=strategy_registry,
        skill_selector=skill_selector,
        llm_router=llm_router,
        refinement_config=refinement_config,
        timezone=timezone,
        event_bus=event_bus,
    )
    event_bus.subscribe(TenTradesCompleted, refinement_loop.on_ten_trades_completed)

    trade_engine = TradeEngine(
        market_data=market_data,
        order_service=order_service,
        account=account,
        risk_manager=risk_manager,
        position_manager=position_manager,
        skill_selector=skill_selector,
        strategy_source=strategy_registry,
        entry_timeframe=engine_config.get("entry_timeframe", "M5"),
        volatility_config=volatility_config,
        regime_config=regime_config,
        signal_decisions=signal_decisions,
        order_book_capture=order_book_capture,
        event_bus=event_bus,
        enabled=engine_config.get("enabled", True),
    )
    event_bus.subscribe(CandleClosed, trade_engine.on_candle_closed)
    event_bus.subscribe(PositionClosed, trade_engine.on_position_closed)
    # NewsWindowEntered subscription is wired by build_container, once every
    # account's bus exists (see its loop after this function returns).

    return AccountRuntime(
        id=account_id,
        config=account_cfg,
        symbols=symbols,
        gateway_client=gateway_client,
        event_bus=event_bus,
        market_data=market_data,
        candle_history=candle_history,
        candle_repository=candle_repository,
        symbol_spec_repository=symbol_spec_repository,
        candle_stream=candle_stream,
        live_candle=live_candle,
        ws_broadcaster=ws_broadcaster,
        account=account,
        broker=broker,
        order_service=order_service,
        manual_trade_gate=manual_trade_gate,
        reconciliation=reconciliation,
        reconciliation_poller=reconciliation_poller,
        health_monitor=health_monitor,
        silence_monitor=silence_monitor,
        position_manager=position_manager,
        trade_journal=trade_journal,
        order_book_capture=order_book_capture,
        activity_log=activity_log,
        risk_manager=risk_manager,
        trade_engine=trade_engine,
        strategy_registry=strategy_registry,
        strategy_versions=strategy_versions,
        pdf_to_strategy=pdf_to_strategy,
        code_regeneration=code_regeneration,
        refinement_loop=refinement_loop,
    )


def _build_provider_factories(
    settings: Settings, provider_secrets: ProviderSecretStore
) -> dict[str, ProviderFactory]:
    """One closure per `KNOWN_PROVIDERS` entry, each capturing only the one
    credential/setting it needs from `Settings` — `LLMRouter` never sees a
    raw secret (AI_PROVIDER_SETTINGS_PLAN.md §4.1). Adding a provider is a
    new adapter class (or, for anything OpenAI-wire-compatible, just a new
    `base_url`) plus one more entry here; `LLMRouter` itself doesn't change.

    Every secret-needing provider resolves its key as
    `provider_secrets.get(id) or settings.<x>_api_key` — a settings-page key
    (`PUT /ai/settings/providers/{id}/key`) always wins over the `.env`
    fallback, so switching providers in the UI never requires a restart.
    """

    def _key(provider: str, env_value: str) -> str:
        return provider_secrets.get(provider) or env_value

    def _require_key(provider: str, env_value: str, env_var: str) -> str:
        api_key = _key(provider, env_value)
        if not api_key:
            raise LLMProviderNotConfiguredError(
                f"provider {provider!r} selected but no API key is set — add one on the "
                f"Settings page or set {env_var} in .env"
            )
        return api_key

    def _claude(spec: ProviderSpec) -> ClaudeAdapter:
        api_key = _require_key("claude", settings.anthropic_api_key, "TB_ANTHROPIC_API_KEY")
        return ClaudeAdapter(api_key, spec.model)

    def _openai(spec: ProviderSpec) -> OpenAICompatibleAdapter:
        api_key = _require_key("openai", settings.openai_api_key, "TB_OPENAI_API_KEY")
        return OpenAICompatibleAdapter("https://api.openai.com/v1", api_key, spec.model)

    def _gemini(spec: ProviderSpec) -> GeminiAdapter:
        api_key = _require_key("gemini", settings.gemini_api_key, "TB_GEMINI_API_KEY")
        return GeminiAdapter(api_key, spec.model)

    def _mistral(spec: ProviderSpec) -> OpenAICompatibleAdapter:
        api_key = _require_key("mistral", settings.mistral_api_key, "TB_MISTRAL_API_KEY")
        return OpenAICompatibleAdapter("https://api.mistral.ai/v1", api_key, spec.model)

    def _groq(spec: ProviderSpec) -> OpenAICompatibleAdapter:
        api_key = _require_key("groq", settings.groq_api_key, "TB_GROQ_API_KEY")
        return OpenAICompatibleAdapter("https://api.groq.com/openai/v1", api_key, spec.model)

    def _deepseek(spec: ProviderSpec) -> OpenAICompatibleAdapter:
        api_key = _require_key("deepseek", settings.deepseek_api_key, "TB_DEEPSEEK_API_KEY")
        return OpenAICompatibleAdapter("https://api.deepseek.com/v1", api_key, spec.model)

    def _xai(spec: ProviderSpec) -> OpenAICompatibleAdapter:
        api_key = _require_key("xai", settings.xai_api_key, "TB_XAI_API_KEY")
        return OpenAICompatibleAdapter("https://api.x.ai/v1", api_key, spec.model)

    def _ollama(spec: ProviderSpec) -> OllamaAdapter:
        return OllamaAdapter(settings.ollama_url, spec.model)

    def _claude_code(spec: ProviderSpec) -> ClaudeCodeAdapter:
        if not shutil.which(settings.claude_code_binary):
            raise LLMProviderNotConfiguredError(
                f"provider 'claude_code' selected but {settings.claude_code_binary!r} "
                "was not found on PATH — install the Claude Code CLI (see "
                "AI_PROVIDERS_CONFIGURATION.md) or switch this task to another "
                "provider in configs/ai.yaml"
            )
        return ClaudeCodeAdapter(
            settings.claude_code_binary,
            spec.model,
            settings.claude_code_extra_args,
            settings.claude_code_timeout_s,
        )

    def _openclaw(spec: ProviderSpec) -> OpenClawAdapter:
        api_key = _key("openclaw", settings.openclaw_api_key)
        if not settings.openclaw_url or not api_key:
            raise LLMProviderNotConfiguredError(
                "provider 'openclaw' selected but no URL/API key is set — set TB_OPENCLAW_URL "
                "in .env and add a key on the Settings page (or TB_OPENCLAW_API_KEY in .env). "
                "Note: OpenClaw's wire contract is unverified (AI_PROVIDER_SETTINGS_PLAN.md "
                "§2.4) — treat this provider as beta."
            )
        return OpenClawAdapter(settings.openclaw_url, api_key, spec.model)

    return {
        "claude": _claude,
        "openai": _openai,
        "gemini": _gemini,
        "mistral": _mistral,
        "groq": _groq,
        "deepseek": _deepseek,
        "xai": _xai,
        "ollama": _ollama,
        "claude_code": _claude_code,
        "openclaw": _openclaw,
    }


def _load_news_skills() -> dict[str, NewsSkill]:
    skills: dict[str, NewsSkill] = {}
    for path in sorted(_NEWS_SKILLS_DIR.glob("*.yaml")):
        with path.open() as f:
            data = yaml.safe_load(f)
        activation = data["activation"]
        window = activation["window"]
        pre = data["rules"]["pre_event"]
        post = data["rules"]["post_event"]
        skill = NewsSkill(
            name=data["name"],
            activation=NewsActivation(
                calendar_events=tuple(activation["calendar_event"]),
                window=NewsActivationWindow(
                    before_min=window["before_min"], after_min=window["after_min"]
                ),
                symbols=tuple(activation["symbols"]),
            ),
            pre_event=PreEventRules(
                close_all=pre.get("close_all", False),
                block_new_entries=pre.get("block_new_entries", True),
            ),
            post_event=PostEventRules(
                wait_candles_m5=post.get("wait_candles_m5", 0),
                strategy_override=post.get("strategy_override", ""),
                max_spread_points=post.get("max_spread_points", 0),
                risk_multiplier=post.get("risk_multiplier", 1.0),
            ),
        )
        skills[skill.name] = skill
    return skills
