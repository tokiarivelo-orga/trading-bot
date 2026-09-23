from src.alerting.domain.models import AlertingConfig
from src.broker.domain.account import AccountConfig
from src.shared.config.loaders import (
    load_accounts_config,
    load_alerting_config,
    load_maintenance_config,
    load_regime_config,
)
from src.shared.config.maintenance import MaintenanceConfig
from src.shared.config.settings import CONFIGS_DIR


def test_load_accounts_config_returns_typed_accounts():
    accounts = load_accounts_config(CONFIGS_DIR)
    assert len(accounts) >= 1
    for account in accounts:
        assert isinstance(account, AccountConfig)
        assert account.id
        assert account.gateway_url.startswith("http")
        assert account.mode in ("paper", "live")


def test_load_accounts_config_defaults_enabled_and_risk_override():
    accounts = load_accounts_config(CONFIGS_DIR)
    default_account = next(a for a in accounts if a.id == "default")
    assert default_account.enabled is True
    assert default_account.risk_override_file is None


def test_load_maintenance_config_returns_typed_config():
    config = load_maintenance_config(CONFIGS_DIR)
    assert isinstance(config, MaintenanceConfig)
    assert config.activity_log_retention_days > 0
    assert config.activity_log_check_interval_hours > 0
    assert config.wal_checkpoint_interval_minutes > 0


def test_load_regime_config_carries_the_volatility_tag_thresholds():
    # The volatility bucket is a tag only now (the guard that acted on it is
    # gone); its thresholds moved from the deleted volatility.yaml into
    # regime.yaml unchanged, so `regime_volatility` stays comparable with
    # historical trades.
    config = load_regime_config(CONFIGS_DIR)
    assert config.volatility_atr_period == 14
    assert config.volatility_lookback_bars == 100
    assert config.volatility_low_percentile == 20
    assert config.volatility_high_percentile == 60
    assert config.volatility_extreme_percentile == 98


def test_volatility_yaml_is_gone():
    assert not (CONFIGS_DIR / "volatility.yaml").exists()


def test_load_alerting_config_includes_silence_tuning():
    # OBSERVABILITY_PLAN.md Phase 5: configs/alerting.yaml's `silence:`
    # section, alongside the existing `events.bot_silence` flag.
    config = load_alerting_config(CONFIGS_DIR)
    assert isinstance(config, AlertingConfig)
    assert config.events.bot_silence is True
    assert config.silence.poll_interval_s == 15 * 60.0
    assert config.silence.lookback_days == 30
    assert config.silence.multiplier == 5.0
    assert config.silence.min_signals == 5
