import pytest

from src.shared.config.settings import CONFIGS_DIR, load_yaml_config


def test_app_config_loads_and_defaults_to_paper_mode():
    cfg = load_yaml_config("app")
    assert cfg["mode"] in ("paper", "live")
    assert {"XAUUSD", "XAGUSD"}.issubset(set(cfg["symbols"]))
    assert cfg["engine"]["entry_timeframe"] == "M5"


def test_risk_config_has_user_owned_caps():
    cfg = load_yaml_config("risk")
    for key in (
        "risk_per_trade_pct",
        "daily_loss_limit_pct",
        "max_open_positions",
        "max_trades_per_day_enabled",
        "consecutive_loss_pause",
    ):
        assert key in cfg, f"risk.yaml missing {key}"
    assert cfg["risk_per_trade_pct"] <= 2.0


def test_every_enabled_symbol_has_a_symbol_config():
    app = load_yaml_config("app")
    for symbol in app["symbols"]:
        cfg = load_yaml_config(f"symbols/{symbol.lower()}")
        assert cfg["symbol"] == symbol
        assert cfg["max_spread_points"] > 0
        assert cfg["min_rr"] >= 1.0


def test_volatility_config_has_classifier_and_regime_fields():
    cfg = load_yaml_config("volatility")
    for key in (
        "atr_period",
        "regime_lookback_bars",
        "low_percentile",
        "high_percentile",
        "extreme_percentile",
        "sl_multiplier_low",
        "sl_multiplier_normal",
        "sl_multiplier_high",
        "tp_multiplier_low",
        "tp_multiplier_normal",
        "tp_multiplier_high",
        "extreme_close_if_losing",
        "extreme_profit_lock_r_mult",
        "chandelier_atr_mult",
        "chandelier_min_profit_r",
    ):
        assert key in cfg, f"volatility.yaml missing {key}"
    assert cfg["low_percentile"] < cfg["high_percentile"] < cfg["extreme_percentile"]


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml_config("does-not-exist", CONFIGS_DIR)


def test_accounts_config_has_at_least_one_account_with_required_fields():
    cfg = load_yaml_config("accounts")
    accounts = cfg["accounts"]
    assert len(accounts) >= 1
    for entry in accounts:
        for key in ("id", "label", "gateway_url", "gateway_shared_secret_env", "mode"):
            assert key in entry, f"accounts.yaml entry missing {key}"
        assert entry["mode"] in ("paper", "live")
    ids = [entry["id"] for entry in accounts]
    assert len(ids) == len(set(ids)), "accounts.yaml has duplicate account ids"
