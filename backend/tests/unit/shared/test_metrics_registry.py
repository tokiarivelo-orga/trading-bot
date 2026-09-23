"""Prometheus metrics registry (OBSERVABILITY_PLAN.md Phase 5) — asserts on
`generate_latest(REGISTRY)` output rather than the private per-child metric
API, so these tests exercise exactly what `GET /metrics` actually serves."""

from __future__ import annotations

from prometheus_client import generate_latest

from src.shared.logging.account_context import bind_account_id
from src.shared.metrics.registry import (
    ENGINE_LOOP_DURATION,
    REGISTRY,
    observe_gateway_rtt,
    observe_signal_fired,
    position_closed,
    position_opened,
    record_signal_outcome,
    set_open_positions,
    set_ws_client_source,
)


def _scrape() -> str:
    return generate_latest(REGISTRY).decode()


def test_observe_signal_fired_uses_bound_account_id_as_a_label():
    with bind_account_id("acct-signals"):
        observe_signal_fired(bot="normal/xauusd/breakout_v1", symbol="XAUUSD")
    text = _scrape()
    assert (
        'tradingbot_signals_total{account_id="acct-signals",'
        'bot="normal/xauusd/breakout_v1",symbol="XAUUSD"}' in text
    )


def test_record_signal_outcome_reuses_the_phase2_vocabulary_label():
    with bind_account_id("acct-outcomes"):
        record_signal_outcome("htf_veto")
    text = _scrape()
    assert 'tradingbot_signal_outcomes_total{account_id="acct-outcomes",outcome="htf_veto"}' in text


def test_record_signal_outcome_folds_unrecognized_strings_into_unknown():
    with bind_account_id("acct-unknown"):
        record_signal_outcome("not_a_real_outcome")
    text = _scrape()
    assert 'outcome="not_a_real_outcome"' not in text
    assert 'tradingbot_signal_outcomes_total{account_id="acct-unknown",outcome="unknown"}' in text


def test_engine_loop_duration_records_elapsed_time_as_a_context_manager():
    with ENGINE_LOOP_DURATION.labels(account_id="acct-loop").time():
        pass
    text = _scrape()
    assert 'tradingbot_engine_loop_duration_seconds_count{account_id="acct-loop"} 1.0' in text


def test_observe_gateway_rtt_labels_method_and_path():
    observe_gateway_rtt(account_id="acct-gw", method="GET", path="/candles", seconds=0.05)
    text = _scrape()
    assert (
        "tradingbot_gateway_request_duration_seconds_count"
        '{account_id="acct-gw",method="GET",path="/candles"} 1.0' in text
    )


def test_position_opened_and_closed_move_the_open_positions_gauge():
    position_opened(account_id="acct-pos")
    position_opened(account_id="acct-pos")
    position_closed(account_id="acct-pos")
    text = _scrape()
    assert 'tradingbot_open_positions{account_id="acct-pos"} 1.0' in text


def test_set_open_positions_seeds_an_explicit_count():
    set_open_positions(account_id="acct-seed", count=3)
    text = _scrape()
    assert 'tradingbot_open_positions{account_id="acct-seed"} 3.0' in text


def test_set_ws_client_source_samples_the_callback_at_scrape_time():
    count = {"n": 0}
    set_ws_client_source(lambda: count["n"])
    count["n"] = 7
    text = _scrape()
    assert "tradingbot_ws_clients_connected 7.0" in text

    # And a later scrape reflects a changed count — proof it's sampled live,
    # not cached from the first call.
    count["n"] = 2
    assert "tradingbot_ws_clients_connected 2.0" in _scrape()
