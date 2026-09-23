from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.activity.adapters.signal_decision_repository import SignalDecisionRepository
from src.activity.domain.models import DecisionCheck, SignalDecision
from src.shared.db.base import Base


@pytest.fixture
def repository(tmp_path) -> SignalDecisionRepository:
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    return SignalDecisionRepository(sessionmaker(bind=engine, expire_on_commit=False))


def _decision(
    signal_id: str,
    *,
    at: int = 1000,
    bot: str = "normal/xauusd/breakout_v1",
    account_id: str = "default",
    checks: tuple[DecisionCheck, ...] = (),
    regime_volatility: str | None = None,
    regime_volatility_percentile: float | None = None,
    regime_trend: str | None = None,
    regime_adx: float | None = None,
    regime_session: str | None = None,
) -> SignalDecision:
    return SignalDecision(
        signal_id=signal_id,
        account_id=account_id,
        bot=bot,
        strategy="breakout_v1",
        symbol="XAUUSD",
        timeframe="M5",
        direction="buy",
        price=2412.35,
        created_at=datetime.fromtimestamp(at, tz=UTC),
        outcome="skipped",
        reason="retest of demand",
        confidence=0.8,
        checks=checks,
        regime_volatility=regime_volatility,
        regime_volatility_percentile=regime_volatility_percentile,
        regime_trend=regime_trend,
        regime_adx=regime_adx,
        regime_session=regime_session,
    )


def test_save_and_list_round_trips_every_field(repository):
    repository.save(_decision("a"))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")

    assert loaded == _decision("a")


def test_save_and_list_round_trips_regime_tag(repository):
    """Regime tagging (OBSERVABILITY_PLAN.md Phase 6)."""
    repository.save(
        _decision(
            "a",
            regime_volatility="high",
            regime_volatility_percentile=82.5,
            regime_trend="trending",
            regime_adx=27.3,
            regime_session="london",
        )
    )

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")

    assert loaded == _decision(
        "a",
        regime_volatility="high",
        regime_volatility_percentile=82.5,
        regime_trend="trending",
        regime_adx=27.3,
        regime_session="london",
    )


def test_regime_fields_default_to_none(repository):
    repository.save(_decision("a"))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")

    assert loaded.regime_volatility is None
    assert loaded.regime_volatility_percentile is None
    assert loaded.regime_trend is None
    assert loaded.regime_adx is None
    assert loaded.regime_session is None


def test_list_for_bot_is_oldest_first_and_scoped(repository):
    repository.save(_decision("a", at=2000))
    repository.save(_decision("b", at=1000))
    repository.save(_decision("c", at=1500, bot="normal/xauusd/other"))
    repository.save(_decision("d", at=1500, account_id="second"))

    ids = [d.signal_id for d in repository.list_for_bot(bot="normal/xauusd/breakout_v1")]

    assert ids == ["b", "a"]


def test_list_for_bot_filters_the_time_range(repository):
    repository.save(_decision("a", at=1000))
    repository.save(_decision("b", at=3000))

    ids = [
        d.signal_id
        for d in repository.list_for_bot(
            bot="normal/xauusd/breakout_v1", created_from=2000, created_to=4000
        )
    ]

    assert ids == ["b"]


def test_set_outcome_updates_outcome_and_reason(repository):
    repository.save(_decision("a"))

    changed = repository.set_outcome("a", "htf_veto", reason="retest of demand — M15 trend down")

    assert changed is True
    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert loaded.outcome == "htf_veto"
    assert loaded.reason == "retest of demand — M15 trend down"


def test_set_outcome_cannot_downgrade_an_opened_decision(repository):
    repository.save(_decision("a"))
    repository.set_outcome("a", "opened")

    changed = repository.set_outcome("a", "risk_rejected", reason="TP2: below min lot")

    assert changed is False
    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert loaded.outcome == "opened"


def test_set_outcome_on_unknown_signal_id_is_a_no_op(repository):
    assert repository.set_outcome("missing", "opened") is False


def test_earliest_created_at_is_none_when_empty(repository):
    assert repository.earliest_created_at() is None


def test_earliest_created_at_is_per_account(repository):
    repository.save(_decision("a", at=5000))
    repository.save(_decision("b", at=1000, account_id="second"))

    assert repository.earliest_created_at() == 5000
    assert repository.earliest_created_at(account_id="second") == 1000


# --- Phase 2: structured per-gate checks --------------------------------------


def _check(name: str, *, passed: bool = True) -> DecisionCheck:
    return DecisionCheck(name=name, value=1.0, threshold=2.0, comparison="<=", passed=passed)


def test_checks_round_trip_through_the_json_column(repository):
    repository.save(_decision("a", checks=(_check("spread_points"),)))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")

    assert loaded.checks == (_check("spread_points"),)


def test_a_decision_saved_without_checks_loads_as_an_empty_tuple(repository):
    repository.save(_decision("a"))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")

    assert loaded.checks == ()


def test_append_checks_accumulates_across_gates_without_touching_the_outcome(repository):
    repository.save(_decision("a"))

    repository.append_checks("a", (_check("htf_confirm"),))
    repository.append_checks("a", (_check("position_volume"),))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert [c.name for c in loaded.checks] == ["htf_confirm", "position_volume"]
    assert loaded.outcome == "skipped"


def test_appending_the_same_check_twice_does_not_duplicate_it(repository):
    """Multi-target entries re-evaluate the same gates per target."""
    repository.save(_decision("a"))

    repository.append_checks("a", (_check("htf_confirm"),))
    repository.append_checks("a", (_check("htf_confirm"),))

    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert [c.name for c in loaded.checks] == ["htf_confirm"]


def test_append_checks_on_an_unknown_signal_id_is_a_no_op(repository):
    assert repository.append_checks("missing", (_check("htf_confirm"),)) is False


def test_set_outcome_with_checks_records_both(repository):
    repository.save(_decision("a", checks=(_check("htf_confirm"),)))

    changed = repository.set_outcome(
        "a", "spread_veto", reason="wide", checks=(_check("spread_points", passed=False),)
    )

    assert changed is True
    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert loaded.outcome == "spread_veto"
    assert [c.name for c in loaded.checks] == ["htf_confirm", "spread_points"]


def test_set_outcome_with_checks_still_cannot_downgrade_an_opened_decision(repository):
    """The check is still worth keeping (it happened), but a later target's
    rejection must not un-open the decision."""
    repository.save(_decision("a"))
    repository.set_outcome("a", "opened")

    changed = repository.set_outcome(
        "a", "max_positions", checks=(_check("open_positions", passed=False),)
    )

    assert changed is False
    (loaded,) = repository.list_for_bot(bot="normal/xauusd/breakout_v1")
    assert loaded.outcome == "opened"
    assert [c.name for c in loaded.checks] == ["open_positions"]


def test_list_between_returns_every_bot_in_the_window_oldest_first(repository):
    repository.save(_decision("a", at=2000))
    repository.save(_decision("b", at=1000, bot="normal/xauusd/other"))
    repository.save(_decision("c", at=9000))
    repository.save(_decision("d", at=1500, account_id="second"))

    ids = [d.signal_id for d in repository.list_between(created_from=500, created_to=5000)]

    assert ids == ["b", "a"]


def test_list_between_can_be_narrowed_to_one_bot(repository):
    repository.save(_decision("a", at=1000))
    repository.save(_decision("b", at=1000, bot="normal/xauusd/other"))

    ids = [d.signal_id for d in repository.list_between(bot="normal/xauusd/other")]

    assert ids == ["b"]
