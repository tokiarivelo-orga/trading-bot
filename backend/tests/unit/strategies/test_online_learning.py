"""Tests for the self-labeling online learner.

The two that matter most are the no-lookahead pair. A self-labeling learner
is one careless line away from being an oracle that backtests beautifully and
loses money live, so "a sample cannot be graded by its own entry bar" and
"re-feeding overlapping windows changes nothing" are asserted directly rather
than assumed from reading the code.
"""

import numpy as np
import pytest

from src.strategies.domain.online_learning import (
    AdaptiveLearner,
    LearnerConfig,
    breakeven_probability,
)

NS = 60 * 1_000_000_000  # one minute


def _features(n=4, fill=0.5):
    return np.full(n, fill, dtype=float)


def _learner(**overrides):
    config = LearnerConfig(
        min_bucket_samples=overrides.pop("min_bucket_samples", 5.0),
        min_model_samples=overrides.pop("min_model_samples", 5),
        min_quantile_samples=overrides.pop("min_quantile_samples", 3),
        **overrides,
    )
    return AdaptiveLearner(4, config)


def _observe(learner, *, entry_ns=0, price=100.0, direction=1, sl=1.0, bucket="B", horizon=50):
    return learner.observe(
        features=_features(),
        bucket=bucket,
        entry_ns=entry_ns,
        entry_price=price,
        direction=direction,
        sl_dist=sl,
        atr=1.0,
        deadline_ns=entry_ns + horizon * NS,
    )


def test_breakeven_probability_matches_the_engines_trailing_buffer():
    # Risking 1R to bank 0.2R needs 1/1.2 to break even. This number is why
    # a 0.62 live secure-rate loses money.
    assert breakeven_probability(0.2) == pytest.approx(1 / 1.2)
    assert breakeven_probability(1.0) == pytest.approx(0.5)


def test_entry_bar_cannot_grade_its_own_sample():
    learner = _learner()
    _observe(learner, entry_ns=10 * NS)
    # A bar at exactly the entry timestamp that would blow through both
    # barriers must resolve nothing: the fill happened on that bar.
    resolved = learner.advance(
        np.array([10 * NS]), np.array([200.0]), np.array([0.0])
    )
    assert resolved == []
    assert learner.pending_count == 1


def test_sample_resolves_on_a_later_bar_only():
    learner = _learner()
    _observe(learner, entry_ns=10 * NS, price=100.0, sl=1.0)  # secure at +0.2
    resolved = learner.advance(
        np.array([10 * NS, 11 * NS]),
        np.array([100.0, 100.3]),
        np.array([100.0, 99.9]),
    )
    assert len(resolved) == 1
    assert resolved[0].label == 1
    assert learner.pending_count == 0


def test_stop_before_secure_labels_zero():
    learner = _learner()
    _observe(learner, entry_ns=0, price=100.0, sl=1.0)
    resolved = learner.advance(
        np.array([0, NS]), np.array([100.05, 100.05]), np.array([100.0, 98.9])
    )
    assert len(resolved) == 1
    assert resolved[0].label == 0
    assert resolved[0].r_outcome == -1.0


def test_same_bar_tie_resolves_pessimistically_as_a_stop():
    learner = _learner()
    _observe(learner, entry_ns=0, price=100.0, sl=1.0)
    # One bar that touches both +0.2 and -1.0; the intrabar path is unknown.
    resolved = learner.advance(
        np.array([0, NS]), np.array([100.0, 101.0]), np.array([100.0, 98.0])
    )
    assert len(resolved) == 1
    assert resolved[0].label == 0


def test_advance_is_idempotent_across_overlapping_windows():
    """The engine hands strategies a sliding window, so the same bars arrive
    on call after call. Re-feeding them must not double-count an excursion
    or re-resolve a sample."""
    a, b = _learner(), _learner()
    _observe(a, entry_ns=0, price=100.0, sl=1.0)
    _observe(b, entry_ns=0, price=100.0, sl=1.0)

    times = np.array([0, NS, 2 * NS, 3 * NS])
    highs = np.array([100.0, 100.1, 100.15, 100.3])
    lows = np.array([100.0, 99.9, 99.8, 99.7])

    a.advance(times, highs, lows)
    # b sees the same tape as three overlapping windows.
    b.advance(times[:2], highs[:2], lows[:2])
    b.advance(times[:3], highs[:3], lows[:3])
    b.advance(times, highs, lows)

    assert a.resolved_count == b.resolved_count == 1
    assert a.global_rate() == pytest.approx(b.global_rate())


def test_pending_samples_do_not_influence_the_score():
    learner = _learner()
    before = learner.score(_features(), "B").p_secure
    for i in range(20):
        _observe(learner, entry_ns=i * NS)
    assert learner.pending_count == 20
    assert learner.score(_features(), "B").p_secure == pytest.approx(before)


def test_time_barrier_expires_at_reduced_weight():
    learner = _learner()
    _observe(learner, entry_ns=0, price=100.0, sl=1.0, horizon=2)
    resolved = learner.advance(
        np.array([0, NS, 2 * NS, 3 * NS]),
        np.array([100.0, 100.05, 100.05, 100.05]),
        np.array([100.0, 99.99, 99.99, 99.99]),
    )
    assert len(resolved) == 1
    assert resolved[0].expired is True
    assert resolved[0].weight == pytest.approx(0.5)


def test_short_side_barriers_are_mirrored():
    learner = _learner()
    _observe(learner, entry_ns=0, price=100.0, direction=-1, sl=1.0)
    resolved = learner.advance(
        np.array([0, NS]), np.array([100.0, 100.05]), np.array([100.0, 99.7])
    )
    assert len(resolved) == 1
    assert resolved[0].label == 1  # price fell 0.3 >= 0.2 secure distance


def test_observe_rejects_ungradeable_inputs():
    learner = _learner()
    assert _observe(learner, sl=0.0) is False
    assert learner.observe(
        features=_features(), bucket="B", entry_ns=0, entry_price=100.0,
        direction=1, sl_dist=1.0, atr=0.0, deadline_ns=NS,
    ) is False
    assert learner.observe(
        features=np.array([np.nan, 0.0, 0.0, 0.0]), bucket="B", entry_ns=0,
        entry_price=100.0, direction=1, sl_dist=1.0, atr=1.0, deadline_ns=NS,
    ) is False
    assert learner.pending_count == 0


def test_prior_logit_shifts_a_cold_estimate_and_evidence_overrides_it():
    learner = _learner(bucket_prior_strength=5.0)
    neutral = learner.score(_features(), "B").p_secure
    penalised = learner.score(_features(), "B", -1.5).p_secure
    boosted = learner.score(_features(), "B", 1.5).p_secure
    assert penalised < neutral < boosted

    # Feed the penalised bucket a run of wins; its own evidence must pull the
    # estimate back up despite the negative prior.
    for i in range(40):
        _observe(learner, entry_ns=i * 10 * NS, price=100.0, sl=1.0, bucket="B")
        learner.advance(
            np.array([i * 10 * NS, (i * 10 + 1) * NS]),
            np.array([100.0, 100.5]),
            np.array([100.0, 99.95]),
        )
    assert learner.score(_features(), "B", -1.5).p_secure > penalised


def test_p_reach_and_best_target_use_the_measured_curve():
    learner = _learner()
    # A curve where 1.5R is reached often and 3.0R never.
    def curve(r):
        return {1.5: 0.6, 3.0: 0.05}[r]

    r_target, expected, p_hit = learner.best_target(
        "empty", grid=(1.5, 3.0), prior_curve=curve
    )
    assert r_target == 1.5
    assert p_hit == pytest.approx(0.6)
    assert expected == pytest.approx(0.6 * 1.5 - 0.4)


def test_p_reach_shrinks_toward_the_prior_when_a_bucket_is_thin():
    learner = _learner(bucket_prior_strength=40.0)
    far = learner.p_reach("unseen", 2.0, prior_p=0.25)
    assert far == pytest.approx(0.25)


def test_blend_clamps_learned_values_into_a_band_around_the_base():
    assert AdaptiveLearner.blend(1.0, None, lo_frac=0.5, hi_frac=2.0) == 1.0
    assert AdaptiveLearner.blend(1.0, 99.0, lo_frac=0.5, hi_frac=2.0) == 2.0
    assert AdaptiveLearner.blend(1.0, 0.01, lo_frac=0.5, hi_frac=2.0) == 0.5
    assert AdaptiveLearner.blend(1.0, 1.3, lo_frac=0.5, hi_frac=2.0) == pytest.approx(1.3)
    assert AdaptiveLearner.blend(2.0, float("nan"), lo_frac=0.5, hi_frac=2.0) == 2.0


def test_adverse_excursion_ignores_losers():
    """A loser's MAE is just the stop distance it was given. Feeding those
    back would make the buffer a measurement of its own previous output."""
    learner = _learner(min_quantile_samples=2)
    for i in range(6):
        _observe(learner, entry_ns=i * 10 * NS, price=100.0, sl=1.0)
        learner.advance(
            np.array([i * 10 * NS, (i * 10 + 1) * NS]),
            np.array([100.0, 100.05]),
            np.array([100.0, 98.5]),  # all stops
        )
    assert learner.resolved_count == 6
    assert learner.adverse_excursion_atr("B", 0.8) is None  # no winners recorded


def test_reset_clears_every_learned_thing():
    learner = _learner()
    for i in range(10):
        _observe(learner, entry_ns=i * 10 * NS, price=100.0, sl=1.0)
        learner.advance(
            np.array([i * 10 * NS, (i * 10 + 1) * NS]),
            np.array([100.0, 100.5]), np.array([100.0, 99.95]),
        )
    assert learner.resolved_count > 0
    learner.reset()
    assert learner.resolved_count == 0
    assert learner.pending_count == 0
    assert learner.snapshot()["buckets"] == 0


def test_pending_ledger_is_bounded():
    learner = _learner(max_pending=10)
    for i in range(50):
        _observe(learner, entry_ns=i * NS)
    assert learner.pending_count == 10
