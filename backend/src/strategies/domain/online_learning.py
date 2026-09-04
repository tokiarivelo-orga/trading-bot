"""Online, self-labeled learning for generated strategies.

A strategy that "learns and re-adapts to the state of the market" has three
hard constraints in this codebase, and every design choice below falls out of
them:

  1. `strategies/sandbox.py` allows generated code to import only `math`,
     `statistics`, `numpy`, `pandas`, and whitelisted pure-domain modules —
     so there is no scikit-learn, no XGBoost, no torch. The model has to be
     something you can write in numpy. It also allows no file or network I/O,
     so learned state lives in the strategy instance's memory and nowhere else.
  2. `Strategy.evaluate()` receives a `MarketContext` — candles and a spread.
     It **never sees a trade outcome**. There is no reward signal handed back
     from the broker or the journal, so supervised learning is only possible
     if the strategy labels its own setups from subsequent price action.
  3. `trade_loop.DEFAULT_CONTEXT_BARS = 200` — every call sees a short
     trailing window, not a growing history. Anything the learner wants to
     remember it must accumulate across calls itself.

This module is that accumulator: a self-labeling triple-barrier ledger, a
per-regime expectancy table, and a small online logistic model, all pure
numpy with no I/O — which is what lets it sit on the sandbox allowlist next
to `domain.fatigue` (same rationale, see that module's docstring).

────────────────────────────────────────────────────────────────────────
WHAT "WIN" MEANS HERE — AND WHY IT IS NOT THE STRATEGY'S TAKE-PROFIT
────────────────────────────────────────────────────────────────────────
The obvious label is "did TP come before SL". On this engine that label
would be measuring the wrong thing.

`PositionManager` rule 2 (secure-base trailing, `DEFAULT_SECURE_BUFFER_R_MULT
= 0.2`) moves the stop to entry + 0.2R as soon as a fresh base is cleared.
On an M1 bot that fires within a couple of bars nearly every time, and the
position is then taken out at exactly +0.20R. Measured on a 2,118-trade
XAUUSD M1 backtest: 93.9% of trades closed at r_multiple 0.1999…, median 2
minutes after entry. The strategy's own take-profit almost never decides the
outcome — so a TP-vs-SL label would train the model on an event that hardly
ever happens.

What *does* decide P&L on this engine is whether the trade survives long
enough to secure. So that is the label:

    y = 1  price travelled `secure_r` x risk in favour **before** touching
           the stop  →  the trailing rule locks in roughly +0.2R
    y = 0  the stop was touched first                    →  roughly -1R

Which makes the break-even probability explicit and unusually high:

    E[R] = p * secure_r - (1 - p) * 1  =  0   →   p* = 1 / (1 + secure_r)

At `secure_r = 0.2` that is **p* = 0.8333**. A base rate near 0.94 leaves
real headroom, and the learner's whole job is to find the pockets where p
falls under p* and decline to trade them. This is why the filter is worth
having even though entry gating has weak leverage on *total* P&L here: the
losing tail is what the gate removes.

────────────────────────────────────────────────────────────────────────
NO LOOKAHEAD, BY CONSTRUCTION
────────────────────────────────────────────────────────────────────────
A self-labeling learner is one careless line away from being a lookahead
oracle that backtests beautifully and loses money live. Two rules keep it
honest, and `tests/unit/strategies/test_online_learning.py` asserts both:

  * A sample is created when the setup fires and is **pending**. It can only
    be resolved by bars strictly later than its entry bar (`advance()`
    ignores any bar whose timestamp is <= the entry timestamp, and re-feeding
    the same bars is idempotent).
  * A pending sample contributes **nothing** to `score()` — not to the
    bucket table, not to the model — until it resolves. The estimate used to
    gate a decision at bar *t* is therefore a function of labels that were
    all fully determined by bar *t*.

When both barriers fall inside the same bar, the bar's internal path is
unknown; the sample resolves as `y = 0` (stop first). Pessimistic on
purpose — the alternative flatters exactly the marginal setups the gate
exists to remove.

────────────────────────────────────────────────────────────────────────
HOW IT ADAPTS RATHER THAN JUST FITTING
────────────────────────────────────────────────────────────────────────
Adaptation is deliberate forgetting, in three places:

  * The logistic model is trained by plain SGD at a **constant** learning
    rate. Constant-rate SGD never converges to a fixed point; it tracks one,
    weighting recent samples exponentially. That is the feature, not a bug —
    a decayed schedule would freeze the model in whatever regime it saw
    first.
  * Bucket counts decay geometrically per observation
    (`half_life_samples`), so an expectancy estimate is always dominated by
    recent evidence for that bucket.
  * Excursion histories are bounded FIFO reservoirs, so learned SL buffers
    and TP distances are quantiles of recent behaviour, not of all history.

────────────────────────────────────────────────────────────────────────
EVIDENCE BEFORE AUTHORITY
────────────────────────────────────────────────────────────────────────
Every accessor takes the caller's base value and returns it unchanged until
the evidence threshold for that specific answer is met (`min_bucket_samples`
for a bucket rate, `min_model_samples` before the model contributes at all,
`min_quantile_samples` before an excursion quantile is trusted). A cold
learner is therefore exactly the un-adapted strategy — the failure mode is
"no adaptation", never "confident nonsense from four samples". Learned
values are additionally clamped to a bounded band around the base value, so
even a pathological run can only nudge the strategy, not redefine it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mirrors `engine.application.position_manager.DEFAULT_SECURE_BUFFER_R_MULT`.
# Duplicated rather than imported because `engine.*` is not on the sandbox
# allowlist and must not be — this module has to stay importable from
# generated strategy code. If that constant moves, this one follows.
DEFAULT_SECURE_R = 0.2

_LOGIT_EPS = 1e-6


def breakeven_probability(secure_r: float = DEFAULT_SECURE_R) -> float:
    """The win rate at which `secure_r`-for-1R trading exactly breaks even.

    Risking 1R to bank `secure_r` needs `1 / (1 + secure_r)` — 0.833 at the
    engine's 0.2 default. Exposed because a caller gating on probability
    needs this number to set a sane threshold, and hardcoding 0.83 in a
    strategy file would silently go wrong the day the engine's buffer moves.
    """
    return 1.0 / (1.0 + max(secure_r, _LOGIT_EPS))


def _sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logit(p: float) -> float:
    clipped = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return float(np.log(clipped / (1.0 - clipped)))


@dataclass(frozen=True, kw_only=True)
class LearnerConfig:
    """Every knob the learner has. Defaults are tuned for an M1 bot seeing a
    few hundred setups a month; a slower timeframe wants smaller sample
    thresholds and a longer half-life."""

    secure_r: float = DEFAULT_SECURE_R
    # Sample-count gates — below each, the corresponding answer is "use the
    # caller's base value" rather than a learned one.
    min_bucket_samples: float = 25.0
    min_model_samples: int = 150
    min_quantile_samples: int = 30
    # Empirical-Bayes shrinkage strengths, in units of pseudo-observations.
    bucket_prior_strength: float = 40.0
    global_prior_strength: float = 60.0
    global_prior_rate: float = 0.85
    # Online SGD. `learning_rate` is constant on purpose (see module
    # docstring); `weight_clip` bounds any single feature's influence so one
    # runaway coefficient cannot dominate the score.
    learning_rate: float = 0.05
    l2: float = 1e-3
    weight_clip: float = 4.0
    # Geometric forgetting for bucket counts, in observations.
    half_life_samples: float = 750.0
    # Bounded memory. `max_pending` caps unresolved setups (oldest evicted),
    # `max_bucket_history` caps each bucket's excursion reservoir.
    max_pending: int = 400
    max_bucket_history: int = 200
    # Time-barrier expiries are genuine information but weaker than a clean
    # barrier touch, so they train at reduced weight.
    expiry_sample_weight: float = 0.5


@dataclass
class _PendingSample:
    """A setup that has fired and is waiting for price to grade it."""

    features: np.ndarray
    bucket: str
    entry_ns: int
    entry_price: float
    direction: int  # +1 long, -1 short
    sl_dist: float
    secure_dist: float
    atr: float
    deadline_ns: int
    last_seen_ns: int
    mfe: float = 0.0  # max favourable excursion so far, price units
    mae: float = 0.0  # max adverse excursion so far, price units


@dataclass(frozen=True, kw_only=True)
class ResolvedSample:
    """A graded setup. `r_outcome` is what this trade would have booked under
    the engine's secure-base trailing: `secure_r` on a win, -1 on a stop."""

    features: np.ndarray
    bucket: str
    label: int
    weight: float
    mfe_r: float
    mae_atr: float
    r_outcome: float
    expired: bool


@dataclass(frozen=True, kw_only=True)
class LearnerVerdict:
    """What the learner thinks of one setup, and how much it actually knows.

    `ready` is False whenever the estimate is not yet backed by enough
    evidence to gate on. A caller must treat that as "no opinion" and fall
    back to its base rules — never as a veto."""

    p_secure: float
    expectancy_r: float
    bucket_samples: float
    model_samples: int
    ready: bool

    def edge_over(self, threshold: float) -> float:
        """Signed distance from a probability threshold — positive means the
        setup clears it. Reported in log-odds so the number stays meaningful
        near the 0.83 break-even point where raw probability differences get
        misleadingly small."""
        return _logit(self.p_secure) - _logit(threshold)


class _Standardizer:
    """Running per-feature mean/variance (Welford), so the model sees inputs
    on a comparable scale without a pre-training pass over data it does not
    have. Outputs are clipped to +/-5 sigma: one freak feature value must not
    be able to swing the decision boundary."""

    def __init__(self, n_features: int) -> None:
        self.n = 0
        self.mean = np.zeros(n_features, dtype=float)
        self.m2 = np.zeros(n_features, dtype=float)

    def observe(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.m2 / (self.n - 1))
        std = np.where(std < 1e-9, 1.0, std)
        return np.clip((x - self.mean) / std, -5.0, 5.0)

    def reset(self) -> None:
        self.n = 0
        self.mean.fill(0.0)
        self.m2.fill(0.0)


class OnlineLogisticModel:
    """Logistic regression trained by constant-rate SGD with L2 and clipped
    weights. Deliberately the simplest model that can express "this
    combination of conditions is worse than usual": with a few hundred
    self-labeled samples and ~20 correlated features, anything with more
    capacity would fit noise, and there is no held-out set inside a live
    strategy to catch it doing so."""

    def __init__(self, n_features: int, config: LearnerConfig) -> None:
        self._config = config
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.n_updates = 0

    def predict_proba(self, x_std: np.ndarray) -> float:
        return float(_sigmoid(float(np.dot(self.weights, x_std)) + self.bias))

    def partial_fit(self, x_std: np.ndarray, label: int, weight: float) -> None:
        cfg = self._config
        p = self.predict_proba(x_std)
        grad = (p - label) * weight
        self.weights -= cfg.learning_rate * (grad * x_std + cfg.l2 * self.weights)
        self.bias -= cfg.learning_rate * grad
        np.clip(self.weights, -cfg.weight_clip, cfg.weight_clip, out=self.weights)
        self.n_updates += 1

    def reset(self) -> None:
        self.weights.fill(0.0)
        self.bias = 0.0
        self.n_updates = 0


@dataclass
class _BucketState:
    """Decayed win/loss counts plus bounded excursion reservoirs for one
    regime bucket."""

    count: float = 0.0
    wins: float = 0.0
    mfe_r: list[float] = field(default_factory=list)
    mae_atr: list[float] = field(default_factory=list)


class AdaptiveLearner:
    """The facade a strategy talks to.

    Lifecycle per `evaluate()` call, in this order:

        learner.advance(times_ns, highs, lows)   # grade what price resolved
        verdict = learner.score(features, bucket)  # ask about a new setup
        learner.observe(...)                       # if you take it, record it

    `advance()` first is what keeps the estimate current without ever letting
    an unresolved sample influence its own gate."""

    def __init__(self, n_features: int, config: LearnerConfig | None = None) -> None:
        self._config = config or LearnerConfig()
        self._n_features = n_features
        self._standardizer = _Standardizer(n_features)
        self._model = OnlineLogisticModel(n_features, self._config)
        self._buckets: dict[str, _BucketState] = {}
        self._pending: list[_PendingSample] = []
        self._global_count = 0.0
        self._global_wins = 0.0
        self._resolved_total = 0
        self._decay = 0.5 ** (1.0 / max(self._config.half_life_samples, 1.0))

    # ── properties ────────────────────────────────────────────────────

    @property
    def config(self) -> LearnerConfig:
        return self._config

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def resolved_count(self) -> int:
        return self._resolved_total

    @property
    def breakeven_p(self) -> float:
        return breakeven_probability(self._config.secure_r)

    # ── recording ─────────────────────────────────────────────────────

    def observe(
        self,
        *,
        features: np.ndarray,
        bucket: str,
        entry_ns: int,
        entry_price: float,
        direction: int,
        sl_dist: float,
        atr: float,
        deadline_ns: int,
    ) -> bool:
        """Record a fired setup as pending. Returns False (and records
        nothing) for inputs that could never be graded — a non-positive stop
        distance or ATR — rather than silently poisoning the table with a
        sample whose barriers are meaningless."""
        if sl_dist <= 0.0 or atr <= 0.0 or direction not in (1, -1):
            return False
        if features.shape != (self._n_features,):
            return False
        if not np.all(np.isfinite(features)):
            return False

        self._pending.append(
            _PendingSample(
                features=np.asarray(features, dtype=float).copy(),
                bucket=bucket,
                entry_ns=int(entry_ns),
                entry_price=float(entry_price),
                direction=int(direction),
                sl_dist=float(sl_dist),
                secure_dist=float(sl_dist) * self._config.secure_r,
                atr=float(atr),
                deadline_ns=int(deadline_ns),
                last_seen_ns=int(entry_ns),
            )
        )
        if len(self._pending) > self._config.max_pending:
            # Oldest-first eviction: a sample that has been pending longest is
            # the one whose regime is most stale, so it is the cheapest to lose.
            del self._pending[: len(self._pending) - self._config.max_pending]
        return True

    def advance(
        self, times_ns: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> list[ResolvedSample]:
        """Feed newly closed bars; grade and train on everything they resolve.

        Idempotent: each pending sample remembers the last bar timestamp it
        has consumed, so re-feeding an overlapping window (which the engine's
        sliding 200-bar context does on every single call) cannot double-count
        an excursion or resolve a sample twice."""
        if len(times_ns) == 0 or not self._pending:
            return []

        resolved: list[ResolvedSample] = []
        still_pending: list[_PendingSample] = []
        for sample in self._pending:
            outcome = self._advance_one(sample, times_ns, highs, lows)
            if outcome is None:
                still_pending.append(sample)
            else:
                resolved.append(outcome)
        self._pending = still_pending

        for outcome in resolved:
            self._learn(outcome)
        return resolved

    def _advance_one(
        self,
        sample: _PendingSample,
        times_ns: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> ResolvedSample | None:
        # Strictly-after: `side="right"` on the sample's last consumed
        # timestamp. On the first call that is the entry bar itself, so the
        # entry bar can never grade its own setup.
        start = int(np.searchsorted(times_ns, sample.last_seen_ns, side="right"))
        if start >= len(times_ns):
            return None

        bar_highs = highs[start:]
        bar_lows = lows[start:]
        bar_times = times_ns[start:]

        if sample.direction == 1:
            favourable = bar_highs - sample.entry_price
            adverse = sample.entry_price - bar_lows
        else:
            favourable = sample.entry_price - bar_lows
            adverse = bar_highs - sample.entry_price

        hit_secure = favourable >= sample.secure_dist
        hit_stop = adverse >= sample.sl_dist
        secure_idx = int(np.argmax(hit_secure)) if bool(hit_secure.any()) else -1
        stop_idx = int(np.argmax(hit_stop)) if bool(hit_stop.any()) else -1

        # Only excursions up to the resolving bar count — beyond it the
        # position no longer exists, so later extremes are not information
        # about this trade.
        cut = len(bar_times) - 1
        for idx in (secure_idx, stop_idx):
            if idx >= 0:
                cut = min(cut, idx)
        sample.mfe = max(sample.mfe, float(favourable[: cut + 1].max()))
        sample.mae = max(sample.mae, float(adverse[: cut + 1].max()))
        sample.last_seen_ns = int(bar_times[cut])

        if stop_idx >= 0 and (secure_idx < 0 or stop_idx <= secure_idx):
            # `<=` makes a same-bar tie resolve as a stop — see module docstring.
            return self._grade(sample, label=0, expired=False)
        if secure_idx >= 0:
            return self._grade(sample, label=1, expired=False)

        sample.last_seen_ns = int(bar_times[-1])
        if sample.last_seen_ns >= sample.deadline_ns:
            # Time barrier: neither level reached. Graded on which way the
            # excursion leaned, at reduced weight.
            label = 1 if sample.mfe > sample.mae else 0
            return self._grade(sample, label=label, expired=True)
        return None

    def _grade(self, sample: _PendingSample, *, label: int, expired: bool) -> ResolvedSample:
        cfg = self._config
        return ResolvedSample(
            features=sample.features,
            bucket=sample.bucket,
            label=label,
            weight=cfg.expiry_sample_weight if expired else 1.0,
            mfe_r=sample.mfe / sample.sl_dist,
            mae_atr=sample.mae / sample.atr,
            r_outcome=(cfg.secure_r if label == 1 else -1.0),
            expired=expired,
        )

    def _learn(self, outcome: ResolvedSample) -> None:
        cfg = self._config
        state = self._buckets.get(outcome.bucket)
        if state is None:
            state = _BucketState()
            self._buckets[outcome.bucket] = state

        state.count = state.count * self._decay + outcome.weight
        state.wins = state.wins * self._decay + outcome.weight * outcome.label
        self._global_count = self._global_count * self._decay + outcome.weight
        self._global_wins = self._global_wins * self._decay + outcome.weight * outcome.label

        state.mfe_r.append(outcome.mfe_r)
        if outcome.label == 1:
            # Winners only. A losing sample's adverse excursion is, by
            # definition, whatever its stop distance was — feeding those back
            # in would make the "how much noise must a stop clear" quantile a
            # measurement of the stop the strategy already chose, and the
            # buffer would ratchet outward on its own output every cycle.
            state.mae_atr.append(outcome.mae_atr)
        if len(state.mfe_r) > cfg.max_bucket_history:
            del state.mfe_r[: len(state.mfe_r) - cfg.max_bucket_history]
        if len(state.mae_atr) > cfg.max_bucket_history:
            del state.mae_atr[: len(state.mae_atr) - cfg.max_bucket_history]

        self._standardizer.observe(outcome.features)
        self._model.partial_fit(
            self._standardizer.transform(outcome.features), outcome.label, outcome.weight
        )
        self._resolved_total += 1

    # ── querying ──────────────────────────────────────────────────────

    def global_rate(self) -> float:
        """Base rate across all buckets, itself shrunk toward
        `global_prior_rate` so the very first resolved sample cannot define
        the prior every bucket is then shrunk toward."""
        cfg = self._config
        numerator = cfg.global_prior_strength * cfg.global_prior_rate + self._global_wins
        return float(numerator / (cfg.global_prior_strength + self._global_count))

    def score(self, features: np.ndarray, bucket: str, prior_logit: float = 0.0) -> LearnerVerdict:
        """Blend three sources of belief about one setup, in log-odds.

            logit(p) = logit(p_bucket | prior)          empirical, this bucket
                     + [logit(p_model) - logit(p_global)]   feature correction

        `prior_logit` is an offset the caller derives from evidence this
        instance has not personally seen — in practice, historical live
        trades summarised per pattern / session / volatility / hour. It sets
        where a *cold* bucket starts, which matters enormously: without it a
        brand-new instance treats a setup with a known-terrible history
        exactly like a known-excellent one, and has to lose real money to
        rediscover the difference. As the bucket accumulates its own
        observations they progressively outweigh the prior, so a prior that
        has gone stale is corrected by evidence rather than believed forever.

        The model enters as a *correction* rather than a competing estimate,
        so a model that has learned nothing (predicting the base rate)
        contributes exactly zero. The correction is capped at +/-1.5 log-odds
        so features can shade a decision but never overturn the empirical
        rate on their own."""
        cfg = self._config
        raw_global = self.global_rate()
        global_p = float(np.clip(_sigmoid(_logit(raw_global) + prior_logit), 0.01, 0.99))
        state = self._buckets.get(bucket)
        bucket_count = state.count if state is not None else 0.0
        bucket_wins = state.wins if state is not None else 0.0
        p_bucket = (cfg.bucket_prior_strength * global_p + bucket_wins) / (
            cfg.bucket_prior_strength + bucket_count
        )

        logit_p = _logit(p_bucket)
        model_ready = self._model.n_updates >= cfg.min_model_samples
        if model_ready and np.all(np.isfinite(features)):
            p_model = self._model.predict_proba(self._standardizer.transform(features))
            # Referenced against the *unconditional* rate, not the
            # prior-adjusted one: the model is trained on every bucket's
            # samples, so its own base rate is the unconditional one. Using
            # `global_p` here would apply `prior_logit` twice.
            correction = float(np.clip(_logit(p_model) - _logit(raw_global), -1.5, 1.5))
            logit_p += correction

        p = float(np.clip(_sigmoid(logit_p), 0.01, 0.999))
        return LearnerVerdict(
            p_secure=p,
            expectancy_r=p * cfg.secure_r - (1.0 - p),
            bucket_samples=float(bucket_count),
            model_samples=self._model.n_updates,
            ready=bucket_count >= cfg.min_bucket_samples,
        )

    def p_reach(self, bucket: str, r_level: float, prior_p: float | None = None) -> float:
        """P(favourable excursion reaches `r_level` x risk) for this bucket.

        The empirical survival function of the MFE reservoir, shrunk toward
        `prior_p` (a caller-supplied estimate from historical trades) by
        `bucket_prior_strength`. This is the quantity a take-profit decision
        actually depends on, and estimating it directly is what lets
        `best_target` optimise expected R instead of guessing an R multiple.
        """
        state = self._buckets.get(bucket)
        observed = np.asarray(state.mfe_r, dtype=float) if state is not None else np.empty(0)
        strength = self._config.bucket_prior_strength
        fallback = self.global_rate() if prior_p is None else prior_p
        if len(observed) == 0:
            return float(np.clip(fallback, 0.01, 0.99))
        hits = float(np.count_nonzero(observed >= r_level))
        p = (strength * fallback + hits) / (strength + len(observed))
        return float(np.clip(p, 0.01, 0.99))

    def best_target(
        self,
        bucket: str,
        *,
        grid: tuple[float, ...],
        prior_curve=None,
    ) -> tuple[float, float, float]:
        """Pick the take-profit that maximises expected R for this bucket.

        Returns `(r_target, expected_r, p_hit)`. Expected R assumes the trade
        either reaches the target or loses its full risk:

            E[R] = p(reach r) * r - (1 - p(reach r))

        which is the honest two-outcome model for a bracket order, and is
        pessimistic in exactly the right direction — it books no credit for
        partial exits or trailing, both of which can only improve on it.

        `grid` must already respect the broker's spread-adjusted `min_rr`
        floor: a target below it is silently rejected by `SpreadGate`, so
        offering one here would just produce a leg that never opens.
        `prior_curve(r)` supplies the historical P(reach r) each bucket
        estimate is shrunk toward while it is still thin."""
        best = (grid[0], -1e9, 0.0)
        for r_target in grid:
            prior_p = None if prior_curve is None else float(prior_curve(r_target))
            p = self.p_reach(bucket, r_target, prior_p)
            expected = p * r_target - (1.0 - p)
            if expected > best[1]:
                best = (float(r_target), float(expected), float(p))
        return best

    def adverse_excursion_atr(self, bucket: str, quantile: float) -> float | None:
        """A quantile of the adverse excursion (in ATR) that *winning* setups
        in this bucket survived — i.e. how much noise a stop has to sit
        behind to not be the reason the trade failed. `None` until the bucket
        has `min_quantile_samples` resolved trades."""
        return self._quantile_of(bucket, "mae_atr", quantile)

    def favourable_excursion_r(self, bucket: str, quantile: float) -> float | None:
        """A quantile of realised favourable excursion in R — what this
        bucket actually tends to give before it turns, which is the honest
        basis for a take-profit distance."""
        return self._quantile_of(bucket, "mfe_r", quantile)

    def _quantile_of(self, bucket: str, attribute: str, quantile: float) -> float | None:
        state = self._buckets.get(bucket)
        if state is None:
            return None
        values = state.mae_atr if attribute == "mae_atr" else state.mfe_r
        if len(values) < self._config.min_quantile_samples:
            return None
        return float(np.quantile(np.asarray(values, dtype=float), quantile))

    @staticmethod
    def blend(base: float, learned: float | None, *, lo_frac: float, hi_frac: float) -> float:
        """Clamp a learned value into a bounded band around the strategy's
        base parameter, and fall back to the base when there is no learned
        value at all.

        This is the guardrail that makes the whole thing safe to run with
        real money: whatever the learner concludes, the strategy stays
        recognisably itself. It can be nudged, not redefined."""
        if learned is None or not np.isfinite(learned):
            return base
        return float(np.clip(learned, base * lo_frac, base * hi_frac))

    # ── lifecycle ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop every learned thing.

        Called by a strategy when the candle stream jumps backwards or gaps
        far enough that this is plainly a different replay — reusing one
        strategy instance across two backtests otherwise leaks the first
        run's learning into the second and makes results irreproducible."""
        self._standardizer.reset()
        self._model.reset()
        self._buckets.clear()
        self._pending.clear()
        self._global_count = 0.0
        self._global_wins = 0.0
        self._resolved_total = 0

    def snapshot(self) -> dict[str, float | int]:
        """Flat diagnostic summary for an INFO log line — CLAUDE.md requires
        money-touching decisions to be explainable, and "the learner declined
        this trade" is one."""
        return {
            "resolved": self._resolved_total,
            "pending": len(self._pending),
            "buckets": len(self._buckets),
            "model_updates": self._model.n_updates,
            "global_rate": round(self.global_rate(), 4),
            "breakeven_p": round(self.breakeven_p, 4),
        }
