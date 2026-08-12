# SMC_STRATEGY_PLAN.md

Smart Money Concepts — research, codebase audit, and a ranked build spec for
the XAUUSD Apex family.

**Status:** specification only. Phase 1 wrote this file and touched nothing
else. Later phases implement.

**Scope of authority:** this document may propose changes only inside
`backend/src/strategies/` and `backend/src/skills/`. It never proposes a
change to `configs/risk.yaml` or to anything under `backend/src/engine/`.

---

## 0. The constraint that governs every recommendation

### 0.1 The measured baseline

`xauusd_snd_apex_trendguard_m1`, XAUUSD, 2026-06 → 2026-08:

| metric | value |
|---|---|
| trades | 1312 |
| profit factor | 6.54 |
| avg R | 0.193 |
| sum R | ~253R |
| max drawdown | 1.03% |
| equity | 10,000 → 17,579 |

### 0.2 That profit factor is an exit artifact, not entry edge

The R distribution decomposes as:

| bucket | trades | share | R each | R total |
|---|---|---|---|---|
| scratch (0 – 0.25R) | ~1238 | 94.4% | +0.200 | +248R |
| reached target | ~32 | 2.4% | +1.50 | +48R |
| full stop | 42 | 3.2% | −1.024 | −43R |
| **total** | **1312** | | **+0.193** | **+253R** |

The scratch bucket is manufactured by the engine, not by the strategy:
`PositionManager` rule 2 (`backend/src/engine/application/position_manager.py:75`,
`DEFAULT_SECURE_BUFFER_R_MULT = 0.2`) ratchets the stop to entry + 0.2R as
soon as a fresh continuation base clears on M5. So **the targeted part of the
strategy — the part an entry rule can actually influence — nets about +5R
over two months** (+48R of targets against −43R of stops).

### 0.3 The arithmetic every proposal below must clear

Let a filter remove `k` trades, of which `s` were destined for a full stop
and `t` were destined to reach target. Using the bucket values above:

```
ΔsumR = +1.024·s  −  1.500·t  −  0.2003·(k − s − t)
      =  1.2243·s  −  1.2997·t  −  0.2003·k
```

Sanity check: at the base composition (s/k = 3.2%, t/k = 2.4%) this gives
ΔsumR/k = −0.193, exactly the average trade. Correct — removing random trades
costs you the average trade.

Solving for break-even with `t/k` held at the base rate:

> **A pure entry filter must be ~6× enriched in stop-outs to be sumR-neutral.**
> It needs a **19.0% stop rate among the trades it removes**, against a
> **3.2% base rate**. If it removes zero target-reachers the bar drops to
> 16.4% (5.1× enrichment). Anything less and it *costs* money even while it
> "improves quality".

This is the single most important number in this document. Almost every SMC
filter you can imagine removes trades at close to the base rate and therefore
**loses ~0.19R per trade removed**.

### 0.4 Where entry-side work can actually pay

Three levers, in descending order of headroom:

1. **Convert scratch → target.** Worth **+1.30R per converted trade**.
   Converting just 20 trades out of 1238 is +26R — five times what the entire
   targeted book nets today. Mechanisms available to entry-side code:
   a tighter, better-anchored **stop** (bigger R for the same price target)
   and a better **target map** (`tp_points` is chosen by the strategy, not the
   engine — see `xauusd_snd_apex_trendguard_m1_v5.py:1385-1416`).
2. **Avoid a full stop.** Worth **+1.224R per avoided stop** — but only if the
   filter is ≥6× enriched, per §0.3.
3. **Tail / clustering.** Max DD is already 1.03%, so there is little headroom
   there; the honest tail metric on this system is
   `BacktestReport.worst_losing_streak` (`backend/src/backtest/domain/models.py`)
   and the calendar clustering of the 42 stops.

**Every hypothesis in Part C is stated in these terms.** None of them promise
a profit-factor or win-rate improvement. Win rate is pinned near 96% by the
trailing artifact and is forbidden as a success metric
(`backend/scripts/fatigue_ab.py:16-21` already says so in code).

### 0.5 Hard engineering constraints

| constraint | source | consequence for SMC |
|---|---|---|
| 200 bars per timeframe, silently | `engine/application/trade_loop.py:65` `DEFAULT_CONTEXT_BARS = 200` | M5 = 16.7h, M15 = 50h, H1 = 8.3 days. Prior-day and prior-week highs/lows **are** reachable from the already-declared H1 frame. A 4-hour Asian range needs 240 M1 bars — over cap on M1, fine on M5 (48 bars). |
| sandbox imports | `strategies/sandbox.py:38-49` | `math`, `statistics`, `numpy`, `pandas`, `domain.models`, `domain.fatigue`. Nothing else, ever. |
| no `getattr`, no `slice()` | `strategies/sandbox.py:_safe_builtins` (neither name is in the list) | Literal slices `arr[a:b]` are fine (AST `Slice` node); the `slice()` *function* is not. Dict `.get()` is fine. |
| no dunder attribute access | `strategies/sandbox.py:_static_scan` | rules out most metaprogramming |
| 2.0s smoke-test timeout | `strategies/sandbox.py:_SMOKE_TIMEOUT_SECONDS` | new detectors must be vectorised or cached per zone-TF bar, like `_zones_for` at `:1030-1057` |
| MT5 comment = 29 chars | `engine/application/trade_loop.py:776-777` — `f"TP{idx+1}:{signal.reason}"[:29]` | only ~25 chars of `reason` reach MT5. New SMC tags belong **after** the existing `APEX/{source}/{pattern}@{tf}` head, so the signal-trail parser and dashboards keep working; they are read from the journal `reason`, not the comment. |
| broker `min_rr` | `SpreadGate`, honoured at `:1408-1416` via `broker_min_rr: 1.60` | any new target map must still clear `tp >= 1.5×(sl + spread)` or the leg is silently refused |

---

# PART A — Research synthesis

Each concept below is given as: **what it claims → exact programmatic rule →
parameters → how a backtest could falsify it**. Concepts are graded for
evidence at the end of the section.

## A.1 Market structure — BOS vs CHoCH

**Claim.** Trend continuation is marked by a *Break of Structure* (BOS: price
takes out the prior swing high in an uptrend); trend change is marked by a
*Change of Character* (CHoCH: price takes out the most recent higher low while
in an uptrend). ([dailypriceaction][1], [quantum-algo][2])

**Programmatic rule** (non-repainting version):

```
swings = fractal pivots with symmetric wing W          # pivot at i confirmed at i+W
last_confirmed_index = n - 1 - W                       # never label the last W bars
state: trend ∈ {+1, -1}, last_swing_high, last_swing_low

for each bar j > last confirmed pivot:
    if trend == +1 and close[j] > last_swing_high:  emit BOS(+1,  level=last_swing_high, j)
    if trend == +1 and close[j] < last_swing_low:   emit CHoCH(-1, level=last_swing_low,  j); trend = -1
    (mirror for trend == -1)
```

`close_break=True` (body close beyond the level) is the standard; the
alternative (`high/low` pierce) is what the `smartmoneyconcepts` Python
package exposes as an option. ([joshyattridge/smart-money-concepts][3])
Nearly every practitioner source agrees that a **body close** is what
separates "structure broken" from "level swept" — this one rule resolves most
BOS-vs-sweep ambiguity. ([chartinglens][4])

**Internal vs swing structure.** Two pivot scales run simultaneously: a short
wing (internal, e.g. 5) for entry-grade shifts and a long wing (swing, e.g.
50) for the bias. ([LuxAlgo SMC][5]) The `smartmoneyconcepts` package defaults
`swing_length=50`. ([3])

**Parameters.** `swing_wing_internal` (3–8), `swing_wing_external` (10–30 on
M5, capped by 200-bar budget), `close_break` (True/False).

**Falsifiable test.** Replace the current majority-vote structure bias
(`:539-553`) with a BOS/CHoCH event machine and A/B on identical bars. The
event version should show *fewer* trades taken against a freshly-flipped
structure and a lower stop-bucket count; if the stop-bucket count does not
fall by ≥6× enrichment, the event machinery is decoration and the vote stays.

**Honest read.** BOS/CHoCH is renamed Dow-theory trend analysis. Trend has a
weak, decaying signal in the literature (Brock/Lakonishok/LeBaron 1992;
Sullivan/Timmermann/White 1999; Park & Irwin 2007) and breakout edges fade
out-of-sample and after costs. ([IndicatorEdge][6]) The value here is
*mechanical*, not predictive: an event with a level and a timestamp is a
better gate input than a rolling label count.

## A.2 Liquidity pools, EQH/EQL, and the sweep

**Claim.** Stops rest above swing highs and below swing lows, especially where
two or more extremes are equal ("equal highs"/"equal lows"). Price is drawn to
those pools, takes them, and reverses.

**This is the one SMC family with real academic backing.** Carol Osler's work
on a major dealer's actual FX order book documents that take-profit orders
cluster *at* round numbers and stop-loss orders cluster *just beyond* them;
that reversal frequency at round numbers is 59.3% vs 54.8% at arbitrary
numbers (dollar–mark, highly significant); and that stop-loss clusters
propagate trends as cascades. ([Osler, NY Fed SR125][7]; [Osler,
*Stop-Loss Orders and Price Cascades in Currency Markets*][8]; Osler 2003
*J. Finance* 58(5) 1791-1819 [9]) The tendency to reverse at round numbers
remains significant for **under thirty minutes** — which is exactly the horizon
an M1 zone bot operates on.

What the literature does **not** support is the narrative layer: that
institutions deliberately hunt retail stops. Cascades are emergent from thin
books and clustered conditional orders, not coordinated. ([6])

**Programmatic rule — EQH/EQL cluster:**

```
pivots = confirmed swing highs (wing W)
for each consecutive pair (p1, p2) of swing highs with p2.index - p1.index <= eq_max_gap:
    if abs(p2.price - p1.price) <= eq_tol_atr * ATR:
        emit LiquidityLevel(price = max(p1.price, p2.price), kind = "buy-side",
                            members = 2, created = p2.index)
merge further pivots into the cluster while they stay within eq_tol_atr * ATR
```

ATR-scaled tolerance is the standard implementation (percentage tolerance is
the cheaper variant). ([LuxAlgo EQH/EQL][10])

**Programmatic rule — sweep:**

```
sweep_of(level L, direction "below" for sell-side):
    exists bar i in the last sweep_lookback bars with
        low[i]  <  L - sweep_min_pierce_atr * ATR        # the run
        close[i] >  L                                     # the failure: closed back inside
    and no close[j] < L for any j in (i, now]             # it has not since been accepted
    and now - i <= sweep_max_age_bars
```

Two-part verification — the run through the level, then the *failure to hold*
— is the canonical definition; a close that holds beyond with continuation is
a breakout, not a sweep. ([LuxAlgo liquidity sweep][11], [chartinglens][4])
Practitioners split on the window: some demand failure within 1–2 candles,
others accept any tag-and-reject resolving within a few bars. That
disagreement is exactly what `sweep_max_age_bars` should sweep in an A/B.

**Parameters.** `eq_tol_atr` (0.05–0.30), `eq_max_gap` (5–40 bars),
`sweep_lookback` (5–40 entry bars), `sweep_min_pierce_atr` (0.0–0.5),
`sweep_max_age_bars` (2–20), `sweep_level_source` ∈ {swing, EQH/EQL, PDH/PDL,
session extreme}.

**Falsifiable test.** Gate demand-zone longs on a prior sell-side sweep. The
hypothesis is *specifically* about the stop bucket: entries taken into a zone
whose underlying liquidity has **not** yet been taken should be the ones that
get run over. Measure the stop rate among trades the gate removes. Kill if
< 19%.

## A.3 Order blocks, breakers, mitigation blocks

**Claim.** The last opposing candle before a displacement leg marks where the
orders that caused the leg were placed; price returns there to fill the rest.

**Programmatic rule — order block:**

```
find displacement leg ending at bar d  (see A.5)
ob_index = the last bar b < d.start with sign(close[b] - open[b]) opposite to the leg
ob_top    = high[ob_index]      # or max(open, close) for a "body OB"
ob_bottom = low[ob_index]       # or min(open, close)
valid only if the leg from ob_index breaks structure or sweeps liquidity
```

Displacement is what separates a meaningful OB from an ordinary candle price
merely traded through. ([quantum-algo OB guide][12], [ictkillzone][13])

**Refinement.** Body-only vs full-range, and the 50% equilibrium of the OB as
the entry line, are the two standard refinements. ([13])

**Mitigation / invalidation.** Only a **body close** beyond the zone boundary
mitigates it; price may wick 70% into an OB and it remains valid. ([12])

**Breaker vs mitigation block.** ICT's own distinction: a **breaker** is a
failed OB *with* a liquidity sweep — price traded through the zone and the
retest arrives from the other side, so the zone flips polarity. A **mitigation
block** is a failed OB *without* a sweep. ([tradingstrategyguides Day 12][14],
[theicttrader][15], [altrady][16])

**Parameters.** `ob_anchor` ∈ {base, last_opposing, ob_body},
`ob_refine_to_equilibrium` (bool), `ob_requires_structure_break` (bool),
`breaker_enabled` (bool), `breaker_requires_sweep` (bool).

**Falsifiable test.** The OB rectangle is typically **narrower** than a
leg-base-leg base band. Narrower zone → tighter stop → larger R for the same
price target. That is a scratch→target conversion mechanism (§0.4 lever 1) but
it cuts both ways: tighter stops also produce *more* stops. The A/B must
report both `mean(sl_points)` and the stop-bucket count, and is only a win if
`ΔsumR > 0` under §0.3.

**Honest read.** "Order block" candles have no validation anywhere in the
literature; support/resistance in general does have genuine predictive power.
([6]) Treat the OB as a *tighter parameterisation of a demand zone*, which is
a real engineering benefit, not as evidence of institutional footprints.

## A.4 Imbalance — FVG, BISI/SIBI, inversion FVG

**Programmatic rule — FVG (three-candle):**

```
bullish FVG at i  ⟺  low[i+1] > high[i-1]        # BISI, "buyside imbalance sellside inefficiency"
bearish FVG at i  ⟺  high[i+1] < low[i-1]        # SIBI
top    = low[i+1]  / bottom = high[i-1]          (bullish)
size   = top - bottom
mitigated when price trades back into [bottom, top]; fully filled at full traverse
```

This is the definition used by the `smartmoneyconcepts` package
(`smc.fvg(ohlc, join_consecutive=False)`, returns FVG/Top/Bottom/
MitigatedIndex). ([3])

**Inversion FVG (IFVG).** A bullish FVG that price *closes through* to the
downside becomes a bearish zone on the retest, and vice versa. ([LuxAlgo
IFVG][17], [FXOpen][18])

```
IFVG(bearish) = bullish FVG F with any close[j] < F.bottom, j > F.created
                → new supply zone [F.bottom, F.top], created at j
```

**Parameters.** `fvg_min_size_atr` (0.10–0.60), `fvg_max_age_bars`,
`fvg_join_consecutive` (bool), `ifvg_enabled` (bool).

**Falsifiable test.** As a *score term* in `_score_zone` (`:786-802`), not a
standalone entry: does `score += w_fvg` on zones overlapping an unmitigated
FVG re-rank candidates toward better outcomes? Measure sumR at fixed trade
count by capping `max_signals_per_bar`.

**Honest read — this is where SMC is weakest.**
- Universal gap fill is explicitly called a **myth** by the gap-anomaly
  literature; Caporale & Plastun and Plastun/Sibande/Gupta/Wohar find price
  tends to move *in the direction of* the gap on gap days, and that behaviour
  is mostly not inconsistent with market efficiency — with FX the partial
  exception. ([Caporale & Plastun, *Price Gaps: Another Market Anomaly?*][19];
  [Plastun et al., *Price Gap Anomaly in the US Stock Market*][20])
- The three-candle visual FVG has **no peer-reviewed literature at all**. What
  is documented is that *signed order-flow imbalance* moves price (Chordia,
  Roll & Subrahmanyam 2002) — a different object entirely. ([6])
- **Cost reality check for this system:** on XAUUSD the broker feed sat at a
  flat 15-point spread (`:170-176` of the flagship docstring). A three-candle
  M1 FVG on gold is frequently smaller than the spread. An M1 FVG entry model
  is structurally uneconomic here; only M5/M15 FVGs are worth measuring.

## A.5 Displacement

**Claim.** A genuine institutional leg is fast, one-directional, large-bodied
and leaves imbalance; noise is not.

**Programmatic rule** — the two standard quantifications, used together:

```
is_displacement(bar i):
    body  = abs(close[i] - open[i])
    range = high[i] - low[i]
    return body >= disp_atr_mult * ATR[i]  and  body / range >= disp_body_ratio

is_displacement_leg(bars a..b):
    net = abs(close[b] - open[a])
    return net >= leg_atr_mult * ATR[b]
       and sum(body of same-sign bars in a..b) / sum(range in a..b) >= disp_body_ratio
       and (b - a + 1) <= disp_max_bars
```

Common defaults in the wild: body > ATR(14) × 1.5, body/range > 0.60.
([FibAlgo ICT Displacement][21], [ictkillzone displacement][22])

**Parameters.** `disp_atr_mult` (1.0–2.5), `disp_body_ratio` (0.45–0.75),
`disp_max_bars` (1–4), `leg_atr_mult` (0.6–1.5).

**Falsifiable test.** The flagship already has `min_impulse_atr` (`:931`)
gating a *net travel* measure (`:343`, `:357`). Displacement adds the
**body/range shape** test, which `impulse_atr` cannot see: a leg made of
three big-wicked indecisive candles scores the same net travel as one clean
marubozu. A/B `disp_body_ratio ∈ {off, 0.45, 0.55, 0.65}` and measure the stop
bucket. This is a cheap, high-information experiment.

## A.6 Premium / discount, equilibrium, OTE

**Programmatic rule:**

```
dealing_range = (last confirmed swing low SL, last confirmed swing high SH)
                on the chosen timeframe, alternating and both confirmed
EQ  = (SH + SL) / 2
discount = price < EQ ;  premium = price > EQ
OTE (bullish) = [SL + 0.21*(SH-SL), SL + 0.38*(SH-SL)]   # i.e. 0.62–0.79 retrace from SH
```

The 62–79% band with 0.705 as the "sweet spot" is the standard ICT
formulation; the range recalculates whenever a new extreme is made.
([GrandAlgo OTE][23], [FibAlgo Premium/Discount][24])

**Parameters.** `pd_timeframe` (M15 or H1 — both already declared at `:834`),
`pd_swing_wing`, `require_discount_for_longs` (bool), `ote_low`/`ote_high`
(0.62/0.79 default), `pd_min_range_atr` (a range narrower than this is not a
range).

**Falsifiable test.** Two arms, deliberately separated because the evidence
differs sharply between them:
- **Arm 1 (EQ half-split).** Longs only in the lower half of the dealing
  range. This is a re-expression of "don't buy after the move has already
  happened", which is what `max_extension_atr` (`:899`) tries to do with an
  EMA anchor instead of a range. Expect overlap; measure whether it removes
  *different* trades.
- **Arm 2 (0.62–0.79 OTE band).** Low prior confidence: in controlled tests
  Fibonacci zones performed **no better than random non-Fibonacci zones**
  (Tsinaslanidis et al. 2022; Gurrib et al. 2022). ([6]) Ship it only as a
  measured arm; do not ship it as a default.

## A.7 Time — killzones, Asian range, judas swing

**Claim.** London open (02:00–05:00 EST), NY open (07:00–10:00 EST) and
London close (10:00–12:00 EST) are privileged windows; the Asian range
(20:00–00:00 EST) is swept at the London open by a "judas swing" false move.
([FXNX killzones][25], [tradingfinder judas][26])

**Evidence.** Time-of-day volatility clustering in FX is genuine and
replicated across decades: Wood/McInish/Ord (1985), Harris (1986),
Andersen & Bollerslev (1997, 1998), Ito & Hashimoto (2006), Clifton & Plumb
(RBA 2007). ([6]) For gold specifically, the London–NY overlap (13:00–17:00
UTC) is the highest-volatility window and 13:30 UTC US data is the sharpest
mover. ([TMGM][27], [NordFX][28])

**But:** volatility clustering is not directional edge. No study shows a
privileged window that beats trading costs, and the classic pattern is fitted
to 1985–2007 data. ([6])

**Programmatic rule** (if built at all):

```
session_active(bar) = utc_hour_of(bar.time) in window
asian_range = (min low, max high) over the Asian window's bars on the zone TF
judas = first sweep of asian_range extreme within the London window,
        followed by a CHoCH on the entry TF within judas_max_bars
```

**Falsifiable test.** Slice the *existing* 1312-trade replay by UTC hour and
compute stop-bucket rate and sumR per hour **before writing any code**. If no
hour band shows ≥6× stop enrichment, a killzone gate cannot pay for itself
and the whole family is dead on arrival. This costs one query and zero
implementation risk — see the "do NOT build" list, §C.7.

## A.8 Multi-timeframe: HTF POI → LTF confirmation

**Claim.** Read the bias and the liquidity pool on H1/M15; execute the sweep,
the structure shift and the FVG entry on M1/M5.

**Canonical sequences.**
- **ICT 2022 / Unicorn model:** sweep → market structure shift →
  displacement → the FVG *inside the breaker block* is the entry zone.
  ([LuxAlgo Model 2022][29], [quantvps Unicorn][30], [innercircletrader
  Unicorn][31])
- The "Unicorn" specifically requires a breaker block and an FVG to **overlap**
  — i.e. it is a confluence-scoring rule, not a new detector.

**Programmatic rule:**

```
htf_bias   = BOS/CHoCH state on H1
htf_pool   = nearest unswept EQH/EQL or PDH/PDL on H1 in bias direction
ltf_trigger= sweep(htf_pool) on M5  →  CHoCH on M1  →  entry at the resulting
             OB/FVG overlap, with SL beyond the sweep extreme
```

**Falsifiable test.** The flagship already implements the *architecture* of
this (§B). What is new is that the POI is a **liquidity level**, not a
supply/demand rectangle. A/B: add liquidity levels to `_overlap_count`
(`:805-817`) and to `score_overlap_weight` (`:872`) and see if candidate
re-ranking alone moves sumR at fixed trade count.

## A.9 Inducement

**Claim.** A deliberate-looking minor swing sits *in front of* the real POI to
bait early entries; price takes it before reacting from the genuine zone. The
rule of thumb is "never take the first pullback after a break of structure —
that is inducement". ([writofinance][32], [innercircletrader IDM][33])

**Why it resists programmatic definition.** Which of two candidate zones is
"the inducement" and which is "the POI" is only knowable after price has
picked one. Every published rule reduces to either (a) "the first pullback
after a BOS" — which is just an ordinal counter — or (b) "the weaker zone",
which restates the scoring problem `_score_zone` (`:786-802`) already solves
explicitly. Practitioner sources concede that inducement can itself *be* an
order block or an S&D zone and that telling them apart is "the challenge".
([32])

**The one testable residue:** an ordinal counter — skip the first retest of a
zone after a structure break, trade the second. Note the flagship's episode
counter (`:599-604`, `max_touch_episode` `:939`) already counts retests, but
from the *opposite* end: it caps them at 8 rather than requiring at least 2.
A `min_touch_episode` param is a one-line falsifiable version of inducement.

## A.10 Evidence grading — the honest summary

| concept | grade | why |
|---|---|---|
| stop clustering / liquidity pools at obvious levels | **supported** | Osler (2000/2003/2005), Harris (1991), Bhattacharya et al. (2012); 59.3% vs 54.8% reversal frequency at round vs arbitrary numbers; effect significant for < 30 min ([7][8][9][6]) |
| session/time-of-day volatility clustering | **supported (volatility only)** | Wood/McInish/Ord (1985), Andersen & Bollerslev (1997/98), Ito & Hashimoto (2006) — clustering yes; directional edge net of costs, no ([6]) |
| support/resistance having predictive power | **weakly supported** | Osler (2000/2003); Brock et al. (1992) — weak and decaying ([6]) |
| market structure / trend | **weakly supported** | Brock et al. (1992), Sullivan/Timmermann/White (1999), Park & Irwin (2007) — fades out-of-sample and after costs ([6]) |
| order blocks as a distinct object | **unsupported** | no validation of "order block" candles anywhere; it is a tighter S&D parameterisation ([6]) |
| FVG / three-candle imbalance | **unsupported** | no peer review; gap-fill is a documented myth (Caporale & Plastun; Plastun et al.) ([19][20][6]) |
| Fibonacci OTE ratios | **falsified in controlled tests** | Fibonacci zones no better than random non-Fibonacci zones (Tsinaslanidis et al. 2022; Gurrib et al. 2022) ([6]) |
| accumulation → manipulation → distribution narrative | **folklore** | Wyckoff (1910/1931) is descriptive text; no rigorous validation of the intraday model ([6]) |
| deliberate institutional stop-hunting | **folklore** | cascades are emergent from thin books, not coordinated ([6]) |

**The most important single data point in this entire research pass:** an
independent mechanical test of the canonical SMC setup (sweep → structure
shift → gap) on **2.5M EURUSD 1-minute bars, 2019–2025** found a best win rate
of 56.3% and **0 of 54 rule variations profitable after 0.5 pips of cost**.
([6]) The underlying phenomena are real; the mechanical rules built on them
were not an edge on that instrument at that cost level.

Meanwhile the SMC backtest results circulating with headline numbers
(85.4% WR, 64.2% WR / PF 2.47 on XAUUSD, "2,600 trades, PF 2.17") are
marketing artefacts: when fetched, the underlying articles contain **no
per-concept statistics, no sample sizes, and no negative results at all**.
Treat them as zero evidence.

**Consequence for this project.** Approach SMC as a source of *better
parameterisations of things the bot already does* — narrower zones, better
target maps, event-based structure — and not as a new alpha. That framing is
consistent with §0: entry-side changes have little leverage on headline P&L
here anyway.

---

# PART B — Codebase coverage matrix

All citations are to
`backend/src/strategies/generated/xauusd_snd_apex_trendguard_m1_v5.py`
unless prefixed. Read-only audit; nothing was modified.

## B.1 What already exists

| SMC concept | status | where | notes |
|---|---|---|---|
| Fractal swing detection | **implemented** | `_detect_swing_points` `:440-451` | symmetric wing, strict-unique extreme (`np.sum(window == h) == 1`), iterates `range(lookback, n-lookback)` so it **never labels the last `wing` bars** — non-repainting by construction. A second, alternating-zigzag implementation lives at `strategies/domain/fatigue.py:343-379` (`_pivots`), vectorised via `sliding_window_view`. |
| HH/HL/LH/LL labelling | **implemented** | `_detect_structure` `:521-536` | |
| Structure as a hard gate | **implemented** | `_structure_bias` `:539-553`, Gate 3 `:1294-1297`, param `require_structure_alignment` `:891` | but it is a **majority vote over the last 4 labels** (`structure_depth` `:862`), not an event |
| BOS as an event (level + index) | **absent** | — | nothing emits a break with a level and a timestamp |
| CHoCH / MSS | **absent** | — | |
| Internal vs swing structure (two scales) | **absent** | single `structure_swing_lookback: 3` `:861` | |
| S&D zone: leg-base-leg | **implemented** | `_detect_zones_v1` `:297-360` (RBR/DBD/RBD/DBR) | supported by `_classify_bars` `:256`, `_build_runs` `:261`, `_make_is_leg` `:272`, `_merge_weak_runs` `:279` |
| Order-block-shaped zone | **partial** | `_detect_zones_v2` `:367-433` | engulf → base → departure. Anchors to the *engulfing* candle, **not** to the last opposing candle before displacement. Rectangle spans engulf extreme + base band (`:414-415`), so it is systematically **wider** than a true OB. Demoted by `source_weights` `:869` (0.5 vs SND_V1's 2.0) after losing $796 at 4.5% WR. |
| Displacement (magnitude) | **partial** | `impulse_atr` computed `:343`/`:357` and `:416`/`:424`; gated by `min_impulse_atr` `:931`; scored by `score_impulse_weight` `:870` at `:797` | measures **net travel of the whole leg** in ATR. No body/range shape test, no consecutive-body test, no per-candle displacement. |
| Zone invalidation (mitigation) | **implemented, stronger than SMC** | `_zone_lifecycle` `:560-615` | far-side **body close** kills the zone `:590-593` (exactly SMC's mitigation rule); plus an "eaten" rule `:610-612` (`max_bars_inside: 20` `:944`) that SMC has no equivalent of |
| Retest counting | **implemented** | rising-edge episode counter `:599-604`, capped by `max_touch_episode: 8` `:939`, enforced `:1197-1199` | |
| Breaker block (polarity flip) | **absent** | dead zones are `continue`d at `:1191-1193` and never reused | |
| Mitigation block | **absent** | — | |
| FVG / BISI / SIBI | **absent** | — | no gap detection anywhere in the file |
| Inversion FVG | **absent** | — | |
| Liquidity pools / EQH / EQL | **absent** | — | no equal-extreme clustering |
| Liquidity sweep / stop hunt | **absent** | closest is `_confirmation` `:714-779` | `_confirmation` requires a rejection close relative to the **zone rectangle**, not relative to a prior swing level. It is a candle-quality test, not a sweep test. |
| Inducement | **absent** (partially inverted) | `max_touch_episode` `:939` caps retests at 8; there is no `min_touch_episode` | |
| Premium / discount / EQ | **absent** | — | `max_entry_depth: 0.75` `:930` is depth **into the zone** (`:1218-1225`), a different quantity. `max_extension_atr: 2.20` `:899` (Gate 3b `:1299-1303`) measures ATR distance from the M5 fast EMA — related in spirit, unrelated in construction. |
| OTE 0.62–0.79 | **absent** | — | |
| Killzones / session clock | **absent by design** | `sessions: []` in `skills/normal/xauusd/xauusd_apex_turbo_m1.yaml:38`; `_bars_since_shock` `:154-192` is a *volatility* proxy for the calendar; `min_atr_median_frac: 0.55` `:996` is the dead-tape filter | the docstring at `:169-179` explains the deliberate choice of a volatility detector over a clock |
| Asian range / judas swing | **absent** | — | |
| PDH / PDL / prior week | **absent** | — | **reachable today**: H1 is already declared (`:834`) and 200 H1 bars = 8.3 days |
| HTF POI → LTF entry (architecture) | **implemented** | zones on M5 (`zone_timeframe` `:845`) and M15 (`htf_zone_timeframe` `:846`), overlap confluence `_overlap_count` `:805-817` + `score_overlap_weight: 0.80` `:872`, entry confirmation on M1 `:714-779` | the POI is an S&D rectangle, never a liquidity level |
| MTF trend agreement | **implemented** | `_timeframe_trend` `:199-223`, `_trend_state` `:226-249`, `trend_weights {M5:1.0, M15:1.5, H1:2.0}` `:880`, `trend_min_score: 2.0` `:883`, Gate 2 `:1277-1292` | strictly stronger than the engine's own EMA20/50-on-M5 veto, which is why `htf_veto=False` `:841` |
| Regime adaptivity | **implemented, no SMC equivalent** | Zone Respect Index `_zone_respect_index` `:622-707`, Gate 1 `:1267-1275` | measures, per zone kind, whether zones are *currently* being respected. Nothing in SMC does this. |
| Exhaustion / fatigue | **implemented** | `strategies/domain/fatigue.py` (8 sub-measures), wired as Gate 3c `:1305-1352`, params `:905-907` — defaults are no-ops (`fatigue_max: 1.01` can never fire) | **do not duplicate this under SMC names** |
| Volatility adaptation | **implemented** | `:1134-1153`, params `:982-996`; `vol_extreme_block` `:991`, ATR-percentile SL/TP scaling | |
| Candidate ranking | **implemented** | `_score_zone` `:786-802`, sorted `:1241` | this is the natural insertion point for any SMC *confluence* signal |
| Zone-anchored stop with ATR cap | **implemented** | `:1362-1376`; `sl_max_atr_mult: 2.20` `:961` skips rather than widens | |
| Target front-run of opposing zone | **implemented** | `:1385-1416`; `tp_frontrun_r_mult: 0.60` `:979` deliberately clears `PositionManager` rule 3's 0.5R band | the target map is built **only** from opposing S&D zones (`:1388-1396`) |
| Broker `min_rr` respected | **implemented** | `broker_min_rr: 1.60` `:971`, enforced `:1408-1416`, scalp leg dropped rather than emitted unfillable `:1416` | |
| Reject-reason funnel | **implemented** | `self.reject_counts` `:1023`, `_reject` `:1027-1028` | every new gate must call `_reject` with a stable key |
| Per-zone-bar caching | **implemented** | `_zones_for` `:1030-1057`, `_regime` `:1059-1091` | any new detector **must** live inside these caches or it re-runs 100k times per replay |

## B.2 Superseded file

`xauusd_snd_qm_structure_m1_v4.py` (764 lines) is the predecessor. It has
`_detect_zones_v1/v2`, `_detect_quasimodo_zones`, `_detect_structure` in
near-identical form (`:157`, `:218`, `:322`, `:434`) but: it **resamples** M1
into M5 (`_resample` `:76-108`) instead of using native HTF candles; it has no
trend filter, no R:R filter, no zone scoring, no ZRI (its own docstring,
`:22-24`: "no higher-timeframe trend filter, no R:R filter — every valid
signal trades"); its structure is annotation-only (`:706-719`); and its zone
tracking is break-only with no episode counting (`_track_zone_on_entry_tf`
`:476-501`). **Nothing in it is worth porting.** It is useful only as the
control arm the flagship's docstring post-mortem was written against.

## B.3 The `xauusd_apex_turbo_m1.yaml` variant

`backend/src/skills/normal/xauusd/xauusd_apex_turbo_m1.yaml` is **pure
`param_overrides`** over the flagship strategy file (`:39-50`) at
`risk_multiplier: 0.6`. It loosens exactly the frequency/quality-tradeoff
gates (`confirm_mode: "off"`, `max_entry_depth: 1.00`, `max_touch_episode: 20`,
`allow_counter_trend: true`, `zri_min: 0.35`) and keeps every gate that stopped
the 2026-08-05 bleed.

**Architectural consequence for every proposal below:** the whole apex ladder
(M1/M5/M15/H1/D1, see `src/skills/normal/xauusd/`) runs one algorithm over
different `trend_weights` ladders. **Any new SMC feature must be a parameter
with a no-op default**, exactly like `fatigue_max: 1.01` (`:905`), so the
measured baseline is byte-identical until an A/B earns the change. Note also
that `scripts/fatigue_ab.py:38-50` hand-mirrors the turbo overrides — a new
param added to the turbo YAML must be mirrored there too.

## B.4 What SMC actually adds beyond what exists

Stripping out everything that already exists under another name, the genuine
deltas are:

1. **Liquidity levels as first-class objects** — swing extremes, EQH/EQL
   clusters, PDH/PDL. The bot currently has *zones* and no *levels*. This is
   the biggest structural gap and it feeds both entries and targets.
2. **Structure as an event with a level and an index** (BOS/CHoCH) rather than
   a rolling label count.
3. **A sweep predicate** — "has the liquidity under this zone already been
   taken?" — which is orthogonal to every existing gate.
4. **Candle-shape displacement** (body/range) on top of the existing net-travel
   `impulse_atr`.
5. **Polarity flip for dead zones** (breaker) instead of discarding them.
6. **Range-relative position** (premium/discount) as distinct from the two
   depth measures already present.

Everything else in SMC is already in the file under a different name, or is in
`fatigue.py`, or is on the "do NOT build" list.

---

# PART C — Build spec, ranked

Ranking is by **expected ΔsumR per unit of implementation cost**, under the
§0.3 arithmetic. Every item ships with a **no-op default**.

Shared conventions for all items:

- New helpers go in the flagship file, computed inside `_zones_for`
  (`:1030-1057`) or `_regime` (`:1059-1091`) so they are cached to the newest
  zone-TF bar. Nothing new may run per-M1-bar unvectorised.
- Every new rejection calls `self._reject("<stable_key>")` (`:1027`).
- Every new gate appends a short token to `head` (`:1460-1471`) **after** the
  existing `APEX/{source}/{pattern}@{tf}` prefix, ≤ 6 chars each, so the MT5
  29-char comment and the signal-trail parser are unaffected.
- Sandbox: literal slices only, no `getattr`, no `slice()`, no dunders.
- A/B via `backend/scripts/fatigue_ab.py` (add variants to `VARIANTS`), which
  already reports PF / avg R / drawdown and deliberately does not lead with
  win rate.

**Reporting requirement for every A/B below.** The harness must emit, per
variant: `n_trades`, `n_stop_bucket` (r_multiple ≤ −0.8), `n_target_bucket`
(r_multiple ≥ 0.8), `n_scratch`, `sumR`, `avg R`, `max_drawdown_pct`,
`worst_losing_streak`, `mean(sl_points)`. Plus, for each removed trade
relative to baseline, its baseline bucket — that is the only way to compute
the enrichment ratio in §0.3. All of this is derivable from
`BacktestReport.trades[].r_multiple`
(`backend/src/backtest/domain/models.py`).

---

## C.1 — Liquidity map + sweep gate + sweep-anchored stop  ★ highest conviction

**Why first.** It is the only SMC family with real microstructure evidence
(§A.2, Osler). It is orthogonal to every existing gate — nothing in the file
knows what a prior swing level is. And it attacks lever 2 (§0.4) directly with
a plausible mechanism: **a demand zone whose sell-side liquidity has not yet
been taken is precisely the zone that gets run over**, and those runs are the
42 full stops.

**Composes with:** it is a new Gate 3d, between Gate 3c (fatigue, `:1305`) and
Gate 4 (confirmation, `:1354`). It **replaces nothing**. The stop-anchor part
modifies the SL block at `:1362-1376`.

### Algorithm

```
# ---- built once per zone-TF bar, inside _regime() ----
def _liquidity_levels(zone_frame, htf_frame, atr, params):
    levels = []                       # each: {price, side, kind, index, members}
    for (frame, tag) in ((zone_frame, "z"), (htf_frame, "h")):
        highs, lows = frame.high.to_numpy(), frame.low.to_numpy()
        swings = _detect_swing_points(highs, lows, params["liq_swing_wing"])
        # 1. every confirmed swing extreme is a level
        for (i, price, kind) in swings:
            side = "buy" if kind == "high" else "sell"   # buy-side liquidity sits ABOVE
            levels.append({price, side, "swing", i, 1})
        # 2. EQH/EQL clusters: consecutive same-kind swings within tolerance
        for consecutive same-kind pairs (p, q) with q.i - p.i <= params["eq_max_gap"]:
            if abs(q.price - p.price) <= params["eq_tol_atr"] * atr:
                merge into a cluster; emit one level at the cluster extreme
                with kind="eq", members=cluster size
    # 3. prior-day extremes from the H1 frame (200 H1 bars = 8.3 days, in budget)
    if params["liq_use_pdh_pdl"]:
        group H1 bars by UTC calendar day; emit yesterday's high (buy-side)
        and low (sell-side) with kind="pd"
    # keep only levels not yet accepted: drop a buy-side level if any
    # zone-TF close since its creation is above it
    return unswept levels, sorted by price

# ---- evaluated per entry bar, cheap ----
def _swept(level, entry_lows, entry_highs, entry_closes, atr, params):
    """Has `level` been swept and rejected, recently, and not since accepted?"""
    w = params["sweep_lookback"]                       # entry-TF bars
    pierce = params["sweep_min_pierce_atr"] * atr
    n = len(entry_closes)
    lo = max(0, n - w)
    if level["side"] == "sell":                        # sell-side pool, below price
        ran  = entry_lows[lo:n]  < level["price"] - pierce
        back = entry_closes[lo:n] > level["price"]
        hit  = ran & back
    else:
        ran  = entry_highs[lo:n] > level["price"] + pierce
        back = entry_closes[lo:n] < level["price"]
        hit  = ran & back
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return None
    j = int(idx[-1]) + lo                              # most recent sweep
    if (n - 1 - j) > params["sweep_max_age_bars"]:
        return None
    # not since accepted
    if level["side"] == "sell" and np.any(entry_closes[j+1:n] < level["price"]):
        return None
    if level["side"] == "buy"  and np.any(entry_closes[j+1:n] > level["price"]):
        return None
    extreme = float(np.min(entry_lows[j:n])) if level["side"] == "sell" \
              else float(np.max(entry_highs[j:n]))
    return {"index": j, "extreme": extreme, "level": level}

# ---- Gate 3d, in _build_signals, after Gate 3c ----
mode = params["require_sweep"]                          # "off" | "soft" | "hard"
sweep = None
if mode != "off":
    want_side = "sell" if demand else "buy"
    # only levels within reach of the zone matter
    near = [L for L in liquidity if L["side"] == want_side
            and abs(L["price"] - zone_edge) <= params["sweep_zone_radius_atr"] * atr_value]
    for L in near:
        s = _swept(L, lows, highs, closes, atr_value, params)
        if s is not None and (sweep is None or s["index"] > sweep["index"]):
            sweep = s
    if sweep is None:
        if mode == "hard":
            self._reject("no_sweep"); return None
        # "soft": no veto, but the candidate loses score — see below
```

Scoring hook (soft mode), in `_score_zone` (`:786-802`):

```
score += params["score_sweep_weight"] * (1.0 if sweep else 0.0)
score += params["score_eq_weight"] * (sweep["level"]["members"] - 1 if sweep else 0)
```

Stop anchor, replacing lines `:1368-1371`:

```
if params["sl_anchor"] == "sweep" and sweep is not None:
    if demand:
        anchor = min(zone["price_low"], sweep["extreme"])
        sl_points = (close - anchor) + buffer
    else:
        anchor = max(zone["price_high"], sweep["extreme"])
        sl_points = (anchor - close) + buffer
# unchanged from here: ATR floor, sl_scale, sl_max_atr_mult cap at :1372-1376
```

Note this stop is **wider**, not tighter — that is deliberate and honest. The
existing `sl_max_atr_mult: 2.20` cap (`:961`) will drop setups whose sweep is
too far away, which is the correct behaviour (`:1374-1376` already skips
rather than widens).

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `require_sweep` | `"off"` | `"soft"` | off / soft / hard |
| `liq_swing_wing` | 3 | 3 | 2–6 |
| `eq_tol_atr` | 0.15 | 0.15 | 0.05–0.35 |
| `eq_max_gap` | 20 | 20 | 5–40 |
| `liq_use_pdh_pdl` | `False` | `True` | bool |
| `sweep_lookback` | 20 | 20 | 5–60 (entry-TF bars) |
| `sweep_min_pierce_atr` | 0.0 | 0.10 | 0.0–0.5 |
| `sweep_max_age_bars` | 10 | 10 | 2–30 |
| `sweep_zone_radius_atr` | 1.5 | 1.5 | 0.5–4.0 |
| `score_sweep_weight` | 0.0 | 0.8 | 0.0–2.0 |
| `score_eq_weight` | 0.0 | 0.3 | 0.0–1.0 |
| `sl_anchor` | `"zone"` | `"sweep"` | zone / sweep |

### Cost

~140 lines. Levels computed once per zone-TF bar (cached in `_regime`);
`_swept` is O(sweep_lookback) numpy per candidate per bar. Bar budget:
`liq_swing_wing`-confirmed pivots over 200 M5 bars — comfortably inside the
cap. PDH/PDL from 200 H1 bars — inside the cap, no new timeframe needed.

### Backtest hypothesis (falsifiable)

> **H1a (hard mode).** Gating on a prior sweep removes trades that are ≥6×
> enriched in the stop bucket. Concretely: trade count falls 40–70%; the
> **stop-bucket count falls from 42 to ≤ 15**; sumR does not fall by more
> than the scratch bucket lost; `worst_losing_streak` improves.
> **Kill if** the enrichment ratio among removed trades is < 5×, i.e. < 16.4%
> of removed trades were stops.
>
> **H1b (sweep-anchored stop).** Stops placed beyond the swept extreme convert
> full stops into scratches. Predicts: `mean(sl_points)` **rises** 10–30%;
> stop-bucket count falls; `sl_too_wide` rejections rise. Net ΔsumR > 0 only
> if `Δ(stop bucket) × 1.224R` exceeds `Δ(trades lost to sl_max_atr_mult) ×
> 0.193R`. Report both terms separately — this one can easily be a wash and
> must not be shipped on vibes.
>
> **H1c (soft mode).** Re-ranking alone (`score_sweep_weight` > 0, no veto)
> at fixed `max_signals_per_bar` raises the **target-bucket count** without
> changing the trade count. This is the cheapest arm and should run first.

---

## C.2 — Displacement shape test on the existing impulse  ★ best cost/benefit

**Why second.** Smallest diff in the whole plan (~25 lines) against a real,
named blind spot: `impulse_atr` (`:343`, `:357`) measures net travel and is
completely blind to candle shape. Three big-wicked indecisive bars that end up
1.5 ATR from where they started score identically to one clean marubozu. The
literature's own displacement definition adds exactly the missing dimension
(§A.5).

**Composes with:** extends `_detect_zones_v1` (`:297-360`) and
`_detect_zones_v2` (`:367-433`); tightens `min_impulse_atr` (`:931`) rather
than replacing it. It does **not** touch the fatigue module.

### Algorithm

```
# inside _detect_zones_v1, at the point impulse is computed (currently :343)
a, b = leg_out[1], leg_out[2]                     # leg-out span
bodies = np.abs(closes[a:b+1] - opens[a:b+1])
ranges = highs[a:b+1] - lows[a:b+1]
range_sum = float(np.sum(ranges))
body_ratio = float(np.sum(bodies)) / range_sum if range_sum > 0 else 0.0
# strongest single displacement candle in the leg
single = float(np.max(bodies)) / atr_at if atr_at > 0 else 0.0
zone["disp_body_ratio"] = body_ratio
zone["disp_single_atr"]  = single

# new quality gate in evaluate(), alongside the existing weak_impulse check :1214-1216
if zone.get("disp_body_ratio", 1.0) < float(params["disp_body_ratio_min"]):
    self._reject("weak_displacement"); continue
if zone.get("disp_single_atr", 9.9) < float(params["disp_single_atr_min"]):
    self._reject("no_disp_candle"); continue

# and a score term in _score_zone (:786-802)
score += float(params["score_disp_weight"]) * min(zone.get("disp_body_ratio", 0.0), 1.0)
```

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `disp_body_ratio_min` | 0.0 | 0.55 | 0.0 / 0.45 / 0.55 / 0.65 / 0.75 |
| `disp_single_atr_min` | 0.0 | 0.8 | 0.0–2.0 |
| `score_disp_weight` | 0.0 | 0.5 | 0.0–1.5 |

### Cost

~25 lines, all inside detectors that are already cached per zone-TF bar. Zero
new bar budget. Lowest-risk change in this document.

### Backtest hypothesis

> **H2.** Zones created by a shape-qualified displacement produce fewer full
> stops than zones created by an equal-travel but sloppy leg. Sweep
> `disp_body_ratio_min ∈ {0.0, 0.45, 0.55, 0.65, 0.75}` and plot
> `n_stop_bucket / n_trades` against it. A genuine effect is **monotone**:
> stop rate falling as the threshold rises. A flat or noisy curve means leg
> shape carries no information on XAUUSD M5 and the parameter ships at 0.0
> forever.
> **Success:** at some threshold, ΔsumR > 0 *and* the removed trades clear the
> 19% enrichment bar.
> **Note:** this test is cheap enough to run on all five apex ladder rungs;
> if the effect only appears on one timeframe it is noise.

---

## C.3 — Liquidity-aware target map + minimum-room veto

**Why third.** This is the only proposal that attacks **lever 1** (§0.4),
which has by far the most headroom (+1.30R per converted trade). The current
target ceiling is built exclusively from opposing S&D zones (`:1388-1396`) —
a rectangle map. Price does not travel to rectangles; it travels to resting
orders. Swing extremes, EQH/EQL clusters and PDH/PDL are a strictly richer
target map, and building it is free once C.1 exists.

**Composes with:** replaces the `ahead`/`ceiling` computation at `:1388-1397`
and adds a pre-trade veto next to the existing `min_rr` check at `:1408-1415`.

### Algorithm

```
# ---- replaces :1388-1397 ----
targets = []
for z in opposing_zones:                                   # unchanged source
    targets.append(z["price_low"] if demand else z["price_high"])
if params["tp_use_liquidity"]:
    for L in liquidity:                                     # from C.1
        if demand and L["side"] == "buy"  and L["price"] > close: targets.append(L["price"])
        if (not demand) and L["side"] == "sell" and L["price"] < close: targets.append(L["price"])

ahead = sorted([(t - close) if demand else (close - t) for t in targets if <ahead of price>])
ceiling = (ahead[0] - tp_buffer) if ahead else None
# tp1/tp2 clipping at :1399-1403 unchanged

# ---- new pre-trade veto, just before the min_rr check at :1408 ----
if params["min_room_r"] > 0.0 and ceiling is not None:
    if ceiling < float(params["min_room_r"]) * risk:
        self._reject("no_room_to_pool"); return None
```

Rationale for the veto: today, a trade whose nearest opposing structure is
0.9R away still fires (tp2 gets clipped, and only fails if it drops under
`tp2_min_rr`). Such a trade is *structurally incapable* of reaching target —
it will scratch at +0.2R or stop out. Those trades are pure dead weight in the
scratch bucket and carry full stop risk. This is the one veto in the plan
whose enrichment argument is a priori rather than empirical.

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `tp_use_liquidity` | `False` | `True` | bool |
| `min_room_r` | 0.0 | 2.2 | 0.0 / 1.8 / 2.2 / 2.6 / 3.0 |
| `tp_liq_kinds` | `("swing",)` | `("swing","eq","pd")` | subsets |

### Cost

~35 lines on top of C.1. **Depends on C.1** for the level map — do not build
C.3 standalone.

### Backtest hypothesis

> **H3a (richer target map).** Adding liquidity levels to the ceiling
> computation moves `tp_points` **down** on average (more candidate obstacles
> ⇒ a lower ceiling). Predicts: `n_target_bucket` **rises** (targets are
> nearer and get hit) while the average R of a target-reaching trade falls.
> Ship only if `Δ(n_target × mean R_target) > 0`. This is a genuine
> two-sided trade and could go either way — it must be measured, not assumed.
>
> **H3b (min-room veto).** Trades with < 2.2R of room to the nearest pool
> never reach target by construction, so removing them costs only their
> scratch value while removing their full stop risk. Predicts: among removed
> trades, target-bucket share ≈ 0 and stop share ≥ base rate. That makes the
> break-even bar 16.4% rather than 19.0%. **Kill if** removed trades contain
> more than a handful of target-bucket trades — that would mean the ceiling
> is mis-measuring room.
>
> Watch for interaction with `broker_min_rr: 1.60` (`:971`): `min_room_r`
> below ~1.7 is a no-op because the existing `min_rr` check already covers it.

---

## C.4 — BOS/CHoCH event machine replacing the structure vote

**Why fourth.** Real but modest expected effect. `_structure_bias`
(`:539-553`) counts labels over the last 4 swings; it has no notion of *when*
structure broke or *at what level*. Two things become possible with events
that are impossible with a vote:

1. a **freshness** gate — refuse continuation entries within N bars of a
   CHoCH against the trade;
2. a **principled counter-trend licence** — `allow_counter_trend` (`:910`,
   `False` on the flagship, `True` on turbo) currently fires on age alone
   (`counter_trend_max_age_bars: 12`, `:912`). "Only fade after a CHoCH in the
   fade's direction" is a far better condition and replaces a crude one.

**Composes with:** extends Gate 3 (`:1294-1297`) and refines Gate 2's
counter-trend branch (`:1280-1286`). `_structure_bias` stays as the fallback
when `structure_mode == "vote"`.

### Algorithm

```
def _structure_events(highs, lows, closes, wing, close_break):
    """Non-repainting BOS/CHoCH. Pivots at index i are confirmed at i+wing;
    _detect_swing_points already refuses to label the last `wing` bars."""
    swings = _detect_swing_points(highs, lows, wing)
    trend = 0
    last_high = None      # (index, price) — most recent CONFIRMED swing high
    last_low  = None
    events = []           # {index, kind: "BOS"|"CHOCH", dir: +1|-1, level}
    si = 0
    for j in range(len(closes)):
        # absorb every pivot whose confirmation bar (i + wing) has passed
        while si < len(swings) and swings[si][0] + wing <= j:
            i, price, kind = swings[si]
            if kind == "high": last_high = (i, price)
            else:              last_low  = (i, price)
            si += 1
        ref_up   = last_high[1] if last_high else None
        ref_down = last_low[1]  if last_low  else None
        probe_up   = closes[j] if close_break else highs[j]
        probe_down = closes[j] if close_break else lows[j]
        if ref_up is not None and probe_up > ref_up:
            events.append({j, "BOS" if trend >= 0 else "CHOCH", +1, ref_up})
            trend = +1; last_high = None
        elif ref_down is not None and probe_down < ref_down:
            events.append({j, "BOS" if trend <= 0 else "CHOCH", -1, ref_down})
            trend = -1; last_low = None
    return events, trend

# ---- Gate 3, replacing :1294-1297 when structure_mode == "event" ----
if params["structure_mode"] == "event":
    if structure_trend != 0 and structure_trend != want:
        self._reject("structure_bias"); return None
    last_choch = most recent event with kind == "CHOCH"
    if last_choch is not None:
        age = newest_bar - last_choch["index"]
        if last_choch["dir"] != want and age <= int(params["choch_cooldown_bars"]):
            self._reject("fresh_choch_against"); return None

# ---- Gate 2 counter-trend branch, refining :1280-1286 ----
if counter_trend and params["counter_trend_requires_choch"]:
    if last_choch is None or last_choch["dir"] != want \
       or (newest_bar - last_choch["index"]) > int(params["counter_trend_max_age_bars"]):
        self._reject("fade_no_choch"); return None
```

`_structure_events` runs on the zone frame inside `_regime` (`:1059-1091`), so
once per M5 bar. Bar budget: 200 M5 bars at `wing=3` — same as the existing
`_detect_structure` call at `:1077-1081`.

**Repainting note.** `_detect_swing_points` (`:440-451`) iterates
`range(lookback, n - lookback)` and therefore never labels the last `wing`
bars — the loop above must preserve that by only absorbing a pivot once
`i + wing <= j`. Any implementation that reads a pivot before its confirmation
bar is look-ahead and will produce a backtest that cannot be reproduced live.

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `structure_mode` | `"vote"` | `"event"` | vote / event |
| `structure_close_break` | `True` | `True` | bool |
| `choch_cooldown_bars` | 0 | 6 | 0–20 (zone-TF bars) |
| `counter_trend_requires_choch` | `False` | `True` | bool |
| `structure_wing_internal` | 3 | 3 | 2–6 |

### Cost

~70 lines. No new timeframe, no new bar budget.

### Backtest hypothesis

> **H4a.** Event-based structure is not worse than the vote at equal trade
> count. This is the *precondition* — run it with `choch_cooldown_bars: 0` and
> confirm sumR is within noise of baseline. If the event machine alone changes
> results materially, that is a bug (a look-ahead), not an edge.
>
> **H4b.** `choch_cooldown_bars > 0` removes continuation entries taken right
> after structure broke against them. Those are textbook "last one in before
> the reversal" trades and should be stop-enriched. Sweep {0, 3, 6, 10, 15};
> require ≥6× enrichment.
>
> **H4c (turbo only).** `counter_trend_requires_choch: True` on the turbo
> variant should cut its counter-trend trade count sharply while raising the
> counter-trend subset's sumR. Compare against the turbo baseline in
> `scripts/fatigue_ab.py:38-50`, not the flagship.

---

## C.5 — Premium/discount (equilibrium half-split)

**Why fifth.** Cheap and conceptually sound, but it likely overlaps heavily
with `max_extension_atr: 2.20` (`:899`, Gate 3b `:1299-1303`), which already
refuses entries far from the M5 fast EMA. The value is measuring whether
**range-relative** position removes *different* trades than **EMA-relative**
distance does.

**Composes with:** a new Gate 3e beside Gate 3b. Does **not** replace 3b.

### Algorithm

```
# inside _regime(), on the HTF frame (M15) so the range is meaningful
swings = _detect_swing_points(htf_highs, htf_lows, params["pd_swing_wing"])
take the last confirmed swing high H and last confirmed swing low L
if H is None or L is None:                  return None      # no opinion
rng = H.price - L.price
if rng < params["pd_min_range_atr"] * htf_atr:  return None  # not a range

eq = (H.price + L.price) / 2.0
pos = (close - L.price) / rng               # 0 at the low, 1 at the high

# ---- Gate 3e ----
if params["pd_mode"] != "off" and pd_reading is not None:
    if params["pd_mode"] == "eq":
        if demand and pos > params["pd_long_max"]:   self._reject("premium_long");  return None
        if (not demand) and pos < params["pd_short_min"]: self._reject("discount_short"); return None
    elif params["pd_mode"] == "ote":                 # measured arm only, see §A.6
        band = (1.0 - params["ote_high"], 1.0 - params["ote_low"])   # 0.21 .. 0.38
        if demand and not (band[0] <= pos <= band[1]):
            self._reject("not_ote"); return None
        if (not demand) and not (band[0] <= 1.0 - pos <= band[1]):
            self._reject("not_ote"); return None
```

Note `pd_reading is None` → **no veto**. Same principle as fatigue's
"unavailable is not exhausted" (`fatigue.py:57-66`): an unknown must never
veto a trade on its own.

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `pd_mode` | `"off"` | `"eq"` | off / eq / ote |
| `pd_timeframe` | `"M15"` | `"M15"` | M15 / H1 (both already declared `:834`) |
| `pd_swing_wing` | 4 | 4 | 3–8 |
| `pd_min_range_atr` | 3.0 | 3.0 | 1.5–6.0 |
| `pd_long_max` | 1.0 | 0.55 | 0.40–0.70 |
| `pd_short_min` | 0.0 | 0.45 | 0.30–0.60 |
| `ote_low` / `ote_high` | 0.62 / 0.79 | — | fixed; arm only |

### Cost

~45 lines.

### Backtest hypothesis

> **H5a.** `pd_mode: "eq"` removes trades ≥6× enriched in the stop bucket.
> **Report the overlap with Gate 3b explicitly**: of the trades `pd_mode` vetoes,
> what fraction would `max_extension_atr` have vetoed at a tighter setting? If
> the overlap is > 80%, this is a re-parameterisation of an existing gate and
> the correct action is to tune `max_extension_atr`, not add a parameter.
>
> **H5b (OTE arm).** Prior: **no effect**, per Tsinaslanidis et al. (2022) and
> Gurrib et al. (2022) — Fibonacci zones did not beat random non-Fibonacci
> zones in controlled tests. Run it precisely so we stop arguing about it, the
> same way `trend_age` ships at weight 0.0 in `fatigue.py:161-163`. Predicted
> outcome: trade count collapses, sumR falls roughly proportionally, no
> enrichment. **Ship at `"off"` unless it surprises us.**

---

## C.6 — Breaker blocks (polarity flip for dead zones)

**Why sixth.** Genuinely additive with near-zero new machinery, but the lowest
prior of the six. Today a zone that dies at `:1191-1193` is discarded forever.
A breaker gives that price area the *opposite* role instead of no role. The
flagship's own post-mortem (`:32-38`) shows the failure mode this addresses:
"the same dead zone was traded over and over" — the fix was to kill the zone;
the breaker asks whether the killed zone still means something from the other
side.

**Composes with:** `_zone_lifecycle` (`:560-615`) already computes everything
needed. The far-side-close branch (`:590-593`) currently returns
`dead=True` and the caller discards. A breaker converts it.

### Algorithm

```
# _zone_lifecycle :590-593 additionally records where and when
if np.any(broke):
    j = int(np.flatnonzero(broke)[0])
    return {dead: True, dead_reason: "closed through far side",
            broke_at: start + j, broke_price: <far side>, ...}

# in evaluate(), replacing the bare `continue` at :1191-1193
if life["dead"]:
    if (params["breaker_enabled"]
            and life["dead_reason"] == "closed through far side"
            and (len(entry_times) - 1 - life["broke_at"]) <= params["breaker_max_age_bars"]
            and (not params["breaker_requires_sweep"] or _zone_was_swept(zone))):
        flipped = dict(zone)
        flipped["kind"]    = SUPPLY if zone["kind"] == DEMAND else DEMAND
        flipped["source"]  = "BREAKER"
        flipped["pattern"] = "BRK"                    # 3 chars, MT5-comment safe
        flipped["created_at"] = life["broke_at"]
        zones_extra.append(flipped)                   # re-enters the normal pipeline
    self._reject("zone_dead")
    continue
```

`_zone_was_swept(zone)` reuses C.1's `_swept` against the zone's own far edge —
that is ICT's own breaker-vs-mitigation distinction (§A.3): breaker = failed OB
**with** a sweep. With `breaker_requires_sweep: False` you get mitigation
blocks too; that is the second arm of the A/B.

A flipped zone must then pass every existing gate unchanged — ZRI (`:1267`),
trend (`:1277`), structure (`:1294`), extension (`:1299`), confirmation
(`:1354`) — and gets its own `source_weights` entry (`:869`) so it can be
scored below `SND_V1` until it earns otherwise.

### Parameters

| param | default (no-op) | proposed live | search range |
|---|---|---|---|
| `breaker_enabled` | `False` | `True` | bool |
| `breaker_requires_sweep` | `True` | `True` | bool (the two arms) |
| `breaker_max_age_bars` | 60 | 60 | 20–150 (entry-TF bars) |
| `source_weights["BREAKER"]` | 0.0 | 0.75 | 0.0–2.0 |

### Cost

~50 lines, plus a dependency on C.1 if `breaker_requires_sweep` is used.

### Backtest hypothesis

> **H6.** Breakers **add** trades rather than filtering them, so the §0.3
> arithmetic runs in reverse: added trades must average **> 0.193R** or they
> dilute. Predicts: trade count rises 5–15%; `n_target_bucket` rises at least
> proportionally; the stop rate among *breaker-sourced* trades is ≤ 3.2%.
> Track breaker trades separately by `pattern == "BRK"` in the signal list.
> **Kill if** breaker-sourced trades average below +0.193R — a diluting
> addition is worse than no addition, because it also consumes the daily trade
> budget and the risk envelope.

---

## C.7 — Explicit build order and dependencies

```
C.2 (displacement shape)   ── independent, ~25 lines ── run first, it is nearly free
C.1 (liquidity + sweep)    ── independent, ~140 lines ── the flagship experiment
   ├── C.3 (target map + min-room)   depends on C.1's level map
   └── C.6 (breakers)                depends on C.1 only for `breaker_requires_sweep`
C.4 (BOS/CHoCH events)     ── independent, ~70 lines
C.5 (premium/discount)     ── independent, ~45 lines ── run last; likely redundant with Gate 3b
```

Run **C.1c (soft/score-only) and C.2 first** — they are the two arms that add
no veto and therefore cannot lose trades. If neither moves `n_target_bucket`
or sumR at fixed trade count, that is strong evidence that SMC adds nothing
here beyond what `_score_zone` already ranks on, and the remaining items
should be cancelled rather than built.

---

# PART D — Do NOT build

Ideas researched and rejected, with reasons. Each of these would look
plausible in a plan; each is a trap.

### D.1 Killzone / session-clock gating

Hardcoding London-open / NY-open / London-close windows.

**Why not.** (a) The evidence supports time-of-day *volatility* clustering,
not directional edge net of costs, and the canonical pattern is fitted to
1985–2007 data. ([6]) (b) The flagship **already made this decision
deliberately** and documented it at `:169-179`: it uses a volatility-shock
detector (`_bars_since_shock` `:154-192`) plus `min_atr_median_frac: 0.55`
(`:996`) precisely so that "it keeps working when the session that matters
moves". (c) The skill layer already has a `sessions:` field
(`xauusd_apex_turbo_m1.yaml:38`) — if a user wants clock gating, it exists
without new strategy code. (d) Broker-server timezone drift makes hardcoded
hours a silent-failure class.
**Instead:** slice the existing 1312-trade replay by UTC hour and check the
stop-bucket rate per hour **before writing any code** (§A.7). One query, zero
risk. Only if some band shows ≥6× enrichment does this become worth
revisiting.

### D.2 Judas swing / Asian-range model

**Why not.** It is C.1's liquidity sweep with a clock bolted on. Build the
clock-free sweep first; if sweeps pay, *then* ask whether adding the time
filter improves them — that is a one-parameter follow-up, not a separate
model. Also: the Asian range (20:00–00:00 EST) is 240 M1 bars, over the 200-bar
cap on the entry timeframe (it would fit on M5, at 48 bars). And the practitioner
claim itself is thin: "3–4 recognisable instances per week on gold, of which
2–3 meet full criteria" ([26]) is ~10 trades/month — not enough sample to
distinguish from noise in a two-month evaluation window.

### D.3 Standalone FVG entry model

**Why not.** (a) An independent mechanical test of exactly this family
(sweep → structure shift → gap) on 2.5M EURUSD M1 bars found **0 of 54
variations profitable after 0.5 pips of cost**. ([6]) (b) The gap-fill premise
is a documented myth (Caporale & Plastun; Plastun et al.) — gaps continue in
their direction more often than they fill. ([19][20]) (c) **Cost arithmetic
kills it here specifically:** XAUUSD on this broker sat at a flat 15-point
spread (`:170-176`); a three-candle M1 FVG on gold is routinely smaller than
the spread, so the "entry at the gap" is inside the bid-ask.
**Instead:** if FVG is built at all, build it as a **score term only** in
`_score_zone` (`:786-802`) on M5/M15 zones, never as an entry trigger, never
on M1.

### D.4 Inducement as a detector

**Why not.** "The obvious level in front of the real one" is only decidable
after price picks one. Every published rule collapses to either "the first
pullback after a BOS" (an ordinal counter) or "the weaker of two zones" —
and the second is exactly the problem `_score_zone` (`:786-802`) already
solves explicitly, with weights derived from measured P&L (`:865-874`).
Adding an inducement detector would be a second, worse, unmeasured ranker
competing with the measured one.
**Instead:** the falsifiable residue is one parameter — a `min_touch_episode`
mirroring the existing `max_touch_episode` (`:939`), i.e. "skip the first
retest, trade the second". If C.1 through C.4 all get built, add it as a
one-line sweep. Not a module.

### D.5 Volume-graded order blocks

The `smartmoneyconcepts` package grades OBs by `OBVolume` (candle volume +
two prior) and a strength `Percentage`. ([3])

**Why not.** The feed provides `tick_volume`, a tick-count proxy, not traded
volume. `fatigue.py` already downweights its volume component for exactly this
reason (module docstring `:83-85`; `DEFAULT_WEIGHTS["volume_decay"] = 0.50`,
the lowest in the set, `fatigue.py:158`). Building a volume-graded OB ranker
on the same feed would repeat a mistake this codebase has already reasoned its
way out of, in writing.

### D.6 Wyckoff AMD / power-of-three phase labelling

**Why not.** Accumulation → manipulation → distribution is century-old
descriptive text with no statistical validation as an intraday model. ([6])
Any programmatic version reduces to "range, then breakout, then trend", which
is `_detect_zones_v1` (`:297-360`) plus `_trend_state` (`:226-249`). It is
unfalsifiable in the specific sense that matters here: there is no rule whose
negation would be observable.

### D.7 A new SMC "exhaustion" / "smart money divergence" module

**Why not.** `backend/src/strategies/domain/fatigue.py` already exists — 8
sub-measures, evidence-weighted, with `trend_age` deliberately shipped at
weight 0.0 because the evidence didn't justify trusting it. It is wired as
Gate 3c (`:1305-1352`) and **its A/B has not been run to completion yet** (the
defaults at `:905-906` are still no-ops). Building a second exhaustion measure
under SMC vocabulary before finishing the first one's evaluation would be pure
duplication. Finish `scripts/fatigue_ab.py`'s sweep first.

### D.8 A new SMC strategy file from scratch

**Why not.** The apex family is one algorithm on five timeframe ladders
(`src/skills/normal/xauusd/xauusd_apex_*.yaml`), regenerated from the M1 file
by `scripts/gen_apex_timeframe_variants.py`. A parallel SMC bot would (a) fork
the maintenance surface, (b) lose the Zone Respect Index, the shock guard, the
episode counter, the ATR stop cap, the TP front-run and the broker-`min_rr`
handling — every one of which was written against measured losses, and (c)
make the A/B meaningless, because two files differ in a hundred ways at once.
**Every item in Part C is a parameter on the existing file with a no-op
default.** That is the only design that keeps the comparison honest.

### D.9 Any target/entry model requiring more than 200 bars of one timeframe

Weekly opening range, monthly levels, 20-day highs (the classic Turtle Soup
lookback), multi-week dealing ranges.

**Why not.** The engine caps context at 200 bars per timeframe
(`engine/application/trade_loop.py:65`) and **fails silently** — no error,
live or backtest. A 20-*day* high needs D1 bars; adding D1 to
`confirmation_timeframes` (`:834`) is possible, but the apex D1 rung already
exists as its own bot and the interaction is untested.
**Instead:** PDH/PDL from the already-declared H1 frame (200 H1 bars = 8.3
days) covers the useful part of this family with zero new timeframe — this is
folded into C.1 as `liq_use_pdh_pdl`.

---

# PART E — Sources

Academic / primary:

- [7] Osler, C. — *Currency Orders and Exchange-Rate Dynamics: Explaining the Success of Technical Analysis*, FRBNY Staff Report 125 — https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr125.pdf
- [8] Osler, C. — *Stop-Loss Orders and Price Cascades in Currency Markets*, FRBNY Staff Report 150 — https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf
- [9] Osler, C. (2003) — *Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive Success of Technical Analysis*, J. Finance 58(5) 1791-1819 — https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6261.00588
- [19] Caporale, G.M. & Plastun, A. — *Price Gaps: Another Market Anomaly?* — https://www.tandfonline.com/doi/full/10.1080/10293523.2017.1333563
- [20] Plastun, Sibande, Gupta & Wohar — *Price Gap Anomaly in the US Stock Market: The Whole Story* — https://repository.up.ac.za/bitstream/handle/2263/78336/Plastun_Price_2020.pdf
- [6] IndicatorEdge — *Smart Money Concepts: the research* (evidence review citing Brock/Lakonishok/LeBaron 1992, Sullivan/Timmermann/White 1999, Lo/Mamaysky/Wang 2000, Park & Irwin 2007, Wood/McInish/Ord 1985, Andersen & Bollerslev 1997/98, Ito & Hashimoto 2006, Chordia/Roll/Subrahmanyam 2002, Tsinaslanidis et al. 2022, Gurrib et al. 2022; plus a 2.5M-bar EURUSD mechanical test) — https://indicatoredge.io/smart-money-research

Implementation references (programmatic definitions):

- [3] joshyattridge/smart-money-concepts (Python) — https://github.com/joshyattridge/smart-money-concepts
- [5] LuxAlgo — Smart Money Concepts (SMC) indicator — https://www.luxalgo.com/library/indicator/smart-money-concepts-smc/
- [10] LuxAlgo — EQH/EQL Liquidity Zones — https://www.luxalgo.com/library/indicator/eqh-eql-liquidity-zones/
- [11] LuxAlgo — Liquidity Sweep — https://www.luxalgo.com/library/concept/liquidity-sweep/
- [17] LuxAlgo — Inversion Fair Value Gaps (IFVG) — https://www.luxalgo.com/library/indicator/inversion-fair-value-gaps-ifvg/
- [29] LuxAlgo — Model 2022 — https://www.luxalgo.com/library/concept/model-2022/
- [21] FibAlgo — ICT Displacement — https://www.tradingview.com/script/9OYOAKNU-FibAlgo-ICT-Displacement/
- [24] FibAlgo — ICT Premium & Discount — https://www.tradingview.com/script/SjBGkYUG-FibAlgo-ICT-Premium-Discount/

Practitioner definitions (used for rules, not for evidence):

- [1] DailyPriceAction — SMC Market Structure: BoS and CHoCH — https://dailypriceaction.com/blog/smc-market-structure/
- [2] Quantum Algo — BOS & CHoCH complete guide — https://www.quantum-algo.com/blog/guides/bos-choch-complete-trading-guide/
- [4] ChartingLens — Liquidity sweeps: BOS vs sweep — https://chartinglens.com/blog/liquidity-sweeps-trading-guide
- [12] Quantum Algo — Order Blocks complete guide — https://www.quantum-algo.com/blog/guides/order-blocks-complete-trading-guide/
- [13] ICTKillzone — What makes an order block valid — https://www.ictkillzone.com/ict-order-block
- [14] TradingStrategyGuides — Breaker & Mitigation Blocks — https://tradingstrategyguides.com/day-12-breaker-blocks-mitigation-blocks-explained-ict-smc-deep-dive/
- [15] TheICTTrader — Order/Breaker/Mitigation blocks — https://theicttrader.com/2024/03/24/order-blocks-breaker-blocks-and-mitigation-blocks/
- [16] Altrady — Mitigation block vs order block — https://www.altrady.com/blog/crypto-trading-strategies/mitigation-block-vs-order-block
- [18] FXOpen — Inverse Fair Value Gaps — https://fxopen.com/blog/en/what-is-an-inverse-fair-value-gap-ifvg-concept-in-trading/
- [22] ICTKillzone — Displacement — https://www.ictkillzone.com/ict-displacement
- [23] GrandAlgo — OTE 62–79% zone — https://grandalgo.com/blog/ict-optimal-trade-entry-ote
- [25] FXNX — ICT killzones for XAUUSD — https://fxnx.com/en/blog/ict-killzones-master-xauusd-timing-maximum-profit
- [26] TradingFinder — ICT Judas Swing — https://tradingfinder.com/education/forex/ict-judas-swing/
- [27] TMGM — Gold trading hours — https://www.tmgm.com/en/academy/trading-academy/gold-trading-hours
- [28] NordFX — Best time to trade XAUUSD — https://nordfx.com/traders-guide/best-time-to-trade-gold-xauusd-sessions-volatility-news
- [30] QuantVPS — ICT Unicorn Model — https://www.quantvps.com/blog/the-ict-unicorn-model-strategy
- [31] InnerCircleTrader — Unicorn model — https://innercircletrader.net/tutorials/ict-unicorn-model/
- [32] WritOfFinance — Inducement (IDM) — https://www.writofinance.com/inducement-in-forex-trading/
- [33] InnerCircleTrader — Inducement — https://innercircletrader.net/tutorials/what-is-inducement-in-forex/

**Explicitly discounted as evidence:** the circulating SMC backtest headline
figures (85.4% WR ML model; "2,600 trades, 61% WR, PF 2.17"; "XAUUSD 64.2% WR
PF 2.47"). When fetched, the source articles contain no per-concept
statistics, no sample sizes, no filter-effectiveness data and no negative
results. They are cited here only so a later phase does not rediscover them
and mistake them for support.
