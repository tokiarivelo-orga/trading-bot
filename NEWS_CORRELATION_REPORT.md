# News correlation report — XAUUSD

Measurement of the dated macro-event table in `backend/scripts/news_event_windows.py`
against this project's own data: `backend/data/trading.db` (opened read-only), M1
candles and the real live/paper trade journal. No backtests were run; backtests do
not write to `trading.db`, so everything in the trade tables below is genuine
executed order flow.

Analysis date: 2026-08-07. All timestamps UTC.

---

## 0. TL;DR

1. **The spread hypothesis is not confirmed, and cannot be tested with the M1 feed.**
   `candles.spread_points` for XAUUSD sits at exactly 15 points through *every*
   high-impact release — median 15 and **max 15** across all 688 impact-phase bars
   of all 9 event kinds. The only wide-spread bars anywhere near an event window are
   174 bars at 20:00 UTC, i.e. the daily rollover. Either the broker holds gold near
   a fixed spread through releases, or MT5's per-bar `spread` snapshot simply misses
   a 2-10 second widening. The data cannot distinguish these. See §3.
2. **The measurable news hazard is single-bar range, not spread.** The probability
   that one M1 bar's range exceeds the M1 bots' *median stop distance* (4.95 USD)
   goes from **4.8 % baseline to 54.7 % (CPI) / 54.2 % (NFP) / 70.8 % (FOMC)** in the
   first 15 minutes — ×11 to ×15. That is the mechanism that breaks a +0.2R-scratch
   strategy, and it is well evidenced. See §4.
3. **Elevated volatility persists far longer after FOMC than the current config
   assumes, and far shorter after everything else.** FOMC stays ≥2× baseline out to
   **+149 min** (the press conference produces a *second* peak at +30…+59 min);
   CPI is back under 1.5× by **+19 min**. The flat `before_min: 30 / after_min: 60`
   is simultaneously too long for CPI/PPI/retail/ISM and much too short for FOMC.
   See §5.
4. **There is no measurable pre-release run-up.** For every kind the median range in
   the 60 minutes before the print is 0.79–1.15× baseline, and the last minute before
   release is *below* baseline (0.58–0.99×). The 20–30 minute pre-blocks are paid for
   with no measured benefit. See §5.3.
5. **Bot attribution is not possible for the families that were asked about.**
   `xauusd_snd_qm_structure_*` has **0** trades inside any high-impact window (of
   3,429). `xauusd_apex_trendguard_*` has **20 trades in total** and 0 inside a
   high-impact window. The whole live book contains **9** trades inside a high-impact
   window (4 FOMC, 5 PCE). No conclusion is drawn from n=9. See §6.
6. **The one comparison with usable n is a clean null.** Jobless claims 2026-08-06:
   n=113 trades in the event window, avg R **+0.183**; n=587 trades in the same clock
   hours on other days, avg R **+0.182**. Identical. A medium-impact weekly release
   has no detectable effect on this bot fleet. See §6.3.
7. **Unplanned finding: the live calendar feed the news gate depends on succeeds
   13.1 % of the time** (320 of 2,447 refreshes, 2026-07-30 → 2026-08-07). Window
   sizes are moot if the gate does not know an event is coming. See §7.

---

## 1. Data inventory and coverage

| Source | Rows | Span |
|---|---|---|
| `candles` XAUUSD M1 | 120,471 | 2026-04-06 20:48 → 2026-08-07 03:12 (99.92 % contiguous minutes) |
| `candles` XAUUSD M5 | 104,330 | 2025-02-16 23:55 → 2026-08-07 |
| `trades` XAUUSD (all) | 3,891 | open 2026-07-15 14:39 → 2026-08-07 00:45 |
| `trades` XAUUSD closed, with an SL (analysis set) | 3,839 | open 2026-07-20 00:15 → 2026-08-07 00:36, **16 distinct dates** |
| `signal_decisions` (all XAUUSD) | 1,754 | 2026-08-05 13:53 → 2026-08-07 03:20 |
| `activity_logs` | 713,335 | 2026-07-29 13:53 → 2026-08-07 03:20 |

The asymmetry matters and constrains everything below: there are **4 months of price
history but only 16 trading dates of real order flow**, and 3,397 of the 3,839 trades
were opened in the last four days (2026-08-03 → 2026-08-06). The high-impact releases
that fall inside the trade period (FOMC 2026-07-29, PCE 2026-07-30) land in the sparse
early era, when the fleet was placing 13–50 trades a day.

### 1.1 Verification census of the event table

Whole table (`EVENTS`, n = 60):

| kind | verified | rule | unverified |
|---|---|---|---|
| CPI | 6 | 0 | 0 |
| FOMC | 4 | 0 | 0 |
| NFP | 4 | 0 | 0 |
| PCE | 4 | 0 | 0 |
| PPI | 4 | 0 | 0 |
| RETAIL_SALES | 6 | 0 | 0 |
| ISM_MFG | 2 | 0 | 2 |
| ISM_SERVICES | 0 | 0 | 3 |
| JOBLESS_CLAIMS | 0 | 25 | 0 |
| **total** | **30** | **25** | **5** |

Restricted to events whose full −60 … +180 min window is inside M1 coverage
(n = 43, the analysis set): **20 verified, 18 rule, 5 unverified.**

**Which conclusions depend on non-`verified` rows:**

- §5 window sizes for **CPI, NFP, FOMC, PCE, PPI, RETAIL_SALES** — 100 % verified rows.
  These carry no verification caveat.
- **ISM_SERVICES** (3 of 3 unverified, third-business-day rule) — every ISM_SERVICES
  number below is caveated. Note the direction of the risk: ISM_SERVICES shows *no*
  volatility signature (peak 1.15× baseline). That is exactly what a table of wrong
  dates would also look like. **"ISM Services has no effect" and "we have the wrong
  ISM Services dates" are not distinguishable here.**
- **ISM_MFG** (2 of 3 unverified) — the one verified instance (2026-08-03) has the
  *smallest* release-minute range of the three (4.47 USD vs 5.42 / 8.76), so the
  measured ISM_MFG signature is mostly carried by unverified rows. Caveated.
- **JOBLESS_CLAIMS** (18 of 18 `rule`-generated Thursdays) — used as the large-n
  control group. The result is a *null*, and a null is robust to a few wrong dates
  (a wrong date only adds noise, pulling toward the null). A *positive* claim from
  this group would not be safe; none is made.

### 1.2 Independent confirmation that the release timestamps are right

The three largest single-minute ranges in four months of XAUUSD M1 are:

| rank | minute (UTC) | range | matching event |
|---|---|---|---|
| 1 | 2026-07-14 12:30 | 60.52 USD | CPI (Jun data), verified |
| 2 | 2026-07-02 12:30 | 56.54 USD | NFP (Jun data), verified |
| 3 | 2026-07-29 18:00 | 42.96 USD | FOMC (Jul meeting), verified |

Ranks 5 and 6 are FOMC 2026-06-17 18:00 (37.98) and CPI 2026-06-10 12:30 (32.16).
The 08:30 ET → 12:30 UTC and 14:00 ET → 18:00 UTC mappings are therefore confirmed
against our own tape, not assumed.

---

## 2. Method

- **Baseline.** For every M1 bar: median spread and median range of the same
  minute-of-day (± 7 min pooled) on the same weekday, computed only over bars more
  than 240 minutes from any event in the table. Minimum 20 samples per cell; typical
  cell ≈ 250 bars.
- **Thursday fallback.** Weekly jobless claims fall every Thursday 12:30 UTC, so
  Thursday midday has *zero* uncontaminated same-weekday control samples. Those cells
  fall back to a weekday-pooled, same-minute-of-day baseline. Affected: all
  JOBLESS_CLAIMS and PCE rows, and one third of NFP / PPI / RETAIL_SALES rows.
  (An earlier pass that lacked this fallback silently dropped every Thursday event —
  worth stating, because it is an easy way to lose NFP-Jul and both PCEs without
  noticing.)
- **Panel.** For each event, offsets −60 … +180 min, matched to the exact M1 bar;
  10,357 event-minute rows over 43 events.
- **Window size definition.** Scanning bucket by bucket forward from t = 0, the last
  bucket whose *median* range ratio is still ≥ X, stopping at the first bucket below
  X. First-crossing, not last — a single noisy late bucket cannot inflate a window.
- **Placebo control for trades.** With 1–3 instances per event kind, "event window"
  and "that particular afternoon" are perfectly confounded. Every inside-window trade
  statistic is therefore paired with the same clock window (release minute-of-day
  −15 … +120) on *every other date* in the trade sample.
- R is computed per trade as `sign · (close_price − open_price) / |open_price − sl|`.
  Profit factor from the `profit` column. Win rate is reported but, per the engine's
  +0.2R secure-base trailing (`DEFAULT_SECURE_BUFFER_R_MULT = 0.2`,
  `backend/src/engine/application/position_manager.py`), it is not used for any
  conclusion.

Scripts were throwaway and live in the session scratchpad, not the repo. The DB was
opened `file:…?mode=ro`; nothing was written.

---

## 3. Spread — the headline, and it is a negative result

### 3.1 The M1 candle feed shows zero news widening

`candles.spread_points` for XAUUSD is heavily quantised:

| value | bars | share |
|---|---|---|
| 15 pts (0.15 USD) | 109,168 | 90.6 % |
| 28 pts | 8,533 | 7.1 % |
| 58 pts | 1,965 | 1.6 % |
| everything else | 790 | 0.7 % |

p50 = 15, p90 = 15, p99 = 58, max = 179.

Wide bars are a pure time-of-day phenomenon. Of 11,050 bars with spread > 15, **all
but 38 are in hours 20, 22 and 23 UTC** — the daily rollover. Hours 06–19 UTC never
exceed 15 points in four months.

Inside the impact phase (0 … +15 min) of every event kind:

| kind | bars | median spread | **max spread** |
|---|---|---|---|
| CPI | 64 | 15 | **15** |
| NFP | 48 | 15 | **15** |
| FOMC | 48 | 15 | **15** |
| PCE | 32 | 15 | **15** |
| PPI | 48 | 15 | **15** |
| RETAIL_SALES | 64 | 15 | **15** |
| ISM_MFG | 48 | 15 | **15** |
| ISM_SERVICES | 48 | 15 | **15** |
| JOBLESS_CLAIMS | 288 | 15 | **15** |

Across the whole 10,357-row event panel, only 174 bars have spread > 15 — every one
of them at hour 20 UTC, inside the far tail (+120 … +180) of an 18:00 FOMC.

**On the 2026-07-14 CPI minute the range was 60.52 USD and the recorded spread was
15 points.** That single fact is the whole section.

### 3.2 Why this is "cannot measure", not "no effect"

Two explanations fit, and this dataset cannot separate them:

- `candles.spread_points` comes from MT5's `MqlRates.spread`
  (`gateway/src/gateway/mt5_client.py:201`) — one snapshot per bar. A widening that
  lasts 2–10 seconds inside a 60-second bar is invisible to it by construction.
- The broker may genuinely hold XAUUSD near a fixed 15-point spread through releases
  and only widen at rollover.

Given that rollover widening *is* captured (28 → 58 → 179 points), the feed is not
simply broken. But absence of evidence at this sampling rate is weak evidence of
absence. **Any "news does not widen spread" claim must be labelled as
feed-limited.**

### 3.3 The one tick-time measurement available

`signal_decisions.checks` records the observed `spread_points` **even when the gate
vetoes the trade**, so it is uncensored. It only spans 2026-08-05 13:53 → 2026-08-07
03:20 (n = 1,681 XAUUSD decisions carrying a spread, gate cap 35 points throughout).

Rest-of-sample baseline: p50 = 16, p90 = 19, p99 = 55, max = 61.

| event | offsets | n obs | p50 | p90 | max |
|---|---|---|---|---|---|
| ISM_SERVICES 2026-08-05 14:00 *(unverified date)* | −30 … −1 | 12 | 16 | 27.7 | 29 |
| | **0 … +5** | **2** | **29** | 29 | 29 |
| | +15 … +60 | 43 | 16 | 16 | 16 |
| JOBLESS_CLAIMS 2026-08-06 12:30 *(rule date)* | −30 … −1 | 45 | 16 | 16 | 30 |
| | **0 … +5** | **6** | **23** | 27.5 | 29 |
| | +5 … +15 | 16 | 16 | 16 | 16 |
| | +15 … +90 | 30 | 16 | 16 | 16 |

So at tick resolution the spread does roughly **double (16 → 23–29 points, 0.16 →
0.29 USD) for about five minutes and then reverts**. That is the shape one expects.
It is also **8 observations across two medium-impact events, one of them on an
unverified date.** It is enough to say "the effect exists and is short"; it is
**not** enough to set a spread cap, and it says nothing about CPI/NFP/FOMC, none of
which fall inside this 1.5-day log window.

Two corroborating details:

- All **107** `spread_veto` decisions in the log occurred at 20:00–22:59 UTC. **Not
  one spread veto has ever fired at a news release.** The bot fleet's real spread
  enemy is the daily rollover.
- `trades.spread_points_at_entry` (n = 3,839): p50 = 16, p90 = 18, p99 = 29, max 51.
  Inside −5/+15 of any event (n = 55) p90 rises to 29 vs 18 outside, and |slippage|
  p90 rises 0.230 → 0.316. But this sample is **censored by the bots' own 35-point
  gate** — trades that would have opened at a 60-point spread do not exist — so it
  understates the tail by construction and cannot be used to size a cap either.
- Slippage is `NULL` for **all 9** high-impact-window trades (execution telemetry
  landed later; only 1,234 of 3,839 rows carry it). **Slippage during a high-impact
  release is entirely unobserved in this database.**

---

## 4. What actually threatens the +0.2R profit engine: single-bar range

The bot fleet's economics, from the real trade journal:

| quantity | value | source |
|---|---|---|
| median SL distance, `snd_qm_structure` M1 family (n = 3,074) | **4.95 USD** | `trades` |
| implied 0.2R scratch distance | **0.99 USD** | 0.2 × above |
| typical spread | 0.16 USD | `spread_points_at_entry` p50 = 16 |
| spread as a fraction of the scratch | **16 %** | |
| spread as a fraction of the scratch at a 29-point news tick | 29 % | |

So even a doubled spread does not by itself destroy the scratch. What does is a
single M1 bar large enough to run the stop before the trail can lock anything in:

**P(one M1 bar's range > 4.95 USD, the median SL distance):**

| population | n bars | P | ratio to baseline |
|---|---|---|---|
| all XAUUSD M1 bars | 120,460 | **4.78 %** | ×1 |
| FOMC 0 … +15 min | 48 | **70.8 %** | **×14.8** |
| CPI 0 … +15 min | 64 | **54.7 %** | **×11.4** |
| NFP 0 … +15 min | 48 | **54.2 %** | **×11.3** |
| PCE 0 … +15 min | 32 | **40.6 %** | ×8.5 |
| PPI 0 … +15 min | 48 | 27.1 % | ×5.7 |
| JOBLESS_CLAIMS 0 … +15 min | 288 | 20.8 % | ×4.4 |
| ISM_MFG 0 … +15 min | 48 | 16.7 % | ×3.5 |
| RETAIL_SALES 0 … +15 min | 64 | 15.6 % | ×3.3 |
| ISM_SERVICES 0 … +15 min *(unverified)* | 48 | 14.6 % | ×3.1 |

For scale, the largest release-minute bars were 60.52 USD (CPI 2026-07-14) and 56.54
USD (NFP 2026-07-02) — **12× the median stop distance of the M1 bots**. A position
open into that bar does not get a fair stop-out; it gets whatever the next tick is.

This, not spread, is the mechanism that should drive the news policy. It is measured
on 43 events and 120k bars, not on 9 trades.

---

## 5. Volatility persistence — the number that sets window sizes

### 5.1 Median M1 range ÷ same-weekday, same-minute-of-day baseline

| kind | n ev | −60…−31 | −30…−11 | −10…−2 | −1 | **0** | 1–2 | 3–4 | 5–9 | 10–14 | 15–19 | 20–29 | 30–44 | 45–59 | 60–89 | 90–119 | 120–149 | 150–180 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPI | 4 | 1.02 | 1.10 | 1.00 | 0.98 | **9.25** | 2.66 | 2.34 | 2.18 | 2.02 | 1.81 | 1.43 | 1.20 | 1.26 | 1.19 | 1.06 | 1.20 | 1.35 |
| NFP | 3 | 1.02 | 1.15 | 0.98 | 0.99 | **5.40** | 2.29 | 2.97 | 2.27 | 1.89 | 1.82 | 1.95 | 1.54 | 1.41 | 1.38 | 1.26 | 1.25 | 1.16 |
| FOMC | 3 | 1.08 | 1.01 | 1.13 | 0.86 | **23.02** | 7.00 | 4.14 | 3.58 | 4.25 | 3.36 | 2.13 | **4.80** | 3.81 | 3.38 | 2.75 | 2.36 | 1.59 |
| PCE | 2 | 0.85 | 1.06 | 1.07 | 0.99 | **7.84** | 2.69 | 1.57 | 1.95 | 2.18 | 1.86 | 1.33 | 1.43 | 1.16 | 1.48 | 1.38 | 1.08 | 1.17 |
| PPI | 3 | 0.79 | 0.86 | 0.99 | 0.60 | **3.62** | 2.25 | 1.51 | 1.65 | 1.50 | 1.32 | 1.10 | 0.94 | 0.84 | 1.09 | 0.88 | 0.91 | 0.97 |
| RETAIL_SALES | 4 | 0.83 | 1.03 | 0.94 | 0.58 | **2.96** | 1.00 | 0.95 | 1.36 | 1.27 | 1.08 | 0.88 | 1.18 | 0.89 | 1.02 | 0.87 | 1.01 | 1.06 |
| ISM_MFG † | 3 | 1.41 | 1.34 | 1.11 | 0.80 | **1.70** | 1.31 | 1.09 | 1.07 | 0.97 | 1.06 | 1.01 | 0.93 | 0.86 | 0.93 | 0.85 | 0.77 | 0.78 |
| ISM_SERVICES † | 3 | 0.95 | 0.99 | 1.02 | 0.73 | **1.15** | 1.20 | 0.92 | 1.07 | 1.01 | 1.15 | 0.91 | 0.92 | 0.76 | 0.89 | 0.94 | 0.96 | 0.86 |
| JOBLESS_CLAIMS ‡ | 18 | 1.03 | 1.07 | 1.04 | 0.76 | **2.34** | 1.41 | 1.41 | 1.46 | 1.31 | 1.31 | 1.14 | 1.23 | 1.05 | 1.19 | 1.10 | 1.04 | 1.05 |

† dates partly (ISM_MFG) or wholly (ISM_SERVICES) unverified — see §1.1.
‡ all dates `rule`-generated.

Bars per cell: 3–4 per kind at offset 0 (one per event), rising to 87–124 in the wide
tail buckets; JOBLESS_CLAIMS 18 → 558. Full n-matrix was computed; the thinnest cells
are exactly the ones at t = 0, so the peak multiples are the least precise numbers in
the table and the tail decay is the most precise.

Absolute median M1 range at the release minute, for a sense of scale: FOMC 37.98 USD,
CPI 23.40, PCE 19.09, NFP 12.15, PPI 8.81, RETAIL_SALES 7.41, JOBLESS_CLAIMS 5.70,
ISM_MFG 5.42, ISM_SERVICES 3.66 — against a 1.4–3.3 USD baseline.

### 5.2 Measured window sizes (first-crossing)

**Minutes after release during which the median stays at or above the threshold:**

| kind | n ev | ≥3× | ≥2× | **≥1.5×** | ≥1.25× |
|---|---|---|---|---|---|
| CPI | 4 | +0 | +14 | **+19** | +29 |
| NFP | 3 | +0 | +9 | **+44** | +149 |
| **FOMC** | 3 | **+19** | **+149** | **+180 (never returns inside the horizon)** | +180 |
| PCE | 2 | +0 | +2 | **+19** | +44 |
| PPI | 3 | +0 | +2 | **+14** | +19 |
| RETAIL_SALES | 4 | — | +0 | **+0** | +0 |
| ISM_MFG † | 3 | — | — | **+0** | +2 |
| ISM_SERVICES † | 3 | — | — | **—** | — |
| JOBLESS_CLAIMS ‡ | 18 | — | +0 | **+0** | +19 |

"+0" means the release minute itself clears the threshold but the very next bucket
does not. "—" means the threshold is never reached.

**The FOMC result is the most important line in this report.** FOMC is the only kind
whose ratio *rises again* after decaying: 2.13× at +20…+29, then **4.80× at +30…+44**
and 3.81× at +45…+59. That is the 14:30 ET press conference, exactly 30 minutes after
the 14:00 ET statement. Every one of the three FOMC instances shows it. A window that
closes at +90 (current `fomc.yaml`) or +120 (the value currently hard-coded in
`news_event_windows.py`) **re-admits the bots into the single most violent stretch of
the month.**

### 5.3 There is no pre-release run-up

Median range ratio before the print:

| kind | −60…−31 | −30…−11 | −10…−2 | −1 |
|---|---|---|---|---|
| CPI | 1.02 | 1.10 | 1.00 | 0.98 |
| NFP | 1.02 | 1.15 | 0.98 | 0.99 |
| FOMC | 1.08 | 1.01 | 1.13 | 0.86 |
| PCE | 0.85 | 1.06 | 1.07 | 0.99 |
| PPI | 0.79 | 0.86 | 0.99 | 0.60 |
| RETAIL_SALES | 0.83 | 1.03 | 0.94 | 0.58 |
| JOBLESS_CLAIMS | 1.03 | 1.07 | 1.04 | 0.76 |

Nothing is elevated, and the **final minute before release is below baseline for
every single kind** (0.58–0.99×) — the market goes quiet into the print. The current
`before_min` of 20–30 minutes buys no volatility protection. Its only defensible
purpose is giving `pre_event: close_all` enough lead time to actually flatten, which
needs single-digit minutes, not thirty.

Cost of the current pre-blocks, measured over the M1 window: current skill-YAML
windows block **2,142 of 120,471 minutes (1.78 %, ≈534 min/month)**; the §8 proposal
blocks **1,540 minutes (1.28 %, ≈384 min/month)** while *tripling* the FOMC block
(363 → 567 min) — i.e. less total downtime, concentrated where the measurement says
the risk actually is.

---

## 6. Bot attribution — mostly a sample-size report

### 6.1 There is almost nothing to attribute

The analysis set is 3,839 closed XAUUSD trades with an SL. Trades whose open time
falls inside a **high-impact** (CPI/NFP/FOMC/PCE) window:

| window definition | inside | outside |
|---|---|---|
| `configs/news.yaml` current default (−30 / +60) | **5** | 3,834 |
| sensitivity (−5 / +90) | **9** | 3,830 |
| wide (−15 / +120) | **9** | 3,830 |
| **§8.1 recommended per-kind windows** † | **6** (4 FOMC, 2 PCE) | 3,876 |

† measured on a slightly later snapshot of the live DB — see the note at the end
of this section.

Per family, using the widest (−15 / +120) definition:

| family | total trades | **inside a high-impact window** |
|---|---|---|
| `xauusd_snd_qm_structure_*` (all TFs) | 3,429 | **0** |
| `xauusd_apex_trendguard_*` | 20 | **0** |
| `xauusd_apex_turbo_*` | 60 | **0** |
| everything else | 330 | 9 |

> **Snapshot note.** `trading.db` was being written by a live session throughout this
> analysis. The trade tables above are a single consistent snapshot (3,839 rows); a
> re-run roughly an hour later saw 3,882 rows, with `snd_qm_structure` at 3,465 and
> `apex_turbo` at 67. Every structural result is unchanged across both snapshots —
> in particular **0 inside a high-impact window** for `snd_qm_structure` (0 of 3,465)
> and `apex_trendguard` (0 of 20) under both the wide and the recommended windows.
> Treat the exact counts as a point-in-time reading, not a fixed quantity.

**Both families named in the task have zero trades inside a high-impact window.**
`xauusd_apex_trendguard_*` — the family whose 1,312-trade backtest is the baseline for
this whole project — has **20 real trades in total**, none of them near a release.
No inside-vs-outside comparison is possible for either family, at any window size.
This is not a weak result; it is no result, and no number should be quoted for it.

### 6.2 The 9 high-impact-window trades, in full

| event | open offset | strategy | side | risk (USD) | R | profit (USD) |
|---|---|---|---|---|---|---|
| FOMC 07-29 18:00 | +15.0 | `scalp_volume_imbalance_v1` | buy | 10.75 | −1.054 | −11.33 |
| FOMC 07-29 18:00 | +75.0 | `trend_structure_v1` | buy | 46.41 | −1.000 | −46.42 |
| FOMC 07-29 18:00 | +75.1 | `trend_structure_v2` | buy | 46.66 | −1.000 | −46.67 |
| FOMC 07-29 18:00 | +75.1 | `trend_structure_v7` | buy | 47.30 | −1.000 | −47.31 |
| PCE 07-30 12:30 | +19.1 | `trend_structure_v1` | buy | 2.74 | −1.000 | −16.44 |
| PCE 07-30 12:30 | +30.0 | `scalp_volume_imbalance_v1` | sell | 6.36 | −1.030 | −26.20 |
| PCE 07-30 12:30 | +45.0 | `trend_structure_v1` | sell | 9.34 | +0.195 | +5.46 |
| PCE 07-30 12:30 | +55.0 | `scalp_volume_imbalance_v1` | sell | 7.29 | +0.147 | +4.28 |
| PCE 07-30 12:30 | +65.0 | `scalp_volume_imbalance_v1` | sell | 7.39 | −1.000 | −29.56 |

Aggregate: n = 9, sum R = −6.74, avg R = **−0.749**, PF = 0.043, net **−214.19 USD**,
2 of 9 winners. Outside: n = 3,830, avg R = −0.006, PF = 0.996.

**This looks damning and it is not evidence.** n = 9, from two calendar days, drawn
entirely from strategies (`trend_structure_*`, `scalp_volume_imbalance_v1`) that are
not the fleet under study. Three of the nine are the *same signal* fired by three
`trend_structure` variants 6 seconds apart on the same bar — effectively one
observation, not three. Effective n is closer to 6.

Note also what the offsets show: nothing opened inside 0 … +15 min of either event.
The earliest entries are +15.0 (FOMC) and +19.1 (PCE). Whether that is the news gate
working or simply no signal firing cannot be determined — see §7 for why it is
probably the latter.

### 6.3 Placebo control — the honest comparison

Same clock window (release minute-of-day −15 … +120) on the event day vs every other
date in the sample:

| event | verif | arm | n | avg R | PF | net USD |
|---|---|---|---|---|---|---|
| FOMC 2026-07-29 18:00 | verified | event day | **4** | −1.014 | 0.000 | −151.73 |
| | | placebo, other days | 451 | −0.168 | 0.734 | −707.62 |
| PCE 2026-07-30 12:30 | verified | event day | **5** | −0.538 | 0.135 | −62.46 |
| | | placebo, other days | 695 | +0.187 | 1.105 | +290.51 |
| ISM_MFG 2026-08-03 14:00 | verified | event day | 49 | +0.086 | 2.168 | +23.68 |
| | | placebo, other days | 666 | +0.184 | 1.369 | +1037.46 |
| ISM_SERVICES 2026-08-05 14:00 | **unverified** | event day | 152 | +0.483 | 3.953 | +885.20 |
| | | placebo, other days | 563 | +0.095 | 1.070 | +175.94 |
| **JOBLESS_CLAIMS 2026-08-06 12:30** | rule | **event day** | **113** | **+0.183** | 1.647 | +273.18 |
| | | **placebo, other days** | **587** | **+0.182** | 0.981 | −45.13 |

Readings:

- **FOMC:** the event-day arm was terrible, but so was that clock window on ordinary
  days (avg R −0.168, PF 0.734 over 451 trades). The 18:00–20:00 UTC slot is a bad
  slot for this fleet generally. n = 4 on the event day. **No conclusion.**
- **PCE:** −0.538 vs +0.187 is the largest gap in the table and points the expected
  way, but n = 5. **No conclusion.**
- **ISM_SERVICES:** the biggest inside-window number in the whole dataset (+885 USD,
  PF 3.95, n = 152) and it should be **ignored**. The date is unverified; it is a
  single afternoon; that same afternoon the whole book was −409 USD on the day; and
  §5 shows ISM_SERVICES has no volatility signature at all. This is a
  one-afternoon effect wearing an event label.
- **JOBLESS_CLAIMS 2026-08-06: the only comparison in this dataset with adequate n on
  both arms.** Avg R +0.183 (n = 113) inside vs +0.182 (n = 587) outside — identical
  to three decimals. Profit factor differs (1.647 vs 0.981) but that is one heavy
  loser's worth of USD on 700 trades and the R measure, which is the noise-robust one,
  is flat. **Finding: a medium-impact weekly release has no detectable effect on this
  fleet's per-trade expectancy.** That is a real result, and it is the only bot-side
  result in this report that clears its own evidence bar.

### 6.4 Context: the live book is not the backtest

For calibration when reading any of the above — over the 3,839-trade analysis set the
whole fleet ran to sum R = −27.4, avg R = −0.007, PF = 0.988, net −254 USD, across 16
dates. That is against a `xauusd_snd_apex_trendguard_m1` backtest of PF 6.54 / avg R
0.193. Any news effect being hunted here is a second-order term on a book whose
first-order term is already negative. Detecting a window effect against that noise
would need far more than 16 dates.

---

## 7. The news gate depends on a feed that is down 87 % of the time

Not part of the brief, but it dominates every recommendation below, so it is measured.

`src.news.application.news_window_service`, 2026-07-30 14:19 → 2026-08-07 02:34:

| outcome | count |
|---|---|
| refresh succeeded | **320 (13.1 %)** |
| `forexfactory unreachable: [Errno -2] Name or service not known` | 1,536 |
| `forexfactory /ff_calendar_thisweek.json -> 429` (rate limited) | 575 |
| other failure | 16 |
| **total attempts** | **2,447** |

Of the 320 successes, **31 returned 0 events**. Daily success rate ranged from
**4.4 %** (2026-07-31, on 1,209 attempts) to 100 % (2026-08-02, 19 attempts).

Two compounding problems:

1. **DNS/network:** 1,536 of 2,447 failures are name-resolution errors, i.e. the host
   running the backend frequently has no working DNS for forexfactory.com.
2. **Self-inflicted rate limiting:** `configs/news.yaml` sets `refresh_minutes: 60`,
   yet 1,209 attempts were made on 2026-07-31 alone — roughly one every 70 seconds.
   ForexFactory's own 429 body says the file "is only updated once per hour.
   Requesting it more than that is unnecessary and can result in being blocked."
   The retry-on-failure path is not honouring the configured interval, so a transient
   DNS outage escalates into a rate-limit ban.

Freshness at the releases inside the log window:

| release | age of last good refresh | failed attempts since |
|---|---|---|
| ISM_MFG 2026-08-03 14:00 | 27.2 min | 1 |
| ISM_SERVICES 2026-08-05 14:00 | 2.8 min | 0 |
| JOBLESS_CLAIMS 2026-08-06 12:30 | **225.1 min** | 0 |

**Implication.** Tuning `before_min`/`after_min` is worth very little while the gate's
knowledge of the calendar is stale 87 % of the time. This is also the strongest
argument for the artefact this phase produced: `news_event_windows.py` is a static,
source-verified, dated table that needs no network. It should become the *fallback*
the gate falls back to, not just an analysis helper.

---

## 8. Recommendations

Every recommendation cites the number that supports it. Where the data does not
support a recommendation, that is said instead of inventing one.

### 8.1 Window sizes — supported

| kind | current (skill YAML / default) | **recommended** | supporting number |
|---|---|---|---|
| **FOMC** | −30 / +90 | **−10 / +180** | ≥2× baseline until **+149 min**; second peak **4.80×** at +30…+59 (the press conference) — §5.2 |
| **NFP** | −30 / +60 | **−5 / +60** | ≥1.5× until +44 min; no pre-run-up (−60…−2 = 0.98–1.15×) — §5.2, §5.3 |
| **CPI** | −20 / +45 | **−5 / +30** | ≥1.5× until +19, ≥1.25× until +29 — §5.2 |
| **PCE** | −15 / +30 (generic) | **−5 / +45** | ≥1.5× until +19, ≥1.25× until +44 — §5.2 |
| **PPI** | −15 / +30 (generic) | **−5 / +20** | ≥1.5× until +14, ≥1.25× until +19 — §5.2 |
| **RETAIL_SALES** | −15 / +30 (generic) | **−5 / +15** | only the release minute clears 1.5× (2.96×); +1 min onward is 0.88–1.36× — §5.1 |
| **ISM_MFG** | −15 / +30 (generic) | **−5 / +10** | peak 1.70× at t=0, ≥1.25× only to +2 min — §5.2. *Caveat: 2 of 3 dates unverified.* |
| **ISM_SERVICES** | −15 / +30 (generic) | **−5 / +5**, or drop entirely | peak **1.15×** — no signature at all — §5.2. *Caveat: all 3 dates unverified; a wrong-date table is indistinguishable from a no-effect event. Verify the dates before acting on this row.* |
| **JOBLESS_CLAIMS** | −15 / +30 (generic) | **−2 / +20** | peak 2.34× at t=0, ≥1.25× to +19 — §5.2; and §6.3 shows **no measurable P&L effect** at n=113 vs 587 |

`before_min` drops to 5 (2 for claims) across the board on the strength of §5.3: no
kind shows any pre-release elevation and every kind is *below* baseline in the final
minute. Five minutes is retained purely as flatten lead-time for
`pre_event: close_all`, not as a volatility buffer.

Net effect on downtime: **−28 % total blocked minutes** (534 → 384 min/month) while
tripling FOMC coverage (363 → 567 min/month). §5.3.

### 8.2 Block entries — supported

Block new entries for the full recommended window on **FOMC, CPI, NFP, PCE**.
Supporting number: P(one M1 bar > the M1 fleet's median 4.95 USD stop) is 70.8 % /
54.7 % / 54.2 % / 40.6 % in the first 15 minutes versus a 4.78 % baseline — ×8.5 to
×14.8 (§4). A strategy whose entire edge is a 0.99 USD scratch cannot survive being
stopped by a bar that is routinely 12–60 USD wide.

### 8.3 Flatten open positions — supported for FOMC/CPI/NFP, weakly

`pre_event: close_all: true` is already set on cpi/fomc/nfp/generic and should stay
for CPI/NFP/FOMC/PCE, on the same §4 number. It should be **removed from
`generic_high_impact`'s effective reach over ISM/retail-class events** (see §8.6),
where the measured hazard is ×3.1–×3.5 and the release-minute range is 3.7–7.4 USD.
Flattening a winning position for a 1.15× volatility bump costs more than it saves.

### 8.4 Spread caps — **not supported; do not change them**

The current caps (cpi 60, nfp 80, fomc 100, generic 50 points) cannot be validated or
falsified with this data.

- The M1 feed shows a maximum of **15 points** across all 688 impact-phase bars of
  all 9 kinds (§3.1), which would imply every cap is 3–7× too loose.
- The only tick-time evidence is **8 observations** on two medium-impact events,
  showing 16 → 23–29 points for ~5 minutes (§3.3).
- The trade-level sample is censored by the bots' own 35-point gate (§3.3).

**Recommendation: leave the caps alone and instrument instead.** Persist a tick-level
spread sample (e.g. every `symbol_info` poll, not only at order-send) for one CPI, one
NFP and one FOMC. Three events at 1-second resolution would settle this permanently;
four months of M1 bars cannot.

Corollary, supported: **the nfp.yaml comment claiming XAUUSD sees "5–15× normal
spread" at NFP is wrong as written for this broker.** The ×5–15 figure is right for
*range* (NFP release minute 12.15 USD median vs 2.25 baseline = 5.4×) and unsupported
for *spread*. Fixing the comment prevents the next reader from tuning the wrong knob.

### 8.5 Risk multiplier — **not supported either way**

Current post-event `risk_multiplier` is 0.4 (fomc) / 0.5 (nfp, generic) / 0.6 (cpi).
There are **9 trades** in the entire journal inside a high-impact window (§6.1), and
the two families the multiplier is meant to protect have **zero**. Nothing in this
dataset can say whether 0.4 is better than 0.6, or whether either beats simply not
trading. Leave them; revisit when the fleet has accumulated real post-event trades.

### 8.6 Rollover, not news, is the live spread hazard — supported

All **107** spread vetoes in the log fired at 20:00–22:59 UTC (§3.3). XAUUSD spread is
28 points median and 58 at p99 in hours 20 and 22, max **179** — versus a flat 15 for
hours 06–19 (§3.1). Yet there is no `configs/` control for this at all, while
considerable machinery exists for news windows that have never once triggered a spread
veto.

Recommendation: add a time-of-day spread guard for 19:55–23:05 UTC on XAUUSD. The
supporting number for the threshold is the p99 of 58 points. This sits outside this
phase's file scope and is offered as a follow-up, not applied.

### 8.7 Is any event worth *trading* rather than avoiding? — **No, on this data**

Directional persistence, 9 high-impact instances:

| | value |
|---|---|
| median absolute 0–15 min impulse | 17.39 USD |
| sign agreement, 0–15 min move vs 15–60 min move | 5/9 (0.556) |
| sign agreement, 0–15 min move vs 15–120 min move | **5/9 (0.556), two-sided binomial p = 1.00** |
| median continuation 15–120 min | +2.19 USD |
| mean continuation 15–120 min | +9.06 USD (one NFP instance dominates) |

Per kind: CPI 2/4 agree at 120 min, NFP 1/3, PCE 2/2, PPI 0/3, RETAIL_SALES 4/4,
ISM_MFG 2/3. Jobless claims, the only group with real n: 12/18 agree, two-sided
binomial **p = 0.238** — not significant.

**A coin flip.** There is no evidence of either a continuation edge or a fade edge, at
any horizon tested. The mean/median split (+9.06 vs +2.19) is the signature of one or
two outliers doing all the work — exactly what an n=9 sample cannot distinguish from
noise. **No event in this table is worth trading on the strength of this data**, and
constructing a news-breakout strategy from these numbers would be fitting to nine
observations.

If this is to be revisited, the honest requirement is stated up front: distinguishing
a 60 % continuation rate from 50 % at p < 0.05 needs roughly 65 event instances —
about **5 years** of monthly releases, or a pooled multi-symbol study.

### 8.8 Fix the calendar feed before tuning windows — supported

13.1 % refresh success over 2,447 attempts, with a 225-minute-stale calendar at the
2026-08-06 claims release (§7). Two changes, in priority order:

1. **Honour `refresh_minutes` on the failure path.** 1,209 attempts in one day against
   a once-per-hour file is what produces the 575 HTTP 429s. Exponential backoff with
   a floor at the configured interval.
2. **Fall back to the static table.** `backend/scripts/news_event_windows.py` is
   source-verified, dated, offline, and already carries a `verification` field the
   gate can respect (e.g. honour `verified` and `rule` rows, ignore `unverified`).
   Promoting it out of `scripts/` into the `news` module as a `NewsCalendarPort`
   implementation would make the gate work with no network at all. **Proposal only —
   not implemented in this phase.**

---

## 9. Proposed config changes — PROPOSAL ONLY, NOT APPLIED

No file under `configs/` or `backend/src/skills/news/` was modified by this phase.

### 9.1 `configs/news.yaml`

```yaml
calendar:
  source: forexfactory
  refresh_minutes: 60
  # NEW — §7: 320/2447 refreshes succeeded (13.1%); 575 of the failures were
  # HTTP 429 caused by ~1200 attempts/day against a once-hourly file.
  min_retry_seconds: 900        # floor on the failure-retry path
  fallback: static_table        # scripts/news_event_windows.py, offline
  fallback_min_verification: rule   # accept "verified" and "rule", skip "unverified"

tracked_events:
  - { name: "Non-Farm Payrolls",     impact: high,   skill: nfp }
  - { name: "CPI",                   impact: high,   skill: cpi }
  - { name: "FOMC Statement",        impact: high,   skill: fomc }
  - { name: "FOMC Press Conference", impact: high,   skill: fomc }
  # NEW — PCE has its own profile (§5.2: >=1.5x to +19, >=1.25x to +44) and is
  # currently swept into generic_high_impact's -15/+30.
  - { name: "PCE Price Index",       impact: high,   skill: pce }
  - { name: "PPI",                   impact: medium, skill: ppi }
  - { name: "Retail Sales",          impact: medium, skill: retail_sales }
  - { name: "*",                     impact: high,   skill: generic_high_impact }

# was { before_min: 30, after_min: 60 } — a guess.
# -5 from §5.3 (no kind shows any pre-release elevation; the final minute before
# release is BELOW baseline for every kind, 0.58-0.99x). +30 from the median of
# the measured >=1.25x crossings for unnamed events.
default_window: { before_min: 5, after_min: 30 }
```

### 9.2 `backend/src/skills/news/fomc.yaml` — the change that matters

```yaml
activation:
  calendar_event: ["FOMC Statement", "FOMC Press Conference"]
  window: { before_min: 10, after_min: 180 }   # was 30 / 90
```

Justification to put in the comment: median M1 range stays ≥2× baseline until
**+149 min** and never returns under 1.5× within +180. Unlike every other kind FOMC
has a **second peak — 4.80× at +30…+59 min** — the 14:30 ET press conference. All
three measured instances (2026-04-29, 2026-06-17, 2026-07-29) show it. The old +90
window re-admitted entries into the highest-volatility stretch of the month.

`max_spread_points: 100` and `risk_multiplier: 0.4` unchanged — §8.4 and §8.5 explain
why neither can be validated on this data.

### 9.3 `backend/src/skills/news/cpi.yaml`, `nfp.yaml`

```yaml
# cpi.yaml   window: { before_min: 5, after_min: 30 }   # was 20 / 45
# nfp.yaml   window: { before_min: 5, after_min: 60 }   # was 30 / 60
```

Also correct the factually wrong line in `nfp.yaml`'s header comment — "XAUUSD/XAGUSD
routinely see 5-15x normal spread" — to refer to **range**, which is what was measured
(NFP release minute 12.15 USD median vs 2.25 USD baseline = 5.4×; CPI 9.25×;
FOMC 23.0×). Recorded spread does not move at all in this feed (§3.1).

### 9.4 New `pce.yaml`, `ppi.yaml`, `retail_sales.yaml`

Generated from §8.1 via the existing `news-skill-gen` skill; windows −5/+45, −5/+20,
−5/+15 respectively. `close_all` on PCE only.

### 9.5 `generic_high_impact.yaml`

```yaml
window: { before_min: 5, after_min: 30 }   # was 15 / 30
```

and drop `pre_event.close_all` to `false` for this catch-all (§8.3): the events it
actually catches in our table (ISM class, retail class) peak at 1.15–2.96× and are
back to baseline within a minute or two. Named high-impact skills keep `close_all`.

### 9.6 Deliberately NOT proposed

- No change to any `max_spread_points` — §8.4, the data cannot support one.
- No change to any `risk_multiplier` — §8.5, n = 9.
- No news-trading strategy — §8.7, sign agreement is 5/9, p = 1.00.
- Nothing under `configs/risk.yaml` or `backend/src/engine/`, which are out of bounds.

---

## 10. What this report cannot tell you

Stated explicitly so the gaps are not mistaken for findings.

1. **Whether spread widens at CPI/NFP/FOMC.** The M1 feed says no (max 15 points); it
   is also structurally incapable of saying yes. The 8 tick-time observations we have
   cover only ISM Services and jobless claims. Needs a tick-resolution spread log
   across three high-impact events.
2. **Whether slippage worsens at releases.** All 9 high-impact-window trades predate
   the execution-telemetry columns; `slippage` is NULL for every one of them.
3. **Whether `snd_qm_structure` or `apex_trendguard` lose money in news windows.**
   Zero trades inside, out of 3,449. Needs the fleet to run through at least one full
   CPI/NFP/FOMC cycle with entries permitted, or an event-tagged backtest slice
   (`tag_index()` in `news_event_windows.py` exists for exactly this).
4. **Whether ISM Services really has no effect** or whether the third-business-day
   rule produced wrong dates. Needs the ISM release calendar.
5. **Any directional edge.** 9 instances. Needs ~65.
6. **Whether the news gate ever actually blocked an entry.** The gate's decisions are
   not distinguishable in `signal_decisions` (whose outcomes are `broker_rejected`,
   `opened`, `spread_veto`, `rr_gate`, `skipped`, `volatility_guard`, `htf_veto` — no
   `news_veto`), and the log only spans 1.5 days. Adding a `news_veto` outcome would
   make this measurable and is the cheapest instrumentation win available.
