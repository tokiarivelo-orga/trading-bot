"""Dated US macro-event table for XAUUSD, plus timestamp -> event-window tagging.

    from scripts.news_event_windows import EVENTS, tag_timestamp, tag_index

This is an *analysis* helper, not engine code and not part of the sandboxed
strategy path: it is imported by research scripts and by backtest slicers that
want to split results into "inside a news window" vs "outside". It deliberately
depends on nothing but the stdlib and pandas so it can be imported from a bare
`uv run python` session with no container wiring.

Times
-----
Every timestamp in `EVENTS` is timezone-aware **UTC**. The US was on EDT
(UTC-4) for the whole 2026-04..2026-09 range covered here, so:

    08:30 ET -> 12:30 UTC   (BLS / BEA / Census 8:30am releases)
    10:00 ET -> 14:00 UTC   (ISM)
    14:00 ET -> 18:00 UTC   (FOMC statement)
    14:30 ET -> 18:30 UTC   (FOMC press conference)

This alignment is not assumed, it is measured: the three largest one-minute
XAUUSD ranges in `data/trading.db` (2026-04-06..2026-08-07) fall exactly on
2026-07-14 12:30 (60.52 USD), 2026-07-02 12:30 (56.54) and 2026-07-29 18:00
(42.96) UTC — the June CPI, June NFP and July FOMC releases. Ranks 5 and 6 are
the June FOMC and June CPI. See NEWS_CORRELATION_REPORT.md section 1.2.

Verification
------------
`Event.verification` is one of:

    "verified"    the date came from the issuing agency's own schedule or
                  release document (BLS / BEA / Census / Federal Reserve).
    "rule"        generated from a deterministic publication rule (weekly
                  jobless claims are every Thursday 08:30 ET). Individual
                  dates are NOT confirmed and holiday weeks can shift.
    "unverified"  the date follows the usual publication pattern but no
                  source was found to confirm it.

Research that quotes a number should filter to `verification == "verified"`
(`events(verified_only=True)` does this). Never silently promote an
"unverified" row: a wrong date corrupts every downstream slice.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import pandas as pd

Verification = Literal["verified", "rule", "unverified"]
Impact = Literal["high", "medium", "low"]

# --------------------------------------------------------------------------
# Window sizes
# --------------------------------------------------------------------------
# Measured from XAUUSD M1 in data/trading.db over 2026-04-06..2026-08-07
# (120,471 bars, 43 events). `after` is the first-crossing point: scanning
# offset buckets forward from the release, the last bucket whose *median* M1
# range is still >= 1.25x its same-weekday, same-minute-of-day baseline.
# `before` is 5 for every kind (2 for the weekly claims) because there is no
# measured pre-release run-up at all — every kind sits at 0.79-1.15x baseline
# through -60..-2 min and *below* baseline in the final minute (0.58-0.99x).
# The 5 minutes are flatten lead-time, not a volatility buffer.
#
# Derivation, per-bucket ratios and sample sizes are in
# NEWS_CORRELATION_REPORT.md section 5. These replace the flat
# `before_min: 30 / after_min: 60` guess in configs/news.yaml.
#
# FOMC is the outlier and the reason this table exists: its median range stays
# >= 2x baseline until +149 min and has a *second* peak of 4.80x at +30..+59 —
# the 14:30 ET press conference, visible in all three measured instances. Any
# window shorter than ~180 min re-admits entries into the most violent stretch
# of the month.
MEASURED_WINDOWS: dict[str, tuple[int, int]] = {
    "CPI": (5, 30),
    "NFP": (5, 60),
    "FOMC": (10, 180),
    "PCE": (5, 45),
    "PPI": (5, 20),
    "RETAIL_SALES": (5, 15),
    "ISM_MFG": (5, 10),
    # No measurable signature at all (peak 1.15x). All three ISM_SERVICES rows
    # are `unverified`, so "no effect" and "wrong dates" are indistinguishable;
    # kept minimal rather than dropped. See NEWS_CORRELATION_REPORT.md 1.1.
    "ISM_SERVICES": (5, 5),
    "JOBLESS_CLAIMS": (2, 20),
}

# Used for any kind with no measured entry.
DEFAULT_WINDOW: tuple[int, int] = (5, 60)


@dataclass(frozen=True, slots=True)
class Event:
    """One scheduled macro release."""

    kind: str
    """Stable machine key, e.g. `"CPI"`. Matches `MEASURED_WINDOWS`."""

    label: str
    """Human name for reports."""

    at: datetime
    """Release instant, timezone-aware UTC."""

    impact: Impact
    verification: Verification
    source: str
    """Where the date came from — an agency URL or the rule that produced it."""

    @property
    def window(self) -> tuple[int, int]:
        """`(minutes_before, minutes_after)` for this event's kind."""
        return MEASURED_WINDOWS.get(self.kind, DEFAULT_WINDOW)

    @property
    def start(self) -> datetime:
        return self.at - timedelta(minutes=self.window[0])

    @property
    def end(self) -> datetime:
        return self.at + timedelta(minutes=self.window[1])


@dataclass(frozen=True, slots=True)
class EventTag:
    """The result of locating a timestamp inside an event window."""

    event: Event
    minutes_from_release: float
    """Signed: negative before the release, 0 at it, positive after."""

    @property
    def kind(self) -> str:
        return self.event.kind

    @property
    def phase(self) -> str:
        """Coarse bucket: `pre`, `impact` (0..15m), `digest` (15m..end)."""
        m = self.minutes_from_release
        if m < 0:
            return "pre"
        if m <= 15:
            return "impact"
        return "digest"


def _utc(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


_BLS = "https://www.bls.gov/schedule/news_release/"
_BEA = "https://www.bea.gov/news/schedule"
_CENSUS = "https://www.census.gov/retail/release_schedule.html"
_FED = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_ISM = "https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/"

# --------------------------------------------------------------------------
# The table. 2026 only. Extend by adding rows — never edit a "verified" row
# without re-checking the agency source named in `source`.
# --------------------------------------------------------------------------
_SCHEDULED: tuple[Event, ...] = (
    # --- Employment Situation / Non-Farm Payrolls, 08:30 ET -------------
    Event("NFP", "Non-Farm Payrolls (Apr data)", _utc(2026, 5, 8, 12, 30),
          "high", "verified", _BLS + "archives/empsit_05082026.htm"),
    Event("NFP", "Non-Farm Payrolls (May data)", _utc(2026, 6, 5, 12, 30),
          "high", "verified", _BLS + "archives/empsit_06052026.htm"),
    Event("NFP", "Non-Farm Payrolls (Jun data)", _utc(2026, 7, 2, 12, 30),
          "high", "verified", _BLS + "archives/empsit_07022026.htm"),
    Event("NFP", "Non-Farm Payrolls (Jul data)", _utc(2026, 8, 7, 12, 30),
          "high", "verified", _BLS),
    # --- CPI, 08:30 ET --------------------------------------------------
    Event("CPI", "US CPI (Mar data)", _utc(2026, 4, 10, 12, 30),
          "high", "verified", _BLS),
    Event("CPI", "US CPI (Apr data)", _utc(2026, 5, 12, 12, 30),
          "high", "verified", _BLS),
    Event("CPI", "US CPI (May data)", _utc(2026, 6, 10, 12, 30),
          "high", "verified", _BLS),
    Event("CPI", "US CPI (Jun data)", _utc(2026, 7, 14, 12, 30),
          "high", "verified", _BLS + "archives/cpi_07142026.htm"),
    Event("CPI", "US CPI (Jul data)", _utc(2026, 8, 12, 12, 30),
          "high", "verified", _BLS),
    Event("CPI", "US CPI (Aug data)", _utc(2026, 9, 11, 12, 30),
          "high", "verified", _BLS),
    # --- PPI, 08:30 ET --------------------------------------------------
    Event("PPI", "US PPI (Apr data)", _utc(2026, 5, 13, 12, 30),
          "medium", "verified", _BLS + "archives/ppi_05132026.htm"),
    Event("PPI", "US PPI (May data)", _utc(2026, 6, 11, 12, 30),
          "medium", "verified", _BLS + "archives/ppi_06112026.htm"),
    Event("PPI", "US PPI (Jun data)", _utc(2026, 7, 15, 12, 30),
          "medium", "verified", _BLS + "archives/ppi_07152026.htm"),
    Event("PPI", "US PPI (Jul data)", _utc(2026, 8, 13, 12, 30),
          "medium", "verified", _BLS),
    # --- PCE / Personal Income & Outlays, 08:30 ET ----------------------
    Event("PCE", "PCE / Personal Income (May data)", _utc(2026, 6, 25, 12, 30),
          "high", "verified", "https://www.bea.gov/news/2026/personal-income-and-outlays-may-2026"),
    Event("PCE", "PCE / Personal Income (Jun data)", _utc(2026, 7, 30, 12, 30),
          "high", "verified", "https://www.bea.gov/news/2026/personal-income-and-outlays-june-2026"),
    Event("PCE", "PCE / Personal Income (Jul data)", _utc(2026, 8, 26, 12, 30),
          "high", "verified", _BEA),
    Event("PCE", "PCE / Personal Income (Aug data)", _utc(2026, 9, 30, 12, 30),
          "high", "verified", _BEA),
    # --- Advance retail sales (Census MARTS), 08:30 ET ------------------
    Event("RETAIL_SALES", "US Retail Sales (Mar data)", _utc(2026, 4, 21, 12, 30),
          "medium", "verified", _CENSUS),
    Event("RETAIL_SALES", "US Retail Sales (Apr data)", _utc(2026, 5, 14, 12, 30),
          "medium", "verified", _CENSUS),
    Event("RETAIL_SALES", "US Retail Sales (May data)", _utc(2026, 6, 17, 12, 30),
          "medium", "verified", _CENSUS),
    Event("RETAIL_SALES", "US Retail Sales (Jun data)", _utc(2026, 7, 16, 12, 30),
          "medium", "verified", _CENSUS),
    Event("RETAIL_SALES", "US Retail Sales (Jul data)", _utc(2026, 8, 14, 12, 30),
          "medium", "verified", _CENSUS),
    Event("RETAIL_SALES", "US Retail Sales (Aug data)", _utc(2026, 9, 16, 12, 30),
          "medium", "verified", _CENSUS),
    # --- FOMC statement, 14:00 ET (press conference 30 min later) -------
    # Dates verified on the Fed calendar; the 14:00 ET statement time is the
    # standing convention and is confirmed for 2026-06-17 and 2026-07-29 by
    # the M1 volatility spikes landing exactly on 18:00 UTC.
    Event("FOMC", "FOMC statement (Apr meeting)", _utc(2026, 4, 29, 18, 0),
          "high", "verified", _FED),
    Event("FOMC", "FOMC statement + SEP (Jun meeting)", _utc(2026, 6, 17, 18, 0),
          "high", "verified", _FED),
    Event("FOMC", "FOMC statement (Jul meeting)", _utc(2026, 7, 29, 18, 0),
          "high", "verified", _FED),
    Event("FOMC", "FOMC statement + SEP (Sep meeting)", _utc(2026, 9, 16, 18, 0),
          "high", "verified", _FED),
    # --- ISM, 10:00 ET --------------------------------------------------
    Event("ISM_MFG", "ISM Manufacturing PMI (Jul data)", _utc(2026, 8, 3, 14, 0),
          "medium", "verified", _ISM),
    Event("ISM_MFG", "ISM Manufacturing PMI (Aug data)", _utc(2026, 9, 1, 14, 0),
          "medium", "verified", _ISM),
    # First business day of the month; not individually confirmed.
    Event("ISM_MFG", "ISM Manufacturing PMI (May data)", _utc(2026, 6, 1, 14, 0),
          "medium", "unverified", "first-business-day rule"),
    Event("ISM_MFG", "ISM Manufacturing PMI (Jun data)", _utc(2026, 7, 1, 14, 0),
          "medium", "unverified", "first-business-day rule"),
    # Third business day of the month; not individually confirmed. July 2026
    # shifts to the 6th because 2026-07-03 is the observed Independence Day.
    Event("ISM_SERVICES", "ISM Services PMI (May data)", _utc(2026, 6, 3, 14, 0),
          "medium", "unverified", "third-business-day rule"),
    Event("ISM_SERVICES", "ISM Services PMI (Jun data)", _utc(2026, 7, 6, 14, 0),
          "medium", "unverified", "third-business-day rule"),
    Event("ISM_SERVICES", "ISM Services PMI (Jul data)", _utc(2026, 8, 5, 14, 0),
          "medium", "unverified", "third-business-day rule"),
)

# Weekly initial jobless claims: DOL publishes every Thursday at 08:30 ET.
# Generated, not sourced — holiday weeks can shift. Kept as a large-n medium
# impact control group for the volatility study.
_CLAIMS_FIRST = _utc(2026, 4, 9, 12, 30)
_CLAIMS_LAST = _utc(2026, 9, 24, 12, 30)


def _jobless_claims() -> list[Event]:
    out: list[Event] = []
    t = _CLAIMS_FIRST
    while t <= _CLAIMS_LAST:
        out.append(
            Event(
                "JOBLESS_CLAIMS",
                "Initial Jobless Claims",
                t,
                "medium",
                "rule",
                "DOL weekly Thursday 08:30 ET rule",
            )
        )
        t += timedelta(days=7)
    return out


EVENTS: tuple[Event, ...] = tuple(
    sorted([*_SCHEDULED, *_jobless_claims()], key=lambda e: e.at)
)

HIGH_IMPACT_KINDS: frozenset[str] = frozenset({"CPI", "NFP", "FOMC", "PCE"})


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------
def events(
    *,
    kinds: Iterable[str] | None = None,
    verified_only: bool = False,
    impact: Impact | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Event]:
    """Filtered view of `EVENTS`, still sorted by release time."""
    wanted = set(kinds) if kinds is not None else None
    out = []
    for e in EVENTS:
        if wanted is not None and e.kind not in wanted:
            continue
        if verified_only and e.verification != "verified":
            continue
        if impact is not None and e.impact != impact:
            continue
        if start is not None and e.at < start:
            continue
        if end is not None and e.at > end:
            continue
        out.append(e)
    return out


def _as_utc(ts: datetime | pd.Timestamp | int | float) -> datetime:
    """Coerce a timestamp to aware UTC. Bare ints/floats are unix epoch seconds."""
    if isinstance(ts, int | float):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def tag_timestamp(
    ts: datetime | pd.Timestamp | int | float,
    *,
    pool: Sequence[Event] | None = None,
) -> EventTag | None:
    """Return the event window `ts` falls inside, or `None`.

    Windows can overlap (CPI and PPI on adjacent days do not, but an 08:30
    double-header does). When several match, the highest-impact one wins, and
    ties break toward the release nearest in time — so a trade at 12:31 on a
    day with both CPI and jobless claims is attributed to CPI.
    """
    pool = list(EVENTS) if pool is None else list(pool)
    when = _as_utc(ts)
    rank = {"high": 0, "medium": 1, "low": 2}
    best: EventTag | None = None
    for e in pool:
        if not (e.start <= when <= e.end):
            continue
        delta = (when - e.at).total_seconds() / 60.0
        if best is None:
            best = EventTag(e, delta)
            continue
        if (rank[e.impact], abs(delta)) < (rank[best.event.impact], abs(best.minutes_from_release)):
            best = EventTag(e, delta)
    return best


def in_any_window(
    ts: datetime | pd.Timestamp | int | float,
    *,
    kinds: Iterable[str] | None = None,
    verified_only: bool = False,
) -> bool:
    """True if `ts` falls in any (optionally filtered) event window."""
    return tag_timestamp(ts, pool=events(kinds=kinds, verified_only=verified_only)) is not None


def minutes_to_next_event(
    ts: datetime | pd.Timestamp | int | float,
    *,
    pool: Sequence[Event] | None = None,
) -> float | None:
    """Minutes until the next release at or after `ts`; `None` past the table end."""
    seq = list(EVENTS) if pool is None else sorted(pool, key=lambda e: e.at)
    when = _as_utc(ts)
    times = [e.at for e in seq]
    i = bisect_left(times, when)
    if i >= len(seq):
        return None
    return (seq[i].at - when).total_seconds() / 60.0


def tag_index(
    index: pd.DatetimeIndex | pd.Series,
    *,
    pool: Sequence[Event] | None = None,
) -> pd.DataFrame:
    """Vectorised tagging of many timestamps.

    Returns a frame aligned to `index` with columns `kind`, `label`,
    `minutes_from_release`, `phase`, `impact`, `verification`. Rows outside
    every window hold `NaN`/`None`. Timestamps are treated as UTC; a naive
    index is assumed to already be UTC.
    """
    seq = list(EVENTS) if pool is None else sorted(pool, key=lambda e: e.at)
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    nan = float("nan")
    frame = pd.DataFrame(
        {
            "kind": pd.Series([None] * len(idx), index=idx, dtype="object"),
            "label": pd.Series([None] * len(idx), index=idx, dtype="object"),
            "minutes_from_release": pd.Series([nan] * len(idx), index=idx, dtype="float64"),
            "phase": pd.Series([None] * len(idx), index=idx, dtype="object"),
            "impact": pd.Series([None] * len(idx), index=idx, dtype="object"),
            "verification": pd.Series([None] * len(idx), index=idx, dtype="object"),
        }
    )
    rank = {"high": 0, "medium": 1, "low": 2}
    # Assign lowest-priority kinds first so higher-impact events overwrite them.
    for e in sorted(seq, key=lambda x: -rank[x.impact]):
        lo = pd.Timestamp(e.start)
        hi = pd.Timestamp(e.end)
        mask = (idx >= lo) & (idx <= hi)
        if not mask.any():
            continue
        delta = (idx[mask] - pd.Timestamp(e.at)).total_seconds() / 60.0
        frame.loc[mask, "kind"] = e.kind
        frame.loc[mask, "label"] = e.label
        frame.loc[mask, "minutes_from_release"] = delta
        frame.loc[mask, "impact"] = e.impact
        frame.loc[mask, "verification"] = e.verification
        frame.loc[mask, "phase"] = pd.cut(
            pd.Series(delta, index=idx[mask]),
            bins=[-float("inf"), -0.0001, 15.0, float("inf")],
            labels=["pre", "impact", "digest"],
        ).astype(object)
    frame.index = pd.DatetimeIndex(index)
    return frame


def summary() -> str:
    """One-line-per-kind coverage summary — handy sanity check on import."""
    lines = []
    kinds = sorted({e.kind for e in EVENTS})
    for k in kinds:
        rows = [e for e in EVENTS if e.kind == k]
        v = sum(1 for e in rows if e.verification == "verified")
        r = sum(1 for e in rows if e.verification == "rule")
        u = sum(1 for e in rows if e.verification == "unverified")
        before, after = MEASURED_WINDOWS.get(k, DEFAULT_WINDOW)
        lines.append(
            f"{k:15s} n={len(rows):3d}  verified={v:3d} rule={r:3d} unverified={u:3d}  "
            f"window=-{before}m/+{after}m"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(summary())
