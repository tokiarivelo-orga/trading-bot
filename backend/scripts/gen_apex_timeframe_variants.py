"""Generate the M5/M15/H1/D1 siblings of xauusd_snd_apex_trendguard_m1_v5.py.

Run from `backend/`:

    uv run python -m scripts.gen_apex_timeframe_variants
    uv run python -m scripts.seed_xauusd_apex_strategies   # then re-seed

**The M1 file is the only one to edit.** Sandbox rules (CLAUDE.md) forbid a
generated strategy importing anything but math/statistics/numpy/pandas and
the domain models, so the siblings cannot share code with it — they have to
be full copies, the same shape the `xauusd_snd_qm_structure_*` family uses.
Edit the M1 file (bumping its own version, e.g. `_v6.py`), point `SRC` at
the new file, re-run this, re-seed.

To keep "full copy" from meaning "silently drifted copy", every edit below is
an exact-string substitution that MUST match exactly once; a shape change in
the M1 source fails the run instead of emitting a wrong file.
`tests/unit/strategies/test_xauusd_snd_apex_trendguard_timeframes.py` asserts
the invariant from the other side, comparing the emitted files' ASTs against
the M1 one, so a hand-edited sibling is caught by the suite.

Each run writes a brand-new `_v{N}.py` file per timeframe rather than
overwriting whatever is already on disk — `next_version_path()` looks at the
existing `xauusd_snd_apex_trendguard_{suffix}_v*.py` files and picks one
past the highest it finds. An already-written version file is somebody's
immutable, possibly-still-`ACTIVE`-in-the-DB historical record (see
`strategies/application/versioning.py`); overwriting it in place would leave
that DB row's `code_hash` pointing at code that no longer matches what's on
disk.
"""

from __future__ import annotations

import pathlib
import re
import textwrap

GEN = pathlib.Path(__file__).resolve().parent.parent / "src" / "strategies" / "generated"
SRC = GEN / "xauusd_snd_apex_trendguard_m1_v5.py"

# entry, class suffix, zone tf, htf zone tf, trend ladder, confirmation tfs,
# ema fast/slow, trend_min, max |score|, shock cooldown, vol lookback
VARIANTS = [
    {
        "entry": "M5",
        "suffix": "M5",
        "zone": "M15",
        "htf": "H1",
        "ladder": '{"M15": 1.0, "H1": 1.5, "H4": 2.0}',
        "confirm": '("M15", "H1", "H4")',
        "ema_fast": 21,
        "ema_slow": 55,
        "trend_min": 2.0,
        "max_score": 4.5,
        "shock_cooldown": 6,
        "vol_lookback": 100,
        "veto_tf": "M15",
        "span": "~16h of M15, ~8 days of H1 and ~33 days of H4",
        "warmup": None,
    },
    {
        "entry": "M15",
        "suffix": "M15",
        "zone": "H1",
        "htf": "H4",
        "ladder": '{"H1": 1.0, "H4": 1.5, "D1": 2.0}',
        "confirm": '("H1", "H4", "D1")',
        "ema_fast": 13,
        "ema_slow": 34,
        "trend_min": 2.0,
        "max_score": 4.5,
        "shock_cooldown": 4,
        "vol_lookback": 100,
        "veto_tf": "M30",
        "span": "~8 days of H1, ~33 days of H4 and ~200 sessions of D1",
        "warmup": (
            "The EMA pair drops to 13/34 (from the M1 file's 21/55) purely so "
            "the D1 vote is alive from the first replayed bar: a backtest only "
            "pre-loads `run_backtest.HISTORY_BUFFER` = 60 days of context, "
            "which is ~43 D1 bars, and a 55-span EMA needs 59. Live this makes "
            "no difference (the engine hands every declared timeframe 200 real "
            "bars), but a warm-up hole would have made the backtest measure a "
            "different bot than the one that trades."
        ),
    },
    {
        "entry": "H1",
        "suffix": "H1",
        "zone": "H4",
        "htf": "D1",
        "ladder": '{"H1": 1.0, "H4": 1.5, "D1": 2.0}',
        "confirm": '("H4", "D1")',
        "ema_fast": 13,
        "ema_slow": 34,
        "trend_min": 2.0,
        "max_score": 4.5,
        "shock_cooldown": 3,
        "vol_lookback": 100,
        "veto_tf": "H4",
        "span": "~33 days of H4 and ~200 sessions of D1",
        "warmup": (
            "The trend ladder starts at the entry timeframe itself rather than "
            "at the zone timeframe: above H4/D1 the only rungs left are W1 and "
            "MN, and a backtest's 60-day context buffer holds ~8 W1 bars, so a "
            "W1 vote would sit at 0 for most of any replay while being alive "
            "live. A vote that means different things in backtest and live is "
            "worse than no vote. The EMA pair is 13/34 for the same reason the "
            "M15 sibling uses it — D1 has only ~43 bars of pre-roll."
        ),
    },
    {
        "entry": "D1",
        "suffix": "D1",
        "zone": "D1",
        "htf": "W1",
        "ladder": '{"H4": 1.0, "D1": 2.0}',
        "confirm": '("H4", "W1")',
        "ema_fast": 13,
        "ema_slow": 34,
        "trend_min": 2.5,
        "max_score": 3.0,
        "shock_cooldown": 1,
        "vol_lookback": 60,
        "veto_tf": "W1",
        "span": "~33 days of H4 and ~4 years of W1",
        "warmup": (
            "Two departures from the finer siblings, both forced by being at "
            "the top of the ladder:\n\n"
            "  * `zone_timeframe` equals the entry timeframe. Every other "
            "sibling draws zones one rung up and triggers on a finer candle; "
            "here there is no finer candle above D1, so the daily bar both "
            "forms the zone and carries the rejection close. There is still no "
            "look-ahead — a zone is only usable from the bar after its leg-out "
            "completes, which `_zone_lifecycle` enforces by timestamp.\n\n"
            "  * The trend ladder is two votes (H4 + D1, max +/-3.0) and "
            "`trend_min_score` rises to 2.5 so the D1 vote alone is not enough "
            "— H4 has to agree. W1 stays out of the ladder because a replay's "
            "60-day context buffer holds ~8 W1 bars; it is used only as the "
            "higher-timeframe *zone* frame, where an empty list costs a "
            "confluence bonus and nothing else."
        ),
    },
]

HEADER = '''"""XAUUSD APEX — S&D + Quasimodo + Trend-Guard + Zone-Respect, {entry} entries.

The {entry} sibling of `xauusd_snd_apex_trendguard_m1_v5.py`. The algorithm is
that file byte-for-byte apart from the timeframe wiring below; every
threshold, gate and default still traces to the forensic post-mortem of the
2026-08-05 XAUUSD session (608 positions, 41.2% win rate, profit factor 0.73,
-$889.58 on the day) documented in full there, plus the v2-v5 calibration
passes on top of it (trend fatigue gate included but off by default — see
`fatigue_max`/`fatigue_fade_min` below). Read the M1 file for *why* each
number is what it is — this docstring only covers what changes.

Sandbox rules (CLAUDE.md) limit a generated strategy to math / statistics /
numpy / pandas, so it cannot import the M1 file and share the code; the
siblings are full copies produced by an exact-substitution generator that
fails rather than emit a drifted copy.

────────────────────────────────────────────────────────────────────────
TIMEFRAME LADDER
────────────────────────────────────────────────────────────────────────
    entry / trigger      {entry}
    zone detection       {zone}
    higher-TF zones      {htf}      (confluence + Quasimodo confirmation)
    trend votes          {ladder_tfs}

The engine hands every declared `confirmation_timeframes` entry its own
200-bar window (`trade_loop.DEFAULT_CONTEXT_BARS`), so these are native
candles, not a resample of the entry frame: {span}.

{warmup}────────────────────────────────────────────────────────────────────────
THE ADAPTIVE CORE: ZONE RESPECT INDEX (ZRI)
────────────────────────────────────────────────────────────────────────
Unchanged from the M1 file. The bot measures, separately for demand and for
supply, whether zones of that kind are actually holding *right now* on this
symbol: of the recently resolved zones of that kind, what fraction produced
a real reversal rather than being closed straight through. A kind whose ZRI
falls under `zri_min` is switched off until it earns its way back. Nothing
is hardcoded to a direction — it is a live read of which side of the book
the market is currently honouring.

Sandbox-safe: only math / numpy / pandas, no I/O, no broker access.
"""'''


def header_for(v: dict) -> str:
    ladder_tfs = ", ".join(
        f"{tf} (w {w})"
        for tf, w in (
            part.strip().split(": ")
            for part in v["ladder"].strip("{}").split(", ")
        )
    ).replace('"', "")
    warmup = ""
    if v["warmup"]:
        warmup = (
            "────────────────────────────────────────────────────────────────────────\n"
            "WHY THIS LADDER AND NOT THE OBVIOUS ONE\n"
            "────────────────────────────────────────────────────────────────────────\n"
            + "\n".join(
                textwrap.fill(para, 72) if not para.startswith("  *") else
                textwrap.fill(para, 72, initial_indent="", subsequent_indent="    ")
                for para in v["warmup"].split("\n\n")
            )
            + "\n\n"
        )
    return HEADER.format(
        entry=v["entry"],
        zone=v["zone"],
        htf=v["htf"],
        ladder_tfs=ladder_tfs,
        span=v["span"],
        warmup=warmup,
    )


def substitutions(v: dict) -> list[tuple[str, str]]:
    e, z, h, suf = v["entry"], v["zone"], v["htf"], v["suffix"]
    return [
        (
            "class XauusdSndApexTrendguardM1:",
            f"class XauusdSndApexTrendguard{suf}:",
        ),
        (
            '            name="xauusd_snd_apex_trendguard_m1",',
            f'            name="xauusd_snd_apex_trendguard_{suf.lower()}",',
        ),
        (
            '            entry_timeframe="M1",',
            f'            entry_timeframe="{e}",',
        ),
        (
            "            # Native higher-timeframe candles: the engine gives every entry\n"
            "            # here its own 200-bar window, so zones are found on real M5 and\n"
            "            # M15 bars instead of a 40-bar resample of the M1 context.\n"
            '            confirmation_timeframes=("M5", "M15", "H1"),',
            "            # Native higher-timeframe candles: the engine gives every entry\n"
            f"            # here its own 200-bar window, so zones are found on real {z}\n"
            f"            # and {h} bars instead of a resample of the {e} context.\n"
            f"            confirmation_timeframes={v['confirm']},",
        ),
        (
            "            # The engine's own veto is EMA20/50 on M5 only. This strategy",
            f"            # The engine's own veto is EMA20/50 on {v['veto_tf']} only. This",
        ),
        (
            "            # runs a strictly stronger three-timeframe gate plus the Zone",
            "            # strategy runs a strictly stronger multi-timeframe gate plus the",
        ),
        (
            "            # Respect Index internally, and records both in `reason`, so the",
            "            # Zone Respect Index internally, and records both in `reason`, so",
        ),
        (
            "            # decision stays in one auditable place. Flip it on per-bot with",
            "            # the decision stays in one auditable place. Flip it on per-bot",
        ),
        (
            "            # `htf_veto_override` in the skill YAML if you want belt and\n"
            "            # braces.",
            "            # with `htf_veto_override` in the skill YAML if you want belt\n"
            "            # and braces.",
        ),
        (
            '                "zone_timeframe": "M5",',
            f'                "zone_timeframe": "{z}",',
        ),
        (
            '                "htf_zone_timeframe": "M15",',
            f'                "htf_zone_timeframe": "{h}",',
        ),
        (
            '                "trend_ema_fast": 21,\n                "trend_ema_slow": 55,',
            f'                "trend_ema_fast": {v["ema_fast"]},\n'
            f'                "trend_ema_slow": {v["ema_slow"]},',
        ),
        (
            '                "trend_weights": {"M5": 1.0, "M15": 1.5, "H1": 2.0},\n'
            "                # Of a possible +/-4.5. Requires real multi-timeframe\n"
            "                # agreement, not one timeframe's opinion.\n"
            '                "trend_min_score": 2.0,',
            f'                "trend_weights": {v["ladder"]},\n'
            f"                # Of a possible +/-{v['max_score']}. Requires real multi-timeframe\n"
            "                # agreement, not one timeframe's opinion.\n"
            f'                "trend_min_score": {v["trend_min"]},',
        ),
        (
            "                # this floor the first working build still took 8 losing\n"
            "                # shorts into a +1.0 (mildly bullish) tape on 2026-08-05.",
            "                # this floor the first working build of the M1 sibling still\n"
            "                # took 8 losing shorts into a mildly bullish tape.",
        ),
        (
            '                "vol_lookback": 100,',
            f'                "vol_lookback": {v["vol_lookback"]},',
        ),
        (
            "                # ~the -5/+15 min blast radius where win rate degraded.\n"
            '                "shock_cooldown_bars": 15,',
            "                # The M1 sibling measured a -5/+15 minute blast radius\n"
            "                # around a range shock and sat out 15 M1 bars for it;\n"
            f"                # {v['shock_cooldown']} {e} bars is the same window here.\n"
            f'                "shock_cooldown_bars": {v["shock_cooldown"]},',
        ),
        # ── comment-level timeframe references ──
        (
            "# Entry confirmation on the M1 candle",
            f"# Entry confirmation on the {e} candle",
        ),
        (
            '    """Does the last M1 candle agree with the trade?',
            f'    """Does the last {e} candle agree with the trade?',
        ),
        (
            "                 zone. Cheap, keeps M1 frequency high. Default.",
            f"                 zone. Cheap, keeps {e} frequency high. Default.",
        ),
        (
            "    Confluence: an M5 RBR sitting inside an M15 demand block is a much\n"
            "    better trade than either alone.",
            f"    Confluence: a {z} RBR sitting inside a {h} demand block is a much\n"
            "    better trade than either alone.",
        ),
        (
            "                # already rejected was. 8 kills the spam without killing\n"
            "                # the M1 trade count.",
            "                # already rejected was. 8 kills the spam without killing\n"
            f"                # the {e} trade count.",
        ),
        (
            "                # In *entry*-timeframe bars. 3 killed essentially every zone\n"
            "                # within minutes (16,456 dead-zone rejections in one\n"
            "                # session); price legitimately works inside an M5-sized\n"
            "                # rectangle for a good while before resolving.",
            "                # In *entry*-timeframe bars. On the M1 sibling 3 killed\n"
            "                # essentially every zone within minutes (16,456 dead-zone\n"
            "                # rejections in one session); price legitimately works\n"
            f"                # inside a {z}-sized rectangle for a good while before\n"
            "                # resolving.",
        ),
        (
            "                # ── M1 entry confirmation ──\n"
            "                # Both thresholds are measured against the *zone*-timeframe\n"
            "                # ATR, so 0.35 of an M5 ATR is roughly a 0.8x M1-ATR body —\n"
            "                # a real rejection candle, not the bare touch the old family\n"
            "                # entered on.",
            f"                # ── {e} entry confirmation ──\n"
            "                # Both thresholds are measured against the *zone*-timeframe\n"
            f"                # ATR, so 0.35 of a {z} ATR is a real rejection candle on\n"
            f"                # the {e} chart, not the bare touch the old family entered\n"
            "                # on.",
        ),
        (
            "        # new zone-TF bar closes, so it is computed once per M5 bar instead\n"
            "        # of once per M1 bar. Without this a 100k-bar M1 replay would redo\n"
            "        # the whole 200-bar scan 100k times.",
            f"        # new zone-TF bar closes, so it is computed once per {z} bar instead\n"
            f"        # of once per {e} bar. Without this a long {e} replay would redo the\n"
            "        # whole 200-bar scan on every single bar.",
        ),
        (
            "        # a release or a rollover gap, no amount of zone quality makes an\n"
            "        # M1 entry a good idea.",
            "        # a release or a rollover gap, no amount of zone quality makes an\n"
            f"        # {e} entry a good idea.",
        ),
        (
            "        # ── Gate 4: the M1 candle must actually reject the zone ───────",
            f"        # ── Gate 4: the {e} candle must actually reject the zone "
            + "─" * max(0, 6 - len(e) + 1)
            + "──",
        ),
    ]


_VERSION_RE = re.compile(r"_v(\d+)\.py$")


def next_version_path(suffix: str) -> pathlib.Path:
    """Next free `xauusd_snd_apex_trendguard_{suffix}_v{N}.py` path — one
    past the highest version already on disk for this timeframe (0 if none
    exist yet), so a re-run never overwrites a prior version's file."""
    stem = f"xauusd_snd_apex_trendguard_{suffix.lower()}"
    existing = [
        int(match.group(1))
        for path in GEN.glob(f"{stem}_v*.py")
        if (match := _VERSION_RE.search(path.name))
    ]
    return GEN / f"{stem}_v{max(existing, default=0) + 1}.py"


def main() -> None:
    source = SRC.read_text()
    doc_end = source.index('"""', 3) + 3
    body = source[doc_end:]

    for v in VARIANTS:
        out = body
        for old, new in substitutions(v):
            count = out.count(old)
            if count != 1:
                raise SystemExit(
                    f"{v['suffix']}: anchor matched {count} times, expected 1:\n{old!r}"
                )
            out = out.replace(old, new)
        text = header_for(v) + out
        target = next_version_path(v["suffix"])
        target.write_text(text)
        print(f"wrote {target} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
