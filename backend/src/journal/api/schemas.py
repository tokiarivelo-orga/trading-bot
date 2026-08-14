"""Wire schema for the `/journal` HTTP API. Mirrors `journal/domain/models.py`
minus the market-context candle snapshots, which are AI-review-only and never
serialized over this API."""

from __future__ import annotations

from datetime import date as date_

from pydantic import BaseModel, Field


class ZoneOut(BaseModel):
    """The supply/demand rectangle the strategy identified before entering
    this trade, for drawing on the chart. Only present for strategies that
    report one (e.g. `pob_price_action_snd`)."""

    kind: str = Field(description="'demand' (buy zone) or 'supply' (sell zone).")
    price_low: float = Field(description="Lower bound of the zone rectangle.")
    price_high: float = Field(description="Upper bound of the zone rectangle.")
    time_start: int = Field(description="Epoch seconds UTC — left edge of the zone rectangle.")
    time_end: int = Field(
        description="Epoch seconds UTC — right edge of the zone rectangle (the entry candle)."
    )
    pattern: str | None = Field(
        default=None,
        description=(
            "The zone's own subtype, e.g. 'RBR', 'DBD', 'RBD', 'DBR' — distinct from the "
            "trade's own `pattern` (confirming candlestick pattern). Null for strategies "
            "that don't label their zone detector's setup type, or trades predating "
            "this field."
        ),
    )


class StructurePointOut(BaseModel):
    """A single labeled swing point from the window the strategy used to
    validate this trade's entry — chart annotation only, not used to gate
    the trade itself."""

    label: str = Field(description="Swing-structure label: 'HH', 'HL', 'LH', or 'LL'.")
    price: float = Field(description="Price of the swing high/low.")
    time: int = Field(description="Epoch seconds UTC of the swing bar.")


class IndicatorReadingOut(BaseModel):
    """A single confluence-check indicator value behind one of the bot's
    entry votes (e.g. RSI, ADX, EMA, Volume) — chart/journal annotation
    only, not used to gate the trade itself. `value` is the indicator's
    raw reading at the entry bar; `comparison` + `threshold` together with
    `value` explain why `passed` is true or false, e.g. `value=62.3`,
    `comparison='>'`, `threshold=50.0`, `passed=True` reads as "RSI 62.3
    is above 50, so this vote passed"."""

    name: str = Field(
        description="Indicator identifier, e.g. 'RSI', 'ADX', 'EMA_FAST_VS_SLOW', 'VOLUME'."
    )
    value: float = Field(
        description="The indicator's raw value at the entry bar (0.0 if not enough warmup)."
    )
    threshold: float = Field(
        description="The value `value` is compared against to decide pass/fail."
    )
    comparison: str = Field(
        description="'>' or '<' — the direction that makes this reading count as a pass."
    )
    passed: bool = Field(
        description="Whether `value` satisfied `comparison`/`threshold` at the entry bar."
    )


class TradeRecordOut(BaseModel):
    """One journaled trade — used both as a chart marker (`/markers`) and in
    the trade history list (`/trades`)."""

    id: str = Field(description="Broker position ticket, as a string.")
    symbol: str
    side: str = Field(description="'buy' or 'sell'.")
    volume: float
    open_price: float
    open_time: int = Field(description="Epoch seconds UTC.")
    sl: float | None
    tp: float | None
    close_price: float | None = Field(default=None, description="Null while the trade is open.")
    close_time: int | None = Field(
        default=None, description="Epoch seconds UTC; null while the trade is open."
    )
    profit: float | None = Field(default=None, description="Realized P/L; null while open.")
    close_reason: str | None = Field(
        default=None,
        description=(
            "Why the position was closed by the engine's position manager, e.g. "
            "'volatility guard: EXTREME regime while losing' or 'time-stop: no progress'. "
            "Null for normal SL/TP fills or manual/API closes, which don't set a reason."
        ),
    )
    comment: str = ""
    strategy_version: str | None = Field(
        default=None, description="e.g. 'breakout_v1:v1'; null for manually placed trades."
    )
    skill: str | None = Field(
        default=None, description="Bot skill that selected this trade, e.g. 'normal/xauusd'."
    )
    reason: str = Field(
        default="",
        description=(
            "Why the strategy took this trade, in its own words (`Signal.reason`) — the full, "
            "untruncated text; `comment` is the same reason truncated to MT5's 29-char comment "
            "limit. Empty for manually/API-placed trades."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description="Strategy's confidence in this signal, 0..1. Null for manual/API trades.",
    )
    zone: ZoneOut | None = Field(
        default=None,
        description="Supply/demand zone this trade was taken from, if the strategy reports one.",
    )
    pattern: str | None = Field(
        default=None,
        description="Confirming candlestick pattern, e.g. 'bullish_engulfing', if reported.",
    )
    structure: list[StructurePointOut] = Field(
        default_factory=list,
        description=(
            "Labeled swing points (HH/HL/LH/LL) from the window the strategy used to validate "
            "this trade's entry, for chart annotation."
        ),
    )
    indicators: list[IndicatorReadingOut] = Field(
        default_factory=list,
        description=(
            "The bot's confluence checklist at the entry bar — e.g. RSI vs 50, ADX vs its "
            "strong-trend threshold, fast EMA vs slow EMA, or last bar's volume vs its "
            "20-period average — one entry per indicator the bot voted on. Empty for trades "
            "from bots that don't report confluence data."
        ),
    )
    regime_volatility: str | None = Field(
        default=None,
        description=(
            "Volatility regime at entry — 'low'/'normal'/'high'/'extreme' — from the "
            "ATR-percentile classifier (OBSERVABILITY_PLAN.md Phase 6). Null for trades "
            "journaled before Phase 6, or whose entry timeframe had no candles to classify."
        ),
    )
    regime_volatility_percentile: float | None = Field(
        default=None, description="ATR percentile rank (0-100) behind `regime_volatility`."
    )
    regime_trend: str | None = Field(
        default=None,
        description="Trend/range regime at entry — 'trending' or 'ranging' — from the "
        "fixed-threshold ADX classifier. Null under the same conditions as `regime_volatility`.",
    )
    regime_adx: float | None = Field(
        default=None, description="Raw ADX reading (0-100) behind `regime_trend`."
    )
    regime_session: str | None = Field(
        default=None,
        description="Trading session at entry — 'asian'/'london'/'overlap'/'new_york'/"
        "'off_session' — bucketed off the UTC hour the signal fired.",
    )
    transaction_cost: float | None = Field(
        default=None,
        description=(
            "Spread + slippage cost of this fill, in account currency: "
            "(spread_points * point + slippage) * volume * contract_size. Null for trades "
            "journaled before Phase 6."
        ),
    )
    mfe: float | None = Field(default=None, description="Max favorable excursion.")
    mfe_time: int | None = Field(default=None, description="Time of MFE, epoch seconds UTC.")
    mae: float | None = Field(default=None, description="Max adverse excursion.")
    mae_time: int | None = Field(default=None, description="Time of MAE, epoch seconds UTC.")
    signal_id: str | None = Field(
        default=None,
        description=(
            "The signal_id of the SignalDecision that led to this trade, joinable against "
            "/activity/... and the order-book snapshot for the same signal; null for trades "
            "opened before this field existed or via manual/API entry."
        ),
    )


class CandleOut(BaseModel):
    """One OHLC candle from a trade's frozen entry-time market-context
    snapshot (see `DecisionContextOut`) — plots directly as a
    `lightweight-charts` candlestick series point."""

    time: int = Field(description="Epoch seconds UTC — candle open time.")
    open: float = Field(description="Open price.")
    high: float = Field(description="High price.")
    low: float = Field(description="Low price.")
    close: float = Field(description="Close price.")
    tick_volume: int = Field(description="Broker tick volume for this candle.")


class DecisionContextOut(BaseModel):
    """The chart snapshot and decision annotations behind one trade — backs
    the "why did the bot take this trade" chart view
    (`GET .../trades/{trade_id}/decision-context`). `entry_candles` and
    `higher_tf_candles` are a **frozen snapshot captured once, at the moment
    of entry** (via the `PositionOpened` event) — not live/refetched market
    data — so this renders identically no matter how much later it's
    viewed, even after the symbol's live candle history has aged the
    original bars out. Never includes exit-time snapshots or any other
    AI-review-only data."""

    trade_id: str = Field(description="Broker position ticket, as a string.")
    symbol: str = Field(description="Broker symbol, e.g. 'XAUUSD'.")
    side: str = Field(description="'buy' or 'sell'.")
    open_price: float = Field(description="Fill price at entry.")
    open_time: int = Field(description="Epoch seconds UTC — when the trade was opened.")
    entry_candles: list[CandleOut] = Field(
        description=(
            "The M5 candle snapshot (typically 50 candles) captured once, at the moment of "
            "entry. A frozen snapshot, not live/refetched data — stays accurate even after the "
            "symbol's live candle history has since aged these bars out. Empty only for very "
            "early trades journaled before enough candle history existed."
        )
    )
    higher_tf_candles: list[CandleOut] = Field(
        description=(
            "The H1 candle snapshot (typically 20 candles) captured once, at the moment of "
            "entry, for higher-timeframe context alongside `entry_candles`. Same frozen-snapshot "
            "caveat: not live/refetched data."
        )
    )
    zone: ZoneOut | None = Field(
        default=None,
        description="Supply/demand zone this trade was taken from, if the strategy reports one.",
    )
    pattern: str | None = Field(
        default=None,
        description="Confirming candlestick pattern, e.g. 'bullish_engulfing', if reported.",
    )
    structure: list[StructurePointOut] = Field(
        default_factory=list,
        description=(
            "Labeled swing points (HH/HL/LH/LL) from the window the strategy used to validate "
            "this trade's entry, for chart annotation."
        ),
    )
    indicators: list[IndicatorReadingOut] = Field(
        default_factory=list,
        description=(
            "The bot's confluence checklist at the entry bar — one entry per indicator the bot "
            "voted on. Empty for trades from bots that don't report confluence data."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Why the strategy took this trade, in its own words (`Signal.reason`). Empty for "
            "manually/API-placed trades."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description="Strategy's confidence in this signal, 0..1. Null for manual/API trades.",
    )


class TradeHistoryPage(BaseModel):
    """One page of the filtered trade history (`GET /journal/history`)."""

    items: list[TradeRecordOut] = Field(description="Trades matching the filters, one page.")
    total: int = Field(description="Total number of trades matching the filters, across all pages.")
    total_profit: float = Field(
        default=0.0, description="Total net profit matching the filters, across all pages."
    )


class SymbolAnalyticsOut(BaseModel):
    """Aggregate performance of every trade (any bot, or manual) on one
    symbol — one entry per symbol on `GET /journal/analytics/symbols`."""

    symbol: str = Field(description="Broker symbol, e.g. 'XAUUSD'.")
    trade_count: int = Field(description="Total trades (open + closed) on this symbol.")
    open_count: int = Field(description="Currently open trades on this symbol.")
    closed_count: int = Field(description="Closed trades on this symbol.")
    win_count: int = Field(description="Closed trades with profit > 0.")
    loss_count: int = Field(description="Closed trades with profit < 0.")
    breakeven_count: int = Field(description="Closed trades with profit == 0.")
    win_rate: float = Field(description="win_count / closed_count, 0..1. 0 if no closed trades.")
    total_profit: float = Field(description="Sum of realized profit across closed trades.")
    gross_profit: float = Field(description="Sum of profit across winning trades only.")
    gross_loss: float = Field(
        description="Sum of |profit| across losing trades only, as a positive number."
    )
    profit_factor: float | None = Field(
        description="gross_profit / gross_loss. Null when there are no losing trades yet "
        "(undefined rather than infinite)."
    )
    avg_win: float = Field(description="gross_profit / win_count. 0 if no wins.")
    avg_loss: float = Field(
        description="gross_loss / loss_count, as a positive number. 0 if no losses."
    )
    avg_profit_per_trade: float = Field(
        description="total_profit / closed_count. 0 if no closed trades."
    )
    largest_win: float = Field(description="Largest single-trade profit, or 0 if no wins.")
    largest_loss: float = Field(
        description="Largest single-trade loss (negative), or 0 if no losses."
    )
    total_volume: float = Field(description="Sum of lot volume across all trades on this symbol.")
    bot_count: int = Field(
        description="Number of distinct bots (skills) that have traded this symbol."
    )
    first_trade_time: int | None = Field(description="Earliest open_time, epoch seconds UTC.")
    last_trade_time: int | None = Field(description="Latest open_time, epoch seconds UTC.")


class EquityPointOut(BaseModel):
    """One step of a bot's cumulative-profit curve, ordered by close time —
    plots directly as a `lightweight-charts` line series."""

    trade_id: str = Field(description="Broker position ticket, as a string.")
    close_time: int = Field(description="Epoch seconds UTC.")
    profit: float = Field(description="This trade's realized profit.")
    cumulative_profit: float = Field(
        description="Running sum of profit up to and including this trade."
    )


class RetcodeCountOut(BaseModel):
    """One bucket of a bot's broker-return-code histogram."""

    retcode: int = Field(
        description="Broker return code as reported on the fill — MT5 10009 means the deal "
        "completed; 10016 is 'invalid stops', 10014 'invalid volume', and so on."
    )
    count: int = Field(description="How many of this bot's fills reported that code.")


class BotAnalyticsOut(BaseModel):
    """Aggregate performance of one bot (skill), plus its equity curve — one
    entry per bot on `GET /journal/analytics/bots`. Trades placed manually
    or via the API (no `skill`) are excluded, since they aren't
    attributable to any bot."""

    skill: str = Field(description="Bot's full id, e.g. 'normal/xauusd/breakout_v1'.")
    bot_name: str = Field(description="This bot's short id — the last segment of `skill`.")
    symbol: str = Field(description="Broker symbol this bot trades.")
    strategy_version: str | None = Field(
        description="This bot's most recent trade's strategy version, e.g. 'breakout_v1:v1'."
    )
    trade_count: int = Field(description="Total trades (open + closed) placed by this bot.")
    open_count: int = Field(description="Currently open trades placed by this bot.")
    closed_count: int = Field(description="Closed trades placed by this bot.")
    win_count: int = Field(description="Closed trades with profit > 0.")
    loss_count: int = Field(description="Closed trades with profit < 0.")
    breakeven_count: int = Field(description="Closed trades with profit == 0.")
    win_rate: float = Field(description="win_count / closed_count, 0..1. 0 if no closed trades.")
    total_profit: float = Field(description="Sum of realized profit across closed trades.")
    gross_profit: float = Field(description="Sum of profit across winning trades only.")
    gross_loss: float = Field(
        description="Sum of |profit| across losing trades only, as a positive number."
    )
    profit_factor: float | None = Field(
        description="gross_profit / gross_loss. Null when there are no losing trades yet "
        "(undefined rather than infinite)."
    )
    avg_win: float = Field(description="gross_profit / win_count. 0 if no wins.")
    avg_loss: float = Field(
        description="gross_loss / loss_count, as a positive number. 0 if no losses."
    )
    expectancy: float = Field(
        description="total_profit / closed_count — average profit per closed trade."
    )
    largest_win: float = Field(description="Largest single-trade profit, or 0 if no wins.")
    largest_loss: float = Field(
        description="Largest single-trade loss (negative), or 0 if no losses."
    )
    max_drawdown: float = Field(
        description="Largest peak-to-trough drop in cumulative profit across this bot's "
        "equity curve, as a positive number. 0 if the curve never fell below a prior peak."
    )
    avg_trade_duration_seconds: float | None = Field(
        description="Average close_time - open_time across closed trades. Null if no closed trades."
    )
    first_trade_time: int | None = Field(description="Earliest open_time, epoch seconds UTC.")
    last_trade_time: int | None = Field(description="Latest open_time, epoch seconds UTC.")
    equity_curve: list[EquityPointOut] = Field(
        description="Cumulative-profit curve over this bot's closed trades, oldest first."
    )
    avg_slippage: float | None = Field(
        description=(
            "Average execution slippage in price units, signed so a POSITIVE number means "
            "the fills cost the trader (bought higher / sold lower than the price the order "
            "asked for). Null when none of this bot's trades carry the measurement — trades "
            "journaled before execution telemetry existed are skipped, not counted as zero."
        )
    )
    measured_slippage_count: int = Field(
        description=(
            "Number of this bot's trades `avg_slippage` averages over — the denominator, so a "
            "large-looking average taken from two fills can be read as the noise it is."
        )
    )
    avg_execution_latency_ms: float | None = Field(
        description=(
            "Average milliseconds from the strategy emitting the signal to the broker "
            "acknowledging the fill. Null when unmeasured (manual/API trades carry no signal)."
        )
    )
    retcode_histogram: list[RetcodeCountOut] = Field(
        description=(
            "Broker return codes across this bot's fills, most frequent first. MT5 10009 is a "
            "clean deal; a recurring other code (e.g. 10016 invalid stops) is a systematic "
            "execution problem, not bad luck. Empty when no fill reported a code."
        )
    )
    avg_mfe: float | None = Field(
        description=(
            "Average maximum favorable excursion in price units over closed trades — how far "
            "the market moved in the trade's favor from entry, at best. Null if unmeasured."
        )
    )
    avg_mae: float | None = Field(
        description=(
            "Average maximum adverse excursion in price units over closed trades — how far "
            "the market moved against the trade from entry, at worst. Null if unmeasured."
        )
    )
    mfe_mae_ratio: float | None = Field(
        description=(
            "avg_mfe / avg_mae. Above 1 means trades generally run further in favor than "
            "against. Null when avg_mae is zero or unmeasured."
        )
    )
    avg_mfe_on_losers: float | None = Field(
        description=(
            "Average MFE across this bot's LOSING closed trades — how far in profit a loser "
            "typically got before dying. Large relative to `avg_win` means the take-profit is "
            "parked beyond where price actually turns. Null if no measured losers."
        )
    )
    avg_mae_on_winners: float | None = Field(
        description=(
            "Average MAE across this bot's WINNING closed trades — how much heat a winner "
            "typically took. Approaching the bot's stop distance means stops are as tight as "
            "they can get before winners start being stopped out. Null if no measured winners."
        )
    )
    total_transaction_cost: float | None = Field(
        description=(
            "Sum of spread + slippage cost (account currency) across this bot's closed, "
            "measured trades. Null when none of this bot's trades carry the measurement."
        )
    )
    avg_transaction_cost_per_trade: float | None = Field(
        description="`total_transaction_cost` divided by how many closed trades it sums. "
        "Null when unmeasured."
    )
    cost_pct_of_gross_edge: float | None = Field(
        description=(
            "Fraction of this bot's gross edge (realized profit before costs) spent on "
            "spread + slippage: total_transaction_cost / (total_profit + "
            "total_transaction_cost). Near or above 1.0 means the bot is spending its whole "
            "edge on execution cost — the headline question for M1 scalps. Null when "
            "unmeasured, or when gross edge isn't positive (undefined, not a divide-by-zero "
            "artifact — mirrors `profit_factor`'s null convention)."
        )
    )


class RegimeAnalyticsOut(BaseModel):
    """One bot's outcome stats within one bucket of one regime dimension —
    one entry per (bot, dimension, bucket) on `GET
    /journal/analytics/regimes`. The same win/PF/expectancy shape
    `BotAnalyticsOut` reports overall, sliced one regime dimension at a
    time (volatility, trend, or session) so a bot's edge in a specific
    market condition isn't averaged away by every other condition it also
    traded through."""

    skill: str = Field(description="Bot's full id, e.g. 'normal/xauusd/breakout_v1'.")
    bot_name: str = Field(description="This bot's short id — the last segment of `skill`.")
    dimension: str = Field(
        description="Which regime axis this bucket is sliced on: 'volatility', 'trend', or "
        "'session'."
    )
    bucket: str = Field(
        description=(
            "The bucket value within `dimension` — e.g. 'low'/'normal'/'high'/'extreme' for "
            "volatility, 'trending'/'ranging' for trend, 'asian'/'london'/'overlap'/"
            "'new_york'/'off_session' for session."
        )
    )
    trade_count: int = Field(description="Total trades (open + closed) in this bucket.")
    closed_count: int = Field(description="Closed trades in this bucket.")
    win_count: int = Field(description="Closed trades with profit > 0.")
    loss_count: int = Field(description="Closed trades with profit < 0.")
    win_rate: float = Field(description="win_count / closed_count, 0..1. 0 if no closed trades.")
    profit_factor: float | None = Field(
        description="gross_profit / gross_loss within this bucket. Null when there are no "
        "losing trades yet (undefined rather than infinite)."
    )
    expectancy: float = Field(
        description="total_profit / closed_count within this bucket — average profit per "
        "closed trade in this regime."
    )
    total_profit: float = Field(
        description="Sum of realized profit across closed trades in this bucket."
    )


class DailyPnlOut(BaseModel):
    """One trading day's realized P&L — one entry per calendar date with at
    least one trade closed that day, on `GET /journal/analytics/daily`.
    Sourced by aggregating the durable `trades` table at read time, not a
    persisted table of its own — the historical counterpart to
    `engine/application/risk_manager.py`'s in-memory `_daily_pnl`, which
    only drives the live daily-loss circuit breaker and is never persisted."""

    date: date_ = Field(description="The trading day, UTC, serialized as 'YYYY-MM-DD'.")
    pnl: float = Field(description="Sum of realized profit across trades closed this day.")
    trade_count: int = Field(description="Trades closed this day.")
    win_count: int = Field(description="Closed trades with profit > 0.")
    loss_count: int = Field(description="Closed trades with profit < 0.")
    breakeven_count: int = Field(description="Closed trades with profit == 0.")
    win_rate: float = Field(description="win_count / trade_count, 0..1. 0 if no trades.")
    gross_profit: float = Field(description="Sum of profit across winning trades only.")
    gross_loss: float = Field(
        description="Sum of |profit| across losing trades only, as a positive number."
    )
    profit_factor: float | None = Field(
        description="gross_profit / gross_loss for this day. Null when there are no losing "
        "trades that day (undefined rather than infinite)."
    )
    avg_win: float = Field(description="gross_profit / win_count. 0 if no wins.")
    avg_loss: float = Field(
        description="gross_loss / loss_count, as a positive number. 0 if no losses."
    )
    largest_win: float = Field(description="Largest single-trade profit this day, or 0.")
    largest_loss: float = Field(description="Largest single-trade loss this day (negative), or 0.")


class DatasetOrderBookLevelOut(BaseModel):
    """One price level of the order-book snapshot captured at the moment a
    trade's signal fired (`order_book/adapters/repository.py`)."""

    side: str = Field(description="'bid' or 'ask'.")
    price: float = Field(description="Price at this level.")
    volume: float = Field(description="Volume/size resting at this level.")


class DatasetRowOut(BaseModel):
    """One training example on `GET .../journal/export/dataset` (`format=json`):
    a closed trade joined to the SMC v2 feature vector (`strategies/generated/
    smc_dl_features_v2.py`) computed at its entry bar, the v1 triple-barrier
    label computed over the same candle history, and every enrichment field
    the dataset export can attach. Fields sourced from data this trade simply
    doesn't have (no `signal_id`, no regime tags, no captured order-book
    depth) are null rather than fabricated — see each field's description."""

    trade_id: str = Field(description="Broker position ticket, as a string.")
    symbol: str = Field(description="Broker symbol, e.g. 'XAUUSD'.")
    strategy_version: str | None = Field(
        description="e.g. 'breakout_v1:v1'; null for manually placed trades."
    )
    timestamp_entry: str = Field(description="ISO-8601 open time.")
    timestamp_exit: str | None = Field(description="ISO-8601 close time; null while open.")

    tp_prob: float = Field(
        description="TP-hit probability parsed from the strategy's logged `reason` string "
        "(e.g. 'DL Buy (tp=0.85, ...)'). 0.0 when the reason carries no such probability."
    )
    bull_prob: float = Field(description="Bullish-model probability parsed the same way.")
    bear_prob: float = Field(description="Bearish-model probability parsed the same way.")
    model_decision: str = Field(description="'BUY', 'SELL', or 'SKIP' — parsed from `reason`.")

    actual_direction: int = Field(
        description="Direction the v1 triple-barrier label resolved to at the entry bar: "
        "1 (long TP hit first), -1 (short TP hit first), 0 (neither)."
    )
    hit_tp_before_sl: int = Field(
        description="1 if the v1 triple-barrier label's take-profit was reached before its "
        "stop-loss from the entry bar, else 0."
    )
    profit_r: float | None = Field(
        description=(
            "This trade's realized profit expressed as a multiple of its initial risk: "
            "profit / (|open_price - sl| * volume * contract_size) — the same formula "
            "`backtest/adapters/bookkeeper.py` uses to compute `BacktestTrade.r_multiple`, "
            "reused here so live and backtest R-multiples mean the same thing. Null when "
            "`sl` is null (risk undefined) or this symbol's contract size was never synced "
            "into `symbol_specs` (see `POST /market-data/backfill`)."
        )
    )
    mfe_atr: float = Field(
        description="Max favorable excursion from the v1 label's forward barrier walk, in "
        "ATR units (distinct from `TradeRecord.mfe`, which is in price units)."
    )
    mae_atr: float = Field(
        description="Max adverse excursion from the v1 label's forward barrier walk, in ATR "
        "units (distinct from `TradeRecord.mae`, which is in price units)."
    )
    bars_to_exit: int = Field(
        description="M5 bars from the entry bar to the label's resolution (TP/SL hit, or the "
        "labeling horizon if neither)."
    )

    entry_price: float = Field(description="Fill price at entry.")
    exit_price: float | None = Field(default=None, description="Fill price at close.")
    sl: float | None = Field(default=None, description="Stop-loss price at entry.")
    tp: float | None = Field(default=None, description="Take-profit price at entry.")
    spread: int = Field(description="Spread in points at entry.")
    slippage: float | None = Field(
        default=None, description="Execution slippage in price units; see `TradeRecordOut`."
    )
    profit_raw: float | None = Field(default=None, description="Realized P/L in account currency.")
    volume: float = Field(description="Lot size.")

    model_correct: int = Field(
        description="1 if `model_decision` agrees with `actual_direction`, else 0."
    )
    confidence_calibration: float = Field(
        description="tp_prob - hit_tp_before_sl — how far the logged TP-probability was from "
        "the label's realized outcome; near 0 is well-calibrated."
    )

    regime_volatility: str | None = Field(default=None, description="See `TradeRecordOut`.")
    regime_volatility_percentile: float | None = Field(
        default=None, description="See `TradeRecordOut`."
    )
    regime_trend: str | None = Field(default=None, description="See `TradeRecordOut`.")
    regime_adx: float | None = Field(default=None, description="See `TradeRecordOut`.")
    regime_session: str | None = Field(default=None, description="See `TradeRecordOut`.")
    transaction_cost: float | None = Field(default=None, description="See `TradeRecordOut`.")
    signal_id: str | None = Field(default=None, description="See `TradeRecordOut`.")

    real_volume: int | None = Field(
        default=None,
        description="Actual traded volume on the M5 candle the entry features were computed "
        "from. Null when the broker never reported it for this bar (see `market_data.domain."
        "models.Candle.real_volume`).",
    )
    atr_14: float | None = Field(
        default=None,
        description="Trailing 14-period ATR stored on that same M5 candle. Null when the "
        "enrichment pass hadn't reached this bar yet.",
    )
    day_of_week: int | None = Field(
        default=None, description="UTC day of week on that candle, 0=Monday..6=Sunday."
    )

    order_book_levels: list[DatasetOrderBookLevelOut] | None = Field(
        default=None,
        description=(
            "Market-depth levels captured at the moment this trade's signal fired, if any. "
            "Null whenever `signal_id` is null, or when a snapshot was never captured for it "
            "— the normal, expected case for most symbols (order-book capture gracefully "
            "degrades to storing nothing for OTC/synthetic symbols that report no depth). "
            "Never fabricated or empty-but-present; see `order_book/domain/models.py`."
        ),
    )

    features: dict[str, float] = Field(
        description=(
            "The scale-free SMC v2 features at the entry bar, keyed by name — see "
            "`strategies.generated.smc_dl_features_v2.FEATURE_NAMES` for the full, canonical "
            "list and what each one means. Not enumerated as individual schema fields since "
            "the feature set is versioned in that module, not here."
        )
    )
