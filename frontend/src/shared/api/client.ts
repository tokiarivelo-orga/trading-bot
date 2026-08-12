/** Thin fetch wrapper for the backend REST API (proxied via /api, see next.config.ts). */

const BASE = "/api";
const TOKEN_KEY = "tb.session";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// ── Session token (§11) ─────────────────────────────────────────────────────
// Token-based (not cookie-based) so the same token works whether a request
// goes through the Next.js /api rewrite or hits the backend directly (Socket.IO
// — see ws.ts), with no cross-origin cookie handling to worry about.

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

/** Dispatched on any 401 from a non-/auth/ request, so the UI can re-lock
 * after a session expires mid-use. See features/auth/LoginGate.tsx. */
export const UNAUTHORIZED_EVENT = "tb:unauthorized";

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function errorMessage(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Not JSON (or no `detail` field) — fall through to the raw text below.
  }
  return text;
}

function handleUnauthorized(path: string, status: number): void {
  if (status !== 401 || path.startsWith("/auth/") || path.endsWith("/account/connect")) return;
  clearToken();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeaders() },
    ...init,
  });
  if (!res.ok) {
    handleUnauthorized(path, res.status);
    throw new ApiError(res.status, await errorMessage(res));
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

async function requestForm<T>(path: string, method: string, form: FormData): Promise<T> {
  // No Content-Type header here on purpose — the browser sets the multipart
  // boundary itself when the body is a FormData.
  const res = await fetch(`${BASE}${path}`, { method, body: form, headers: authHeaders() });
  if (!res.ok) {
    handleUnauthorized(path, res.status);
    throw new ApiError(res.status, await errorMessage(res));
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>(path, signal ? { signal } : undefined),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  postForm: <T>(path: string, form: FormData) => requestForm<T>(path, "POST", form),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

// ── Auth (§11) ───────────────────────────────────────────────────────────────

export interface AuthStatus {
  auth_required: boolean;
}

export const getAuthStatus = () => api.get<AuthStatus>("/auth/status");
export const login = async (password: string) => {
  const res = await api.post<{ token: string; expires_in_seconds: number }>("/auth/login", {
    password,
  });
  setToken(res.token);
  return res;
};
export const logout = async () => {
  clearToken();
  try {
    await api.post("/auth/logout");
  } catch {
    // Logout is best-effort client-side (token already cleared above).
  }
};

export interface AppConfig {
  mode: "paper" | "live";
  symbols: string[];
  engine: { enabled: boolean; entry_timeframe: string };
}

export const getHealth = () => api.get<{ status: string }>("/health");
export const getAppConfig = () => api.get<AppConfig>("/config/app");

// ── Accounts (multi-account switcher, Phase 8) ──────────────────────────────

export interface AccountSummary {
  id: string;
  label: string;
  mode: "paper" | "live";
  enabled: boolean;
}

/** Every enabled account from `configs/accounts.yaml` — the source of every
 * other call's `accountId` param and the account switcher's own list. Global,
 * unprefixed — called before any account is known. */
export const getAccounts = () => api.get<AccountSummary[]>("/accounts");

function acctPath(accountId: string, path: string): string {
  return `/accounts/${encodeURIComponent(accountId)}${path}`;
}

// ── Account (MT5 login, F11) ────────────────────────────────────────────────

export interface AccountInfo {
  login: number;
  server: string;
  name: string;
  currency: string;
  balance: number;
  equity: number;
  leverage: number;
}

export interface AccountStatus {
  gateway_up: boolean;
  connected: boolean;
  account: AccountInfo | null;
  has_saved_credentials: boolean;
}

export const getAccountStatus = (accountId: string) =>
  api.get<AccountStatus>(acctPath(accountId, "/account/status"));
export const connectAccount = (
  accountId: string,
  body: { login: number; password: string; server: string; remember: boolean },
) =>
  api.post<{ connected: boolean; account: AccountInfo }>(
    acctPath(accountId, "/account/connect"),
    body,
  );
export const disconnectAccount = (accountId: string, forget = false) =>
  api.post<{ connected: boolean }>(acctPath(accountId, "/account/disconnect"), { forget });

// ── Market data ─────────────────────────────────────────────────────────────

export interface Candle {
  symbol: string;
  timeframe: "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1" | "W1" | "MN";
  time: number; // bar open, epoch seconds UTC (lightweight-charts native)
  open: number;
  high: number;
  low: number;
  close: number;
  tick_volume: number;
  spread_points: number;
}

/** `before` (epoch seconds) pages further back than the most recent `count`
 * bars — pass the oldest loaded candle's `time` to fetch older history. */
export const getCandles = (
  accountId: string,
  symbol: string,
  timeframe: Candle["timeframe"],
  count = 300,
  before?: number,
  signal?: AbortSignal,
) => {
  const params = new URLSearchParams({ symbol, timeframe, count: String(count) });
  if (before !== undefined) params.set("before", String(before));
  return api.get<Candle[]>(acctPath(accountId, `/market-data/candles?${params}`), signal);
};

/** One stretch of bars missing from stored candle history. The chart draws
 * straight across a hole like this, and every indicator, zone detector and
 * backtest over the window treats the bars either side as adjacent when they
 * can be hours apart — see GET /market-data/candle-gaps. */
export interface CandleGap {
  start: number; // open time the first missing bar would have had, epoch seconds
  end: number; // open time of the first bar present after the hole
  missing_bars: number;
  /** Hole that fits inside a normal weekend closure — expected, not damage.
   * `repairCandleGaps` skips these unless `include_weekend` is set. */
  weekend: boolean;
}

export interface CandleGapScan {
  symbol: string;
  timeframe: Candle["timeframe"];
  start: number;
  end: number;
  gaps: CandleGap[];
  missing_bars: number;
}

/** Read-only scan of stored history for `[start, end]` (epoch seconds, both
 * inclusive) — typically the chart's currently loaded window. */
export const getCandleGaps = (
  accountId: string,
  symbol: string,
  timeframe: Candle["timeframe"],
  start: number,
  end: number,
  signal?: AbortSignal,
) => {
  const params = new URLSearchParams({
    symbol,
    timeframe,
    start: String(start),
    end: String(end),
  });
  return api.get<CandleGapScan>(
    acctPath(accountId, `/market-data/candle-gaps?${params}`),
    signal,
  );
};

export interface GapRepairResult {
  symbol: string;
  timeframe: Candle["timeframe"];
  found: CandleGap[];
  repaired: CandleGap[];
  /** Holes the broker itself cannot fill (holiday, halt, symbol listed
   * later) — retrying will not change them. */
  remaining: CandleGap[];
  bars_downloaded: number;
  bars_recovered: number;
}

/** Re-downloads every hole in `[start, end]` from the broker and reports
 * which ones actually closed. Safe to call repeatedly — bars are overwritten
 * in place, never duplicated. */
export const repairCandleGaps = (
  accountId: string,
  body: {
    symbol: string;
    timeframe: Candle["timeframe"];
    start: string; // ISO 8601, inclusive
    end: string; // ISO 8601, inclusive
    include_weekend?: boolean;
  },
) => api.post<GapRepairResult>(acctPath(accountId, "/market-data/candle-gaps/repair"), body);

export interface SymbolInfo {
  symbol: string;
  bid: number;
  ask: number;
  spread_points: number;
  point: number;
  digits: number;
  stops_level: number;
  contract_size: number;
  volume_min: number;
  volume_max: number;
  volume_step: number;
}

export const getSymbolInfo = (accountId: string, symbol: string) =>
  api.get<SymbolInfo>(
    acctPath(accountId, `/market-data/symbol-info?symbol=${encodeURIComponent(symbol)}`),
  );

export interface BrokerSymbol {
  name: string;
  description: string;
  path: string; // broker's Market Watch group, e.g. "Forex\\Majors"
  visible: boolean;
}

export interface BrokerSymbolPage {
  items: BrokerSymbol[];
  total: number; // count matching `search` (or full catalog), before limit/offset
}

/** Browse the connected broker's full symbol catalog (chart/watchlist only —
 * does not configure the engine; see configs/app.yaml: symbols for that).
 * Pass `offset` to page through the full catalog when `search` is omitted. */
export const getBrokerSymbols = (accountId: string, search?: string, limit = 50, offset = 0) => {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (search) params.set("search", search);
  return api.get<BrokerSymbolPage>(acctPath(accountId, `/market-data/broker-symbols?${params}`));
};

// ── Journal (trade markers, F7) ─────────────────────────────────────────────

export interface TradeMarker {
  id: string;
  symbol: string;
  side: "buy" | "sell";
  volume: number;
  open_price: number;
  open_time: number; // epoch seconds UTC
  sl: number | null;
  tp: number | null;
  close_price: number | null;
  close_time: number | null; // epoch seconds UTC, null while open
  profit: number | null;
  comment: string;
}

/** `skill` (a bot's full id from `getSkillAssignments`, e.g.
 * 'normal/xauusd/breakout_v1'), when given, scopes markers to just that
 * bot's own trades instead of every trade (any bot, or manual) on the
 * symbol — used by the chart's per-bot "eye" overlay. */
export const getTradeMarkers = (accountId: string, symbol: string, skill?: string) => {
  const params = new URLSearchParams({ symbol });
  if (skill) params.set("skill", skill);
  return api.get<TradeMarker[]>(acctPath(accountId, `/journal/markers?${params}`));
};

// ── Journal (trade history, filterable/paginated) ───────────────────────────

/** One confluence-check reading behind a bot's entry vote, e.g.
 * name: 'RSI', value: 28.4, threshold: 30, comparison: '<', passed: true. */
export interface IndicatorReading {
  name: string;
  value: number;
  threshold: number;
  comparison: string;
  passed: boolean;
}

export interface TradeHistoryItem {
  id: string;
  symbol: string;
  side: "buy" | "sell";
  volume: number;
  open_price: number;
  open_time: number; // epoch seconds UTC
  sl: number | null;
  tp: number | null;
  close_price: number | null;
  close_time: number | null; // epoch seconds UTC, null while open
  profit: number | null;
  /** Why the engine's position manager closed this trade, e.g. "volatility
   * guard: EXTREME regime while losing" or "time-stop: no progress". Null
   * for normal SL/TP fills or manual/API closes. */
  close_reason: string | null;
  comment: string;
  strategy_version: string | null;
  skill: string | null;
  /** Why the strategy took this trade, full text (unlike `comment`, which
   * is truncated to MT5's 29-char limit). Empty for manual/API trades. */
  reason: string;
  /** Strategy's confidence in this signal, 0..1. Null for manual/API trades. */
  confidence: number | null;
  zone: Zone | null;
  pattern: string | null;
  structure: StructurePoint[];
  /** Confluence-check readings behind the bot's entry vote — RSI/ADX/EMA/Volume
   * for the bots that report it. Empty otherwise. */
  indicators: IndicatorReading[];
  mfe: number | null;
  mfe_time: number | null; // epoch seconds UTC
  mae: number | null;
  mae_time: number | null; // epoch seconds UTC
}

export interface TradeHistoryPage {
  items: TradeHistoryItem[];
  total: number; // count matching the filters, before limit/offset
  total_profit?: number; // total realized profit matching the filters, across all pages
}

export type TradeOutcome = "win" | "loss" | "breakeven" | "open";
export type TradeHistoryOrderBy = "open_time" | "close_time" | "profit";
export type SortDir = "asc" | "desc";

export interface TradeHistoryFilters {
  symbol?: string;
  side?: OrderSide;
  strategy_version?: string;
  skill?: string;
  outcome?: TradeOutcome;
  open_from?: number; // epoch seconds UTC
  open_to?: number;
  close_from?: number;
  close_to?: number;
  order_by?: TradeHistoryOrderBy;
  order_dir?: SortDir;
  limit?: number;
  offset?: number;
}

/** Filtered, paginated trade history across any symbol — backs the trade
 * history page's filter and group-by controls. Unlike `getTradeMarkers`
 * (single symbol, chart overlay only), this supports the full filter set. */
export const getTradeHistory = (accountId: string, filters: TradeHistoryFilters = {}) => {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const qs = params.toString();
  return api.get<TradeHistoryPage>(acctPath(accountId, `/journal/history${qs ? `?${qs}` : ""}`));
};

/** Frozen candle snapshot (M5 entry timeframe + H1 higher timeframe) plus the
 * same zone/pattern/structure/indicators/reason/confidence fields as
 * `TradeHistoryItem`, captured at the moment of entry — backs the mini chart
 * in `TradeDecisionModal`. */
export interface DecisionContext {
  trade_id: string;
  symbol: string;
  side: "buy" | "sell";
  open_price: number;
  open_time: number; // epoch seconds UTC
  entry_candles: Candle[];
  higher_tf_candles: Candle[];
  zone: Zone | null;
  pattern: string | null;
  structure: StructurePoint[];
  indicators: IndicatorReading[];
  reason: string;
  confidence: number | null;
}

/** Decision context for a single trade — used by `TradeDecisionModal` to
 * render the frozen entry-candle snapshot behind the bot's "Why" answer. */
export const getTradeDecisionContext = (
  accountId: string,
  tradeId: string | number,
  signal?: AbortSignal,
) => api.get<DecisionContext>(acctPath(accountId, `/journal/trades/${tradeId}/decision-context`), signal);

// ── Journal analytics (per-symbol and per-bot performance) ─────────────────

export interface SymbolAnalytics {
  symbol: string;
  trade_count: number;
  open_count: number;
  closed_count: number;
  win_count: number;
  loss_count: number;
  breakeven_count: number;
  win_rate: number; // 0..1
  total_profit: number;
  gross_profit: number;
  gross_loss: number; // positive number
  profit_factor: number | null; // null when there are no losing trades yet
  avg_win: number;
  avg_loss: number; // positive number
  avg_profit_per_trade: number;
  largest_win: number;
  largest_loss: number; // negative or zero
  total_volume: number;
  bot_count: number;
  first_trade_time: number | null; // epoch seconds UTC
  last_trade_time: number | null;
}

export interface BotEquityPoint {
  trade_id: string;
  close_time: number; // epoch seconds UTC
  profit: number;
  cumulative_profit: number;
}

export interface BotAnalytics {
  skill: string;
  bot_name: string;
  symbol: string;
  strategy_version: string | null;
  trade_count: number;
  open_count: number;
  closed_count: number;
  win_count: number;
  loss_count: number;
  breakeven_count: number;
  win_rate: number; // 0..1
  total_profit: number;
  gross_profit: number;
  gross_loss: number;
  profit_factor: number | null;
  avg_win: number;
  avg_loss: number;
  expectancy: number;
  largest_win: number;
  largest_loss: number;
  max_drawdown: number; // positive number
  avg_trade_duration_seconds: number | null;
  first_trade_time: number | null;
  last_trade_time: number | null;
  equity_curve: BotEquityPoint[];
  // Execution quality (OBSERVABILITY_PLAN.md Phase 3). Null throughout means
  // "never measured" — trades journaled before execution telemetry existed
  // are skipped by the backend's averages rather than counted as zero.
  avg_slippage: number | null; // price units, positive = the fills cost the trader
  measured_slippage_count: number; // how many trades avg_slippage averages over
  avg_execution_latency_ms: number | null; // signal emit -> broker ack
  retcode_histogram: BotRetcodeCount[]; // most frequent first; MT5 10009 = clean deal
  avg_mfe: number | null; // max favorable excursion, price units
  avg_mae: number | null; // max adverse excursion, price units
  mfe_mae_ratio: number | null;
  avg_mfe_on_losers: number | null; // high vs avg_win => take-profits are too far
  avg_mae_on_winners: number | null; // near the stop distance => stops are too tight
  // Cost-as-%-of-gross-edge (OBSERVABILITY_PLAN.md Phase 6). Null throughout
  // means "never measured" — same skip-don't-zero convention as the Phase 3
  // fields above.
  total_transaction_cost: number | null; // sum of spread+slippage cost, account currency
  avg_transaction_cost_per_trade: number | null;
  cost_pct_of_gross_edge: number | null; // near/above 1.0 => spending the whole edge on costs
}

/** One bot's outcome stats within one bucket of one regime dimension — one
 * entry per (bot, dimension, bucket) on `GET .../journal/analytics/regimes`
 * (OBSERVABILITY_PLAN.md Phase 6). The same win/PF/expectancy shape
 * `BotAnalytics` reports overall, sliced to one regime dimension at a time. */
export interface RegimeAnalytics {
  skill: string;
  bot_name: string;
  dimension: "volatility" | "trend" | "session";
  bucket: string; // e.g. "high" (volatility), "trending" (trend), "london" (session)
  trade_count: number;
  closed_count: number;
  win_count: number;
  loss_count: number;
  win_rate: number; // 0..1
  profit_factor: number | null;
  expectancy: number;
  total_profit: number;
}

/** One bucket of a bot's broker-return-code histogram. */
export interface BotRetcodeCount {
  retcode: number; // MT5 code — 10009 done, 10016 invalid stops, 10014 invalid volume, ...
  count: number;
}

export interface AnalyticsDateFilters {
  open_from?: number; // epoch seconds UTC
  open_to?: number; // epoch seconds UTC
}

function toAnalyticsQs(filters: AnalyticsDateFilters = {}): string {
  const params = new URLSearchParams();
  if (filters.open_from !== undefined && !isNaN(filters.open_from)) {
    params.set("open_from", String(filters.open_from));
  }
  if (filters.open_to !== undefined && !isNaN(filters.open_to)) {
    params.set("open_to", String(filters.open_to));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

/** Per-symbol aggregate stats (any bot, or manual) — sorted by total_profit
 * descending. Powers the analytics page's symbol comparison table. */
export const getSymbolAnalytics = (accountId: string, filters?: AnalyticsDateFilters) =>
  api.get<SymbolAnalytics[]>(acctPath(accountId, `/journal/analytics/symbols${toAnalyticsQs(filters)}`));

/** Per-bot aggregate stats plus equity curves — sorted by total_profit
 * descending. Trades with no `skill` (manual/API) are excluded. */
export const getBotAnalytics = (accountId: string, filters?: AnalyticsDateFilters) =>
  api.get<BotAnalytics[]>(acctPath(accountId, `/journal/analytics/bots${toAnalyticsQs(filters)}`));

/** Per-bot win-rate/PF/expectancy split by market regime (volatility, trend,
 * session) at entry — one entry per (bot, dimension, bucket) with at least
 * one attributable trade (OBSERVABILITY_PLAN.md Phase 6). */
export const getRegimeAnalytics = (
  accountId: string,
  filters?: AnalyticsDateFilters,
  signal?: AbortSignal,
) =>
  api.get<RegimeAnalytics[]>(
    acctPath(accountId, `/journal/analytics/regimes${toAnalyticsQs(filters)}`),
    signal,
  );

// ── Activity log (persisted "what is the bot doing and why") ───────────────

export interface LogEntry {
  id: number;
  created_at: number; // epoch seconds UTC
  level: string; // "INFO" | "WARNING" | "ERROR" | ...
  logger: string; // e.g. "src.engine.application.trade_loop"
  message: string;
}

export interface LogHistoryPage {
  items: LogEntry[];
  total: number; // count matching the filters, before limit/offset
}

export interface LogHistoryFilters {
  level?: string;
  logger_contains?: string;
  q?: string;
  created_from?: number; // epoch seconds UTC
  created_to?: number;
  limit?: number;
  offset?: number;
}

/** Filtered, paginated activity log across every backend module — the
 * durable record of signals, vetoes, fills, and circuit breakers, beyond
 * what scrolls past in stdout. */
export const getActivityLog = (accountId: string, filters: LogHistoryFilters = {}) => {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const qs = params.toString();
  return api.get<LogHistoryPage>(acctPath(accountId, `/activity/history${qs ? `?${qs}` : ""}`));
};

export interface LogDeleteResult {
  deleted: number;
}

/** Deletes specific activity log rows by id — backs single-row delete and
 * multi-select bulk delete in the activity log UI. */
export const deleteActivityLogByIds = (accountId: string, ids: number[]) =>
  api.post<LogDeleteResult>(acctPath(accountId, "/activity/history/delete-by-ids"), { ids });

/** Deletes every activity log row matching the given filters — backs
 * "delete all matching" in the activity log UI. Mirrors `LogHistoryFilters`
 * (minus pagination); omitting all fields deletes every row. */
export const deleteActivityLogByFilter = (
  accountId: string,
  filters: Omit<LogHistoryFilters, "limit" | "offset"> = {},
) => api.post<LogDeleteResult>(acctPath(accountId, "/activity/history/delete-by-filter"), filters);

// ── Backtest (Phase 5 reports) ──────────────────────────────────────────────

export interface Zone {
  kind: "demand" | "supply";
  price_low: number;
  price_high: number;
  time_start: number; // epoch seconds UTC
  time_end: number; // epoch seconds UTC
  pattern: string | null; // zone subtype, e.g. "RBR"/"DBD"/"RBD"/"DBR"; null if unreported
}

export interface StructurePoint {
  label: "HH" | "HL" | "LH" | "LL";
  price: number;
  time: number; // epoch seconds UTC
}

export interface BacktestTrade {
  side: "buy" | "sell";
  volume: number;
  open_time: number; // epoch seconds UTC
  open_price: number;
  sl: number | null;
  tp: number | null;
  close_time: number; // epoch seconds UTC
  close_price: number;
  profit: number;
  r_multiple: number | null;
  zone: Zone | null;
  pattern: string | null;
  structure: StructurePoint[];
  /** Why the strategy took this trade, full text. Empty for report files
   * predating this field. */
  reason: string;
  /** Strategy's confidence in this signal, 0..1. Null if not reported. */
  confidence: number | null;
}

export interface EquityPoint {
  time: number; // epoch seconds UTC
  balance: number;
}

export interface ActivityLogEntry {
  time: number; // epoch seconds UTC — simulated bot clock, not wall-clock
  level: string; // "INFO" | "WARNING" | "ERROR"
  logger: string; // e.g. "src.engine.application.trade_loop"
  message: string;
}

/** One strategy signal emitted during the replay — including signals that
 * never became trades (vetoed or rejected by the engine), so the report
 * page and chart can show every valid setup the strategy saw. Also reused
 * as-is for a *live* bot's signal trail (`getLiveBotSignals` below,
 * `GET /activity/signals`) — same shape, same chart rendering, just sourced
 * from the live decision-trail log instead of a backtest replay. */
export interface BacktestSignal {
  time: number; // epoch seconds UTC — simulated bot clock (bar close time), or live wall clock
  direction: "buy" | "sell";
  /** 'opened' (became a trade), 'htf_veto' (higher-TF trend opposed it),
   * 'volatility_guard' (ATR regime EXTREME), 'max_positions' (open-position
   * cap), 'risk_sizing' (no tradable lot size), 'spread_veto' (live spread
   * over the cap), 'rr_gate' (spread-adjusted risk-reward floor),
   * 'daily_loss_breaker' (a circuit breaker had the engine paused),
   * 'broker_rejected' (the broker/MT5 itself refused the order — live only),
   * 'risk_rejected' (any other pre-trade risk block; also every row written
   * before the buckets were split — OBSERVABILITY_PLAN.md Phase 2), or
   * 'skipped'. Every member must have an entry in
   * `features/backtest/signalOutcome.ts`, which is indexed unguarded. */
  outcome:
    | "opened"
    | "htf_veto"
    | "risk_rejected"
    | "spread_veto"
    | "rr_gate"
    | "volatility_guard"
    | "max_positions"
    | "risk_sizing"
    | "daily_loss_breaker"
    | "broker_rejected"
    | "skipped";
  /** The strategy's own reason string — pattern, zone rect, entry/SL/TP. */
  reason: string;
  /** Intended entry price at the moment of the signal, when the source log
   * line carried one. Optional: older persisted signals (and any backtest
   * report written before the field existed) have none. */
  price?: number | null;
}

/** Reconstructs one live bot's own signal→outcome trail — every setup its
 * strategy saw, whether it became a trade or was vetoed/rejected — for the
 * chart's per-bot "eye" overlay. `skill` is a bot's full id from
 * `getSkillAssignments` (e.g. 'normal/xauusd/breakout_v1') and already
 * fully identifies the symbol, so no separate `symbol` param is needed.
 * Defaults to the last 14 days server-side if `from` is omitted. */
export const getLiveBotSignals = (accountId: string, skill: string, from?: number, to?: number) => {
  const params = new URLSearchParams({ skill });
  if (from !== undefined) params.set("from", String(from));
  if (to !== undefined) params.set("to", String(to));
  return api.get<BacktestSignal[]>(acctPath(accountId, `/activity/signals?${params}`));
};

/** One reason a bot's signals stopped at a given funnel stage. */
export interface FunnelDrop {
  /** The stage these signals failed to reach. */
  stage: "passed_htf" | "sized_ok" | "passed_spread" | "filled";
  outcome: BacktestSignal["outcome"] | string;
  count: number;
  /** One dropped signal's reason text, so the count is actionable. */
  example_reason: string;
}

/** One bot's signal→fill funnel over a period. Counts are monotonically
 * non-increasing and follow the engine's real gate order. */
export interface BotFunnel {
  bot: string;
  symbols: string[];
  fired: number;
  passed_htf: number;
  sized_ok: number;
  passed_spread: number;
  filled: number;
  drops: FunnelDrop[];
}

/** `GET /accounts/{id}/activity/signals/funnel` — "of N signals this bot
 * fired, why did only M trade?". Built only from the typed decision trail, so
 * a period predating it returns an empty list rather than a misleading
 * funnel. Defaults to the last 14 days server-side when `from` is omitted. */
export const getSignalFunnel = (
  accountId: string,
  opts: { skill?: string; from?: number; to?: number } = {},
) => {
  const params = new URLSearchParams();
  if (opts.skill !== undefined) params.set("skill", opts.skill);
  if (opts.from !== undefined) params.set("from", String(opts.from));
  if (opts.to !== undefined) params.set("to", String(opts.to));
  const qs = params.toString();
  return api.get<BotFunnel[]>(
    acctPath(accountId, `/activity/signals/funnel${qs ? `?${qs}` : ""}`),
  );
};

/** How many entries the simulated broker refused for one reason. A rejection
 * is a signal the strategy produced, the risk manager sized and the spread
 * gate cleared — that a real broker would then have thrown away. */
export interface BacktestRejection {
  /** 'stops_level' (SL/TP closer to price than the symbol's minimum
   * distance), 'volume_below_min' or 'volume_above_max'. */
  reason: string;
  count: number;
  /** The MT5 code a live server would have returned — 10016 invalid stops,
   * 10014 invalid volume. */
  retcode: number;
  /** The first refusal's message, with the concrete distances/lot sizes. */
  example: string;
}

/** Which broker constraints a backtest run simulated, and what they cost
 * (OBSERVABILITY_PLAN.md Phase 4). Reports written before this existed report
 * `enabled: false`, which correctly describes them: they filled every order
 * at the bar's closing quote, so their trade list may include entries a live
 * broker would have refused. */
export interface BrokerRealism {
  enabled: boolean;
  stops_level_enforced: boolean;
  volume_grid_enforced: boolean;
  /** Research mode: a too-close SL/TP was widened to the broker minimum
   * rather than the entry being refused, so those trades risked more than
   * the risk manager sized them for. */
  clamp_stops: boolean;
  spread_widening_factor: number;
  /** Price units, positive = the fill cost the trader. */
  slippage_mean: number;
  slippage_stddev: number;
  /** 'live' — calibrated from real measured fills; 'fallback' — not enough
   * live fills yet, so these numbers are a documented guess; 'none'. */
  slippage_source: "live" | "fallback" | "none" | string;
  slippage_sample_count: number;
  accepted_count: number;
  clamped_count: number;
  rejected_count: number;
  rejections: BacktestRejection[];
}

export interface BacktestReportSummary {
  id: string;
  strategy: string;
  symbol: string;
  period: string;
  trade_count: number;
  win_rate: number;
  profit_factor: number | null; // null means no losing trades (infinite)
  max_drawdown_pct: number;
  avg_r: number;
  worst_losing_streak: number;
  starting_balance: number;
  ending_balance: number;
  /** Spread-adjusted minimum reward:risk ratio SpreadGate enforced for this
   * run — a run parameter like starting_balance, not a fixed strategy
   * property. */
  min_rr: number;
  // The full RiskCaps actually enforced for this run (configs/risk.yaml's
  // values, or this run's own overrides) — see RiskManager.size_position /
  // record_trade_closed. daily_loss_limit_pct and consecutive_loss_pause
  // are circuit breakers that pause the engine and never auto-resume, so a
  // low trade_count relative to the period often means one of these
  // tripped early, not that no more setups occurred.
  risk_per_trade_pct: number;
  daily_loss_limit_pct: number;
  max_open_positions: number;
  /** Manual daily kill switch for this run, not a count: true blocked every
   * new trade for the rest of the trading day once flipped on. */
  max_trades_per_day_enabled: boolean;
  consecutive_loss_pause: number;
  min_lot_fallback_enabled: boolean;
  max_risk_per_trade_pct: number | null;
  /** Broker constraints simulated for this run and how many entries they
   * refused. Reports predating it report `enabled: false`. */
  broker_realism: BrokerRealism;
}

export interface BacktestReportDetail extends BacktestReportSummary {
  trades: BacktestTrade[];
  equity_curve: EquityPoint[];
  activity_log: ActivityLogEntry[];
  /** Every signal the strategy emitted (taken or vetoed), oldest first —
   * empty for report files predating this field. */
  signals: BacktestSignal[];
}

export interface BacktestReportPage {
  items: BacktestReportSummary[];
  total: number;
  limit: number;
  offset: number;
}

export const getBacktestReports = (limit: number, offset: number) =>
  api.get<BacktestReportPage>(
    `/backtest/reports?${new URLSearchParams({ limit: String(limit), offset: String(offset) })}`,
  );
export const getBacktestReport = (id: string) =>
  api.get<BacktestReportDetail>(`/backtest/reports/${encodeURIComponent(id)}`);
/** Hard-deletes this report's file. This cannot be undone. */
export const deleteBacktestReport = (id: string) =>
  api.delete<void>(`/backtest/reports/${encodeURIComponent(id)}`);
/** Saves a new report from JSON shaped exactly like getBacktestReport()'s
 * response — download an existing report to get a valid example. A fresh
 * id is always assigned (any `id` in `body` is ignored), so this never
 * overwrites an existing report. */
export const importBacktestReport = (body: BacktestReportDetail) =>
  api.post<BacktestReportSummary>("/backtest/reports/import", body);

/** One measurement compared between live trading and a backtest. */
export interface DivergenceMetric {
  /** 'fill_rate' | 'avg_slippage' | 'win_rate' | 'avg_profit' | 'avg_r' | 'avg_volume' */
  name: string;
  /** 'execution' — how the order was filled; 'outcome' — what it then earned.
   * Execution metrics diverging points at the simulator, outcome metrics
   * alone at the strategy. */
  kind: "execution" | "outcome" | string;
  live_value: number | null;
  backtest_value: number | null;
  /** live_value - backtest_value. */
  delta: number | null;
  /** delta / |backtest_value|. Null when the backtest value is 0 or missing. */
  relative_delta: number | null;
  significant: boolean;
  live_sample_count: number;
  backtest_sample_count: number;
  note: string;
}

/** Live-vs-backtest comparison for one strategy on one symbol — is the
 * simulator lying about fills, or has the edge decayed? */
export interface DivergenceReport {
  strategy: string;
  symbol: string;
  report_id: string;
  live_trade_count: number;
  backtest_trade_count: number;
  /** False when either side has too few trades for the comparison to mean
   * anything; the metrics are still returned, but draw no conclusion. */
  comparable: boolean;
  verdict:
    | "aligned"
    | "simulator_optimistic"
    | "edge_decayed"
    | "both"
    | "insufficient_data"
    | string;
  summary: string;
  metrics: DivergenceMetric[];
}

/** Compares every closed live trade journalled for this report's strategy and
 * symbol against the report's own trades. Read-only, computed on demand. */
export const getBacktestDivergence = (id: string) =>
  api.get<DivergenceReport>(`/backtest/reports/${encodeURIComponent(id)}/divergence`);


// ── Backtest bots + on-demand run ─────────────────────────────────────────────

export interface BacktestBot {
  /** Stable identifier — pass this to `startBacktest`, never `name`. The
   * literal string "breakout_v1" for the hardcoded baseline, or a strategy
   * version id (UUID) for everything else. */
  id: string;
  /** Display label only — human-typed, not guaranteed unique or stable
   * (a family can be renamed, or its generated code can hardcode the same
   * internal name as an unrelated family). Never use this to look anything
   * up; use `id`. */
  name: string;
  symbols: string[];
}

export interface BacktestJobStatus {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  report_id: string | null;
  error: string | null;
}

export const getBacktestBots = () => api.get<BacktestBot[]>("/backtest/bots");
export const startBacktest = (
  strategyId: string,
  symbol: string,
  period: string,
  startingBalance?: number,
  /** Override configs/risk.yaml's min-lot fallback for this run only — null/omitted
   * uses whatever's currently configured (file default, or the live engine override
   * from putMinLotFallback). See RunBacktestPanel's "small balance" section. */
  minLotFallbackEnabled?: boolean,
  maxRiskPerTradePct?: number,
  /** Override configs/symbols/<symbol>.yaml's min_rr for this run only — null/omitted
   * uses whatever's currently configured (file default, or the live override from
   * putSymbolMinRr). A tighter-stop strategy can fail the RR floor a swing-trading
   * min_rr was tuned for. */
  minRr?: number,
) =>
  api.post<BacktestJobStatus>("/backtest/run", {
    strategy_id: strategyId,
    symbol,
    period,
    ...(startingBalance != null ? { starting_balance: startingBalance } : {}),
    ...(minLotFallbackEnabled != null
      ? { min_lot_fallback_enabled: minLotFallbackEnabled }
      : {}),
    ...(maxRiskPerTradePct != null ? { max_risk_per_trade_pct: maxRiskPerTradePct } : {}),
    ...(minRr != null ? { min_rr: minRr } : {}),
  });
export const getBacktestJobStatus = (jobId: string) =>
  api.get<BacktestJobStatus>(`/backtest/run/${encodeURIComponent(jobId)}`);

// ── AI: PDF -> StrategySpec pipeline (Phase 6, F4) ──────────────────────────

export type IndicatorType = "ema" | "sma" | "rsi" | "macd" | "bollinger";

export interface IndicatorSpec {
  type: IndicatorType;
  period: number;
  label: string;
  source: string;
  params: Record<string, number>;
}

export type PriceLevelAnnotationType = "support" | "resistance" | "level";

export interface PriceLevelAnnotation {
  type: PriceLevelAnnotationType;
  price: number;
  label: string;
}

export interface ExtractedStrategySpec {
  name: string;
  symbols: string[];
  entry_timeframe: string;
  confirmation_timeframes: string[];
  indicators: IndicatorSpec[];
  entry_rules: string;
  exit_rules: string;
  risk_notes: string;
  params: Record<string, unknown>;
  unrecognized_indicators: string[];
  price_levels: PriceLevelAnnotation[];
  chart_notes: string[];
}

export type DraftStatus = "pending_review" | "approved" | "rejected" | "code_generated";

export interface StrategyDraft {
  id: string;
  source_filename: string;
  created_at: number; // epoch seconds UTC
  status: DraftStatus;
  extracted_spec: ExtractedStrategySpec;
  edited_spec: ExtractedStrategySpec | null;
  effective_spec: ExtractedStrategySpec;
}

export interface GeneratedCode {
  draft_id: string;
  code: string;
  is_valid: boolean;
  sandbox_errors: string[];
  version_id: string | null;
  backtest_report_id: string | null;
}

export const uploadStrategyPdf = (accountId: string, file: File, symbol?: string) => {
  const form = new FormData();
  form.append("file", file);
  if (symbol) form.append("symbol", symbol);
  return api.postForm<StrategyDraft>(acctPath(accountId, "/ai/pdf-strategy/upload"), form);
};

/** Same draft pipeline as uploadStrategyPdf, but from a typed description —
 * no PDF required. Lands on the same review/approve/generate-code flow. */
export const createStrategyDraftFromText = (
  accountId: string,
  description: string,
  symbol?: string,
) => api.post<StrategyDraft>(acctPath(accountId, "/ai/pdf-strategy/from-prompt"), { description, symbol });

/** Same draft pipeline, but `spec` is already structured JSON (e.g. a file
 * upload) — skips LLM extraction entirely, `spec` becomes the draft's
 * extracted_spec as-is. Lands on the same review/approve/generate flow. */
export const createStrategyDraftFromJson = (
  accountId: string,
  spec: ExtractedStrategySpec,
  symbol?: string,
) => api.post<StrategyDraft>(acctPath(accountId, "/ai/pdf-strategy/from-spec"), { spec, symbol });

export const getStrategyDrafts = (accountId: string) =>
  api.get<StrategyDraft[]>(acctPath(accountId, "/ai/pdf-strategy/drafts"));
export const getStrategyDraft = (accountId: string, id: string) =>
  api.get<StrategyDraft>(
    acctPath(accountId, `/ai/pdf-strategy/drafts/${encodeURIComponent(id)}`),
  );
export const updateStrategyDraftSpec = (
  accountId: string,
  id: string,
  editedSpec: ExtractedStrategySpec,
) =>
  api.patch<StrategyDraft>(acctPath(accountId, `/ai/pdf-strategy/drafts/${encodeURIComponent(id)}`), {
    edited_spec: editedSpec,
  });
export const approveStrategyDraft = (accountId: string, id: string) =>
  api.post<StrategyDraft>(
    acctPath(accountId, `/ai/pdf-strategy/drafts/${encodeURIComponent(id)}/approve`),
  );
export const rejectStrategyDraft = (accountId: string, id: string) =>
  api.post<StrategyDraft>(
    acctPath(accountId, `/ai/pdf-strategy/drafts/${encodeURIComponent(id)}/reject`),
  );
export const generateStrategyCode = (accountId: string, id: string) =>
  api.post<GeneratedCode>(
    acctPath(accountId, `/ai/pdf-strategy/drafts/${encodeURIComponent(id)}/generate-code`),
  );

// ── Strategy versions & activation (Phase 6, §6.5) ──────────────────────────

export type StrategyVersionStatus = "validated" | "active" | "archived";
export type StrategySource = "ai_generated" | "ai_refined" | "manual";

export interface StrategyVersionSummary {
  id: string;
  name: string;
  version: number;
  file_path: string;
  code_hash: string;
  source: StrategySource;
  status: StrategyVersionStatus;
  /** Only meaningful while status is "active": true if suspended from live
   * evaluation via pauseStrategyVersion without being deactivated. */
  paused: boolean;
  created_at: number; // epoch seconds UTC
  parent_version_id: string | null;
  draft_id: string | null;
  spec: ExtractedStrategySpec | null;
  backtest_report_id: string | null;
}

export interface StrategyVersionDetail extends StrategyVersionSummary {
  code: string;
}

export const getStrategyVersions = (
  accountId: string,
  name?: string,
  status?: StrategyVersionStatus,
) => {
  const params = new URLSearchParams();
  if (name) params.set("name", name);
  if (status) params.set("status", status);
  const query = params.toString();
  return api.get<StrategyVersionSummary[]>(
    acctPath(accountId, `/strategies/versions${query ? `?${query}` : ""}`),
  );
};
export const getStrategyVersion = (accountId: string, id: string) =>
  api.get<StrategyVersionDetail>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}`),
  );
export const activateStrategyVersion = (accountId: string, id: string) =>
  api.post<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/activate`),
  );
/** Clones a version's code into a new, independent strategy family (fork,
 * not a new version of the same family). Pass `symbols` to also retarget
 * the clone — rewrites the generated code's `StrategySpec(symbols=...)` and
 * re-validates it in the sandbox; omit to keep the source's symbols. */
export const duplicateStrategyVersion = (
  accountId: string,
  id: string,
  body: { name: string; symbols?: string[] },
) =>
  api.post<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/duplicate`),
    body,
  );
/** Renames the display name shared by every version of this strategy
 * family, not just this one. */
export const renameStrategyVersion = (accountId: string, id: string, name: string) =>
  api.patch<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/rename`),
    { name },
  );
/** Retires this version: marks it archived and, if it was the live version,
 * stops the engine from evaluating it. No replacement version is required —
 * the strategy family can end up with nothing active. */
export const archiveStrategyVersion = (accountId: string, id: string) =>
  api.post<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/archive`),
  );
/** Hard-deletes this version's record and generated file. Rejected with a
 * 409 if the version is currently active — archive it first. */
export const deleteStrategyVersion = (accountId: string, id: string) =>
  api.delete<void>(acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}`));
/** Suspends live trading for this active version without deactivating it —
 * distinct from the engine-wide kill switch, which pauses every strategy. */
export const pauseStrategyVersion = (accountId: string, id: string) =>
  api.post<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/pause`),
  );
/** Reverses pauseStrategyVersion. */
export const resumeStrategyVersion = (accountId: string, id: string) =>
  api.post<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/resume`),
  );
/** Saves a hand-edited source. Leave `newName` unset to save as the next
 * version of this version's own strategy family, parented on `id` itself
 * (not necessarily the active version). Pass `newName` to fork the edit
 * into a brand-new strategy family at version 1 instead — throws
 * ApiError(409) if that name is already in use by another family.
 * Re-validated in the sandbox either way — throws ApiError(422) if that
 * fails. The new version's status is "validated", never "active". */
export const editStrategyVersionCode = (
  accountId: string,
  id: string,
  code: string,
  newName?: string,
) =>
  api.post<StrategyVersionDetail>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/edit`),
    { code, new_name: newName },
  );
/** Overwrites this version's spec snapshot in place — annotation only, never
 * touches the generated code or creates a new version (the same way
 * renameStrategyVersion mutates in place rather than forking). */
export const updateStrategyVersionSpec = (
  accountId: string,
  id: string,
  spec: ExtractedStrategySpec,
) =>
  api.patch<StrategyVersionSummary>(
    acctPath(accountId, `/strategies/versions/${encodeURIComponent(id)}/spec`),
    spec,
  );

// ── AI: user-triggered code regeneration (§6.5 code editor) ────────────────

export interface RegeneratedCode {
  version_id: string;
  instructions: string;
  code: string;
  is_valid: boolean;
  sandbox_errors: string[];
  new_version_id: string | null;
}

/** Runs the trader's free-form instructions through the `code_generation`
 * task's configured LLM (the same provider setting as the PDF-to-code
 * pipeline — see the Settings page) against this version's current code
 * and spec — or,
 * if `spec` is given, that edited spec instead, letting the trader tweak
 * symbols/timeframes/entry-exit rules before regenerating — then
 * sandbox-validates the result. Leave `newName` unset to save as the next
 * version of this version's own family; pass it to fork into a brand-new
 * family at version 1 instead (throws ApiError(409) if already in use). On
 * success `new_version_id` points at the new "validated" StrategyVersion;
 * on sandbox rejection `sandbox_errors` explains why and no version is
 * created — the caller can still show `code` for manual fixup via
 * `editStrategyVersionCode`. */
export const regenerateStrategyVersionCode = (
  accountId: string,
  id: string,
  instructions: string,
  spec?: ExtractedStrategySpec,
  newName?: string,
) =>
  api.post<RegeneratedCode>(
    acctPath(accountId, `/ai/strategies/versions/${encodeURIComponent(id)}/regenerate`),
    { instructions, spec, new_name: newName },
  );

export interface CustomSignal {
  time: number;
  direction: "buy" | "sell";
  sl_points: number;
  tp_points: number;
  confidence: number;
  reason: string;
}

export interface EvaluateCustomCodeResponse {
  signals: CustomSignal[];
  indicators: Record<string, (number | null)[]>;
  candles: {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    tick_volume: number;
  }[];
  error: string | null;
}

export const evaluateCustomCode = (body: {
  code: string;
  symbol: string;
  timeframe: string;
  period: string;
}) => api.post<EvaluateCustomCodeResponse>("/strategies/evaluate-custom", body);

// ── Custom indicators (sandboxed Python, independent of the chart's ────────
// ── built-in client-side indicators) ────────────────────────────────────────

export interface IndicatorSummary {
  id: string;
  name: string;
  code_hash: string;
  default_params: Record<string, number>;
  created_at: number; // epoch seconds UTC
  updated_at: number; // epoch seconds UTC
}

export interface IndicatorDetail extends IndicatorSummary {
  code: string;
}

export interface ComputeIndicatorResponse {
  times: number[];
  series: Record<string, (number | null)[]>;
  error: string | null;
}

export const listIndicators = () => api.get<IndicatorSummary[]>("/indicators");
export const getIndicator = (id: string) =>
  api.get<IndicatorDetail>(`/indicators/${encodeURIComponent(id)}`);
/** Sandbox-validates `code` and, if it passes, saves a new indicator that
 * immediately shows up in the chart's indicator picker. Throws
 * ApiError(409) if `name` is already in use, ApiError(422) on sandbox
 * rejection. */
export const createIndicator = (body: {
  name: string;
  code: string;
  default_params?: Record<string, number>;
}) => api.post<IndicatorDetail>("/indicators", body);
/** Re-validates `code` and, if it passes, updates this indicator's row in
 * place — no version history, since indicators never trade live. Every
 * chart currently using it picks up the new code on its next compute. */
export const editIndicatorCode = (
  id: string,
  code: string,
  defaultParams?: Record<string, number>,
) =>
  api.post<IndicatorDetail>(`/indicators/${encodeURIComponent(id)}/edit`, {
    code,
    default_params: defaultParams,
  });
/** Clones this indicator's code and default params into a brand-new row.
 * Throws ApiError(409) if `name` is already in use. */
export const duplicateIndicator = (id: string, name: string) =>
  api.post<IndicatorDetail>(`/indicators/${encodeURIComponent(id)}/duplicate`, { name });
export const deleteIndicator = (id: string) =>
  api.delete<void>(`/indicators/${encodeURIComponent(id)}`);
/** Computes a saved indicator against real candle history for the chart.
 * `period` is "YYYY-MM:YYYY-MM". Sandbox/history/runtime failures come back
 * as `error` in the response body, not an HTTP error. */
export const computeIndicator = (
  id: string,
  body: { symbol: string; timeframe: string; period: string; params?: Record<string, number> },
) => api.post<ComputeIndicatorResponse>(`/indicators/${encodeURIComponent(id)}/compute`, body);
/** Same as computeIndicator, but for ad-hoc code that hasn't been saved yet
 * — nothing is persisted. Used by the create/edit UI's Preview button. */
export const previewIndicatorCode = (body: {
  code: string;
  params?: Record<string, number>;
  symbol: string;
  timeframe: string;
  period: string;
}) => api.post<ComputeIndicatorResponse>("/indicators/preview", body);

// ── Symbol -> strategy routing (§6.6) ───────────────────────────────────────

export interface SessionWindowWire {
  start: string; // HH:MM
  end: string; // HH:MM
}

/** JSON-scalar value a strategy param can hold — mirrors the backend's
 * `dict[str, float | int | str | bool]`. */
export type ParamValue = number | string | boolean;

export interface NormalSkillAssignment {
  name: string;
  /** This bot's short id on its symbol (the last segment of `name`) — the
   * path segment used by updateBotAssignment/removeBotFromSymbol. */
  bot_name: string;
  symbol: string;
  strategy: string;
  risk_multiplier: number;
  sessions: SessionWindowWire[];
  /** Per-bot overrides of this strategy's tunable params, keyed by param
   * name — only explicitly overridden keys appear here; every other param
   * runs at its `strategy_default_params` value. Set via updateBotConfig. */
  param_overrides: Record<string, ParamValue>;
  /** Per-bot override of the engine's HTF veto. `null` means this bot
   * inherits `strategy_default_htf_veto`. */
  htf_veto_override: boolean | null;
  /** This bot's strategy's own declared param defaults (its registered
   * StrategySpec.params) — the base every key in `param_overrides` layers
   * on top of. Empty if the strategy isn't currently registered (paused). */
  strategy_default_params: Record<string, ParamValue>;
  /** This bot's strategy's own declared StrategySpec.htf_veto — the base
   * `htf_veto_override` layers on top of. */
  strategy_default_htf_veto: boolean;
  /** True only on the addBotToSymbol response when that call just
   * activated a previously-inactive symbol for live automated trading
   * (persisted to configs/app.yaml, hot-added to candle streaming and the
   * spread gate). Always false in getSkillAssignments()'s list — a listing
   * isn't an action outcome. */
  newly_activated: boolean;
}

/** Every bot currently routed for live trading, on every symbol — the real
 * "which bots trade this symbol live" state (`TradeEngine._try_enter` reads
 * this via `SkillSelector`), distinct from a version's `spec.symbols`
 * membership. A symbol may appear multiple times, once per active bot.
 * Includes any bot activated at runtime via `addBotToSymbol`, not just bots
 * configured at backend startup. */
export const getSkillAssignments = () => api.get<NormalSkillAssignment[]>("/skills/normal");

/** Activates a new bot on `symbol`, alongside any bots already routed there
 * — never replaces one (see `updateBotAssignment` to reassign an existing
 * bot instead). Writes skills/normal/<symbol>/<bot_name>.yaml and hot-swaps
 * the live SkillSelector, no restart needed. If `symbol` isn't yet
 * live-traded, this is also the action that activates it: persists it into
 * configs/app.yaml and hot-adds it to candle streaming/the spread gate (see
 * the response's `newly_activated`). `strategyName` must currently have an
 * active, non-paused StrategyVersion (422 otherwise); 404 if `symbol` has no
 * configs/symbols/<symbol>.yaml; 409 if `botName` is already taken on this
 * symbol. */
export const addBotToSymbol = (symbol: string, strategyName: string, botName?: string) =>
  api.post<NormalSkillAssignment>(`/skills/normal/${encodeURIComponent(symbol)}/bots`, {
    strategy_name: strategyName,
    bot_name: botName,
  });

/** Reassigns `botName`'s strategy on `symbol` in place, keeping its
 * sessions/risk_multiplier and leaving every other bot on the symbol
 * untouched — hot-swaps the live SkillSelector, no restart needed. */
export const updateBotAssignment = (symbol: string, botName: string, strategyName: string) =>
  api.put<NormalSkillAssignment>(
    `/skills/normal/${encodeURIComponent(symbol)}/bots/${encodeURIComponent(botName)}`,
    { strategy_name: strategyName },
  );

/** Renames `botName` on `symbol` in place, keeping its strategy, risk_multiplier,
 * sessions, and param/htf_veto overrides — hot-swaps the live SkillSelector, no
 * restart needed. Changes this bot's MT5 magic number (derived from the full skill
 * name), so any position the broker already has open under the old name will no
 * longer be recognized as this bot's — avoid renaming a bot that holds an open
 * position. 409 if `newBotName` (slugified) is already taken on this symbol. */
export const renameBot = (symbol: string, botName: string, newBotName: string) =>
  api.put<NormalSkillAssignment>(
    `/skills/normal/${encodeURIComponent(symbol)}/bots/${encodeURIComponent(botName)}/name`,
    { new_bot_name: newBotName },
  );

/** Replaces `botName`'s risk_multiplier, sessions, and per-bot strategy
 * param/htf_veto overrides in one call — every field is a full replacement,
 * not a partial patch. Doesn't change which strategy family the bot trades
 * (see `updateBotAssignment`) — reassigning strategy resets overrides
 * server-side, since they may not apply to the new strategy's params. */
export const updateBotConfig = (
  symbol: string,
  botName: string,
  config: {
    risk_multiplier: number;
    sessions: SessionWindowWire[];
    param_overrides: Record<string, ParamValue>;
    htf_veto_override: boolean | null;
  },
) =>
  api.put<NormalSkillAssignment>(
    `/skills/normal/${encodeURIComponent(symbol)}/bots/${encodeURIComponent(botName)}/config`,
    config,
  );

/** Stops `botName` from trading `symbol` — every other bot on the symbol
 * keeps trading unaffected. */
export const removeBotFromSymbol = (symbol: string, botName: string) =>
  api.delete<void>(
    `/skills/normal/${encodeURIComponent(symbol)}/bots/${encodeURIComponent(botName)}`,
  );

// ── AI: 10-trade self-refinement loop (Phase 7, F5) ─────────────────────────

export type ReportVerdict = "no_action" | "refinement_proposed";

export interface AnalysisReport {
  id: string;
  symbol: string;
  strategy_name: string;
  base_version_id: string;
  trade_ids: string[];
  created_at: number; // epoch seconds UTC
  win_rate: number;
  avg_r: number;
  common_failure_pattern: string;
  session_or_news_correlation: string;
  verdict: ReportVerdict;
  raw_llm_response: string;
  proposal_id: string | null;
}

export type ProposalStatus = "pending" | "backtested" | "applied" | "rejected";

export interface RefinementProposalDetail {
  id: string;
  report_id: string;
  strategy_name: string;
  base_version_id: string;
  rationale: string;
  proposed_code: string;
  status: ProposalStatus;
  created_at: number; // epoch seconds UTC
  sandbox_errors: string[];
  new_version_id: string | null;
  improvement_pct: number | null; // candidate avg_r % improvement over baseline
  applied_mode: "suggest" | "auto" | null;
  diff: string[]; // unified diff lines, computed fresh server-side on every read
  baseline_backtest: BacktestReportSummary | null;
  candidate_backtest: BacktestReportSummary | null;
}

export const getAnalysisReports = (accountId: string, symbol?: string) =>
  api.get<AnalysisReport[]>(
    acctPath(
      accountId,
      `/ai/refinement/reports${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ""}`,
    ),
  );
export const getAnalysisReport = (accountId: string, id: string) =>
  api.get<AnalysisReport>(
    acctPath(accountId, `/ai/refinement/reports/${encodeURIComponent(id)}`),
  );
export const getRefinementProposal = (accountId: string, id: string) =>
  api.get<RefinementProposalDetail>(
    acctPath(accountId, `/ai/refinement/proposals/${encodeURIComponent(id)}`),
  );
export const rejectRefinementProposal = (accountId: string, id: string) =>
  api.post<RefinementProposalDetail>(
    acctPath(accountId, `/ai/refinement/proposals/${encodeURIComponent(id)}/reject`),
  );

// ── News: economic calendar & active windows (Phase 8, F8) ─────────────────

export type ImpactLevel = "low" | "medium" | "high";

export interface NewsEvent {
  name: string;
  time: number; // epoch seconds UTC
  impact: ImpactLevel;
  currency: string;
  skill: string | null; // matched news skill, or null if this event never activates one
  forecast: string | null; // consensus estimate, formatted as the source publishes it
  previous: string | null; // prior period's reading
  actual: string | null; // released value, once the event has happened (source-dependent)
}

export interface NewsWindow {
  event: NewsEvent;
  skill: string;
  window_start: number; // epoch seconds UTC
  window_end: number; // epoch seconds UTC
  phase: "pre" | "post";
  symbols: string[]; // symbols this window affects
}

export const getUpcomingNews = (daysAhead = 7) =>
  api.get<NewsEvent[]>(`/news/upcoming?days_ahead=${daysAhead}`);
export const getActiveNewsWindows = () => api.get<NewsWindow[]>("/news/active-windows");

// ── Engine: status + kill switch (Phase 9, §11) ─────────────────────────────

export interface EngineStatus {
  enabled: boolean;
  paused: boolean;
  pause_reason: string;
  consecutive_losses: number;
  trades_today: number;
  daily_pnl: number;
}

export const getEngineStatus = (accountId: string) =>
  api.get<EngineStatus>(acctPath(accountId, "/engine/status"));
export const killSwitch = (accountId: string) =>
  api.post<EngineStatus>(acctPath(accountId, "/engine/kill"));
export const resumeEngine = (accountId: string) =>
  api.post<EngineStatus>(acctPath(accountId, "/engine/resume"));

export interface RiskCaps {
  risk_per_trade_pct: number;
  daily_loss_limit_pct: number;
  max_open_positions: number;
  /** Manual daily kill switch, not a count: true blocks every new trade for
   * the rest of the trading day; false leaves trade count today unlimited. */
  max_trades_per_day_enabled: boolean;
  consecutive_loss_pause: number;
  /** When false, the consecutive-loss circuit breaker never pauses the engine,
   * regardless of consecutive_loss_pause's count. */
  consecutive_loss_pause_enabled: boolean;
  /** When true, a balance too small for risk_per_trade_pct to reach the broker's
   * minimum lot trades that minimum lot anyway, capped by max_risk_per_trade_pct. */
  min_lot_fallback_enabled: boolean;
  max_risk_per_trade_pct: number | null;
}

export const getRiskCaps = (accountId: string) =>
  api.get<RiskCaps>(acctPath(accountId, "/engine/risk-caps"));
/** Live-updates the min-lot fallback on the running engine. Not persisted —
 * a backend restart reverts to configs/risk.yaml. */
export const putMinLotFallback = (
  accountId: string,
  enabled: boolean,
  maxRiskPerTradePct: number | null,
) =>
  api.put<RiskCaps>(acctPath(accountId, "/engine/risk-caps/min-lot-fallback"), {
    enabled,
    max_risk_per_trade_pct: maxRiskPerTradePct,
  });
/** Live-updates the daily trading kill switch on the running engine. Not
 * persisted — a backend restart reverts to configs/risk.yaml. */
export const putMaxTradesPerDayEnabled = (accountId: string, enabled: boolean) =>
  api.put<RiskCaps>(acctPath(accountId, "/engine/risk-caps/max-trades-per-day-enabled"), {
    enabled,
  });

export interface CoreRiskCapsUpdate {
  risk_per_trade_pct?: number;
  daily_loss_limit_pct?: number;
  max_open_positions?: number;
  consecutive_loss_pause?: number;
  consecutive_loss_pause_enabled?: boolean;
}

/** Live-updates any of the core sizing/circuit-breaker caps on the running
 * engine — omitted fields keep their current value. Not persisted — a
 * backend restart reverts to configs/risk.yaml. Free to loosen or tighten,
 * unlike a per-account risk_override_file. */
export const putCoreRiskCaps = (accountId: string, update: CoreRiskCapsUpdate) =>
  api.put<RiskCaps>(acctPath(accountId, "/engine/risk-caps/core"), update);

/** ATR-percentile-based volatility regime classifier that scales bots' SL/TP
 * and can force-close/trail positions in high volatility. */
export interface VolatilityConfig {
  atr_period: number;
  regime_lookback_bars: number;
  low_percentile: number;
  high_percentile: number;
  extreme_percentile: number;
  sl_multiplier_low: number;
  sl_multiplier_normal: number;
  sl_multiplier_high: number;
  tp_multiplier_low: number;
  tp_multiplier_normal: number;
  tp_multiplier_high: number;
  extreme_close_if_losing: boolean;
  extreme_profit_lock_r_mult: number;
  chandelier_atr_mult: number;
  chandelier_min_profit_r: number;
  /** Live on/off switch for the whole volatility guard. */
  enabled: boolean;
}

export const getVolatilityConfig = (accountId: string) =>
  api.get<VolatilityConfig>(acctPath(accountId, "/engine/volatility-config"));
/** Live-updates the volatility guard on/off switch on the running engine.
 * Not persisted — a backend restart reverts to configs/volatility.yaml
 * (default: enabled). */
export const putVolatilityGuardEnabled = (accountId: string, enabled: boolean) =>
  api.put<VolatilityConfig>(acctPath(accountId, "/engine/volatility-config/enabled"), {
    enabled,
  });

// ── Broker: manual trading (chart buttons, click-to-trade, draggable SL/TP) ─

export type OrderSide = "buy" | "sell";
export type PendingOrderType = "limit" | "stop";

export interface OpenOrderRequest {
  symbol: string;
  side: OrderSide;
  volume: number;
  sl?: number | null;
  tp?: number | null;
  comment?: string;
}

export interface ExecutionResultOut {
  ticket: number;
  symbol: string;
  side: OrderSide;
  volume: number;
  price: number;
  sl: number | null;
  tp: number | null;
  time: string;
  spread_points: number;
  comment: string;
  profit: number | null;
}

export interface PositionOut {
  ticket: number;
  symbol: string;
  side: OrderSide;
  volume: number;
  open_price: number;
  sl: number | null;
  tp: number | null;
  open_time: string;
  profit: number;
  comment: string;
}

export interface PlacePendingOrderRequest {
  symbol: string;
  side: OrderSide;
  order_type: PendingOrderType;
  volume: number;
  price: number;
  sl?: number | null;
  tp?: number | null;
  comment?: string;
}

export interface PendingOrderOut {
  ticket: number;
  symbol: string;
  side: OrderSide;
  order_type: PendingOrderType;
  volume: number;
  price: number;
  sl: number | null;
  tp: number | null;
  placed_time: string;
  comment: string;
}

export const openOrder = (accountId: string, body: OpenOrderRequest) =>
  api.post<ExecutionResultOut>(acctPath(accountId, "/broker/orders"), body);
export const closePosition = (accountId: string, ticket: number, volume?: number) =>
  api.post<ExecutionResultOut>(
    acctPath(accountId, `/broker/positions/${ticket}/close`),
    volume ? { volume } : undefined,
  );
export const closeAllPositions = (accountId: string, symbol: string) =>
  api.post<ExecutionResultOut[]>(
    acctPath(accountId, `/broker/positions/close-all?symbol=${encodeURIComponent(symbol)}`),
    undefined,
  );
export const modifyPosition = (
  accountId: string,
  ticket: number,
  sl: number | null,
  tp: number | null,
) => api.post<{ status: string }>(acctPath(accountId, `/broker/positions/${ticket}/modify`), { sl, tp });
export const getPositions = (accountId: string, symbol?: string) =>
  api.get<PositionOut[]>(
    acctPath(accountId, `/broker/positions${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ""}`),
  );

export const placePendingOrder = (accountId: string, body: PlacePendingOrderRequest) =>
  api.post<PendingOrderOut>(acctPath(accountId, "/broker/orders/pending"), body);
export const getPendingOrders = (accountId: string, symbol?: string) =>
  api.get<PendingOrderOut[]>(
    acctPath(
      accountId,
      `/broker/orders/pending${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ""}`,
    ),
  );
export const modifyPendingOrder = (
  accountId: string,
  ticket: number,
  price: number | null,
  sl: number | null,
  tp: number | null,
) =>
  api.post<{ status: string }>(acctPath(accountId, `/broker/orders/pending/${ticket}/modify`), {
    price,
    sl,
    tp,
  });
export const cancelPendingOrder = (accountId: string, ticket: number) =>
  api.delete<{ status: string }>(acctPath(accountId, `/broker/orders/pending/${ticket}`));

export interface SymbolSpreadConfig {
  symbol: string;
  max_spread_points: number;
  /** Minimum spread-adjusted reward:risk ratio required to open —
   * tp_distance >= min_rr * (sl_distance + spread_value). */
  min_rr: number;
}

export const getSymbolSpreadConfig = (symbol: string) =>
  api.get<SymbolSpreadConfig>(`/broker/symbols/${encodeURIComponent(symbol)}/spread-config`);
/** Live-updates min_rr on the running engine. Not persisted — a backend
 * restart reverts to configs/symbols/<symbol>.yaml. */
export const putSymbolMinRr = (symbol: string, minRr: number) =>
  api.put<SymbolSpreadConfig>(`/broker/symbols/${encodeURIComponent(symbol)}/min-rr`, {
    min_rr: minRr,
  });

// ── AI: provider settings (per-task LLM selection, Phase 10.4) ─────────────
export interface TaskProviderStatus {
  task: string;
  provider: string;
  model: string;
  source: "override" | "default";
  configured: boolean;
}

export interface ProviderPresetModel {
  label: string;
  model: string;
}

export interface ProviderInfo {
  id: string;
  label: string;
  description: string;
  needsSecret: boolean;
  configured: boolean;
  presetModels?: ProviderPresetModel[];
}

export interface ProviderTestResult {
  provider: string;
  ok: boolean;
  message: string | null;
  reply: string | null;
}

export const listTaskProviders = () => api.get<TaskProviderStatus[]>("/ai/settings/tasks");
export const setTaskProvider = (task: string, provider: string, model: string) =>
  api.put<TaskProviderStatus>(`/ai/settings/tasks/${task}`, { provider, model });
export const clearTaskProvider = (task: string) =>
  api.delete<TaskProviderStatus>(`/ai/settings/tasks/${task}`);
export const testProvider = (provider: string, message?: string) =>
  api.post<ProviderTestResult>(
    `/ai/settings/providers/${provider}/test`,
    message ? { message } : undefined,
  );
interface ProviderInfoRaw {
  id: string;
  label: string;
  description: string;
  needs_secret: boolean;
  configured: boolean;
  preset_models: ProviderPresetModel[] | null;
}

function fromRawProviderInfo(p: ProviderInfoRaw): ProviderInfo {
  return {
    id: p.id,
    label: p.label,
    description: p.description,
    needsSecret: p.needs_secret,
    configured: p.configured,
    presetModels: p.preset_models ?? undefined,
  };
}

export const listProviders = () =>
  api.get<ProviderInfoRaw[]>("/ai/settings/providers").then((raw) => raw.map(fromRawProviderInfo));

/** Saves `provider`'s API key, encrypted at rest — takes effect immediately,
 * no backend restart. The key is never returned by this or any other call. */
export const setProviderKey = (provider: string, apiKey: string) =>
  api
    .put<ProviderInfoRaw>(`/ai/settings/providers/${provider}/key`, { api_key: apiKey })
    .then(fromRawProviderInfo);

export const clearProviderKey = (provider: string) =>
  api.delete<ProviderInfoRaw>(`/ai/settings/providers/${provider}/key`).then(fromRawProviderInfo);
