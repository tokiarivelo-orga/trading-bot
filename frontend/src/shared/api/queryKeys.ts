/**
 * TanStack Query key convention for this codebase.
 *
 * Adopted starting with `features/trading/` (see OPTIMIZATION_CHECKLIST.md,
 * "app-wide optimization pass" — request-level caching/dedup), extended to
 * `features/strategies/` and `features/analytics/` in the same pass.
 * `features/chart/` was attempted (replacing `sharedPoll.ts`'s ref-counted
 * poller with TanStack Query's query-key sharing) but reverted: TanStack's
 * `refetchInterval` is scheduled per-observer, not once per shared query key,
 * so N mounted windows on the same symbol produced ~N independent polls
 * instead of one — confirmed via live browser network monitoring (multiple
 * requests per interval tick with 2 windows open, where sharedPoll gave
 * exactly one). `features/chart/` keeps `sharedPoll.ts` until a version of
 * this migration exists that reproduces true single-timer sharing (e.g. a
 * singleton poller driving `queryClient.invalidateQueries` while individual
 * `useQuery` calls set `refetchInterval: false`), not attempted in this pass.
 *
 * Shape: `queryKeys.<feature>.<resource>(accountId, ...args)` returning a
 * `[feature, resource, accountId, ...args] as const` tuple.
 *   - `<feature>` namespaces by feature folder, so two features' resources
 *     never collide even if they happen to share a name.
 *   - Every per-account resource takes `accountId: string | null` as its
 *     first arg and is spelled into the key as-is (including `null`) — this
 *     mirrors `useActiveAccount()`'s "null until resolved" contract, keeps
 *     cache entries scoped per account (switching accounts never serves
 *     another account's cached positions/orders), and callers should gate
 *     `enabled` on `accountId !== null` the same way pre-migration hooks
 *     gated their manual fetch on a non-null accountId.
 *   - Extra args (symbol, ticket, filters, ...) append after `accountId`.
 */
export const queryKeys = {
  trading: {
    /** `GET /accounts/{id}/broker/positions` — every open position, account-wide. */
    positions: (accountId: string | null) => ["trading", "positions", accountId] as const,
    /** `GET /accounts/{id}/broker/orders/pending` — every pending order, account-wide. */
    pendingOrders: (accountId: string | null) => ["trading", "pendingOrders", accountId] as const,
    /** `GET /accounts/{id}/journal/history?outcome=open` — journaled open
     * trades, source of the ticket -> skill / ticket -> full-trade lookups. */
    openTrades: (accountId: string | null) => ["trading", "openTrades", accountId] as const,
  },
  strategies: {
    /** `GET /accounts/{id}/strategies/versions` — every non-archived strategy
     * version, account-wide (BotSelector filters client-side by symbol). */
    versions: (accountId: string | null) => ["strategies", "versions", accountId] as const,
    /** `GET /accounts/{id}/strategies/versions?status=active` — only the
     * active versions, used to resolve the one (if any) whose spec targets a
     * given symbol. */
    activeVersions: (accountId: string | null) =>
      ["strategies", "activeVersions", accountId] as const,
    /** `GET /skills/normal` — every live bot -> symbol assignment. Not
     * per-account (the endpoint takes no account id), so no `accountId` arg. */
    skillAssignments: () => ["strategies", "skillAssignments"] as const,
    /** Per-bot chip-count fan-out (`GET .../journal/markers` +
     * `GET .../signals/live`, one pair per active bot) for BotSelector's
     * Sig/Rej/Ord/Trd chips. Keyed on `botNamesKey` (the sorted, joined list
     * of active bot names) rather than the assignment array itself, since
     * that array's identity churns every assignments-poll tick even when the
     * assigned bots haven't changed. */
    botCounts: (accountId: string | null, symbol: string, botNamesKey: string) =>
      ["strategies", "botCounts", accountId, symbol, botNamesKey] as const,
  },
  analytics: {
    /** `GET /accounts/{id}/analytics/symbols` — per-symbol win-rate/profit
     * breakdown. */
    symbols: (accountId: string | null, openFrom?: number, openTo?: number) =>
      ["analytics", "symbols", accountId, openFrom, openTo] as const,
    /** `GET /accounts/{id}/analytics/bots` — per-bot win-rate/profit
     * breakdown. */
    bots: (accountId: string | null, openFrom?: number, openTo?: number) =>
      ["analytics", "bots", accountId, openFrom, openTo] as const,
    /** `GET /accounts/{id}/activity/signals/funnel` — per-bot signal→fill
     * veto funnel for the period (OBSERVABILITY_PLAN.md Phase 2). Keyed on
     * the same date filters as the other analytics queries. */
    signalFunnel: (accountId: string | null, from?: number, to?: number) =>
      ["analytics", "signalFunnel", accountId, from, to] as const,
    /** `GET /accounts/{id}/journal/analytics/regimes` — per-bot win-rate/PF
     * breakdown split by market regime (volatility/trend/session)
     * (OBSERVABILITY_PLAN.md Phase 6). Keyed on the same date filters as the
     * other analytics queries. */
    regimes: (accountId: string | null, openFrom?: number, openTo?: number) =>
      ["analytics", "regimes", accountId, openFrom, openTo] as const,
    /** `GET /accounts/{id}/journal/analytics/daily` — realized P&L grouped
     * by calendar date (OBSERVABILITY_PLAN.md Phase 7). Keyed on the same
     * date filters as the other analytics queries. */
    dailyPnl: (accountId: string | null, openFrom?: number, openTo?: number) =>
      ["analytics", "dailyPnl", accountId, openFrom, openTo] as const,
  },
  news: {
    /** `GET /accounts/{id}/news/events?start=&end=&impact=` — persisted
     * calendar-event history (Phase 6 Part B), joined client-side against
     * `analytics.dailyPnl` to flag high-impact-news days (Phase 7). Not
     * actually account-scoped server-side (`news_events` has no per-account
     * rows — see `news/api/routes.py`'s module docstring) but keyed per
     * account here anyway, matching every other resource's convention. */
    eventsForRange: (accountId: string | null, start?: number, end?: number) =>
      ["news", "eventsForRange", accountId, start, end] as const,
  },
  history: {
    /** Root key for all trade-history queries for an account — pass to
     * invalidateQueries to invalidate all filter and page combinations at once. */
    all: (accountId: string | null) => ["history", "trades", accountId] as const,
    /** `GET /accounts/{id}/journal/history` — paginated, filtered trade history. */
    trades: (accountId: string | null, filtersKey: string, page: number) =>
      ["history", "trades", accountId, filtersKey, page] as const,
  },
  engine: {
    /** `GET /accounts/{id}/engine/volatility-config` — the ATR-percentile
     * volatility guard's on/off switch and thresholds/multipliers. Shared by
     * `features/settings/VolatilityGuardPanel.tsx` and the chart toolbar
     * toggle so both stay in sync via the same cache entry. */
    volatilityConfig: (accountId: string | null) =>
      ["engine", "volatilityConfig", accountId] as const,
  },
  modelTraining: {
    /** `GET /model-training/runs` — process-wide, not scoped by account (see
     * that router's docstring: one set of model weights shared by every
     * account), so unlike every other resource here this takes no
     * `accountId` arg. */
    runs: () => ["modelTraining", "runs"] as const,
    /** `GET /model-training/models` — every trained model on disk with its
     * out-of-sample walk-forward summary. Also process-wide. */
    models: () => ["modelTraining", "models"] as const,
    /** `GET /model-training/models/{name}/walk-forward` — per-fold detail
     * for one model, keyed on its name. */
    walkForward: (modelName: string) => ["modelTraining", "walkForward", modelName] as const,
  },
} as const;

