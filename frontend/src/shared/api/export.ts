/**
 * Shared constants for the trade-history/analytics CSV/JSON export flows
 * (`features/analytics/export.ts`, `features/history/exportHistory.ts`).
 */

/** `/journal/history` caps `limit` at 500 (see `backend/src/journal/api/routes.py`)
 * — exports page through it at this size so a bot/account with a long trade
 * history still exports in full. */
export const EXPORT_PAGE_SIZE = 500;
