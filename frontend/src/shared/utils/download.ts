import { getToken } from "@/shared/api/client";

/**
 * Triggers a browser download of a server-generated file, e.g. a bulk CSV
 * export endpoint that already sets its own `Content-Disposition: attachment`
 * header (`journal/export/dataset`, `market-data/candles/export`,
 * `order-book/export`). Unlike `downloadJson`/`downloadCsv`, which build the
 * blob client-side from data already in memory, this fetches `path` (with
 * the same bearer-token auth every other `/api` call uses — a plain
 * `<a href>`/`window.location` download can't attach that header) and
 * re-blobs the response, since there is no other precedent in this app for
 * downloading an authenticated server-generated attachment.
 *
 * `path` is relative to `/api` (mirrors `shared/api/client.ts`'s `BASE`).
 * `fallbackFilename` is used only if the response carries no
 * `Content-Disposition` filename to read.
 */
export async function downloadFileFromApi(path: string, fallbackFilename: string): Promise<void> {
  const token = getToken();
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(`/api${path}`, { headers });
  if (!res.ok) {
    throw new Error(`Export failed (${res.status})`);
  }
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(disposition);
  const filename = match?.[1] ?? fallbackFilename;
  const blob = await res.blob();
  downloadBlob(blob, filename);
}

/**
 * Helper utility to trigger a browser download of JSON data.
 */
export function downloadJson(data: unknown, filename: string): void {
  try {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    downloadBlob(blob, filename);
  } catch (err) {
    console.error("Failed to download JSON:", err);
  }
}

/**
 * Helper utility to trigger a browser download of tabular data as CSV.
 * Column set is the union of keys across all rows (rows need not be
 * uniform), in first-seen order, so callers can pass loosely-shaped data.
 */
export function downloadCsv(rows: Record<string, unknown>[], filename: string): void {
  try {
    if (rows.length === 0) {
      console.error("Failed to download CSV: no rows to export");
      return;
    }
    const headers: string[] = [];
    const seen = new Set<string>();
    for (const row of rows) {
      for (const key of Object.keys(row)) {
        if (!seen.has(key)) {
          seen.add(key);
          headers.push(key);
        }
      }
    }
    const escape = (value: unknown): string => {
      if (value === null || value === undefined) return "";
      const str = typeof value === "string" ? value : JSON.stringify(value);
      return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
    };
    const lines = [
      headers.join(","),
      ...rows.map((row) => headers.map((h) => escape(row[h])).join(",")),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    downloadBlob(blob, filename);
  } catch (err) {
    console.error("Failed to download CSV:", err);
  }
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
