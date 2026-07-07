const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});
const usd0 = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

export const fmtCurrency = (v: number | null | undefined) =>
  v == null ? "—" : usd.format(v);

export const fmtCurrency0 = (v: number | null | undefined) =>
  v == null ? "—" : usd0.format(v);

/** 0.4267 → "43%" */
export const fmtPct = (v: number | null | undefined) =>
  v == null ? "—" : `${Math.round(v * 100)}%`;

/** 0.4267 → "42.7%" */
export const fmtPct1 = (v: number | null | undefined) =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;

/** 0.042 → "+4.2%" (signed) */
export const fmtSignedPct1 = (v: number | null | undefined) =>
  v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;

/** "2026-07-03T00:13:22.821430+00:00" → "2026-07-03 00:13:22" */
export const fmtTimestamp = (ts: string | null | undefined) =>
  ts ? ts.slice(0, 19).replace("T", " ") : "";

export const fmtDate = (ts: string | null | undefined) =>
  ts ? ts.slice(0, 10) : "—";
