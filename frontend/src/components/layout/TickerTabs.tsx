import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTicker } from "../../hooks/useTicker";
import {
  pricesQuery,
  quotesQuery,
  sentimentSummaryQuery,
} from "../../api/queries";
import { fmtCurrency } from "../../lib/format";

/** Minimal sentiment chip: trend arrow + latest daily sentiment score,
 * colored by sign. Renders nothing until sentiment data exists. */
function SentimentChip({
  s,
  selected,
}: {
  s: { score: number; trend: number } | undefined;
  selected: boolean;
}) {
  if (!s) return null;
  const color =
    s.score > 0.05 ? "#30d158" : s.score < -0.05 ? "#ff453a" : "#8e8e93";
  const arrow = s.trend > 0.02 ? "▲" : s.trend < -0.02 ? "▼" : "▬";
  return (
    <span
      className="tnum text-[11px] leading-none"
      style={{ color: selected ? "#fff" : color }}
      title={`Sentiment ${s.score >= 0 ? "+" : ""}${s.score.toFixed(2)} · trend ${
        s.trend >= 0 ? "+" : ""
      }${s.trend.toFixed(2)}`}
    >
      {arrow} {s.score >= 0 ? "+" : ""}
      {s.score.toFixed(2)}
    </span>
  );
}

/** Prominent per-ticker tab bar (replaces the header dropdown). Selection is
 * held in the ?t= param via useTicker, so it persists and syncs across tabs.
 * Selecting a ticker prefetches its 1y chart so the Price Chart tab is warm. */
function TickerTabs() {
  const reduced = useReducedMotion();
  const [ticker, setTicker, tickers] = useTicker();
  const { data: quotes } = useQuery(quotesQuery);
  const { data: sentiment } = useQuery(sentimentSummaryQuery);
  const queryClient = useQueryClient();

  if (tickers.length === 0) return null;

  const select = (t: string) => {
    setTicker(t);
    void queryClient.prefetchQuery(pricesQuery(t, "1y"));
  };

  return (
    <nav
      className="flex max-w-full gap-[3px] overflow-x-auto rounded-2xl border border-hairline bg-surface p-[4px]"
      aria-label="Select ticker"
    >
      {tickers.map((t) => {
        const selected = t === ticker;
        const price = quotes?.[t]?.price ?? null;
        return (
          <button
            key={t}
            type="button"
            aria-pressed={selected}
            onClick={() => select(t)}
            className={`relative flex-1 rounded-[13px] px-4 py-2 text-center whitespace-nowrap transition-colors duration-150 ${
              selected ? "text-white" : "text-secondary hover:text-body"
            }`}
          >
            {selected && (
              <motion.span
                layoutId="ticker-thumb"
                className="absolute inset-0 rounded-[13px] bg-elev shadow-[0_1px_4px_rgba(0,0,0,0.35)]"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 40 }
                }
              />
            )}
            <span className="relative z-10 flex flex-col items-center gap-0.5">
              <span className="text-[15px] font-semibold tracking-[-0.01em]">{t}</span>
              <span
                className={`tnum text-[12px] ${selected ? "text-white/70" : "text-tertiary"}`}
              >
                {price != null ? fmtCurrency(price) : "—"}
              </span>
              <SentimentChip s={sentiment?.[t]} selected={selected} />
            </span>
          </button>
        );
      })}
    </nav>
  );
}

export default memo(TickerTabs);
