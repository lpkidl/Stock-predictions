import { memo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  predictionsQuery,
  sentimentSummaryQuery,
  tradesQuery,
} from "../../api/queries";
import { useTicker } from "../../hooks/useTicker";
import { DIRECTION_ARROWS, DIRECTION_COLORS, HORIZON_LABELS } from "../../lib/constants";
import { fmtPct } from "../../lib/format";
import type { Direction } from "../../api/types";

const SURFACE = "#1c1c1e";
const ORDER: Direction[] = ["up", "flat", "down"];
const DIR_LABEL: Record<Direction, string> = { up: "Up", flat: "Flat", down: "Down" };

/** Donut of the 3-class forecast probabilities. Every segment is gapped and
 * directly labeled (color is never the sole channel — green/amber sit in the
 * CVD floor band), with the winning call as the center hero number. */
function ForecastDonut({
  probs,
  direction,
  confidence,
}: {
  probs: { up: number; flat: number; down: number };
  direction: Direction;
  confidence: number;
}) {
  const R = 46;
  const SW = 13;
  const C = 2 * Math.PI * R;
  const GAP = 7; // circumference px between segments (secondary encoding)

  let offset = 0;
  const arcs = ORDER.map((k) => {
    const val = Math.max(0, Math.min(1, probs[k] ?? 0));
    const len = val * C;
    const drawn = Math.max(0, len - GAP);
    const arc = (
      <circle
        key={k}
        cx="60"
        cy="60"
        r={R}
        fill="none"
        stroke={DIRECTION_COLORS[k]}
        strokeWidth={SW}
        strokeDasharray={`${drawn} ${C - drawn}`}
        strokeDashoffset={-offset}
        transform="rotate(-90 60 60)"
      />
    );
    offset += len;
    return arc;
  });

  return (
    <div className="relative h-[120px] w-[120px] shrink-0">
      <svg viewBox="0 0 120 120" className="h-full w-full">
        <circle cx="60" cy="60" r={R} fill="none" stroke={SURFACE} strokeWidth={SW} />
        {arcs}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="text-[26px] leading-none font-bold"
          style={{ color: DIRECTION_COLORS[direction] }}
        >
          {DIRECTION_ARROWS[direction]}
        </span>
        <span className="tnum mt-0.5 text-[15px] font-semibold text-content">
          {fmtPct(confidence)}
        </span>
        <span className="text-[10px] tracking-wide text-tertiary uppercase">1-day</span>
      </div>
    </div>
  );
}

/** Diverging sentiment meter: -1 … 0 … +1 with a marker at the score. */
function SentimentMeter({ score, trend }: { score: number; trend: number }) {
  const pct = ((Math.max(-1, Math.min(1, score)) + 1) / 2) * 100;
  const color = score > 0.05 ? "#30d158" : score < -0.05 ? "#ff453a" : "#8e8e93";
  const arrow = trend > 0.02 ? "▲" : trend < -0.02 ? "▼" : "▬";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <span className="text-secondary">Sentiment</span>
        <span className="tnum font-semibold" style={{ color }}>
          {arrow} {score >= 0 ? "+" : ""}
          {score.toFixed(2)}
        </span>
      </div>
      <div
        className="relative h-[8px] rounded-full"
        style={{
          background:
            "linear-gradient(to right, rgba(255,69,58,0.45), rgba(255,255,255,0.10) 50%, rgba(48,209,88,0.45))",
        }}
      >
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/25" />
        <div
          className="absolute top-1/2 h-[12px] w-[12px] -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-black/40"
          style={{ left: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

/** Trade-confidence meter: 0…100% fill with a 50% tradeable threshold tick. */
function TradeMeter({ value }: { value: number | null }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(100, value * 100));
  const tradeable = value != null && value >= 0.5;
  const color = tradeable ? "#30d158" : "#8e8e93";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <span className="text-secondary">Trade score</span>
        <span className="tnum font-semibold" style={{ color }}>
          {value == null ? "—" : fmtPct(value)}
          {tradeable && <span className="ml-1 text-[10px] uppercase">tradeable</span>}
        </span>
      </div>
      <div className="relative h-[8px] overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
        {/* 50% tradeable threshold */}
        <div className="absolute top-[-2px] bottom-[-2px] left-1/2 w-px bg-white/40" />
      </div>
    </div>
  );
}

/** Per-ticker signal board for the selected tab: forecast donut + probability
 * legend + sentiment & trade meters + a compact multi-horizon strip. */
function TickerStatsPanel({ className = "" }: { className?: string }) {
  const [ticker] = useTicker();
  const { data: predictions } = useQuery(predictionsQuery);
  const { data: sentiment } = useQuery(sentimentSummaryQuery);
  const { data: trades } = useQuery(tradesQuery);

  const tp = predictions?.[ticker];
  const h1 = tp?.horizons?.["1"];
  const s = sentiment?.[ticker];

  let blended: number | null = null;
  for (const t of [...(trades?.active ?? []), ...(trades?.skipped ?? [])]) {
    if (t.ticker === ticker && t.blended_confidence != null) blended = t.blended_confidence;
  }

  if (!h1) {
    return (
      <div className={`rounded-[20px] border border-hairline bg-surface px-5 py-4 ${className}`}>
        <p className="text-[13px] text-secondary">No forecast for {ticker} yet.</p>
      </div>
    );
  }

  const dir = (h1.direction ?? "flat") as Direction;
  const probs = h1.probabilities;

  return (
    <div className={`rounded-[20px] border border-hairline bg-surface px-5 py-4 ${className}`}>
      <h3 className="mb-3 text-[13px] font-semibold tracking-[0.03em] text-secondary uppercase">
        {ticker} — Signals
      </h3>

      <div className="flex items-center gap-4">
        <ForecastDonut probs={probs} direction={dir} confidence={h1.confidence} />

        {/* probability legend — direct labels satisfy the CVD secondary-encoding rule */}
        <div className="flex-1 space-y-1.5">
          {ORDER.map((k) => (
            <div key={k} className="flex items-center gap-2 text-[13px]">
              <span
                className="inline-block h-[10px] w-[10px] rounded-[3px]"
                style={{ background: DIRECTION_COLORS[k] }}
              />
              <span className="text-body">{DIR_LABEL[k]}</span>
              <span className="tnum ml-auto font-semibold text-content">
                {fmtPct(probs[k])}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        {s ? (
          <SentimentMeter score={s.score} trend={s.trend} />
        ) : (
          <div className="text-[12px] text-tertiary">Sentiment — no data yet</div>
        )}
        <TradeMeter value={blended} />
      </div>

      {/* multi-horizon forecast strip */}
      <div className="mt-4 grid grid-cols-4 gap-2 border-t border-hairline pt-3">
        {["1", "3", "5", "10"].map((h) => {
          const hp = tp?.horizons?.[h];
          const hd = (hp?.direction ?? null) as Direction | null;
          return (
            <div key={h} className="text-center">
              <div className="text-[11px] text-tertiary">{HORIZON_LABELS[h] ?? `${h}d`}</div>
              <div
                className="tnum text-[13px] font-semibold"
                style={{ color: hd ? DIRECTION_COLORS[hd] : "#6e6e73" }}
              >
                {hd ? `${DIRECTION_ARROWS[hd]} ${fmtPct(hp?.confidence)}` : "—"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default memo(TickerStatsPanel);
