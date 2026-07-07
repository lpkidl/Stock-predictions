import { useQuery } from "@tanstack/react-query";
import { tradesQuery } from "../api/queries";
import type { TradeLog } from "../api/types";
import { REASON_LABELS, SIGNAL_LABELS } from "../lib/constants";
import {
  fmtCurrency,
  fmtCurrency0,
  fmtPct1,
  fmtTimestamp,
} from "../lib/format";
import Badge from "../components/ui/Badge";
import Card from "../components/ui/Card";
import DataTable, { type Column } from "../components/ui/DataTable";
import Disclosure from "../components/ui/Disclosure";
import Metric from "../components/ui/Metric";
import Progress from "../components/ui/Progress";
import Skeleton from "../components/ui/Skeleton";

function prettifyReason(log: TradeLog): string {
  const raw = log.reason ?? "—";
  if (REASON_LABELS[raw]) return REASON_LABELS[raw];
  if (raw.startsWith("low_blended_confidence"))
    return `Blended confidence too low (${fmtPct1(log.blended_confidence)} < 50% threshold)`;
  return raw;
}

function ActiveTradeCard({ log }: { log: TradeLog }) {
  const isLong = log.action === "long";
  const color = isLong ? "#30d158" : "#ff453a";
  const slDir = isLong
    ? `-$${(log.sl_distance ?? 0).toFixed(2)} below entry`
    : `+$${(log.sl_distance ?? 0).toFixed(2)} above entry`;
  const tpDir = isLong
    ? `+$${(log.tp_distance ?? 0).toFixed(2)} above entry`
    : `-$${(log.tp_distance ?? 0).toFixed(2)} below entry`;

  return (
    <Card>
      <div
        className="flex flex-wrap items-center gap-3 rounded-md py-1"
        style={{ borderLeft: `4px solid ${color}`, paddingLeft: 14 }}
      >
        <Badge color={color}>{isLong ? "LONG ↑" : "SHORT ↓"}</Badge>
        <span className="text-[19px] font-bold">{log.ticker}</span>
        <span className="text-[13px] text-secondary">
          1-day horizon · ATR ${(log.atr_used ?? 0).toFixed(2)}/day
        </span>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          label="Entry Price"
          value={fmtCurrency(log.entry_price)}
          title="Last closing price — this is where the trade would open"
        />
        <Metric
          label="Stop-Loss"
          value={fmtCurrency(log.stop_loss)}
          sub={slDir}
          title={`Exit with a loss if price reaches this level. Distance = 2 × ATR ($${(log.atr_used ?? 0).toFixed(2)})`}
        />
        <Metric
          label="Take-Profit"
          value={fmtCurrency(log.take_profit)}
          sub={tpDir}
          title={`Exit with a profit if price reaches this level. Distance = 3 × ATR ($${(log.atr_used ?? 0).toFixed(2)})`}
        />
        <Metric
          label="Position Size"
          value={`${log.position_size ?? 0} shares`}
          title={`Sized so a stop-out costs exactly ${fmtCurrency0(log.dollar_risk)} (1% of account). Risk/Reward = 1:${log.risk_reward_ratio}`}
        />
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3">
        <Metric
          label="$ at Risk"
          value={fmtCurrency0(log.dollar_risk)}
          title="Max loss if Stop-Loss is hit (1% of $100k account)"
        />
        <Metric
          label="$ Potential Gain"
          value={fmtCurrency0(log.dollar_reward)}
          title="Profit if Take-Profit is reached"
        />
        <Metric
          label="Risk / Reward"
          value={`1 : ${log.risk_reward_ratio ?? "—"}`}
          title="For every $1 risked, the target profit is $1.50"
        />
      </div>

      <div className="mt-5 text-[14px] text-body">
        <b className="text-content">Confidence breakdown</b> — ML model:{" "}
        <b className="tnum text-content">{fmtPct1(log.ml_confidence)}</b> × 65% +
        Technical confirmation:{" "}
        <b className="tnum text-content">{fmtPct1(log.tech_conf_score)}</b> × 35% ={" "}
        Blended:{" "}
        <b className="tnum text-content">{fmtPct1(log.blended_confidence)}</b> |
        Indicators aligned:{" "}
        <b className="text-content">{log.confirming_signals ?? "?"}</b>
      </div>
      <div className="mt-2">
        <Progress
          value={Math.min(log.blended_confidence ?? 0, 1)}
          threshold={0.5}
          label={`Blended confidence ${fmtPct1(log.blended_confidence)} (threshold 50%)`}
        />
      </div>

      {Object.keys(log.tech_signals).length > 0 && (
        <div className="mt-5">
          <p className="mb-2 text-[14px] font-semibold text-content">
            Technical signal breakdown
          </p>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {Object.entries(log.tech_signals).map(([key, confirmed]) => {
              const [label, tooltip] = SIGNAL_LABELS[key] ?? [key, ""];
              const c = confirmed ? "#30d158" : "#ff453a";
              return (
                <div
                  key={key}
                  className="rounded-lg p-2 text-center"
                  style={{ border: `1px solid ${c}33`, background: `${c}1a` }}
                  title={tooltip}
                >
                  <div className="text-[20px]">{confirmed ? "✅" : "❌"}</div>
                  <div className="mt-1 text-[12px] font-bold">{label}</div>
                  <div className="text-[11px]" style={{ color: c }}>
                    {confirmed ? "Confirmed" : "Not confirmed"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Card>
  );
}

export default function TradesPage() {
  const { data, isLoading } = useQuery(tradesQuery);

  if (isLoading) return <Skeleton className="h-[500px]" />;

  const skippedColumns: Column<TradeLog>[] = [
    { header: "Ticker", render: (r) => <b className="text-content">{r.ticker}</b> },
    { header: "ML Direction", render: (r) => (r.direction ?? "flat").toUpperCase() },
    { header: "ML Confidence", align: "right", render: (r) => fmtPct1(r.ml_confidence) },
    {
      header: "Tech Score",
      align: "right",
      render: (r) =>
        r.reason === "direction_flat" ? "N/A (skipped early)" : fmtPct1(r.tech_conf_score),
    },
    {
      header: "Blended",
      align: "right",
      render: (r) =>
        r.reason === "direction_flat" ? "N/A (skipped early)" : fmtPct1(r.blended_confidence),
    },
    {
      header: "Signals",
      render: (r) =>
        r.reason === "direction_flat" ? "N/A" : (r.confirming_signals ?? "—"),
    },
    { header: "Skip Reason", render: (r) => prettifyReason(r) },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-[24px] font-semibold tracking-[-0.015em]">
          Trade Execution — ATR-Based Risk Management
        </h2>
        <p className="mt-1 max-w-[75ch] text-[13px] text-secondary">
          Each ML prediction is cross-checked against 4 technical indicators
          before a trade fires. The final decision uses a blended score: 65% ML
          model confidence + 35% technical confirmation. A trade only executes
          when the blended score clears 50% and the model has a clear direction
          (UP or DOWN).
        </p>
        {data?.last_run && (
          <p className="tnum mt-1 text-[13px] text-tertiary">
            Last pipeline run: {fmtTimestamp(data.last_run)} UTC
          </p>
        )}
      </div>

      {!data || data.summary.evaluated === 0 ? (
        <Card>
          <p className="text-secondary">
            No trade log data yet — run the pipeline first (
            <code>python main.py</code>).
          </p>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Metric label="Tickers Evaluated" value={String(data.summary.evaluated)} />
            <Metric
              label="Active Trades"
              value={String(data.summary.active)}
              title="Trades that passed both confidence threshold and direction check"
            />
            <Metric
              label="Skipped"
              value={String(data.summary.skipped)}
              title="Insufficient confidence, flat direction, or no clear ML signal"
            />
            <Metric
              label="Total Exposure"
              value={`${fmtCurrency0(data.summary.total_risk)} risked`}
              sub={`Max reward if all TPs hit: ${fmtCurrency0(data.summary.total_reward)}`}
            />
          </div>

          {data.active.length > 0 ? (
            <div className="space-y-5">
              <div>
                <h3 className="text-[19px] font-semibold tracking-[-0.015em]">
                  Active Trade Signals
                </h3>
                <p className="mt-1 text-[13px] text-secondary">
                  These tickers passed all filters. Position size is calculated
                  so that hitting the Stop-Loss costs exactly 1% of the $100,000
                  account ($1,000).
                </p>
              </div>
              {data.active.map((log) => (
                <ActiveTradeCard key={log.ticker} log={log} />
              ))}
            </div>
          ) : (
            <Card>
              <p className="text-secondary">
                No active trades this run — all tickers were either flat, or the
                blended confidence was below the 50% threshold.
              </p>
            </Card>
          )}

          <div>
            <h3 className="text-[19px] font-semibold tracking-[-0.015em]">
              Why Each Ticker Was Skipped
            </h3>
            <p className="mt-1 mb-3 text-[13px] text-secondary">
              Tickers skipped for 'FLAT direction' never reached the blending
              stage — the ML model itself couldn't pick a clear direction, so no
              trade is warranted.
            </p>
            <DataTable
              columns={skippedColumns}
              rows={data.skipped}
              rowKey={(r) => r.ticker}
            />
          </div>

          <Disclosure title="📖 How the execution engine works — plain English">
            <div className="space-y-4 text-[14px] leading-relaxed">
              <div>
                <b className="text-content">Step 1 — ML Direction check.</b> The
                model must predict either UP or DOWN with any confidence. If it
                outputs FLAT (too uncertain to call), the ticker is skipped
                immediately — no point blending indecision.
              </div>
              <div>
                <b className="text-content">
                  Step 2 — Technical indicator confirmation (4 signals).
                </b>{" "}
                Four independent chart signals are checked. Each one either
                agrees or disagrees with the ML direction. The fraction that
                agree becomes the technical score (0% = none agree, 100% = all
                agree): Ichimoku T/K Cross (Tenkan-sen crossed above/below
                Kijun-sen with price confirming), Ichimoku Cloud (price fully
                above or below both Senkou A &amp; B), ADX Trend Strength (ADX
                &gt; 20 plus +DI/−DI direction), and the BBW Volatility Gate
                (Bollinger Band Width between 1%–12%: too narrow = coiling, too
                wide = panic).
              </div>
              <div>
                <b className="text-content">Step 3 — Confidence blending.</b>{" "}
                Blended = ML confidence × 0.65 + Technical score × 0.35. A 52%
                ML signal with 3/4 indicators confirming → ~62% blended. Below
                50% → skip.
              </div>
              <div>
                <b className="text-content">
                  Step 4 — Position sizing (ATR-based).
                </b>{" "}
                Shares = floor($1,000 risk ÷ Stop-Loss distance). Stop-Loss
                distance = 2 × ATR; Take-Profit = 3 × ATR above/below entry →
                always a 1.5:1 reward/risk ratio. $1,000 = 1% of the $100,000
                simulated account. Maximum loss per trade is always $1,000.
              </div>
            </div>
          </Disclosure>
        </>
      )}
    </div>
  );
}
