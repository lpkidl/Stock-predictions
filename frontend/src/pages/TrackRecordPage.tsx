import { useQuery } from "@tanstack/react-query";
import { configQuery, trackRecordQuery } from "../api/queries";
import type { OutcomeRecord } from "../api/types";
import { RANDOM_BASELINE } from "../lib/constants";
import {
  fmtCurrency,
  fmtDate,
  fmtPct1,
  fmtSignedPct1,
} from "../lib/format";
import Card from "../components/ui/Card";
import DataTable, { type Column } from "../components/ui/DataTable";
import Disclosure from "../components/ui/Disclosure";
import Metric from "../components/ui/Metric";
import Skeleton from "../components/ui/Skeleton";

const vsRandom = (acc: number) => fmtSignedPct1(acc - RANDOM_BASELINE);

export default function TrackRecordPage() {
  const { data, isLoading } = useQuery(trackRecordQuery);
  const { data: config } = useQuery(configQuery);

  if (isLoading) return <Skeleton className="h-[500px]" />;

  const summary = data?.summary;
  const hasHistory =
    !!data && (summary!.correct + summary!.incorrect + summary!.pending > 0);
  const hasResolved = !!summary && summary.correct + summary.incorrect > 0;

  const recentColumns: Column<OutcomeRecord>[] = [
    { header: "Date", render: (r) => fmtDate(r.predicted_at) },
    { header: "Ticker", render: (r) => <b className="text-content">{r.ticker}</b> },
    { header: "Horizon", render: (r) => `${r.horizon_days}d` },
    { header: "Predicted", render: (r) => r.predicted_direction.toUpperCase() },
    { header: "Conf", align: "right", render: (r) => fmtPct1(r.predicted_confidence) },
    { header: "Actual", render: (r) => (r.actual_direction ?? "—").toUpperCase() },
    {
      header: "Move %",
      align: "right",
      render: (r) =>
        r.actual_pct_change == null
          ? "—"
          : `${r.actual_pct_change >= 0 ? "+" : ""}${r.actual_pct_change.toFixed(2)}%`,
    },
    {
      header: "Result",
      render: (r) =>
        r.status === "correct" ? (
          <span className="text-up">✓ Correct</span>
        ) : (
          <span className="text-down">✕ Wrong</span>
        ),
    },
  ];

  const pendingColumns: Column<OutcomeRecord>[] = [
    { header: "Ticker", render: (r) => <b className="text-content">{r.ticker}</b> },
    { header: "Horizon", render: (r) => `${r.horizon_days}d` },
    { header: "Predicted", render: (r) => r.predicted_direction.toUpperCase() },
    { header: "Confidence", align: "right", render: (r) => fmtPct1(r.predicted_confidence) },
    { header: "Entry Price", align: "right", render: (r) => fmtCurrency(r.entry_price) },
    { header: "Outcome Date", render: (r) => r.outcome_date ?? "—" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-[24px] font-semibold tracking-[-0.015em]">
          Real-World Prediction Track Record
        </h2>
        <p className="mt-1 max-w-[75ch] text-[13px] text-secondary">
          Every time the pipeline runs, predictions are saved with an expected
          outcome date. When that date arrives, the next run fetches the actual
          closing price, applies the same deadband thresholds used during
          training, and records whether the call was right or wrong. This is
          the only accuracy that matters — not backtest accuracy, but live
          predictions.
        </p>
      </div>

      {!hasHistory ? (
        <Card>
          <p className="text-secondary">
            No prediction history yet. Run the pipeline once to start recording.
            Outcomes are resolved automatically on subsequent runs once each
            horizon passes.
          </p>
        </Card>
      ) : (
        <>
          {hasResolved ? (
            <>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <Metric
                  label="Overall Accuracy"
                  value={fmtPct1(summary!.overall_accuracy)}
                  title="Across all tickers and all horizons. Random baseline = 33%."
                />
                <Metric
                  label="Correct"
                  value={String(summary!.correct)}
                  title="Predictions that matched the actual direction"
                />
                <Metric label="Incorrect" value={String(summary!.incorrect)} />
                <Metric
                  label="Pending"
                  value={String(summary!.pending)}
                  title="Horizon not yet reached — outcome unknown"
                />
              </div>

              <div>
                <h3 className="text-[19px] font-semibold tracking-[-0.015em]">
                  Accuracy by Forecast Horizon
                </h3>
                <p className="mt-1 mb-3 text-[13px] text-secondary">
                  A random 3-class model scores 33%. Anything consistently above
                  that has edge.
                </p>
                <DataTable
                  columns={[
                    {
                      header: "Horizon",
                      render: (r: (typeof data.by_horizon)[number]) =>
                        `${r.horizon_days} day${r.horizon_days > 1 ? "s" : ""}`,
                    },
                    { header: "Evaluated", align: "right", render: (r) => String(r.evaluated) },
                    { header: "Correct", align: "right", render: (r) => String(r.correct) },
                    { header: "Accuracy", align: "right", render: (r) => fmtPct1(r.accuracy) },
                    { header: "vs Random", align: "right", render: (r) => vsRandom(r.accuracy) },
                  ]}
                  rows={data.by_horizon}
                  rowKey={(r) => String(r.horizon_days)}
                />
              </div>

              <div>
                <h3 className="mb-3 text-[19px] font-semibold tracking-[-0.015em]">
                  Accuracy by Ticker
                </h3>
                <DataTable
                  columns={[
                    {
                      header: "Ticker",
                      render: (r: (typeof data.by_ticker)[number]) => (
                        <b className="text-content">{r.ticker}</b>
                      ),
                    },
                    { header: "Evaluated", align: "right", render: (r) => String(r.evaluated) },
                    { header: "Correct", align: "right", render: (r) => String(r.correct) },
                    { header: "Accuracy", align: "right", render: (r) => fmtPct1(r.accuracy) },
                    { header: "vs Random", align: "right", render: (r) => vsRandom(r.accuracy) },
                  ]}
                  rows={data.by_ticker}
                  rowKey={(r) => r.ticker}
                />
              </div>

              <div>
                <h3 className="mb-3 text-[19px] font-semibold tracking-[-0.015em]">
                  Recent Resolved Predictions
                </h3>
                <DataTable
                  columns={recentColumns}
                  rows={data.recent}
                  rowKey={(r) => r.id}
                />
              </div>
            </>
          ) : (
            <Card>
              <p className="text-body">
                <b className="text-content">
                  {summary!.pending} prediction(s) pending.
                </b>{" "}
                No outcomes have resolved yet — the model needs at least one
                horizon to pass before accuracy can be measured. For 1-day
                predictions, run the pipeline again tomorrow.
              </p>
            </Card>
          )}

          {data.pending.length > 0 && (
            <Disclosure
              title={`⏳ ${data.pending.length} pending prediction(s) — awaiting outcome`}
            >
              <DataTable
                columns={pendingColumns}
                rows={data.pending}
                rowKey={(r) => r.id}
              />
            </Disclosure>
          )}

          <Disclosure title="📖 How outcomes are resolved">
            <div className="space-y-3 text-[14px] leading-relaxed">
              <p>
                The pipeline uses the same deadband thresholds during outcome
                resolution as it does when creating training labels. This
                ensures the accuracy score is apples-to-apples with what the
                model was actually trained to predict.
              </p>
              <table className="tnum w-full max-w-md text-left text-[13px]">
                <thead>
                  <tr className="border-b border-hairline text-secondary">
                    <th className="py-1.5 pr-4">Horizon</th>
                    <th className="py-1.5 pr-4">Deadband</th>
                    <th className="py-1.5">Meaning</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(config?.deadbands ?? {}).map(([h, db]) => (
                    <tr key={h} className="border-b border-hairline last:border-0">
                      <td className="py-1.5 pr-4">
                        {h} day{Number(h) > 1 ? "s" : ""}
                      </td>
                      <td className="py-1.5 pr-4">±{db}%</td>
                      <td className="py-1.5">
                        Move &lt; {db}% either way = FLAT
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                <b className="text-content">Outcome date</b> is calculated as
                the N-th business day after the prediction date (weekends
                excluded; note: no holiday calendar applied).
              </p>
              <p>
                <b className="text-content">Why 33% is the random baseline</b> —
                the model predicts one of three classes (UP, FLAT, DOWN). A
                coin-flip model scores 33%. Any consistent reading above ~38–40%
                over many predictions suggests the model is finding a real
                signal.
              </p>
            </div>
          </Disclosure>
        </>
      )}
    </div>
  );
}
