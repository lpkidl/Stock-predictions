import { useQuery } from "@tanstack/react-query";
import { configQuery, metricsQuery } from "../api/queries";
import { useTicker } from "../hooks/useTicker";
import { HORIZON_LABELS } from "../lib/constants";
import { fmtPct1 } from "../lib/format";
import Card from "../components/ui/Card";
import DataTable from "../components/ui/DataTable";
import Disclosure from "../components/ui/Disclosure";
import Metric from "../components/ui/Metric";
import Skeleton from "../components/ui/Skeleton";

export default function PerformancePage() {
  const [ticker] = useTicker();
  const { data: metrics, isLoading } = useQuery(metricsQuery(ticker));
  const { data: config } = useQuery(configQuery);

  const horizons = metrics?.horizons ?? {};
  const horizonKeys = Object.keys(horizons).sort((a, b) => Number(a) - Number(b));

  if (isLoading) return <Skeleton className="h-[500px]" />;

  return (
    <div className="space-y-6">
      <h2 className="text-[24px] font-semibold tracking-[-0.015em]">
        Model Performance — {ticker}
      </h2>

      <Card className="border-accent/30 bg-accent/[0.06]">
        <p className="text-[14px] leading-relaxed text-body">
          <b className="text-content">How to read these numbers:</b>{" "}
          <b>Accuracy</b> — what % of the time the model correctly called UP /
          FLAT / DOWN (3 classes; random guessing = 33%). <b>F1 Score</b> —
          balanced accuracy across all three classes, accounting for class
          imbalance. 1.0 = perfect; 0.33 = random. <b>LOOCV</b> — temporal
          Leave-One-Out Cross-Validation across rolling expanding windows; the
          most honest estimate of real-world performance.
        </p>
      </Card>

      {horizonKeys.length === 0 && (
        <Card>
          <p className="text-secondary">
            No model metrics — run <code>python main.py</code> first.
          </p>
        </Card>
      )}

      {horizonKeys.map((h) => {
        const m = horizons[h];
        return (
          <Disclosure
            key={h}
            title={`${HORIZON_LABELS[h] ?? `${h} days`} model`}
            defaultOpen={h === "1"}
          >
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div>
                <h4 className="text-[16px] font-semibold text-content">
                  Validation
                </h4>
                <p className="mt-0.5 mb-3 text-[12px] text-secondary">
                  Used only for early stopping — not for evaluation.
                </p>
                <div className="space-y-3">
                  <Metric
                    label="Accuracy"
                    value={fmtPct1(m.val?.accuracy ?? 0)}
                    title="% of val days called correctly (up/flat/down)."
                  />
                  <Metric
                    label="F1 Score"
                    value={(m.val?.f1 ?? 0).toFixed(3)}
                    title="Macro-averaged F1 across 3 classes."
                  />
                </div>
              </div>
              <div>
                <h4 className="text-[16px] font-semibold text-content">
                  Test — unseen data
                </h4>
                <p className="mt-0.5 mb-3 text-[12px] text-secondary">
                  Recent data the model never saw. This is what matters.
                </p>
                <div className="space-y-3">
                  <Metric
                    label="Accuracy"
                    value={fmtPct1(m.test?.accuracy ?? 0)}
                    title="% of test days called correctly."
                  />
                  <Metric
                    label="F1 Score"
                    value={(m.test?.f1 ?? 0).toFixed(3)}
                    title="Macro-averaged F1 on test set."
                  />
                </div>
              </div>
            </div>

            {m.loocv && (
              <div className="mt-6 border-t border-hairline pt-5">
                <h4 className="text-[16px] font-semibold text-content">
                  Temporal LOOCV
                </h4>
                <p className="mt-0.5 mb-3 text-[12px] text-secondary">
                  Each fold trains on all data up to a point and tests on the
                  single next sample — no look-ahead bias.
                </p>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <Metric
                    label="LOOCV Accuracy"
                    value={fmtPct1(m.loocv.accuracy ?? 0)}
                  />
                  <Metric label="LOOCV F1" value={(m.loocv.f1 ?? 0).toFixed(3)} />
                </div>
                {m.loocv.n_folds != null && (
                  <p className="tnum mt-2 text-[12px] text-tertiary">
                    Evaluated over {m.loocv.n_folds} folds.
                  </p>
                )}
              </div>
            )}
          </Disclosure>
        );
      })}

      {config && (
        <div>
          <h3 className="mb-3 text-[19px] font-semibold tracking-[-0.015em]">
            Model Settings
          </h3>
          <DataTable
            columns={[
              { header: "Setting", render: (r: { setting: string; value: string }) => r.setting },
              { header: "Value", render: (r) => r.value },
            ]}
            rows={config.model_settings}
            rowKey={(r) => r.setting}
          />
        </div>
      )}
    </div>
  );
}
