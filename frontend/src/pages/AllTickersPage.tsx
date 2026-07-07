import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { predictionsQuery, quotesQuery } from "../api/queries";
import type { Direction } from "../api/types";
import {
  DIRECTION_ARROWS,
  DIRECTION_COLORS,
  HORIZON_LABELS,
} from "../lib/constants";
import { fmtCurrency, fmtPct } from "../lib/format";
import Card from "../components/ui/Card";
import DataTable, { type Column } from "../components/ui/DataTable";
import Disclosure from "../components/ui/Disclosure";
import Skeleton from "../components/ui/Skeleton";

const HORIZON_KEYS = ["1", "3", "5", "10"];

interface Row {
  ticker: string;
  price: number | null;
  cells: Record<string, { direction: Direction; confidence: number } | null>;
  regime: string;
  updated: string;
}

function DirectionCell({
  cell,
}: {
  cell: { direction: Direction; confidence: number } | null;
}) {
  if (!cell) return <span className="text-tertiary">—</span>;
  return (
    <span style={{ color: DIRECTION_COLORS[cell.direction] }}>
      {DIRECTION_ARROWS[cell.direction]} {cell.direction.toUpperCase()}{" "}
      <span className="text-secondary">({fmtPct(cell.confidence)})</span>
    </span>
  );
}

export default function AllTickersPage() {
  const { data: predictions, isLoading } = useQuery(predictionsQuery);
  const { data: quotes } = useQuery(quotesQuery);

  const rows: Row[] = useMemo(() => {
    if (!predictions) return [];
    return Object.entries(predictions).map(([ticker, pred]) => {
      const cells: Row["cells"] = {};
      for (const h of HORIZON_KEYS) {
        const p = pred.horizons?.[h];
        cells[h] = p
          ? { direction: p.direction, confidence: p.confidence }
          : null;
      }
      return {
        ticker,
        price: quotes?.[ticker]?.price ?? null,
        cells,
        regime: pred.horizons?.["1"]?.regime ?? "—",
        updated: (pred.timestamp ?? "").slice(0, 16).replace("T", " "),
      };
    });
  }, [predictions, quotes]);

  const columns: Column<Row>[] = useMemo(
    () => [
      {
        header: "Ticker",
        render: (r) => <b className="text-content">{r.ticker}</b>,
      },
      {
        header: "Current Price",
        align: "right",
        render: (r) => fmtCurrency(r.price),
      },
      ...HORIZON_KEYS.map(
        (h): Column<Row> => ({
          header: HORIZON_LABELS[h],
          render: (r) => <DirectionCell cell={r.cells[h]} />,
        }),
      ),
      { header: "Regime", render: (r) => r.regime },
      { header: "Updated", render: (r) => r.updated },
    ],
    [],
  );

  if (isLoading) return <Skeleton className="h-[400px]" />;

  return (
    <div className="space-y-6">
      <h2 className="text-[24px] font-semibold tracking-[-0.015em]">
        Multi-Day Forecasts — All Tickers
      </h2>

      {rows.length > 0 ? (
        <DataTable columns={columns} rows={rows} rowKey={(r) => r.ticker} />
      ) : (
        <Card>
          <p className="text-secondary">
            No predictions yet — run the pipeline first.
          </p>
        </Card>
      )}

      <Disclosure title="📖 Full Glossary — plain English explanations">
        <div className="space-y-3 text-[14px] leading-relaxed">
          <p>
            <b className="text-content">Tomorrow / 3 Days / 5 Days / 10 Days</b>{" "}
            — each is a separate per-regime ensemble model (XGBoost + Logistic
            Regression) trained to predict whether the stock will be UP, FLAT,
            or DOWN after N trading days.
          </p>
          <p>
            <b className="text-content">Direction</b> — UP / FLAT / DOWN.
            Classified using a per-horizon deadband: ±0.3% for 1d, ±0.7% for
            3d, ±1.0% for 5d, ±1.5% for 10d.
          </p>
          <p>
            <b className="text-content">Confidence</b> — the winning class
            probability from the ensemble.
          </p>
          <p>
            <b className="text-content">Probability Bars</b> — full 3-class
            probability distribution (up / flat / down). All three bars sum to
            100%.
          </p>
          <p>
            <b className="text-content">Regime</b> — bull (close &gt; SMA50 &gt;
            SMA200), bear (reverse), or sideways. The model selects a
            regime-specific sub-model when enough training data exists.
          </p>
          <p>
            <b className="text-content">Accuracy</b> — % of time the model
            called the correct direction (3 classes; random = 33%); above 40%+
            consistently is a meaningful edge.
          </p>
          <p>
            <b className="text-content">F1 Score</b> — macro-averaged F1 across
            all three classes, accounting for class imbalance. 1.0 = perfect;
            0.33 = random.
          </p>
          <p>
            <b className="text-content">Temporal LOOCV</b> — expanding-window
            Leave-One-Out CV: each fold trains on all past data and tests on a
            single future sample. No look-ahead bias.
          </p>
          <hr className="border-hairline" />
          <p>
            <b className="text-content">RSI</b> — 0–100. Above 70 = overbought;
            below 30 = oversold; 30–70 = neutral.
          </p>
          <p>
            <b className="text-content">MACD</b> — gap between 12-day and
            26-day averages. Positive = short-term above long-term.
          </p>
          <p>
            <b className="text-content">Bollinger Bands</b> — price channel ±2
            std deviations around the 20-day average. Near upper = potentially
            expensive; near lower = potentially cheap.
          </p>
          <p>
            <b className="text-content">ATR-14</b> — average daily price range
            including gaps. High = volatile; low = calm.
          </p>
          <p>
            <b className="text-content">Stochastic %K / %D</b> — close vs.
            High-Low range over 14 days. Above 80 = overbought; below 20 =
            oversold.
          </p>
          <p>
            <b className="text-content">52-Week Ratios</b> — where today's
            price sits within its yearly range.
          </p>
          <p>
            <b className="text-content">1 / 5 / 20-Day Return %</b> — recent
            momentum signals.
          </p>
          <p>
            <b className="text-content">VIX</b> — the 'fear index'. High =
            stressed market; model uses it for macro context.
          </p>
          <p>
            <b className="text-content">Yield Spread (10Y-3M)</b> — negative
            spread often precedes recessions.
          </p>
          <p>
            <b className="text-content">Sector ETF Momentum</b> — how the
            stock's sector (e.g., QQQ for tech) is trending.
          </p>
          <p>
            <b className="text-content">Days to Earnings / Earnings Imminent</b>{" "}
            — captures pre-earnings volatility.
          </p>
          <p>
            <b className="text-content">Volume Ratio</b> — today's volume vs.
            20-day average. Above 1.5 = unusually heavy.
          </p>
        </div>
      </Disclosure>
    </div>
  );
}
