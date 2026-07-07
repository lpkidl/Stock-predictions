import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  featuresQuery,
  predictionsQuery,
  pricesQuery,
} from "../api/queries";
import { useTicker } from "../hooks/useTicker";
import { FEATURE_EXPLANATIONS, PERIOD_OPTIONS } from "../lib/constants";
import CandlestickChart from "../components/charts/CandlestickChart";
import FeatureImportanceBars from "../components/charts/FeatureImportanceBars";
import RsiChart from "../components/charts/RsiChart";
import Card from "../components/ui/Card";
import Disclosure from "../components/ui/Disclosure";
import SegmentedControl from "../components/ui/SegmentedControl";
import Skeleton from "../components/ui/Skeleton";

export default function PriceChartPage() {
  const [ticker] = useTicker();
  const [period, setPeriod] = useState("1y");
  const [topN, setTopN] = useState(10);

  const { data: prices, isLoading, isError } = useQuery(pricesQuery(ticker, period));
  const { data: predictions } = useQuery(predictionsQuery);
  const { data: features } = useQuery(featuresQuery(ticker, topN));

  const horizons = predictions?.[ticker]?.horizons ?? {};
  const explained = (features?.features ?? []).filter(
    (f) => FEATURE_EXPLANATIONS[f.name],
  );

  return (
    <div className="space-y-6">
      <Card>
        <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
          <h2 className="text-[21px] font-semibold tracking-[-0.015em]">
            {ticker} — Price History + Directional Forecast
          </h2>
          <SegmentedControl
            layoutId="period-thumb"
            options={PERIOD_OPTIONS}
            value={period}
            onChange={setPeriod}
          />
        </div>
        {isLoading ? (
          <Skeleton className="h-[460px]" />
        ) : isError || !prices ? (
          <p className="py-20 text-center text-secondary">
            No price data for {ticker}.
          </p>
        ) : (
          <CandlestickChart
            candles={prices.candles}
            sma20={prices.sma20}
            sma50={prices.sma50}
            horizons={horizons}
            lastClose={prices.last_close}
          />
        )}
      </Card>

      <Card>
        <h3 className="mb-4 text-[19px] font-semibold tracking-[-0.015em]">
          RSI Momentum
        </h3>
        {isLoading ? (
          <Skeleton className="h-[220px]" />
        ) : prices && prices.rsi14.length > 0 ? (
          <RsiChart rsi={prices.rsi14} lastRsi={prices.last_rsi} />
        ) : (
          <p className="text-secondary">Not enough history for RSI.</p>
        )}
      </Card>

      <Card>
        <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-[19px] font-semibold tracking-[-0.015em]">
              Which indicators drove the forecast?
            </h3>
            <p className="mt-1 text-[13px] text-secondary">
              Scores from the 1-day model. Longer bar = the model leaned on this
              indicator more heavily.
            </p>
          </div>
          <label className="flex items-center gap-3 text-[13px] text-secondary">
            Show {topN}
            <input
              type="range"
              min={3}
              max={32}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="w-36 accent-accent"
              aria-label="Indicators shown in chart"
            />
          </label>
        </div>
        {features && features.features.length > 0 ? (
          <FeatureImportanceBars features={features.features} />
        ) : (
          <p className="text-secondary">
            No feature data — run <code>python main.py</code> first.
          </p>
        )}
      </Card>

      {explained.length > 0 && (
        <Disclosure title="📖 What do these indicators mean?">
          <ul className="list-disc space-y-2 pl-5">
            {explained.map((f) => (
              <li key={f.name}>{FEATURE_EXPLANATIONS[f.name]}</li>
            ))}
          </ul>
        </Disclosure>
      )}
    </div>
  );
}
