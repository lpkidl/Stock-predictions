import { memo, useEffect, useRef } from "react";
import {
  AreaSeries,
  ColorType,
  LineStyle,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";
import type { SeriesPoint } from "../../api/types";
import { CHART_COLORS } from "../../lib/constants";

interface Props {
  rsi: SeriesPoint[];
  lastRsi: number | null;
}

function rsiStatus(v: number): string {
  if (v > 70) return "Overbought — could pull back";
  if (v < 30) return "Oversold — could bounce";
  return "Neutral";
}

/** RSI-14 area chart pinned to a 0–100 scale with 30/70 guides. */
function RsiChart({ rsi, lastRsi }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: CHART_COLORS.text,
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
      },
      grid: {
        vertLines: { color: CHART_COLORS.grid },
        horzLines: { color: CHART_COLORS.grid },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false },
      crosshair: {
        horzLine: { labelBackgroundColor: "#2c2c2e" },
        vertLine: { labelBackgroundColor: "#2c2c2e" },
      },
      height: 220,
      autoSize: true,
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor: CHART_COLORS.accent,
      lineWidth: 2,
      topColor: "rgba(10,132,255,0.16)",
      bottomColor: "rgba(10,132,255,0)",
      priceLineVisible: false,
      autoscaleInfoProvider: () => ({
        priceRange: { minValue: 0, maxValue: 100 },
      }),
    });

    series.createPriceLine({
      price: 70,
      color: "#ff453a",
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      title: "Overbought",
    });
    series.createPriceLine({
      price: 30,
      color: "#30d158",
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      title: "Oversold",
    });

    chartRef.current = chart;
    seriesRef.current = series;
    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!chart || !series) return;
    series.setData(rsi.map((p) => ({ time: p.time as Time, value: p.value })));
    chart.timeScale().fitContent();
  }, [rsi]);

  return (
    <div>
      {lastRsi != null && (
        <p className="tnum mb-2 text-[14px] text-body">
          Currently <b className="text-content">{lastRsi.toFixed(1)}</b> ·{" "}
          {rsiStatus(lastRsi)}
        </p>
      )}
      <div ref={containerRef} className="h-[220px] w-full" />
      <p className="mt-2 text-[12px] text-tertiary">
        RSI above 70 = overbought · below 30 = oversold · 30–70 = neutral
      </p>
    </div>
  );
}

export default memo(RsiChart);
