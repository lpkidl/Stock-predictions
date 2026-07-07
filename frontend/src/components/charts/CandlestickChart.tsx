import { memo, useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";
import type { Candle, HorizonPrediction, SeriesPoint } from "../../api/types";
import {
  CHART_COLORS,
  DIRECTION_ARROWS,
  DIRECTION_COLORS,
  HORIZON_LABELS,
} from "../../lib/constants";

interface Props {
  candles: Candle[];
  sma20: SeriesPoint[];
  sma50: SeriesPoint[];
  horizons: Record<string, HorizonPrediction>;
  lastClose: number | null;
}

/** N business days after `date` (weekends skipped — mirrors ui/app.py bday()). */
function addBusinessDays(date: Date, n: number): Date {
  const d = new Date(date);
  let remaining = n;
  while (remaining > 0) {
    d.setDate(d.getDate() + 1);
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) remaining -= 1;
  }
  return d;
}

const toTime = (d: Date): Time => d.toISOString().slice(0, 10) as Time;

const BASE_OPTIONS = {
  layout: {
    background: { type: ColorType.Solid, color: "transparent" },
    textColor: CHART_COLORS.text,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif',
    panes: { separatorColor: CHART_COLORS.grid },
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
};

/** Candles + SMA20/50 overlays + volume pane + forecast markers at future
 * business days. Chart instance is created once and fed via setData. */
function CandlestickChart({ candles, sma20, sma50, horizons, lastClose }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<{
    candle: ISeriesApi<"Candlestick">;
    sma20: ISeriesApi<"Line">;
    sma50: ISeriesApi<"Line">;
    volume: ISeriesApi<"Histogram">;
    forecasts: ISeriesApi<"Line">[];
  } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      ...BASE_OPTIONS,
      height: 460,
      autoSize: true,
    });

    const candle = chart.addSeries(CandlestickSeries, {
      upColor: DIRECTION_COLORS.up,
      downColor: DIRECTION_COLORS.down,
      wickUpColor: DIRECTION_COLORS.up,
      wickDownColor: DIRECTION_COLORS.down,
      borderVisible: false,
    });
    const sma20Series = chart.addSeries(LineSeries, {
      color: CHART_COLORS.sma20,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const sma50Series = chart.addSeries(LineSeries, {
      color: CHART_COLORS.sma50,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const volume = chart.addSeries(
      HistogramSeries,
      { priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false },
      1,
    );

    chartRef.current = chart;
    seriesRef.current = { candle, sma20: sma20Series, sma50: sma50Series, volume, forecasts: [] };

    const panes = chart.panes();
    if (panes[1]) panes[1].setStretchFactor(22);
    if (panes[0]) panes[0].setStretchFactor(78);

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  const volumeData = useMemo(
    () =>
      candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color:
          c.close >= c.open ? "rgba(48,209,88,0.55)" : "rgba(255,69,58,0.55)",
      })),
    [candles],
  );

  useEffect(() => {
    const chart = chartRef.current;
    const s = seriesRef.current;
    if (!chart || !s || candles.length === 0) return;

    s.candle.setData(candles.map((c) => ({ ...c, time: c.time as Time })));
    s.sma20.setData(sma20.map((p) => ({ time: p.time as Time, value: p.value })));
    s.sma50.setData(sma50.map((p) => ({ time: p.time as Time, value: p.value })));
    s.volume.setData(volumeData);

    // Rebuild forecast series: a dotted line from the last candle to the
    // horizon's business day at last-close level, ending in a labeled marker.
    for (const f of s.forecasts) chart.removeSeries(f);
    s.forecasts = [];

    if (lastClose != null && Object.keys(horizons).length > 0) {
      const lastDate = new Date(candles[candles.length - 1].time);
      for (const h of Object.keys(horizons).sort((a, b) => Number(a) - Number(b))) {
        const pred = horizons[h];
        const d = pred.direction ?? "flat";
        const color = DIRECTION_COLORS[d] ?? "#888";
        const futTime = toTime(addBusinessDays(lastDate, Number(h)));
        const line = chart.addSeries(LineSeries, {
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        line.setData([
          { time: candles[candles.length - 1].time as Time, value: lastClose },
          { time: futTime, value: lastClose },
        ]);
        createSeriesMarkers(line, [
          {
            time: futTime,
            position: "inBar",
            shape: "circle",
            color,
            text: `${HORIZON_LABELS[h] ?? `${h}d`} ${DIRECTION_ARROWS[d]} ${Math.round(
              (pred.confidence ?? 0) * 100,
            )}%`,
          },
        ]);
        s.forecasts.push(line);
      }
    }

    chart.timeScale().fitContent();
  }, [candles, sma20, sma50, volumeData, horizons, lastClose]);

  return (
    <div>
      <div ref={containerRef} className="h-[460px] w-full" />
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-secondary">
        <span>
          <span style={{ color: CHART_COLORS.sma20 }}>┄</span> 20-Day Avg
        </span>
        <span>
          <span style={{ color: CHART_COLORS.sma50 }}>┄</span> 50-Day Avg
        </span>
        <span>● Forecast markers at each horizon (colored by direction)</span>
      </div>
    </div>
  );
}

export default memo(CandlestickChart);
