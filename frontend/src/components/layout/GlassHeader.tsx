import { memo } from "react";

interface Props {
  tickers: string[];
  ticker: string;
  onTickerChange: (t: string) => void;
}

/** Sticky translucent header — app identity left, ticker picker right. */
function GlassHeader({ tickers, ticker, onTickerChange }: Props) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-black/60 backdrop-blur-xl backdrop-saturate-150">
      <div className="mx-auto flex max-w-[1100px] items-center justify-between px-6 py-3">
        <div className="flex items-baseline gap-3">
          <span className="text-[17px] font-semibold tracking-[-0.015em]">
            Stock Forecaster
          </span>
          <span className="hidden text-[13px] text-secondary sm:inline">
            ML forecasts, grounded in sentiment and market structure
          </span>
        </div>
        <label className="flex items-center gap-2 text-[13px] text-secondary">
          <span className="hidden sm:inline">Ticker</span>
          <select
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value)}
            className="rounded-[10px] border border-white/[0.12] bg-surface-2 px-3 py-1.5 text-[14px] font-medium text-content outline-none focus:border-accent"
            aria-label="Stock to analyse"
          >
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>
    </header>
  );
}

export default memo(GlassHeader);
