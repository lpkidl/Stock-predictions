import { memo } from "react";

/** Sticky translucent header — app identity. Ticker selection lives in the
 * prominent TickerTabs bar below the header. */
function GlassHeader() {
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
      </div>
    </header>
  );
}

export default memo(GlassHeader);
