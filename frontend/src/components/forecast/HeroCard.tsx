import { memo } from "react";
import { useCountUp } from "../../hooks/useCountUp";
import { fmtTimestamp } from "../../lib/format";

interface Props {
  ticker: string;
  price: number | null;
  timestamp: string | null;
}

function HeroCard({ ticker, price, timestamp }: Props) {
  const displayed = useCountUp(price);
  return (
    <div className="inline-block min-w-[300px] rounded-[20px] border border-hairline bg-surface px-8 pt-[26px] pb-[22px]">
      <div className="text-[15px] font-semibold tracking-[0.05em] text-secondary uppercase">
        {ticker}
      </div>
      <div className="tnum text-[56px] leading-[1.1] font-bold tracking-[-0.03em] text-content">
        {displayed == null
          ? "Loading…"
          : `$${displayed.toLocaleString("en-US", {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}`}
      </div>
      {timestamp && (
        <div className="mt-1.5 text-[13px] text-secondary">
          Forecast generated {fmtTimestamp(timestamp)}
        </div>
      )}
    </div>
  );
}

export default memo(HeroCard);
