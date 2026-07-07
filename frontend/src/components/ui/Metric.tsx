import { memo } from "react";

interface Props {
  label: string;
  value: string;
  sub?: string;
  title?: string;
}

/** Apple-style stat tile: quiet label, prominent tabular value. */
function Metric({ label, value, sub, title }: Props) {
  return (
    <div
      className="rounded-2xl border border-hairline bg-surface px-[18px] py-[14px]"
      title={title}
    >
      <div className="text-[13px] font-medium text-secondary">{label}</div>
      <div className="tnum mt-0.5 text-[26px] font-semibold tracking-[-0.02em] text-content">
        {value}
      </div>
      {sub && <div className="tnum mt-0.5 text-[12px] text-tertiary">{sub}</div>}
    </div>
  );
}

export default memo(Metric);
