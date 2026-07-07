import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { fmtPct } from "../../lib/format";

interface Props {
  label: string;
  value: number; // 0..1
  color: string;
}

function ProbabilityBar({ label, value, color }: Props) {
  const reduced = useReducedMotion();
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="mt-[9px]">
      <div className="mb-[3px] flex justify-between text-[12px] text-secondary">
        <span>{label}</span>
        <span className="tnum">{fmtPct(value)}</span>
      </div>
      <div className="h-[6px] overflow-hidden rounded-[5px] bg-white/10">
        <motion.div
          className="h-full rounded-[5px]"
          style={{ background: color }}
          initial={reduced ? { width: `${pct}%` } : { width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: [0.25, 1, 0.5, 1] }}
        />
      </div>
    </div>
  );
}

export default memo(ProbabilityBar);
