import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";

interface Props {
  /** 0..1 */
  value: number;
  color?: string;
  /** Optional threshold tick, 0..1 (e.g. 0.5 for the trade gate). */
  threshold?: number;
  label?: string;
}

function Progress({ value, color = "#0a84ff", threshold, label }: Props) {
  const reduced = useReducedMotion();
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div>
      <div className="relative h-[7px] overflow-hidden rounded-full bg-white/10">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: color }}
          initial={reduced ? { width: `${pct}%` } : { width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: [0.25, 1, 0.5, 1] }}
        />
        {threshold != null && (
          <div
            className="absolute inset-y-0 w-[2px] bg-white/40"
            style={{ left: `${threshold * 100}%` }}
            aria-hidden
          />
        )}
      </div>
      {label && <div className="tnum mt-1.5 text-[12px] text-secondary">{label}</div>}
    </div>
  );
}

export default memo(Progress);
