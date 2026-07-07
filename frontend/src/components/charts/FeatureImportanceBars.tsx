import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { FEATURE_LABELS } from "../../lib/constants";

interface Props {
  features: { name: string; score: number }[];
}

/** Horizontal importance bars — width-animated, opacity scaled by weight,
 * matching the old Plotly rendering. */
function FeatureImportanceBars({ features }: Props) {
  const reduced = useReducedMotion();
  if (features.length === 0) return null;
  const maxScore = Math.max(...features.map((f) => f.score)) || 1;

  return (
    <div className="space-y-3">
      {features.map((f, i) => {
        const frac = f.score / maxScore;
        const alpha = 0.35 + 0.65 * frac;
        return (
          <div key={f.name}>
            <div className="mb-1 flex items-baseline justify-between gap-4">
              <span className="truncate text-[13px] text-body">
                {FEATURE_LABELS[f.name] ?? f.name}
              </span>
              <span className="tnum shrink-0 text-[12px] text-secondary">
                {f.score.toFixed(3)}
              </span>
            </div>
            <div className="h-[10px] overflow-hidden rounded-[5px] bg-white/[0.06]">
              <motion.div
                className="h-full rounded-[5px]"
                style={{ background: `rgba(10,132,255,${alpha.toFixed(2)})` }}
                initial={reduced ? { width: `${frac * 100}%` } : { width: 0 }}
                animate={{ width: `${frac * 100}%` }}
                transition={{
                  duration: 0.5,
                  delay: reduced ? 0 : i * 0.03,
                  ease: [0.25, 1, 0.5, 1],
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default memo(FeatureImportanceBars);
