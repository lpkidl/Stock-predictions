import { memo } from "react";
import { motion } from "framer-motion";
import type { HorizonPrediction } from "../../api/types";
import { DIRECTION_ARROWS, DIRECTION_COLORS } from "../../lib/constants";
import { fmtPct } from "../../lib/format";
import ProbabilityBar from "./ProbabilityBar";

interface Props {
  label: string;
  prediction: HorizonPrediction;
}

const entrance = {
  hidden: { opacity: 0, scale: 0.98, y: 6 },
  show: { opacity: 1, scale: 1, y: 0 },
};

function ForecastCard({ label, prediction }: Props) {
  const d = prediction.direction ?? "flat";
  const color = DIRECTION_COLORS[d] ?? "#888";
  const arrow = DIRECTION_ARROWS[d] ?? "→";
  const probs = prediction.probabilities ?? { up: 0, flat: 0, down: 0 };

  return (
    <motion.div
      variants={entrance}
      className="rounded-card border border-hairline bg-surface px-[26px] pt-[22px] pb-[18px] transition-transform duration-150 hover:-translate-y-[2px]"
    >
      <div className="text-[12px] font-semibold tracking-[0.06em] text-secondary uppercase">
        {label}
      </div>
      <div
        className="mt-2 text-[30px] leading-[1.1] font-bold tracking-[-0.02em]"
        style={{ color }}
      >
        {arrow}&nbsp;&nbsp;{d.toUpperCase()}
      </div>
      <div className="mt-1 mb-3 text-[14px] text-[#aeaeb2]">
        Confidence: <b className="text-content">{fmtPct(prediction.confidence)}</b>
      </div>
      <ProbabilityBar label="↑ Up" value={probs.up ?? 0} color="#30d158" />
      <ProbabilityBar label="→ Flat" value={probs.flat ?? 0} color="#ff9f0a" />
      <ProbabilityBar label="↓ Down" value={probs.down ?? 0} color="#ff453a" />
      {prediction.regime && (
        <div className="mt-2.5 border-t border-hairline pt-2.5 text-[12px] text-secondary">
          Market regime: <b className="font-medium text-body">{prediction.regime}</b>
        </div>
      )}
    </motion.div>
  );
}

export default memo(ForecastCard);
