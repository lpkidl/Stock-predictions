import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";

interface Option {
  label: string;
  value: string;
}

interface Props {
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  /** Distinguishes multiple controls' shared-layout animations. */
  layoutId: string;
}

/** iOS-style segmented control with a sliding selection thumb. */
function SegmentedControl({ options, value, onChange, layoutId }: Props) {
  const reduced = useReducedMotion();
  return (
    <div
      role="tablist"
      className="inline-flex gap-[2px] rounded-xl border border-hairline bg-surface p-[3px]"
    >
      {options.map((opt) => {
        const selected = opt.value === value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(opt.value)}
            className={`relative rounded-[9px] px-4 py-1.5 text-[14px] font-medium transition-colors duration-150 ${
              selected ? "text-white" : "text-secondary hover:text-body"
            }`}
          >
            {selected && (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-0 rounded-[9px] bg-elev shadow-[0_1px_3px_rgba(0,0,0,0.3)]"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 40 }
                }
              />
            )}
            <span className="relative z-10">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export default memo(SegmentedControl);
