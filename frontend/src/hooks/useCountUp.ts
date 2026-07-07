import { useEffect, useRef, useState } from "react";
import { animate, useReducedMotion } from "framer-motion";

/** Animates a number toward `target` with a swift ease-out.
 * Renders instantly when reduced motion is preferred or on first paint of 0. */
export function useCountUp(target: number | null | undefined, duration = 0.6) {
  const reduced = useReducedMotion();
  const [display, setDisplay] = useState<number | null>(target ?? null);
  const prev = useRef<number | null>(null);

  useEffect(() => {
    if (target == null) {
      setDisplay(null);
      return;
    }
    if (reduced || prev.current === null) {
      prev.current = target;
      setDisplay(target);
      return;
    }
    const controls = animate(prev.current, target, {
      duration,
      ease: [0.25, 1, 0.5, 1],
      onUpdate: (v) => setDisplay(v),
    });
    prev.current = target;
    return () => controls.stop();
  }, [target, reduced, duration]);

  return display;
}
