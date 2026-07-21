import { memo } from "react";
import { NavLink, useLocation } from "react-router";
import { motion, useReducedMotion } from "framer-motion";

const TABS = [
  { to: "/chart", label: "Price Chart" },
  { to: "/performance", label: "Model Performance" },
  { to: "/tickers", label: "All Tickers" },
  { to: "/trades", label: "Trade Execution" },
  { to: "/track-record", label: "Track Record" },
  { to: "/data-sources", label: "Data Sources" },
];

/** Segmented-control page navigation; the selected ticker (?t=) is carried
 * across tabs so switching pages never loses context. */
function NavTabs() {
  const reduced = useReducedMotion();
  const { pathname, search } = useLocation();

  return (
    <nav
      className="inline-flex max-w-full gap-[2px] overflow-x-auto rounded-xl border border-hairline bg-surface p-[3px]"
      aria-label="Sections"
    >
      {TABS.map((tab) => {
        const selected = pathname.startsWith(tab.to);
        return (
          <NavLink
            key={tab.to}
            to={{ pathname: tab.to, search }}
            className={`relative rounded-[9px] px-[18px] py-1.5 text-[14px] font-medium whitespace-nowrap transition-colors duration-150 ${
              selected ? "text-white" : "text-secondary hover:text-body"
            }`}
          >
            {selected && (
              <motion.span
                layoutId="nav-thumb"
                className="absolute inset-0 rounded-[9px] bg-elev shadow-[0_1px_3px_rgba(0,0,0,0.3)]"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 40 }
                }
              />
            )}
            <span className="relative z-10">{tab.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}

export default memo(NavTabs);
