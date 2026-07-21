import { Suspense, useMemo } from "react";
import { Outlet, useLocation } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { predictionsQuery, quotesQuery } from "../../api/queries";
import { useTicker } from "../../hooks/useTicker";
import { HORIZON_LABELS } from "../../lib/constants";
import ForecastCard from "../forecast/ForecastCard";
import HeroCard from "../forecast/HeroCard";
import TickerStatsPanel from "../forecast/TickerStatsPanel";
import Skeleton from "../ui/Skeleton";
import GlassHeader from "./GlassHeader";
import NavTabs from "./NavTabs";
import TickerTabs from "./TickerTabs";

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04 } },
};

export default function AppLayout() {
  const reduced = useReducedMotion();
  const { pathname } = useLocation();

  const [ticker] = useTicker();
  const { data: predictions, isLoading: predsLoading } =
    useQuery(predictionsQuery);
  const { data: quotes } = useQuery(quotesQuery);

  const tickerPred = predictions?.[ticker];
  const horizons = useMemo(() => {
    const h = tickerPred?.horizons ?? {};
    return Object.keys(h)
      .sort((a, b) => Number(a) - Number(b))
      .map((key) => ({ key, pred: h[key] }));
  }, [tickerPred]);

  const price = quotes?.[ticker]?.price ?? null;

  return (
    <div className="min-h-screen">
      <GlassHeader />

      <main className="mx-auto max-w-[1100px] px-6 pb-16">
        <section className="pt-8">
          <TickerTabs />
        </section>

        <section className="flex flex-col gap-4 pt-6 lg:flex-row lg:items-stretch">
          <HeroCard
            ticker={ticker || "—"}
            price={price}
            timestamp={tickerPred?.timestamp ?? null}
          />
          <TickerStatsPanel className="flex-1" />
        </section>

        <section className="mt-8" aria-label="Forecasts">
          {predsLoading ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-[280px]" />
              ))}
            </div>
          ) : horizons.length > 0 ? (
            <motion.div
              className="grid grid-cols-1 gap-4 md:grid-cols-2"
              variants={reduced ? undefined : stagger}
              initial="hidden"
              animate="show"
              key={ticker}
            >
              {horizons.map(({ key, pred }) => (
                <ForecastCard
                  key={key}
                  label={HORIZON_LABELS[key] ?? `${key} days`}
                  prediction={pred}
                />
              ))}
            </motion.div>
          ) : (
            <div className="rounded-card border border-hairline bg-surface p-6 text-secondary">
              No forecast data — run <code className="text-body">python main.py</code>{" "}
              first.
            </div>
          )}
        </section>

        <section className="mt-10">
          <NavTabs />
        </section>

        <section className="mt-6">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={pathname}
              initial={reduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduced ? undefined : { opacity: 0, y: -4 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <Suspense fallback={<Skeleton className="h-[400px]" />}>
                <Outlet />
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </section>

        <footer className="mt-16 border-t border-hairline pt-6 text-[13px] text-tertiary">
          Stock Forecaster · 32-feature XGBoost+LR ensemble · Ternary
          classification · Regime-aware models · Re-run{" "}
          <code>python main.py</code> to refresh forecasts.
        </footer>
      </main>
    </div>
  );
}
