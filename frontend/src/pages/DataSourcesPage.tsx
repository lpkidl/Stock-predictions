import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, useReducedMotion } from "framer-motion";
import { dataSourcesQuery } from "../api/queries";
import type { RecordedPost } from "../api/types";
import { useTicker } from "../hooks/useTicker";
import { useCountUp } from "../hooks/useCountUp";
import { fmtSignedPct1, fmtTimestamp } from "../lib/format";
import Badge from "../components/ui/Badge";
import Card from "../components/ui/Card";
import Metric from "../components/ui/Metric";
import SegmentedControl from "../components/ui/SegmentedControl";
import Skeleton from "../components/ui/Skeleton";

const SOURCE_LABELS: Record<string, string> = {
  news: "News",
  reddit: "Reddit",
  x: "X / Twitter",
  unknown: "Unknown",
};

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#30d158",
  negative: "#ff453a",
  neutral: "#8e8e93",
};

const feedStagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.022 } },
};
const rowVariant = {
  hidden: { opacity: 0, y: 8 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.28, ease: [0.25, 1, 0.5, 1] as const },
  },
};

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url.slice(0, 40);
  }
}

/** Metric whose numeric value counts up smoothly on change. */
function CountMetric({
  label,
  value,
  sub,
  title,
}: {
  label: string;
  value: number;
  sub?: string;
  title?: string;
}) {
  const n = useCountUp(value);
  return (
    <Metric
      label={label}
      value={n == null ? "—" : Math.round(n).toLocaleString()}
      sub={sub}
      title={title}
    />
  );
}

function PostRow({ post, reduced }: { post: RecordedPost; reduced: boolean | null }) {
  const color = SENTIMENT_COLORS[post.sentiment_label ?? "neutral"] ?? "#8e8e93";
  return (
    <motion.a
      variants={rowVariant}
      whileHover={reduced ? undefined : { y: -2 }}
      transition={{ type: "spring", stiffness: 400, damping: 30 }}
      href={post.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-3 rounded-xl border border-hairline bg-surface px-4 py-3 transition-[border-color,box-shadow] duration-200 hover:border-elev hover:shadow-[0_6px_20px_rgba(0,0,0,0.35)]"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <span className="mt-0.5 shrink-0 rounded-md bg-elev px-2 py-[2px] text-[12px] font-bold text-content">
        {post.ticker}
      </span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[14px] text-content">
          {post.title ?? "(untitled)"}
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[12px] text-tertiary">
          <span>{SOURCE_LABELS[post.source] ?? post.source}</span>
          <span className="text-secondary">{hostOf(post.url)}</span>
          {post.subreddit && <span>r/{post.subreddit}</span>}
          {post.posted_at && <span className="tnum">{fmtTimestamp(post.posted_at)}</span>}
        </div>
      </div>
      <div className="shrink-0 text-right">
        <div
          className="text-[12px] font-semibold uppercase tracking-wide"
          style={{ color }}
        >
          {post.sentiment_label ?? "—"}
        </div>
        <div className="tnum text-[12px] text-tertiary">
          {fmtSignedPct1(post.sentiment_score)}
        </div>
      </div>
    </motion.a>
  );
}

export default function DataSourcesPage() {
  const reduced = useReducedMotion();
  const [ticker] = useTicker();
  // Scope the feed: default to the header-selected ticker so picking "AAPL"
  // shows only Apple posts; the toggle lets the user widen back to All.
  const [showAll, setShowAll] = useState(false);
  const scoped = !showAll && !!ticker;
  const { data, isLoading } = useQuery(dataSourcesQuery(scoped ? ticker : undefined));

  if (isLoading) return <Skeleton className="h-[500px]" />;

  const rec = data?.recording;
  const hasData = !!rec && rec.total_posts > 0;

  const scopeOptions = ticker
    ? [
        { label: ticker, value: "ticker" },
        { label: "All Tickers", value: "all" },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-[24px] font-semibold tracking-[-0.015em]">
            Data Sources — Live Recording
          </h2>
          {rec?.db_enabled ? (
            <Badge color="#30d158">● Recording to database</Badge>
          ) : (
            <Badge color="#8e8e93">Database writes disabled</Badge>
          )}
        </div>
        <p className="mt-1 max-w-[75ch] text-[13px] text-secondary">
          Every pipeline run persists the individual posts it analyzed — with
          their source links — into the SQLite database (<code>results/stocks.db</code>,
          the <code>posts</code> table). Their daily-aggregated sentiment score,
          spread, and post-count feed directly into the ML model as features, so
          this feed is the raw signal behind each forecast.
        </p>
      </div>

      {!hasData ? (
        <Card>
          <p className="text-secondary">
            No posts recorded yet. Run the pipeline (<code>python main.py</code>)
            to fetch and store sentiment sources.
          </p>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <CountMetric
              label="Posts Recorded"
              value={rec.total_posts}
              title="Total rows in the posts table (deduplicated by URL)"
            />
            <CountMetric
              label="Sentiment Days"
              value={rec.sentiment_days}
              title="Distinct dates in the daily_sentiment index feeding the ML model"
            />
            <CountMetric
              label="Pipeline Runs"
              value={rec.runs}
              title="Recorded executions of the pipeline"
            />
            <Metric
              label="Last Recorded"
              value={rec.last_recorded ? fmtTimestamp(rec.last_recorded).slice(0, 10) : "—"}
              sub={rec.last_run?.status ? `run #${rec.last_run.id} · ${rec.last_run.status}` : undefined}
              title={rec.last_recorded ? `${fmtTimestamp(rec.last_recorded)} UTC` : undefined}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Card>
              <h3 className="text-[15px] font-semibold text-content">By Source</h3>
              <div className="mt-3 space-y-2">
                {data.by_source.map((s) => (
                  <div key={s.source} className="flex items-center justify-between text-[14px]">
                    <span className="text-body">{SOURCE_LABELS[s.source] ?? s.source}</span>
                    <span className="tnum font-semibold text-content">{s.count}</span>
                  </div>
                ))}
                {(["reddit", "x"] as const)
                  .filter((src) => !data.by_source.some((s) => s.source === src))
                  .map((src) => (
                    <div key={src} className="flex items-center justify-between text-[14px]">
                      <span className="text-tertiary">{SOURCE_LABELS[src]}</span>
                      <span className="tnum text-tertiary">0</span>
                    </div>
                  ))}
              </div>
            </Card>

            <Card>
              <h3 className="text-[15px] font-semibold text-content">By Sentiment</h3>
              <div className="mt-3 space-y-2">
                {data.by_sentiment
                  .slice()
                  .sort((a, b) => b.count - a.count)
                  .map((s) => (
                    <div key={s.label} className="flex items-center justify-between text-[14px]">
                      <span
                        className="font-medium"
                        style={{ color: SENTIMENT_COLORS[s.label] ?? "#8e8e93" }}
                      >
                        {s.label}
                      </span>
                      <span className="tnum font-semibold text-content">{s.count}</span>
                    </div>
                  ))}
              </div>
            </Card>

            <Card>
              <h3 className="text-[15px] font-semibold text-content">
                Avg Sentiment by Ticker
              </h3>
              <div className="mt-3 space-y-2">
                {data.by_ticker.map((t) => {
                  const active = scoped && t.ticker === ticker;
                  return (
                    <div
                      key={t.ticker}
                      className={`flex items-center justify-between rounded-md px-1.5 text-[14px] transition-colors ${
                        active ? "bg-elev/60" : ""
                      }`}
                    >
                      <span className={active ? "font-semibold text-content" : "text-body"}>
                        {t.ticker}{" "}
                        <span className="text-tertiary">({t.count})</span>
                      </span>
                      <span
                        className="tnum font-semibold"
                        style={{
                          color:
                            (t.avg_score ?? 0) > 0.05
                              ? "#30d158"
                              : (t.avg_score ?? 0) < -0.05
                                ? "#ff453a"
                                : "#8e8e93",
                        }}
                      >
                        {fmtSignedPct1(t.avg_score)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>

          <div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h3 className="text-[19px] font-semibold tracking-[-0.015em]">
                Recorded Sentiment Feed
              </h3>
              {scopeOptions.length > 0 && (
                <SegmentedControl
                  options={scopeOptions}
                  value={scoped ? "ticker" : "all"}
                  onChange={(v) => setShowAll(v === "all")}
                  layoutId="feed-scope"
                />
              )}
            </div>
            <p className="mt-1 mb-3 text-[13px] text-secondary">
              {scoped ? (
                <>
                  Showing the {data.posts.length} most recent <b className="text-body">{ticker}</b>{" "}
                  posts. Switch the header ticker or the toggle to change scope.
                </>
              ) : (
                <>Showing the {data.posts.length} most recent posts across all tickers.</>
              )}{" "}
              Each row links to its original source.
            </p>
            {data.posts.length === 0 ? (
              <Card>
                <p className="text-secondary">
                  No posts recorded for {ticker} in the latest run.
                </p>
              </Card>
            ) : (
              <motion.div
                className="space-y-2"
                variants={reduced ? undefined : feedStagger}
                initial="hidden"
                animate="show"
                key={scoped ? ticker : "all"}
              >
                {data.posts.map((p, i) => (
                  <PostRow key={`${p.url}-${i}`} post={p} reduced={reduced} />
                ))}
              </motion.div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
