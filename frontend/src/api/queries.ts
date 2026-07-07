import { queryOptions } from "@tanstack/react-query";
import { fetchJson } from "./client";
import type {
  AppConfig,
  FeaturesResponse,
  MetricsResponse,
  PredictionsResponse,
  PricesResponse,
  QuotesResponse,
  TrackRecordResponse,
  TradesResponse,
} from "./types";

export const configQuery = queryOptions({
  queryKey: ["config"],
  queryFn: () => fetchJson<AppConfig>("/api/config"),
  staleTime: Infinity, // static per server process
});

export const predictionsQuery = queryOptions({
  queryKey: ["predictions"],
  queryFn: () => fetchJson<PredictionsResponse>("/api/predictions"),
});

export const quotesQuery = queryOptions({
  queryKey: ["quotes"],
  queryFn: () => fetchJson<QuotesResponse>("/api/quotes"),
});

export const pricesQuery = (ticker: string, period: string) =>
  queryOptions({
    queryKey: ["prices", ticker, period],
    queryFn: () =>
      fetchJson<PricesResponse>(`/api/prices/${ticker}?period=${period}`),
    enabled: !!ticker,
  });

export const featuresQuery = (ticker: string, topN: number) =>
  queryOptions({
    queryKey: ["features", ticker, topN],
    queryFn: () =>
      fetchJson<FeaturesResponse>(`/api/features/${ticker}?top_n=${topN}`),
    enabled: !!ticker,
  });

export const metricsQuery = (ticker: string) =>
  queryOptions({
    queryKey: ["metrics", ticker],
    queryFn: () => fetchJson<MetricsResponse>(`/api/metrics/${ticker}`),
    enabled: !!ticker,
  });

export const tradesQuery = queryOptions({
  queryKey: ["trades"],
  queryFn: () => fetchJson<TradesResponse>("/api/trades"),
});

export const trackRecordQuery = queryOptions({
  queryKey: ["track-record"],
  queryFn: () => fetchJson<TrackRecordResponse>("/api/track-record"),
});
