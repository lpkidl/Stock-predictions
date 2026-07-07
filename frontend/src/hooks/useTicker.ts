import { useCallback } from "react";
import { useSearchParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { configQuery } from "../api/queries";

/** Selected ticker, held in the ?t= search param so it survives reloads
 * and is shareable. Falls back to the first configured ticker. */
export function useTicker(): [string, (t: string) => void, string[]] {
  const { data: config } = useQuery(configQuery);
  const tickers = config?.tickers ?? [];
  const [params, setParams] = useSearchParams();

  const fromUrl = params.get("t");
  const ticker =
    fromUrl && tickers.includes(fromUrl) ? fromUrl : (tickers[0] ?? "");

  const setTicker = useCallback(
    (t: string) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("t", t);
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return [ticker, setTicker, tickers];
}
