import { createBrowserRouter, Navigate } from "react-router";
import AppLayout from "./components/layout/AppLayout";

// Route-level code splitting: each page ships as its own chunk.
// A redeploy re-hashes every chunk, so a tab opened before the deploy points at
// filenames that no longer exist → "Failed to fetch dynamically imported module".
// Recover by forcing a one-time full reload, which pulls the fresh index.html and
// the new chunk names. A sessionStorage guard keyed per-chunk prevents a reload
// loop when the failure is a genuine network/404 error rather than a stale hash.
const page = (loader: () => Promise<{ default: React.ComponentType }>) =>
  async () => {
    try {
      return { Component: (await loader()).default };
    } catch (err) {
      const key = `chunk-reload:${String(err)}`;
      if (!sessionStorage.getItem(key)) {
        sessionStorage.setItem(key, "1");
        window.location.reload();
        // Return a placeholder while the reload takes over.
        return { Component: () => null };
      }
      throw err;
    }
  };

export const router = createBrowserRouter([
  {
    path: "/",
    Component: AppLayout,
    children: [
      { index: true, element: <Navigate to="/chart" replace /> },
      { path: "chart", lazy: page(() => import("./pages/PriceChartPage")) },
      { path: "performance", lazy: page(() => import("./pages/PerformancePage")) },
      { path: "tickers", lazy: page(() => import("./pages/AllTickersPage")) },
      { path: "trades", lazy: page(() => import("./pages/TradesPage")) },
      { path: "track-record", lazy: page(() => import("./pages/TrackRecordPage")) },
      { path: "data-sources", lazy: page(() => import("./pages/DataSourcesPage")) },
      { path: "*", element: <Navigate to="/chart" replace /> },
    ],
  },
]);
