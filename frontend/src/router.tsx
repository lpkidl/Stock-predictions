import { createBrowserRouter, Navigate } from "react-router";
import AppLayout from "./components/layout/AppLayout";

// Route-level code splitting: each page ships as its own chunk.
const page = (loader: () => Promise<{ default: React.ComponentType }>) =>
  async () => ({ Component: (await loader()).default });

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
      { path: "*", element: <Navigate to="/chart" replace /> },
    ],
  },
]);
