"""
Performance tracker — appends each pipeline run's metrics to a persistent JSON
ledger and generates aggregate summary reports.

One entry is written per ticker × horizon per run, capturing walk-forward
accuracy (the number to trust), single-split test accuracy, and the live
prediction made at the end of that run.  Over time the ledger lets you see
whether model quality is drifting and whether confidence scores are calibrated.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Appends walk-forward and live-prediction metrics to a persistent ledger."""

    def __init__(self, ledger_path: str = "results/performance_ledger.json"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict]:
        if self.ledger_path.exists():
            try:
                return json.loads(self.ledger_path.read_text())
            except Exception as exc:
                logger.warning(f"Could not parse ledger — starting fresh: {exc}")
        return []

    def _save(self, records: List[Dict]) -> None:
        self.ledger_path.write_text(json.dumps(records, indent=2, default=str))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_run(
        self,
        ticker: str,
        horizon: int,
        walk_forward: Dict,
        test_metrics: Dict,
        prediction: Dict,
    ) -> None:
        """Append one run entry to the ledger."""
        records = self._load()
        records.append({
            "timestamp": datetime.now().isoformat(),
            "ticker":    ticker,
            "horizon":   horizon,
            "walk_forward": {
                "mean_accuracy": walk_forward.get("mean_accuracy"),
                "std_accuracy":  walk_forward.get("std_accuracy"),
                "mean_f1":       walk_forward.get("mean_f1"),
                "n_folds":       walk_forward.get("n_folds"),
            },
            "test_accuracy": test_metrics.get("accuracy"),
            "test_f1":       test_metrics.get("f1"),
            "prediction": {
                "direction":  prediction.get("direction"),
                "confidence": prediction.get("confidence"),
                "regime":     prediction.get("regime"),
            },
        })
        self._save(records)
        logger.info(f"Recorded performance: {ticker} h{horizon}d")

    def record_batch(
        self,
        ticker: str,
        walk_forward_by_horizon: Dict[int, Dict],
        test_metrics_by_horizon: Dict[str, Dict],
        predictions_by_horizon: Dict[str, Dict],
    ) -> None:
        """Convenience wrapper: record all horizons for one ticker in one call."""
        for horizon, wf in walk_forward_by_horizon.items():
            h_str = str(horizon)
            self.record_run(
                ticker=ticker,
                horizon=horizon,
                walk_forward=wf,
                test_metrics=test_metrics_by_horizon.get(h_str, {}),
                prediction=predictions_by_horizon.get(h_str, {}),
            )

    def generate_report(self) -> Dict:
        """
        Summarize the ledger: per-ticker, per-horizon mean walk-forward accuracy
        across all recorded runs plus the most recent prediction.
        """
        records = self._load()
        if not records:
            return {"message": "No runs recorded yet.", "_meta": {}}

        # Group walk-forward accuracies by ticker → horizon
        wf_acc_by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        latest_pred: Dict[str, Dict[str, Dict]]       = defaultdict(dict)

        for r in records:
            t = r["ticker"]
            h = str(r["horizon"])
            wf = r.get("walk_forward", {})
            if wf.get("mean_accuracy") is not None:
                wf_acc_by[t][h].append(float(wf["mean_accuracy"]))
            if r.get("prediction"):
                latest_pred[t][h] = r["prediction"]

        report: Dict = {}
        for ticker, horizons in wf_acc_by.items():
            report[ticker] = {}
            for h, accs in horizons.items():
                report[ticker][h] = {
                    "runs":                  len(accs),
                    "mean_wf_accuracy":      round(float(sum(accs) / len(accs)), 4),
                    "latest_wf_accuracy":    round(float(accs[-1]), 4),
                    "latest_prediction":     latest_pred.get(ticker, {}).get(h),
                }

        report["_meta"] = {
            "total_entries": len(records),
            "generated":     datetime.now().isoformat(),
        }
        return report

    def print_report(self) -> None:
        """Log a human-readable summary of the ledger to the console."""
        report = self.generate_report()
        meta = report.pop("_meta", {})
        logger.info("=" * 55)
        logger.info(f"PERFORMANCE REPORT  ({meta.get('generated', '')[:19]})")
        logger.info(f"Total ledger entries: {meta.get('total_entries', 0)}")
        logger.info("=" * 55)
        for ticker, horizons in report.items():
            for h, stats in horizons.items():
                wf     = stats.get("mean_wf_accuracy", "n/a")
                latest = stats.get("latest_wf_accuracy", "n/a")
                pred   = stats.get("latest_prediction") or {}
                logger.info(
                    f"  {ticker} h{h:>2}d | WF acc mean={wf:.1%} latest={latest:.1%} | "
                    f"last pred={pred.get('direction','?')} "
                    f"conf={pred.get('confidence', 0):.1%} "
                    f"regime={pred.get('regime','?')}"
                )
        logger.info("=" * 55)
