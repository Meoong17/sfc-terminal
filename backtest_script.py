#!/usr/bin/env python3
"""
backtest_script.py — Monthly Backtest Report Generator
======================================================
Menghitung akurasi historis model SFC dan menghasilkan laporan
untuk backtest display dashboard.

Usage:
    python3 backtest_script.py              # Generate report
    python3 backtest_script.py --update     # Update + inject into data.json
"""

import json, os, sys, math
from datetime import datetime, timezone
from collections import defaultdict

DATA_FILE = os.path.join(os.path.dirname(__file__), "data_collection.json")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "backtest_report.json")


class ModelBacktester:
    """
    Hitung akurasi model SFC dari historical data.
    """

    def __init__(self, crash_threshold=0.25):
        """
        crash_threshold: SFC score above this is considered a "crash warning"
        """
        self.crash_threshold = crash_threshold
        self.results = []
        self.monthly = []

    def run(self, features, labels, dates=None):
        """
        Run backtest on historical data.

        features: list of feature vectors (method scores)
        labels: list of actual stress labels (0=no stress, 1=stress)
        dates: optional list of date strings
        """
        results = []
        n = min(len(features), len(labels))

        for i in range(n):
            obs = features[i]
            label = labels[i]

            if label is None:
                continue

            # Compute composite SFC score (same blend as collect.py)
            vals = [float(v) if v is not None else 0.5 for v in obs]
            if len(vals) >= 6:
                m1m6 = sum(vals[:6]) / 6
                m7m19 = sum(vals[6:19]) / min(13, max(len(vals)-6, 1)) if len(vals) > 6 else 0.5
                m20m31 = sum(vals[19:]) / max(len(vals[19:]), 1) if len(vals) > 19 else 0.5
                sfc = 0.86*m1m6 + 0.08*m7m19 + 0.06*m20m31
            else:
                sfc = 0.5

            predicted = 1 if sfc > self.crash_threshold else 0
            actual = int(float(label))

            results.append({
                'date': str(dates[i]) if dates and i < len(dates) else f"obs_{i}",
                'sfc_score': round(sfc, 4),
                'predicted': predicted,
                'actual': actual,
                'correct': predicted == actual,
                'tp': predicted == 1 and actual == 1,
                'fp': predicted == 1 and actual == 0,
                'tn': predicted == 0 and actual == 0,
                'fn': predicted == 0 and actual == 1,
            })

        self.results = results

        # Monthly aggregation
        monthly_map = defaultdict(list)
        for r in results:
            month_key = r['date'][:7] if len(r['date']) >= 7 else 'unknown'
            monthly_map[month_key].append(r)

        self.monthly = []
        for month_key in sorted(monthly_map.keys()):
            entries = monthly_map[month_key]
            total = len(entries)
            correct = sum(1 for e in entries if e['correct'])
            predicted = sum(1 for e in entries if e['predicted'] == 1)
            actual = sum(1 for e in entries if e['actual'] == 1)
            tp = sum(1 for e in entries if e['tp'])
            fp = sum(1 for e in entries if e['fp'])
            fn = sum(1 for e in entries if e['fn'])

            self.monthly.append({
                'month': month_key,
                'total': total,
                'correct': correct,
                'accuracy': round(correct / total * 100, 1) if total > 0 else 0,
                'predicted_crashes': predicted,
                'actual_crashes': actual,
                'tp': tp, 'fp': fp, 'fn': fn,
            })

        return self.get_report()

    def get_report(self):
        """Generate consolidated report."""
        if not self.results:
            return {"error": "No results. Run backtest first."}

        total = len(self.results)
        correct = sum(1 for r in self.results if r['correct'])
        tp = sum(1 for r in self.results if r['tp'])
        fp = sum(1 for r in self.results if r['fp'])
        tn = sum(1 for r in self.results if r['tn'])
        fn = sum(1 for r in self.results if r['fn'])

        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
        recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
        f1 = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0

        # Recent accuracy (last 30 and 90)
        n = len(self.results)
        last_30 = self.results[-30:] if n >= 30 else self.results
        last_90 = self.results[-90:] if n >= 90 else self.results
        acc_30 = round(sum(1 for r in last_30 if r['correct']) / len(last_30) * 100, 1) if last_30 else 0
        acc_90 = round(sum(1 for r in last_90 if r['correct']) / len(last_90) * 100, 1) if last_90 else 0

        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_observations': total,
            'overall_accuracy': accuracy,
            'accuracy_30d': acc_30,
            'accuracy_90d': acc_90,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'monthly': self.monthly,
        }
        return report

    def save(self, path=None):
        """Save report to JSON file."""
        path = path or REPORT_FILE
        report = self.get_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Backtest report saved to {path}")
        print(f"   Overall Accuracy: {report['overall_accuracy']}%")
        print(f"   Precision: {report['precision']}% | Recall: {report['recall']}% | F1: {report['f1_score']}")
        print(f"   TP:{report['true_positives']} FP:{report['false_positives']} TN:{report['true_negatives']} FN:{report['false_negatives']}")
        return report


if __name__ == "__main__":
    print("═" * 50)
    print("  SFC BACKTEST REPORT GENERATOR")
    print("═" * 50)

    if not os.path.exists(DATA_FILE):
        print(f"✗ data_collection.json not found at {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE) as f:
        data = json.load(f)

    features = data.get("features", [])
    labels = data.get("labels", [])
    dates = data.get("dates", [])

    print(f"\nLoaded {len(features)} observations, {len(labels)} labels")
    print(f"Label distribution: {sum(1 for l in labels if l == 1)} stress / {sum(1 for l in labels if l == 0)} normal")

    backtester = ModelBacktester(crash_threshold=0.25)
    report = backtester.run(features, labels, dates)
    backtester.save()

    print(f"\nMonthly Breakdown:")
    for m in report.get('monthly', []):
        bar = '█' * int(m['accuracy'] / 5) + '░' * (20 - int(m['accuracy'] / 5))
        print(f"  {m['month']}: {m['accuracy']:5.1f}% {bar} ({m['correct']}/{m['total']})")
