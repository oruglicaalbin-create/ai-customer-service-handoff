"""Is the baseline's underconfidence a property of the data or of one setting?

Recomputes out-of-fold predictions for several regularisation strengths C and
reports accuracy, calibration and the cost-optimal threshold for each.
ECE (expected calibration error) = average gap between confidence and accuracy,
weighted by how many messages fall in each confidence band; 0 = perfectly calibrated.
Positive "confidence minus accuracy" = overconfident, negative = underconfident.

No API calls. Takes a few minutes (5-fold CV per setting; results are cached).
Run: python src/calibration_check.py
"""
import json

import numpy as np

from threshold import RESULTS_DIR, arrays, best_threshold, out_of_fold_baseline


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


rows = {}
for C in [1, 10, 100]:
    oof = out_of_fold_baseline(C=C)
    conf, correct = oof["confidence"].to_numpy(), oof["correct"].to_numpy(bool)
    t, _ = best_threshold(*arrays(oof, "confidence", "predicted_intent", "correct"))
    rows[f"C={C}"] = {
        "oof_accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "confidence_minus_accuracy": round(float(conf.mean() - correct.mean()), 4),
        "ece": round(float(ece(conf, correct)), 4),
        "cost_optimal_threshold": float(t),
    }
    print(f"C={C:>3}: {rows[f'C={C}']}")

(RESULTS_DIR / "calibration_check.json").write_text(json.dumps(rows, indent=2))
