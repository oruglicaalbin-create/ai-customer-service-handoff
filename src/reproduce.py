"""Reproduce every result in the README and report with one command. No API key, no cost.

Run: python src/reproduce.py

Steps (about 6-8 minutes the first time, mostly cross-validation):
  1. baseline.py            train TF-IDF + logistic regression, predict the test set
  2. evaluate.py            LLM vs baseline on the 154 test messages (reads saved LLM results)
  3. threshold.py           routing cost, optimal threshold, safety curve, charts
  4. calibration_check.py   how the threshold depends on regularisation C
  5. hybrid.py analyse      measured end-to-end hybrid costs (reads saved replies and verdicts)
  6. report_figures.py      extra figures for the report

The LLM runs themselves (run_agent.py, hybrid.py without "analyse") call the paid API;
their outputs are committed in results/, so they are not needed to reproduce the numbers.
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ["baseline.py"],
    ["evaluate.py"],
    ["threshold.py"],
    ["calibration_check.py"],
    ["hybrid.py", "analyse"],
    ["report_figures.py"],
]

for step in STEPS:
    start = time.perf_counter()
    print(f"\n=== {' '.join(step)} ===", flush=True)
    result = subprocess.run([sys.executable, str(SRC / step[0]), *step[1:]], cwd=SRC.parent,
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
        sys.exit(f"Step {' '.join(step)} failed.")
    print(f"ok ({time.perf_counter() - start:.0f} s)")

print("\nDone. Key numbers: results/threshold_metrics.json, results/hybrid_metrics.json, "
      "results/evaluation_metrics.json. Charts: results/*.png, report/figures/*.png")
