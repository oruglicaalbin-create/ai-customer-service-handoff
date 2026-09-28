"""Extra figures for the deep-dive report (report/). No API calls.

Run: python src/report_figures.py
Output: report/figures/reliability.png, report/figures/confidence_hist.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threshold import BLUE, GRAY, INK, ORANGE, SURFACE, style

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "report" / "figures"
BINS = [0, 0.5, 0.7, 0.9, 1.0]

baseline = pd.read_csv(RESULTS_DIR / "baseline_test_predictions.csv")
llm = pd.read_csv(RESULTS_DIR / "agent_test_2_per_intent.csv")


def reliability_points(conf: pd.Series, correct: pd.Series) -> pd.DataFrame:
    band = pd.cut(conf, BINS, include_lowest=True)
    return pd.DataFrame({"conf": conf, "correct": correct, "band": band}).groupby("band", observed=True).agg(
        mean_conf=("conf", "mean"), accuracy=("correct", "mean"), n=("correct", "size"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Reliability diagram -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--", label="Perfect calibration")
    for name, df, conf_col, color in [
        ("Baseline (3,080 test msgs)", baseline, "confidence", BLUE),
        ("LLM (154 test msgs)", llm, "confidence", ORANGE),
    ]:
        pts = reliability_points(df[conf_col], df["correct"].astype(bool))
        ax.plot(pts["mean_conf"], pts["accuracy"], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        for _, p in pts.iterrows():
            ax.annotate(f"n={int(p['n'])}", (p["mean_conf"], p["accuracy"]), xytext=(6, -12),
                        textcoords="offset points", fontsize=7, color=GRAY)
    ax.text(0.62, 0.36, "below the line:\noverconfident", color=GRAY, fontsize=8, ha="center")
    ax.text(0.30, 0.72, "above the line:\nunderconfident", color=GRAY, fontsize=8, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Average confidence in the band", color=INK)
    ax.set_ylabel("Share actually correct (accuracy)", color=INK)
    ax.set_title("Calibration: does confidence match accuracy?", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reliability.png", facecolor=SURFACE)
    plt.close(fig)

    # --- Distribution of confidence scores (small multiples, shared x) --------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150, facecolor=SURFACE, sharex=True)
    for ax in axes:
        style(ax)
    edges = np.linspace(0, 1, 21)
    axes[0].hist(baseline["confidence"], bins=edges, color=BLUE, edgecolor=SURFACE, linewidth=2)
    axes[0].set_title("Baseline: probability of top intent", color=INK, fontsize=10, loc="left")
    values = llm["confidence"].round(2).value_counts().sort_index()
    axes[1].bar(values.index, values.values, width=0.03, color=ORANGE, edgecolor=SURFACE, linewidth=1)
    axes[1].set_title("LLM: self-reported confidence", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Messages", color=INK)
    for ax in axes:
        ax.set_xlabel("Confidence", color=INK)
        ax.set_xlim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confidence_hist.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
