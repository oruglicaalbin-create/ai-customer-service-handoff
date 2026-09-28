"""Step 7: choose the confidence threshold that minimises expected cost.

Decision rule for each message:
  - predicted intent is on the always-handoff list  -> human     (cost c2)
  - confidence >= threshold                          -> automated (cost 0 if right, c1 if wrong)
  - otherwise                                        -> human     (cost c2)
  Every automated answer also pays c_auto (API cost of writing the reply).

Theory: automating a message with confidence p costs (1-p)*c1 on average, a
human costs c2. So automate when (1-p)*c1 < c2, i.e. p > 1 - c2/c1.
If the confidence is well calibrated, the best threshold is 1 - c2/c1.

To avoid fooling ourselves, the baseline's threshold is chosen on held-out
("out-of-fold") predictions for the TRAIN set and then evaluated on the TEST set.
The LLM has only 154 test messages, so its threshold is chosen on those
(in-sample, optimistic).

No API calls. Run: python src/threshold.py
Output: results/threshold_metrics.json, results/cost_curve.png, results/sensitivity.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from baseline import make_model
from load_data import load_banking77
from policy import must_handoff

C1 = 30.0       # € cost of a wrong automated answer (assumption, agreed with the business)
C2 = 5.0        # € cost of a human handling the message (assumption)
C_AUTO = 0.01   # € API cost of an automated reply (measured ~$0.008 per message)
THRESHOLDS = np.round(np.arange(0, 1.0001, 0.005), 3)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CACHE_FILE = ROOT / "cache" / "baseline_oof_predictions.csv"

# Chart style (validated palette: blue / orange pass colour-blindness checks)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
INK, SURFACE, GRID = "#0b0b0b", "#fcfcfb", "#e6e5e1"


def cost_per_message(conf, correct, handoff_by_rule, t, c1=C1, c2=C2, c_auto=C_AUTO):
    """Average cost per message for threshold t (numpy arrays in, one number out)."""
    human = handoff_by_rule | (conf < t)
    cost = np.where(human, c2, np.where(correct, c_auto, c1 + c_auto))
    return cost.mean()


def best_threshold(conf, correct, handoff_by_rule, c1=C1, c2=C2):
    costs = np.array([cost_per_message(conf, correct, handoff_by_rule, t, c1, c2) for t in THRESHOLDS])
    i = int(costs.argmin())
    return THRESHOLDS[i], costs


def summary(conf, correct, handoff_by_rule, t):
    automated = ~(handoff_by_rule | (conf < t))
    return {
        "threshold": float(t),
        "cost_per_message_eur": round(float(cost_per_message(conf, correct, handoff_by_rule, t)), 3),
        "automation_rate": round(float(automated.mean()), 3),
        "error_rate_of_automated": round(float((~correct[automated]).mean()), 3) if automated.any() else None,
        "wrong_automated_per_1000_messages": round(float((automated & ~correct).mean() * 1000), 1),
    }


def arrays(df, conf_col, pred_col, correct_col):
    return (df[conf_col].to_numpy(), df[correct_col].to_numpy(bool),
            df[pred_col].map(must_handoff).to_numpy(bool))


def out_of_fold_baseline() -> pd.DataFrame:
    """Baseline predictions for each TRAIN message from a model that did not see it (5-fold CV)."""
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)
    train = load_banking77("train")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_model()
    proba = cross_val_predict(model, train["text"], train["intent"], cv=folds, method="predict_proba")
    classes = np.array(sorted(train["intent"].unique()))  # predict_proba columns are sorted classes
    pred = classes[proba.argmax(axis=1)]
    oof = pd.DataFrame({"true_intent": train["intent"], "predicted_intent": pred,
                        "confidence": proba.max(axis=1), "correct": pred == train["intent"]})
    CACHE_FILE.parent.mkdir(exist_ok=True)
    oof.to_csv(CACHE_FILE, index=False)
    return oof


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=GRAY)


def main():
    theory_t = 1 - C2 / C1
    print(f"c1 = €{C1:.0f}, c2 = €{C2:.0f}  ->  theoretical threshold 1 - c2/c1 = {theory_t:.3f}\n")

    # --- Baseline: choose on out-of-fold train predictions, evaluate on test ---
    print("Computing out-of-fold baseline predictions (5-fold CV, ~1 min the first time)...")
    oof = out_of_fold_baseline()
    base_t, _ = best_threshold(*arrays(oof, "confidence", "predicted_intent", "correct"))

    test = pd.read_csv(RESULTS_DIR / "baseline_test_predictions.csv")
    base_test = arrays(test, "confidence", "predicted_intent", "correct")
    _, base_curve = best_threshold(*base_test)

    # --- LLM: 154 test messages, in-sample ---
    llm = pd.read_csv(RESULTS_DIR / "agent_test_2_per_intent.csv")
    llm_arr = arrays(llm, "confidence", "intent", "correct")
    llm_t, llm_curve = best_threshold(*llm_arr)

    # --- Same 154 messages, both systems, for a like-for-like comparison ---
    same = llm[["message", "true_intent"]].merge(
        test.drop_duplicates(["text", "true_intent"]).rename(columns={"text": "message"}),
        on=["message", "true_intent"])
    base_same = arrays(same, "confidence", "predicted_intent", "correct")

    results = {
        "assumptions": {"c1_wrong_automated_eur": C1, "c2_human_eur": C2, "c_auto_eur": C_AUTO,
                        "theoretical_threshold": round(theory_t, 3)},
        "all_to_human_cost_per_message_eur": C2,
        "baseline_full_test_3080": {
            "chosen_on_train_out_of_fold": summary(*base_test, base_t),
            "at_theoretical_threshold": summary(*base_test, theory_t),
        },
        "llm_test_154_in_sample": {
            "best": summary(*llm_arr, llm_t),
            "at_theoretical_threshold": summary(*llm_arr, theory_t),
        },
        "baseline_same_154": summary(*base_same, base_t),
    }
    RESULTS_DIR.joinpath("threshold_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # --- Chart 1: cost curve -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.axhline(C2, color=GRAY, linewidth=1, linestyle=":")
    ax.text(0.01, C2 + 0.12, f"Everything to a human: €{C2:.2f}", color=GRAY, fontsize=8)
    ax.axvline(theory_t, color=GRAY, linewidth=1, linestyle="--")
    ax.text(theory_t - 0.01, 0.3, f"theory: 1 − c2/c1 = {theory_t:.2f}", color=GRAY, fontsize=8,
            ha="right")
    ax.plot(THRESHOLDS, base_curve, color=BLUE, linewidth=2, label="Baseline (TF-IDF), 3,080 test msgs")
    ax.plot(THRESHOLDS, llm_curve, color=ORANGE, linewidth=2, label="LLM (Claude Haiku), 154 test msgs")
    base_cost = cost_per_message(*base_test, base_t)
    llm_cost = llm_curve[list(THRESHOLDS).index(llm_t)]
    ax.plot(base_t, base_cost, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(llm_t, llm_cost, "o", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"chosen on training data: t = {base_t:.3f}, test cost €{base_cost:.2f}",
                (base_t, base_cost), xytext=(0, -18), textcoords="offset points", ha="center",
                fontsize=8, color=INK)
    ax.text(theory_t - 0.01, llm_cost - 0.6, f"LLM best: automate nothing (€{llm_cost:.2f}) →",
            ha="right", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(C2 + 1, float(llm_curve.max()) + 0.5))
    ax.set_xlabel("Confidence threshold (automate only above this)", color=INK)
    ax.set_ylabel("Expected cost per message (€)", color=INK)
    ax.set_title(f"Expected cost per message vs. threshold (c1 = €{C1:.0f} wrong answer, c2 = €{C2:.0f} human)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cost_curve.png", facecolor=SURFACE)
    plt.close(fig)

    # --- Chart 2: sensitivity of the optimal threshold to c1/c2 -------------
    ratios = np.geomspace(1.25, 50, 40)
    base_opt = [best_threshold(*arrays(oof, "confidence", "predicted_intent", "correct"), c1=r * C2)[0]
                for r in ratios]
    llm_opt = [best_threshold(*llm_arr, c1=r * C2)[0] for r in ratios]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.plot(ratios, 1 - 1 / ratios, color=GRAY, linewidth=2, linestyle="--", label="Theory: 1 − c2/c1")
    ax.plot(ratios, base_opt, color=BLUE, linewidth=2, label="Baseline (out-of-fold train data)")
    ax.plot(ratios, llm_opt, color=ORANGE, linewidth=2, drawstyle="steps-post",
            label="LLM (154 test msgs, noisy)")
    ax.axvline(C1 / C2, color=GRAY, linewidth=1, linestyle=":")
    ax.text(C1 / C2 * 1.04, 0.05, f"our case: c1/c2 = {C1 / C2:.0f}", color=GRAY, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1.25, 2, 5, 10, 20, 50], labels=["1.25", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cost ratio c1/c2 (wrong automated answer ÷ human handling)", color=INK)
    ax.set_ylabel("Cost-optimal threshold", color=INK)
    ax.set_title("How the optimal threshold moves with the cost ratio", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sensitivity.png", facecolor=SURFACE)
    plt.close(fig)

    print("\nSensitivity (baseline, out-of-fold): ratio -> optimal threshold")
    for r in [1.5, 2, 3, 6, 10, 20, 50]:
        t, _ = best_threshold(*arrays(oof, "confidence", "predicted_intent", "correct"), c1=r * C2)
        print(f"  c1/c2 = {r:>4}: t* = {t:.3f}   (theory {1 - 1 / r:.3f})")
    print("\nSaved results/threshold_metrics.json, results/cost_curve.png, results/sensitivity.png")


if __name__ == "__main__":
    main()
